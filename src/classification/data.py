import json
import os
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


MNIST_CLASS_NAMES = tuple(str(index) for index in range(10))
MNIST_TRAIN_SIZE = 60_000


def _dataset_config(config):
    dataset_config = config.get("dataset", {})
    if not isinstance(dataset_config, dict):
        raise TypeError(
            "Classification dataset config must be a mapping, for example "
            "dataset: {name: MNIST}."
        )
    name = str(dataset_config.get("name", "MNIST")).upper()
    if name != "MNIST":
        raise ValueError(f"Classification dataset '{name}' is not supported.")
    return dataset_config


def build_mnist_transform(config):
    dataset_config = _dataset_config(config)
    input_size = int(dataset_config.get("input_size", 224))
    if input_size < 63:
        raise ValueError("AlexNet requires dataset.input_size to be at least 63.")
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,) * 3, (0.3081,) * 3),
        ]
    )


def _mnist_root(config, project_root):
    dataset_config = _dataset_config(config)
    root = os.path.expanduser(dataset_config.get("root", "data"))
    if not os.path.isabs(root):
        root = os.path.join(project_root, root)
    return root


def _mnist_split_dir(config, project_root):
    split_dir = _dataset_config(config).get("split_dir")
    if split_dir is None:
        return None
    split_dir = os.path.expanduser(split_dir)
    if not os.path.isabs(split_dir):
        split_dir = os.path.join(project_root, split_dir)
    return split_dir


def _ensure_mnist_split(config, project_root, num_clients):
    """Auto-create a configured split only when its directory is absent/empty."""
    dataset_config = _dataset_config(config)
    split_dir = _mnist_split_dir(config, project_root)
    if split_dir is None:
        return

    required_shards = [
        os.path.join(split_dir, f"client_{client_id}_indices.pt")
        for client_id in range(1, num_clients + 1)
    ]
    if all(os.path.isfile(path) for path in required_shards):
        return

    split_path = Path(split_dir)
    directory_is_empty = split_path.is_dir() and not any(split_path.iterdir())
    directory_is_absent = not split_path.exists()
    if not bool(dataset_config.get("auto_create_split", True)):
        raise FileNotFoundError(
            f"MNIST split is missing under {split_dir}. Run "
            "scripts/split_mnist_dirichlet.py or enable dataset.auto_create_split."
        )
    if not directory_is_absent and not directory_is_empty:
        missing = [path for path in required_shards if not os.path.isfile(path)]
        raise FileNotFoundError(
            f"MNIST split directory is incomplete; missing {missing}. "
            "Remove/fix the directory or regenerate it with --overwrite."
        )

    from scripts.split_mnist_dirichlet import create_mnist_split

    mode = str(dataset_config.get("split_mode", "dirichlet")).lower()
    alpha = float(dataset_config.get("dirichlet_alpha", 0.5))
    seed = int(dataset_config.get("split_seed", 42))
    subset_fraction = float(dataset_config.get("subset_fraction", 1.0))
    print(
        f"MNIST split not found. Creating {mode} split for {num_clients} "
        f"clients at {split_dir} (alpha={alpha}, seed={seed})."
    )
    create_mnist_split(
        root=Path(_mnist_root(config, project_root)),
        output_dir=split_path,
        num_clients=num_clients,
        mode=mode,
        alpha=alpha,
        seed=seed,
        subset_fraction=subset_fraction,
        download=bool(dataset_config.get("download", True)),
        overwrite=False,
    )


def _load_client_split_indices(
    config, project_root, client_index, num_clients, dataset_size
):
    dataset_config = _dataset_config(config)
    split_dir = _mnist_split_dir(config, project_root)
    if split_dir is None:
        return None

    metadata_path = os.path.join(split_dir, "split_metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        configured_clients = int(metadata.get("num_clients", num_clients))
        if configured_clients != num_clients:
            raise ValueError(
                f"MNIST split contains {configured_clients} clients, but the "
                f"SFL config expects {num_clients}."
            )

    shard_path = os.path.join(
        split_dir, f"client_{client_index + 1}_indices.pt"
    )
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(f"MNIST client shard does not exist: {shard_path}")
    indices = torch.load(shard_path, map_location="cpu", weights_only=True)
    return _validate_indices(indices, dataset_size, shard_path)


def _validate_indices(indices, dataset_size, source):
    indices = torch.as_tensor(indices, dtype=torch.int64).flatten()
    if indices.numel() == 0:
        raise ValueError(f"MNIST client shard is empty: {source}")
    if indices.min() < 0 or indices.max() >= dataset_size:
        raise ValueError(f"MNIST client shard contains an invalid index: {source}")
    if indices.unique().numel() != indices.numel():
        raise ValueError(f"MNIST client shard contains duplicate indices: {source}")
    return indices.tolist()


def build_mnist_client_descriptors(config, project_root, num_clients):
    """Read precomputed shards for transmission in per-client start messages."""
    dataset_config = _dataset_config(config)
    if dataset_config.get("split_dir") is None:
        return [
            {
                "client_index": client_index,
                "num_clients": num_clients,
                "partition": "generated_iid",
            }
            for client_index in range(num_clients)
        ]

    _ensure_mnist_split(config, project_root, num_clients)

    descriptors = []
    assigned_indices = set()
    for client_index in range(num_clients):
        indices = _load_client_split_indices(
            config,
            project_root,
            client_index,
            num_clients,
            MNIST_TRAIN_SIZE,
        )
        overlap = assigned_indices.intersection(indices)
        if overlap:
            raise ValueError(
                f"MNIST split assigns {len(overlap)} samples to multiple clients."
            )
        assigned_indices.update(indices)

        split_dir = _mnist_split_dir(config, project_root)
        client_metadata_path = os.path.join(
            split_dir, f"client_{client_index + 1}.json"
        )
        client_metadata = {}
        if os.path.isfile(client_metadata_path):
            with open(client_metadata_path, encoding="utf-8") as metadata_file:
                client_metadata = json.load(metadata_file)
        descriptors.append(
            {
                "client_index": client_index,
                "num_clients": num_clients,
                "partition": "precomputed",
                "indices": indices,
                "num_samples": len(indices),
                "class_counts": client_metadata.get("class_counts", {}),
            }
        )
    return descriptors


def build_mnist_client_dataset(
    config,
    project_root,
    client_index,
    num_clients,
    dataset_descriptor=None,
):
    """Build one deterministic IID shard of the official MNIST train split."""
    if isinstance(client_index, bool) or not isinstance(client_index, int):
        raise TypeError("client_index must be an integer.")
    if not 0 <= client_index < num_clients:
        raise ValueError(
            f"client_index must be between 0 and {num_clients - 1}; "
            f"received {client_index}."
        )

    dataset_config = _dataset_config(config)
    full_dataset = datasets.MNIST(
        root=_mnist_root(config, project_root),
        train=True,
        download=bool(dataset_config.get("download", True)),
        transform=build_mnist_transform(config),
    )
    split_indices = None
    if dataset_descriptor and dataset_descriptor.get("indices") is not None:
        if int(dataset_descriptor.get("client_index", -1)) != client_index:
            raise ValueError("Received an MNIST shard for a different client.")
        if int(dataset_descriptor.get("num_clients", -1)) != num_clients:
            raise ValueError("Received an MNIST shard for a different client group.")
        split_indices = _validate_indices(
            dataset_descriptor["indices"],
            len(full_dataset),
            f"RabbitMQ descriptor for client_{client_index + 1}",
        )
    elif dataset_config.get("split_dir") is not None:
        # Backward-compatible local loading if an older coordinator only sends
        # the client index instead of transmitting the shard indices.
        split_indices = _load_client_split_indices(
            config, project_root, client_index, num_clients, len(full_dataset)
        )
    if split_indices is not None:
        return Subset(full_dataset, split_indices)

    generator = torch.Generator().manual_seed(
        int(dataset_config.get("split_seed", 42))
    )
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()

    subset_fraction = float(dataset_config.get("subset_fraction", 1.0))
    if not 0 < subset_fraction <= 1:
        raise ValueError("dataset.subset_fraction must be in the interval (0, 1].")
    indices = indices[: max(1, int(len(indices) * subset_fraction))]

    base_size, remainder = divmod(len(indices), num_clients)
    start = client_index * base_size + min(client_index, remainder)
    length = base_size + (client_index < remainder)
    return Subset(full_dataset, indices[start:start + length])


def build_mnist_validation_dataset(config, project_root):
    """Build the official MNIST test split used as global validation data."""
    dataset_config = _dataset_config(config)
    dataset = datasets.MNIST(
        root=_mnist_root(config, project_root),
        train=False,
        download=bool(dataset_config.get("download", True)),
        transform=build_mnist_transform(config),
    )
    max_samples = dataset_config.get("validation_max_samples")
    if max_samples is None:
        return dataset
    max_samples = int(max_samples)
    if max_samples < 1:
        raise ValueError("dataset.validation_max_samples must be positive.")
    return Subset(dataset, range(min(max_samples, len(dataset))))
