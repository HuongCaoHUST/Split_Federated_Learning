"""Dataset registry and shard loading for classification split learning."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


MNIST_CLASS_NAMES = tuple(str(index) for index in range(10))
CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
MNIST_TRAIN_SIZE = 60_000
CIFAR10_TRAIN_SIZE = 50_000


@dataclass(frozen=True)
class ClassificationDatasetSpec:
    name: str
    slug: str
    dataset_class: type
    class_names: tuple[str, ...]
    train_size: int
    validation_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    split_script: str
    split_module: str
    split_factory: str


DATASET_REGISTRY = {
    "MNIST": ClassificationDatasetSpec(
        name="MNIST",
        slug="mnist",
        dataset_class=datasets.MNIST,
        class_names=MNIST_CLASS_NAMES,
        train_size=MNIST_TRAIN_SIZE,
        validation_size=10_000,
        mean=(0.1307,),
        std=(0.3081,),
        split_script="scripts/split_mnist_dirichlet.py",
        split_module="scripts.split_mnist_dirichlet",
        split_factory="create_mnist_split",
    ),
    "CIFAR10": ClassificationDatasetSpec(
        name="CIFAR10",
        slug="cifar10",
        dataset_class=datasets.CIFAR10,
        class_names=CIFAR10_CLASS_NAMES,
        train_size=CIFAR10_TRAIN_SIZE,
        validation_size=10_000,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        split_script="scripts/split_cifar10_dirichlet.py",
        split_module="scripts.split_cifar10_dirichlet",
        split_factory="create_cifar10_split",
    ),
}


def _normalize_dataset_name(name) -> str:
    normalized = str(name).strip().upper().replace("-", "").replace("_", "")
    aliases = {"MNIST": "MNIST", "CIFAR10": "CIFAR10"}
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(DATASET_REGISTRY)
        raise ValueError(
            f"Classification dataset '{name}' is not supported; choose {supported}."
        ) from exc


def _dataset_config(config):
    dataset_config = config.get("dataset", {})
    if not isinstance(dataset_config, dict):
        raise TypeError(
            "Classification dataset config must be a mapping, for example "
            "dataset: {name: CIFAR10}."
        )
    _normalize_dataset_name(dataset_config.get("name", "MNIST"))
    return dataset_config


def get_dataset_spec(config) -> ClassificationDatasetSpec:
    dataset_config = _dataset_config(config)
    return DATASET_REGISTRY[
        _normalize_dataset_name(dataset_config.get("name", "MNIST"))
    ]


def get_dataset_name(config) -> str:
    return get_dataset_spec(config).name


def get_class_names(config) -> tuple[str, ...]:
    return get_dataset_spec(config).class_names


def build_classification_transform(config, train=False):
    """Build preprocessing from dataset config without assuming a model type."""
    dataset_config = _dataset_config(config)
    spec = get_dataset_spec(config)
    input_size = int(dataset_config.get("input_size", 224))
    if input_size < 1:
        raise ValueError("dataset.input_size must be positive.")

    operations = []
    if spec.name == "CIFAR10" and train and bool(
        dataset_config.get("augmentation", False)
    ):
        operations.extend(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
    operations.append(transforms.Resize((input_size, input_size)))

    channels = int(dataset_config.get("channels", 3))
    if spec.name == "MNIST":
        if channels not in (1, 3):
            raise ValueError("MNIST dataset.channels must be 1 or 3.")
        operations.append(transforms.Grayscale(num_output_channels=channels))
        mean = spec.mean * channels
        std = spec.std * channels
    else:
        if channels != 3:
            raise ValueError("CIFAR10 dataset.channels must be 3.")
        mean, std = spec.mean, spec.std

    operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(operations)


def _dataset_root(config, project_root):
    dataset_config = _dataset_config(config)
    root = os.path.expanduser(dataset_config.get("root", "data"))
    if not os.path.isabs(root):
        root = os.path.join(project_root, root)
    return root


def _alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def _split_dir(config, project_root):
    dataset_config = _dataset_config(config)
    split_dir = dataset_config.get("split_dir")
    if split_dir is None:
        return None
    if str(split_dir).strip().lower() == "auto":
        spec = get_dataset_spec(config)
        mode = str(dataset_config.get("split_mode", "dirichlet")).lower()
        alpha = float(dataset_config.get("dirichlet_alpha", 0.5))
        split_dir = os.path.join(
            _dataset_root(config, project_root),
            f"{spec.slug}_splits",
            f"{spec.slug}_{mode}_alpha_{_alpha_tag(alpha)}",
        )
    split_dir = os.path.expanduser(str(split_dir))
    if not os.path.isabs(split_dir):
        split_dir = os.path.join(project_root, split_dir)
    return split_dir


def _get_split_factory(spec: ClassificationDatasetSpec) -> Callable:
    module = importlib.import_module(spec.split_module)
    try:
        return getattr(module, spec.split_factory)
    except AttributeError as exc:
        raise ValueError(
            f"Dataset {spec.name} registered missing split factory "
            f"{spec.split_module}.{spec.split_factory}."
        ) from exc


def _ensure_classification_split(config, project_root, num_clients):
    """Auto-create a configured split only when its directory is absent/empty."""
    dataset_config = _dataset_config(config)
    spec = get_dataset_spec(config)
    split_dir = _split_dir(config, project_root)
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
            f"{spec.name} split is missing under {split_dir}. Run "
            f"{spec.split_script} or enable dataset.auto_create_split."
        )
    if not directory_is_absent and not directory_is_empty:
        missing = [path for path in required_shards if not os.path.isfile(path)]
        raise FileNotFoundError(
            f"{spec.name} split directory is incomplete; missing {missing}. "
            "Remove/fix the directory or regenerate it with --overwrite."
        )

    mode = str(dataset_config.get("split_mode", "dirichlet")).lower()
    alpha = float(dataset_config.get("dirichlet_alpha", 0.5))
    seed = int(dataset_config.get("split_seed", 42))
    subset_fraction = float(dataset_config.get("subset_fraction", 1.0))
    print(
        f"{spec.name} split not found. Creating {mode} split for {num_clients} "
        f"clients at {split_dir} (alpha={alpha}, seed={seed})."
    )
    _get_split_factory(spec)(
        root=Path(_dataset_root(config, project_root)),
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
    spec = get_dataset_spec(config)
    split_dir = _split_dir(config, project_root)
    if split_dir is None:
        return None

    metadata_path = os.path.join(split_dir, "split_metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        metadata_dataset = metadata.get("dataset")
        if metadata_dataset and _normalize_dataset_name(metadata_dataset) != spec.name:
            raise ValueError(
                f"Split under {split_dir} belongs to {metadata_dataset}, not {spec.name}."
            )
        configured_clients = int(metadata.get("num_clients", num_clients))
        if configured_clients != num_clients:
            raise ValueError(
                f"{spec.name} split contains {configured_clients} clients, but the "
                f"SFL config expects {num_clients}."
            )

    shard_path = os.path.join(split_dir, f"client_{client_index + 1}_indices.pt")
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(
            f"{spec.name} client shard does not exist: {shard_path}"
        )
    indices = torch.load(shard_path, map_location="cpu", weights_only=True)
    return _validate_indices(indices, dataset_size, shard_path, spec.name)


def _validate_indices(indices, dataset_size, source, dataset_name="classification"):
    indices = torch.as_tensor(indices, dtype=torch.int64).flatten()
    if indices.numel() == 0:
        raise ValueError(f"{dataset_name} client shard is empty: {source}")
    if indices.min() < 0 or indices.max() >= dataset_size:
        raise ValueError(
            f"{dataset_name} client shard contains an invalid index: {source}"
        )
    if indices.unique().numel() != indices.numel():
        raise ValueError(
            f"{dataset_name} client shard contains duplicate indices: {source}"
        )
    return indices.tolist()


def build_client_descriptors(config, project_root, num_clients):
    """Read precomputed shards for transmission in per-client start messages."""
    dataset_config = _dataset_config(config)
    spec = get_dataset_spec(config)
    if dataset_config.get("split_dir") is None:
        return [
            {
                "client_index": client_index,
                "num_clients": num_clients,
                "dataset": spec.name,
                "partition": "generated_iid",
            }
            for client_index in range(num_clients)
        ]

    _ensure_classification_split(config, project_root, num_clients)
    descriptors = []
    assigned_indices = set()
    for client_index in range(num_clients):
        indices = _load_client_split_indices(
            config, project_root, client_index, num_clients, spec.train_size
        )
        overlap = assigned_indices.intersection(indices)
        if overlap:
            raise ValueError(
                f"{spec.name} split assigns {len(overlap)} samples to multiple clients."
            )
        assigned_indices.update(indices)

        split_dir = _split_dir(config, project_root)
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
                "dataset": spec.name,
                "partition": "precomputed",
                "indices": indices,
                "num_samples": len(indices),
                "class_counts": client_metadata.get("class_counts", {}),
            }
        )
    return descriptors


def _build_torchvision_dataset(config, project_root, train):
    dataset_config = _dataset_config(config)
    spec = get_dataset_spec(config)
    return spec.dataset_class(
        root=_dataset_root(config, project_root),
        train=train,
        download=bool(dataset_config.get("download", True)),
        transform=build_classification_transform(config, train=train),
    )


def build_client_dataset(
    config,
    project_root,
    client_index,
    num_clients,
    dataset_descriptor=None,
):
    """Build one client shard of the configured classification train set."""
    if isinstance(client_index, bool) or not isinstance(client_index, int):
        raise TypeError("client_index must be an integer.")
    if not 0 <= client_index < num_clients:
        raise ValueError(
            f"client_index must be between 0 and {num_clients - 1}; "
            f"received {client_index}."
        )

    dataset_config = _dataset_config(config)
    spec = get_dataset_spec(config)
    full_dataset = _build_torchvision_dataset(config, project_root, train=True)
    split_indices = None
    if dataset_descriptor and dataset_descriptor.get("indices") is not None:
        descriptor_dataset = dataset_descriptor.get("dataset")
        if (
            descriptor_dataset
            and _normalize_dataset_name(descriptor_dataset) != spec.name
        ):
            raise ValueError(
                f"Received a {descriptor_dataset} shard while training {spec.name}."
            )
        if int(dataset_descriptor.get("client_index", -1)) != client_index:
            raise ValueError("Received a shard for a different client.")
        if int(dataset_descriptor.get("num_clients", -1)) != num_clients:
            raise ValueError("Received a shard for a different client group.")
        split_indices = _validate_indices(
            dataset_descriptor["indices"],
            len(full_dataset),
            f"RabbitMQ descriptor for client_{client_index + 1}",
            spec.name,
        )
    elif dataset_config.get("split_dir") is not None:
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


def build_validation_dataset(config, project_root):
    """Build the official test split used as shared global validation data."""
    dataset_config = _dataset_config(config)
    dataset = _build_torchvision_dataset(config, project_root, train=False)
    max_samples = dataset_config.get("validation_max_samples")
    if max_samples is None:
        return dataset
    max_samples = int(max_samples)
    if max_samples < 1:
        raise ValueError("dataset.validation_max_samples must be positive.")
    return Subset(dataset, range(min(max_samples, len(dataset))))


# Backward-compatible MNIST API used by existing callers and external scripts.
def build_mnist_transform(config):
    return build_classification_transform(config, train=False)


def _mnist_root(config, project_root):
    return _dataset_root(config, project_root)


def _mnist_split_dir(config, project_root):
    return _split_dir(config, project_root)


def _ensure_mnist_split(config, project_root, num_clients):
    return _ensure_classification_split(config, project_root, num_clients)


def build_mnist_client_descriptors(config, project_root, num_clients):
    return build_client_descriptors(config, project_root, num_clients)


def build_mnist_client_dataset(
    config, project_root, client_index, num_clients, dataset_descriptor=None
):
    return build_client_dataset(
        config,
        project_root,
        client_index,
        num_clients,
        dataset_descriptor=dataset_descriptor,
    )


def build_mnist_validation_dataset(config, project_root):
    return build_validation_dataset(config, project_root)
