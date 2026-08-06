"""Split the MNIST train set into reproducible client index shards.

The script supports balanced IID splitting and label-skewed Dirichlet
splitting. It stores dataset indices rather than copying images, and writes
per-client class counts, ratios, a sample manifest, and a distribution heatmap.
The official MNIST test set remains shared global validation data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "split_mnist_matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "split_mnist_cache")
)
import numpy as np
import torch
from torchvision.datasets import MNIST

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_NAMES = tuple(str(class_id) for class_id in range(10))


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive number") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def fraction(value: str) -> float:
    number = positive_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError("subset-fraction must not exceed one")
    return number


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def iid_partition(
    sample_indices: np.ndarray,
    num_clients: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    shuffled = rng.permutation(sample_indices)
    return [chunk.astype(int).tolist() for chunk in np.array_split(shuffled, num_clients)]


def dirichlet_partition(
    sample_indices: np.ndarray,
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Allocate every class independently using Dirichlet client proportions."""
    partitions = [[] for _ in range(num_clients)]
    for class_id in range(len(CLASS_NAMES)):
        class_indices = sample_indices[targets[sample_indices] == class_id]
        class_indices = rng.permutation(class_indices)
        proportions = rng.dirichlet(
            np.full(num_clients, alpha, dtype=np.float64)
        )
        client_sizes = rng.multinomial(len(class_indices), proportions)
        boundaries = np.cumsum(client_sizes)[:-1]
        for client_id, chunk in enumerate(np.split(class_indices, boundaries)):
            partitions[client_id].extend(chunk.astype(int).tolist())

    for partition in partitions:
        rng.shuffle(partition)
    _ensure_nonempty_partitions(partitions, rng)
    return partitions


def _ensure_nonempty_partitions(
    partitions: list[list[int]], rng: np.random.Generator
) -> None:
    for partition in partitions:
        if partition:
            continue
        donor = max(partitions, key=len)
        if len(donor) <= 1:
            raise ValueError("Cannot give every client at least one MNIST sample.")
        partition.append(donor.pop(int(rng.integers(len(donor)))))


def class_counts(
    partitions: list[list[int]], targets: np.ndarray
) -> np.ndarray:
    counts = np.zeros((len(partitions), len(CLASS_NAMES)), dtype=np.int64)
    for client_id, indices in enumerate(partitions):
        if indices:
            counts[client_id] = np.bincount(
                targets[np.asarray(indices, dtype=np.int64)],
                minlength=len(CLASS_NAMES),
            )
    return counts


def validate_partitions(
    partitions: list[list[int]], selected_indices: np.ndarray
) -> None:
    flattened = [index for partition in partitions for index in partition]
    if len(flattened) != len(selected_indices):
        raise ValueError("The split lost or duplicated the number of selected samples.")
    if len(set(flattened)) != len(flattened):
        raise ValueError("A sample was assigned to more than one client.")
    if set(flattened) != set(selected_indices.astype(int).tolist()):
        raise ValueError("The split indices do not match the selected MNIST subset.")
    if any(not partition for partition in partitions):
        raise ValueError("Every client must receive at least one sample.")


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already contains files: {output_dir}; "
                "use --overwrite"
            )
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_index_shards(
    output_dir: Path,
    partitions: list[list[int]],
    counts: np.ndarray,
) -> list[Path]:
    shard_paths = []
    for client_id, indices in enumerate(partitions, start=1):
        shard_path = output_dir / f"client_{client_id}_indices.pt"
        torch.save(torch.tensor(indices, dtype=torch.int64), shard_path)
        shard_paths.append(shard_path)
        client_metadata = {
            "client_id": client_id,
            "num_samples": len(indices),
            "class_counts": {
                class_name: int(counts[client_id - 1, class_index])
                for class_index, class_name in enumerate(CLASS_NAMES)
            },
        }
        (output_dir / f"client_{client_id}.json").write_text(
            json.dumps(client_metadata, indent=2), encoding="utf-8"
        )
    return shard_paths


def write_distribution_files(
    output_dir: Path,
    partitions: list[list[int]],
    targets: np.ndarray,
    counts: np.ndarray,
) -> None:
    fieldnames = ["client_id", "num_samples", *CLASS_NAMES]
    with (output_dir / "class_distribution_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for client_id, indices in enumerate(partitions, start=1):
            row = {"client_id": f"client_{client_id}", "num_samples": len(indices)}
            row.update(
                {
                    class_name: int(counts[client_id - 1, class_index])
                    for class_index, class_name in enumerate(CLASS_NAMES)
                }
            )
            writer.writerow(row)

    with (output_dir / "class_distribution_ratios.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for client_id, indices in enumerate(partitions, start=1):
            denominator = max(len(indices), 1)
            row = {"client_id": f"client_{client_id}", "num_samples": len(indices)}
            row.update(
                {
                    class_name: float(counts[client_id - 1, class_index] / denominator)
                    for class_index, class_name in enumerate(CLASS_NAMES)
                }
            )
            writer.writerow(row)

    with (output_dir / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["client_id", "dataset_index", "class_id"])
        for client_id, indices in enumerate(partitions, start=1):
            for dataset_index in indices:
                writer.writerow(
                    [client_id, dataset_index, int(targets[dataset_index])]
                )


def save_heatmap(
    output_dir: Path,
    counts: np.ndarray,
    mode: str,
    alpha: float,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, max(4.8, counts.shape[0] * 0.7)))
    image = axis.imshow(counts, aspect="auto", cmap="YlOrRd")
    axis.set_xlabel("MNIST class")
    axis.set_ylabel("Client")
    axis.set_xticks(np.arange(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES)
    axis.set_yticks(np.arange(counts.shape[0]))
    axis.set_yticklabels(
        [f"client_{client_id}" for client_id in range(1, counts.shape[0] + 1)]
    )
    title = f"MNIST train distribution: {mode}"
    if mode == "dirichlet":
        title += f", alpha={alpha}"
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Number of samples")

    max_count = counts.max(initial=0)
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            value = counts[row, column]
            color = "white" if max_count and value > max_count * 0.55 else "black"
            axis.text(
                column, row, str(value), ha="center", va="center", color=color
            )
    figure.tight_layout()
    figure.savefig(
        output_dir / "class_distribution_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--mode", choices=("iid", "dirichlet"), default="dirichlet")
    parser.add_argument("--alpha", type=positive_float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-fraction", type=fraction, default=1.0)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def create_mnist_split(
    root: Path,
    output_dir: Path,
    num_clients: int = 4,
    mode: str = "dirichlet",
    alpha: float = 0.5,
    seed: int = 42,
    subset_fraction: float = 1.0,
    download: bool = True,
    overwrite: bool = False,
) -> dict:
    """Create an MNIST split programmatically and return its metadata."""
    if num_clients < 1:
        raise ValueError("num-clients must be at least one.")
    if mode not in ("iid", "dirichlet"):
        raise ValueError("mode must be 'iid' or 'dirichlet'.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if not 0 < subset_fraction <= 1:
        raise ValueError("subset_fraction must be in the interval (0, 1].")

    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    train_dataset = MNIST(
        root=root,
        train=True,
        download=download,
    )
    validation_dataset = MNIST(
        root=root,
        train=False,
        download=download,
    )
    targets = np.asarray(train_dataset.targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected_count = max(1, int(len(train_dataset) * subset_fraction))
    if selected_count < num_clients:
        raise ValueError("The selected subset has fewer samples than clients.")
    selected_indices = rng.permutation(len(train_dataset))[:selected_count]

    if mode == "iid":
        partitions = iid_partition(selected_indices, num_clients, rng)
    else:
        partitions = dirichlet_partition(
            selected_indices, targets, num_clients, alpha, rng
        )
    validate_partitions(partitions, selected_indices)
    counts = class_counts(partitions, targets)

    prepare_output(output_dir, overwrite)
    shard_paths = write_index_shards(output_dir, partitions, counts)
    write_distribution_files(output_dir, partitions, targets, counts)
    save_heatmap(output_dir, counts, mode, alpha)

    metadata = {
        "dataset": "MNIST",
        "root": str(root),
        "mode": mode,
        "alpha": alpha,
        "seed": seed,
        "num_clients": num_clients,
        "subset_fraction": subset_fraction,
        "selected_train_samples": selected_count,
        "validation_samples": len(validation_dataset),
        "client_train_samples": [len(indices) for indices in partitions],
        "class_names": list(CLASS_NAMES),
        "index_files": [path.name for path in shard_paths],
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output
    if output_dir is None:
        output_dir = root / "mnist_splits" / (
            f"mnist_{args.mode}_alpha_{alpha_tag(args.alpha)}"
        )
    metadata = create_mnist_split(
        root=root,
        output_dir=output_dir,
        num_clients=args.num_clients,
        mode=args.mode,
        alpha=args.alpha,
        seed=args.seed,
        subset_fraction=args.subset_fraction,
        download=not args.no_download,
        overwrite=args.overwrite,
    )

    print(f"Created {args.num_clients} MNIST clients in {Path(output_dir).resolve()}")
    for client_id, sample_count in enumerate(
        metadata["client_train_samples"], start=1
    ):
        client_metadata = json.loads(
            (Path(output_dir).resolve() / f"client_{client_id}.json").read_text(
                encoding="utf-8"
            )
        )
        distribution = ", ".join(
            f"{class_name}:{count}"
            for class_name, count in client_metadata["class_counts"].items()
        )
        print(f"client_{client_id}: {sample_count} samples ({distribution})")
    print(f"Shared validation samples: {metadata['validation_samples']}")
    print(f"Heatmap: {output_dir / 'class_distribution_heatmap.png'}")
    print("Set dataset.split_dir in config_alexnet_mnist.yaml to this output path.")


if __name__ == "__main__":
    main()
