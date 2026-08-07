"""Split the CIFAR-10 train set into reproducible client index shards.

The script supports balanced IID and label-skewed Dirichlet splitting. It
stores dataset indices rather than copying images. The official CIFAR-10 test
set remains shared global validation data.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "split_cifar10_matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "split_cifar10_cache")
)

import numpy as np
from torchvision.datasets import CIFAR10

if __package__:
    from .split_mnist_dirichlet import (
        alpha_tag,
        class_counts,
        dirichlet_partition,
        fraction,
        iid_partition,
        positive_float,
        prepare_output,
        save_heatmap,
        validate_partitions,
        write_distribution_files,
        write_index_shards,
    )
else:
    from split_mnist_dirichlet import (
        alpha_tag,
        class_counts,
        dirichlet_partition,
        fraction,
        iid_partition,
        positive_float,
        prepare_output,
        save_heatmap,
        validate_partitions,
        write_distribution_files,
        write_index_shards,
    )


CLASS_NAMES = (
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


def create_cifar10_split(
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
    """Create a CIFAR-10 split programmatically and return its metadata."""
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
    train_dataset = CIFAR10(root=root, train=True, download=download)
    validation_dataset = CIFAR10(root=root, train=False, download=download)
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
            selected_indices,
            targets,
            num_clients,
            alpha,
            rng,
            class_names=CLASS_NAMES,
            dataset_name="CIFAR10",
        )
    validate_partitions(partitions, selected_indices, dataset_name="CIFAR10")
    counts = class_counts(partitions, targets, class_names=CLASS_NAMES)

    prepare_output(output_dir, overwrite)
    shard_paths = write_index_shards(
        output_dir, partitions, counts, class_names=CLASS_NAMES
    )
    write_distribution_files(
        output_dir, partitions, targets, counts, class_names=CLASS_NAMES
    )
    save_heatmap(
        output_dir,
        counts,
        mode,
        alpha,
        class_names=CLASS_NAMES,
        dataset_name="CIFAR10",
    )

    metadata = {
        "dataset": "CIFAR10",
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
        output_dir = root / "cifar10_splits" / (
            f"cifar10_{args.mode}_alpha_{alpha_tag(args.alpha)}"
        )
    metadata = create_cifar10_split(
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

    print(f"Created {args.num_clients} CIFAR10 clients in {Path(output_dir).resolve()}")
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
    print("Set dataset.name=CIFAR10 and dataset.split_dir to this output path.")


if __name__ == "__main__":
    main()
