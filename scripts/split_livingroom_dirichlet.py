"""Split a YOLO detection dataset into client datasets.

The split is performed at image level, so an image and its label file always
stay together and cannot be assigned to more than one client. The validation
split is copied unchanged to every client. Both the conventional
``train/images`` layout and the converted Pascal VOC layout
``images/{train2007,train2012,...}`` are supported.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "split_livingroom_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "split_livingroom_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VOC_DEFAULT_TRAIN_SPLITS = (
    "train2007",
    "val2007",
    "train2012",
    "val2012",
)
VOC_DEFAULT_VAL_SPLITS = ("test2007",)


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    class_ids: tuple[int, ...]

    @property
    def classes(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.class_ids)))


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("alpha must be a positive number") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("alpha must be greater than zero")
    return number


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def find_dataset_yaml(source_dir: Path) -> Path | None:
    for candidate in (
        source_dir.with_suffix(".yaml"),
        source_dir.with_suffix(".yml"),
        source_dir / "dataset.yaml",
        source_dir / "data.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_class_names(source_dir: Path, observed_class_ids: Iterable[int]) -> list[str]:
    yaml_path = find_dataset_yaml(source_dir)
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path else {}
    config = config or {}
    names = config.get("names")

    if isinstance(names, dict):
        numeric_names = {}
        for key, value in names.items():
            numeric_names[int(key)] = str(value)
        class_count = max(
            int(config.get("nc", 0)),
            max(numeric_names, default=-1) + 1,
            max(observed_class_ids, default=-1) + 1,
        )
        return [numeric_names.get(index, f"class_{index}") for index in range(class_count)]

    if isinstance(names, list):
        class_count = max(len(names), int(config.get("nc", 0)), max(observed_class_ids, default=-1) + 1)
        return [str(names[index]) if index < len(names) else f"class_{index}" for index in range(class_count)]

    class_count = max(int(config.get("nc", 0)), max(observed_class_ids, default=-1) + 1)
    return [f"class_{index}" for index in range(class_count)]


def read_label_classes(label_path: Path) -> tuple[int, ...]:
    class_ids = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        try:
            class_id = int(fields[0])
        except ValueError as error:
            raise ValueError(f"Invalid class id at {label_path}:{line_number}") from error
        if class_id < 0:
            raise ValueError(f"Class id must be non-negative at {label_path}:{line_number}")
        class_ids.append(class_id)
    return tuple(class_ids)


def normalize_source_dir(source_dir: Path) -> Path:
    """Accept either the dataset root or its ``images`` directory."""
    source_dir = source_dir.resolve()
    looks_like_voc_images = any(
        (source_dir / split_name).is_dir()
        for split_name in (
            *VOC_DEFAULT_TRAIN_SPLITS,
            *VOC_DEFAULT_VAL_SPLITS,
        )
    )
    if (
        source_dir.name == "images"
        and (
            (source_dir.parent / "labels").is_dir()
            or looks_like_voc_images
        )
    ):
        return source_dir.parent
    return source_dir


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _validate_unique_names(paths: list[Path], description: str) -> None:
    names = [path.name for path in paths]
    duplicates = sorted(
        name for name, count in Counter(names).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Duplicate {description} filenames across splits: {duplicates[:5]}. "
            "Client output uses a flat image directory, so filenames must be unique."
        )


def _load_flat_yolo_samples(
    source_dir: Path,
) -> tuple[list[Sample], list[Path], list[Path], list[str]]:
    train_images_dir = source_dir / "train" / "images"
    train_labels_dir = source_dir / "train" / "labels"

    validation_name = "valid" if (source_dir / "valid").is_dir() else "val"
    validation_dir = source_dir / validation_name
    validation_images_dir = validation_dir / "images"
    validation_labels_dir = validation_dir / "labels"
    if not validation_images_dir.is_dir() or not validation_labels_dir.is_dir():
        raise FileNotFoundError("Expected valid/images and valid/labels, or val/images and val/labels")

    train_images = _image_files(train_images_dir)
    if not train_images:
        raise FileNotFoundError("No train images were found")

    samples = []
    for image_path in train_images:
        label_path = train_labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for image: {image_path}")
        samples.append(Sample(image_path, label_path, read_label_classes(label_path)))

    validation_images = _image_files(validation_images_dir)
    validation_labels = []
    for image_path in validation_images:
        label_path = validation_labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing validation label for image: {image_path}")
        validation_labels.append(label_path)

    observed_class_ids = {class_id for sample in samples for class_id in sample.class_ids}
    observed_class_ids.update(
        class_id
        for label_path in validation_labels
        for class_id in read_label_classes(label_path)
    )
    class_names = load_class_names(source_dir, observed_class_ids)
    max_class_id = len(class_names) - 1
    invalid_ids = sorted(class_id for class_id in observed_class_ids if class_id > max_class_id)
    if invalid_ids:
        raise ValueError(f"Class ids {invalid_ids} do not fit the dataset class names")

    return samples, validation_images, validation_labels, class_names


def _resolve_voc_split_dir(root: Path, kind: str, split_name: str) -> Path:
    split_path = Path(split_name)
    candidates = []
    if split_path.is_absolute():
        candidates.append(split_path)
    else:
        candidates.extend(
            [
                root / split_path,
                root / kind / split_path,
            ]
        )
        if split_path.parts and split_path.parts[0] in ("images", "labels"):
            candidates.insert(0, root / split_path)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve VOC {kind} split '{split_name}' under {root}."
    )


def _load_voc_samples(
    source_dir: Path,
    train_splits: tuple[str, ...],
    val_splits: tuple[str, ...],
) -> tuple[list[Sample], list[Path], list[Path], list[str]]:
    images_root = source_dir / "images"
    labels_root = source_dir / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(
            "VOC layout requires sibling images/ and labels/ directories. "
            "Run the VOC XML-to-YOLO conversion before splitting."
        )

    train_images = []
    validation_images = []
    for split_name in train_splits:
        train_images.extend(
            _image_files(_resolve_voc_split_dir(source_dir, "images", split_name))
        )
    for split_name in val_splits:
        validation_images.extend(
            _image_files(_resolve_voc_split_dir(source_dir, "images", split_name))
        )
    if not train_images:
        raise FileNotFoundError("No VOC training images were found.")
    if not validation_images:
        raise FileNotFoundError("No VOC validation images were found.")
    _validate_unique_names(train_images, "training image")
    _validate_unique_names(validation_images, "validation image")

    def label_for_image(image_path: Path) -> Path:
        relative_split = image_path.parent.relative_to(images_root)
        label_path = labels_root / relative_split / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing VOC YOLO label for image: {image_path}")
        return label_path

    samples = [
        Sample(image_path, label_for_image(image_path), read_label_classes(label_for_image(image_path)))
        for image_path in train_images
    ]
    validation_labels = [label_for_image(image_path) for image_path in validation_images]
    observed_class_ids = {
        class_id for sample in samples for class_id in sample.class_ids
    }
    observed_class_ids.update(
        class_id
        for label_path in validation_labels
        for class_id in read_label_classes(label_path)
    )
    class_names = load_class_names(source_dir, observed_class_ids)
    invalid_ids = sorted(
        class_id
        for class_id in observed_class_ids
        if class_id >= len(class_names)
    )
    if invalid_ids:
        raise ValueError(f"VOC class ids do not fit class names: {invalid_ids}.")
    return samples, validation_images, validation_labels, class_names


def load_samples(
    source_dir: Path,
    train_splits: tuple[str, ...] | None = None,
    val_splits: tuple[str, ...] | None = None,
) -> tuple[list[Sample], list[Path], list[Path], list[str]]:
    source_dir = normalize_source_dir(source_dir)
    is_flat_yolo = (
        (source_dir / "train" / "images").is_dir()
        and (source_dir / "train" / "labels").is_dir()
    )
    if is_flat_yolo:
        if train_splits is not None or val_splits is not None:
            raise ValueError(
                "--train-splits/--val-splits are only valid for VOC layout."
            )
        return _load_flat_yolo_samples(source_dir)

    return _load_voc_samples(
        source_dir,
        train_splits or VOC_DEFAULT_TRAIN_SPLITS,
        val_splits or VOC_DEFAULT_VAL_SPLITS,
    )


def iid_partition(sample_count: int, num_clients: int, rng: np.random.Generator) -> list[list[int]]:
    shuffled_indices = rng.permutation(sample_count)
    return [chunk.tolist() for chunk in np.array_split(shuffled_indices, num_clients)]


def dirichlet_partition(
    samples: list[Sample], num_clients: int, alpha: float, rng: np.random.Generator
) -> list[list[int]]:
    class_frequency = Counter(class_id for sample in samples for class_id in sample.classes)
    classes = sorted(class_frequency)
    class_client_probabilities = {
        class_id: rng.dirichlet(np.full(num_clients, alpha, dtype=float)) for class_id in classes
    }

    primary_classes = [
        min(sample.classes, key=lambda class_id: (class_frequency[class_id], class_id))
        if sample.classes
        else None
        for sample in samples
    ]

    partitions = [[] for _ in range(num_clients)]
    for sample_index in rng.permutation(len(samples)):
        primary_class = primary_classes[sample_index]
        if primary_class is None:
            client_id = int(rng.integers(num_clients))
        else:
            probabilities = class_client_probabilities[primary_class]
            client_id = int(rng.choice(num_clients, p=probabilities))
        partitions[client_id].append(sample_index)

    for client_id, partition in enumerate(partitions):
        if partition:
            continue
        donor_id = max(range(num_clients), key=lambda candidate: len(partitions[candidate]))
        if len(partitions[donor_id]) <= 1:
            raise ValueError("The Dirichlet split could not give every client at least one image")
        partition.append(partitions[donor_id].pop())

    return partitions


def class_counts(
    partitions: list[list[int]], samples: list[Sample], class_count: int
) -> np.ndarray:
    counts = np.zeros((len(partitions), class_count), dtype=int)
    for client_id, sample_indices in enumerate(partitions):
        for sample_index in sample_indices:
            for class_id in samples[sample_index].class_ids:
                counts[client_id, class_id] += 1
    return counts


def ensure_output_is_safe(source_dir: Path, output_dir: Path) -> None:
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()
    try:
        output_resolved.relative_to(source_resolved)
    except ValueError:
        return
    raise ValueError("The output directory must not be inside the source dataset")


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already contains files: {output_dir}; use --overwrite")
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_files(files: Iterable[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_path in files:
        shutil.copy2(source_path, destination_dir / source_path.name)


def write_client_yaml(output_dir: Path, client_dir: Path, class_names: list[str]) -> Path:
    yaml_path = output_dir / f"{client_dir.name}.yaml"
    config = {
        "train": f"{client_dir.name}/train/images",
        "val": f"{client_dir.name}/valid/images",
        "nc": len(class_names),
        "names": {index: name for index, name in enumerate(class_names)},
        "kpt_shape": [0, 0],
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def write_distribution_files(
    output_dir: Path,
    partitions: list[list[int]],
    samples: list[Sample],
    class_names: list[str],
    counts: np.ndarray,
) -> None:
    fieldnames = ["client_id", "num_images", *class_names]
    with (output_dir / "class_distribution_counts.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for client_id, sample_indices in enumerate(partitions, start=1):
            row = {"client_id": f"client_{client_id}", "num_images": len(sample_indices)}
            row.update({name: int(counts[client_id - 1, index]) for index, name in enumerate(class_names)})
            writer.writerow(row)

    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["client_id", "image", "label", "class_ids"])
        for client_id, sample_indices in enumerate(partitions, start=1):
            for sample_index in sample_indices:
                sample = samples[sample_index]
                writer.writerow([client_id, sample.image.name, sample.label.name, " ".join(map(str, sample.class_ids))])


def save_heatmap(output_dir: Path, counts: np.ndarray, class_names: list[str], mode: str, alpha: float) -> None:
    figure_width = max(13, len(class_names) * 0.65)
    figure, axis = plt.subplots(figsize=(figure_width, 5.8))
    image = axis.imshow(counts, aspect="auto", cmap="YlOrRd")
    axis.set_xlabel("Class")
    axis.set_ylabel("Client")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=65, ha="right")
    axis.set_yticks(np.arange(counts.shape[0]))
    axis.set_yticklabels([f"client_{index}" for index in range(1, counts.shape[0] + 1)])
    axis.set_title(f"Train label distribution: {mode}, alpha={alpha}")
    figure.colorbar(image, ax=axis, label="Number of objects")

    max_count = counts.max(initial=0)
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            value = counts[row, column]
            text_color = "white" if value > max_count * 0.55 else "black"
            axis.text(column, row, str(value), ha="center", va="center", color=text_color, fontsize=7)

    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def copy_client_dataset(
    client_dir: Path,
    sample_indices: list[int],
    samples: list[Sample],
    validation_images: list[Path],
    validation_labels: list[Path],
) -> None:
    train_images_dir = client_dir / "train" / "images"
    train_labels_dir = client_dir / "train" / "labels"
    for sample_index in sample_indices:
        sample = samples[sample_index]
        train_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image, train_images_dir / sample.image.name)
        shutil.copy2(sample.label, train_labels_dir / sample.label.name)

    copy_files(validation_images, client_dir / "valid" / "images")
    copy_files(validation_labels, client_dir / "valid" / "labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/livingroom_2"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--alpha", type=positive_float, default=0.5)
    parser.add_argument("--mode", choices=("iid", "dirichlet"), default="dirichlet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=None,
        help="VOC image split directories used for training.",
    )
    parser.add_argument(
        "--val-splits",
        nargs="+",
        default=None,
        help="VOC image split directories copied as shared validation data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_clients < 1:
        raise ValueError("num-clients must be at least one")

    source_dir = normalize_source_dir(args.source)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")

    output_dir = args.output
    if output_dir is None:
        output_dir = source_dir.parent / "subdataset" / (
            f"{source_dir.name}_{args.mode}_alpha_{alpha_tag(args.alpha)}"
        )
    output_dir = output_dir.resolve()
    ensure_output_is_safe(source_dir, output_dir)

    samples, validation_images, validation_labels, class_names = load_samples(
        source_dir,
        train_splits=tuple(args.train_splits) if args.train_splits else None,
        val_splits=tuple(args.val_splits) if args.val_splits else None,
    )
    rng = np.random.default_rng(args.seed)
    if args.mode == "iid":
        partitions = iid_partition(len(samples), args.num_clients, rng)
    else:
        partitions = dirichlet_partition(samples, args.num_clients, args.alpha, rng)
    counts = class_counts(partitions, samples, len(class_names))

    prepare_output(output_dir, args.overwrite)
    yaml_paths = []
    for client_id, sample_indices in enumerate(partitions, start=1):
        client_dir = output_dir / f"client_{client_id}"
        copy_client_dataset(client_dir, sample_indices, samples, validation_images, validation_labels)
        yaml_paths.append(write_client_yaml(output_dir, client_dir, class_names))

    write_distribution_files(output_dir, partitions, samples, class_names, counts)
    save_heatmap(output_dir, counts, class_names, args.mode, args.alpha)
    metadata = {
        "source": str(source_dir),
        "mode": args.mode,
        "alpha": args.alpha,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "train_splits": args.train_splits,
        "val_splits": args.val_splits,
        "train_images": len(samples),
        "validation_images_per_client": len(validation_images),
        "client_train_images": [len(indices) for indices in partitions],
        "yaml_files": [path.name for path in yaml_paths],
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Created {args.num_clients} clients in {output_dir}")
    print(f"Train images per client: {[len(indices) for indices in partitions]}")
    print(f"Validation images copied per client: {len(validation_images)}")
    print(f"Heatmap: {output_dir / 'class_distribution_heatmap.png'}")
    print(f"YAML files: {', '.join(path.name for path in yaml_paths)}")


if __name__ == "__main__":
    main()
