import json

import numpy as np
import torch

import scripts.split_mnist_dirichlet as split_mnist
from scripts.split_mnist_dirichlet import (
    class_counts,
    dirichlet_partition,
    iid_partition,
    validate_partitions,
    write_index_shards,
)
from src.classification.data import (
    _load_client_split_indices,
    build_mnist_client_descriptors,
)


def synthetic_targets(samples_per_class=20):
    return np.repeat(np.arange(10, dtype=np.int64), samples_per_class)


def test_iid_partition_is_complete_disjoint_and_balanced():
    targets = synthetic_targets()
    indices = np.arange(len(targets))
    partitions = iid_partition(indices, 4, np.random.default_rng(42))

    validate_partitions(partitions, indices)
    assert [len(partition) for partition in partitions] == [50, 50, 50, 50]
    assert class_counts(partitions, targets).sum() == len(targets)


def test_dirichlet_partition_is_complete_disjoint_and_nonempty():
    targets = synthetic_targets()
    indices = np.arange(len(targets))
    partitions = dirichlet_partition(
        indices,
        targets,
        num_clients=4,
        alpha=0.1,
        rng=np.random.default_rng(7),
    )

    validate_partitions(partitions, indices)
    counts = class_counts(partitions, targets)
    assert counts.sum() == len(targets)
    assert np.all(counts.sum(axis=1) > 0)
    assert np.any(counts == 0)


def test_generated_index_shards_are_loadable_by_classification_data(tmp_path):
    targets = synthetic_targets(samples_per_class=2)
    indices = np.arange(len(targets))
    partitions = iid_partition(indices, 2, np.random.default_rng(3))
    counts = class_counts(partitions, targets)
    write_index_shards(tmp_path, partitions, counts)
    (tmp_path / "split_metadata.json").write_text(
        json.dumps({"num_clients": 2}), encoding="utf-8"
    )
    config = {"dataset": {"name": "MNIST", "split_dir": str(tmp_path)}}

    loaded = _load_client_split_indices(
        config,
        project_root=str(tmp_path),
        client_index=1,
        num_clients=2,
        dataset_size=len(targets),
    )

    assert loaded == partitions[1]
    assert torch.equal(
        torch.load(tmp_path / "client_2_indices.pt", weights_only=True),
        torch.tensor(partitions[1]),
    )

    descriptors = build_mnist_client_descriptors(
        config, project_root=str(tmp_path), num_clients=2
    )
    assert descriptors[0]["partition"] == "precomputed"
    assert descriptors[0]["indices"] == partitions[0]
    assert descriptors[0]["num_samples"] == len(partitions[0])
    assert descriptors[0]["class_counts"] == {
        str(class_id): int(counts[0, class_id]) for class_id in range(10)
    }
    assert set(descriptors[0]["indices"]).isdisjoint(descriptors[1]["indices"])


def test_missing_split_is_created_by_coordinator(monkeypatch, tmp_path):
    split_dir = tmp_path / "auto_split"
    calls = []

    def fake_create_mnist_split(**kwargs):
        calls.append(kwargs)
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True)
        partitions = [[0, 2], [1, 3]]
        counts = np.zeros((2, 10), dtype=np.int64)
        counts[0, 0] = 2
        counts[1, 1] = 2
        write_index_shards(output_dir, partitions, counts)
        (output_dir / "split_metadata.json").write_text(
            json.dumps({"num_clients": 2}), encoding="utf-8"
        )
        return {"num_clients": 2}

    monkeypatch.setattr(split_mnist, "create_mnist_split", fake_create_mnist_split)
    config = {
        "dataset": {
            "name": "MNIST",
            "root": str(tmp_path / "data"),
            "split_dir": str(split_dir),
            "auto_create_split": True,
            "split_mode": "dirichlet",
            "dirichlet_alpha": 0.3,
            "split_seed": 9,
            "subset_fraction": 1.0,
            "download": False,
        }
    }

    descriptors = build_mnist_client_descriptors(
        config, project_root=str(tmp_path), num_clients=2
    )

    assert len(calls) == 1
    assert calls[0]["mode"] == "dirichlet"
    assert calls[0]["alpha"] == 0.3
    assert calls[0]["seed"] == 9
    assert descriptors[0]["indices"] == [0, 2]
    assert descriptors[1]["indices"] == [1, 3]
