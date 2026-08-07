import json

import numpy as np
import torch

import scripts.split_cifar10_dirichlet as split_cifar10
from src.classification.data import (
    CIFAR10_CLASS_NAMES,
    _split_dir,
    get_class_names,
    get_dataset_name,
)
from src.classification.models import get_model_name, validate_classification_config


class FakeCIFAR10:
    def __init__(self, root, train, download):
        self.targets = np.repeat(np.arange(10), 4).tolist() if train else [0] * 10

    def __len__(self):
        return len(self.targets)


def test_create_cifar10_split_writes_compatible_index_shards(monkeypatch, tmp_path):
    monkeypatch.setattr(split_cifar10, "CIFAR10", FakeCIFAR10)
    output_dir = tmp_path / "split"

    metadata = split_cifar10.create_cifar10_split(
        root=tmp_path / "data",
        output_dir=output_dir,
        num_clients=4,
        mode="dirichlet",
        alpha=0.5,
        seed=42,
        download=False,
    )

    assert metadata["dataset"] == "CIFAR10"
    assert metadata["selected_train_samples"] == 40
    assert metadata["validation_samples"] == 10
    all_indices = []
    for client_id in range(1, 5):
        shard = torch.load(
            output_dir / f"client_{client_id}_indices.pt", weights_only=True
        ).tolist()
        assert shard
        all_indices.extend(shard)
        client_metadata = json.loads(
            (output_dir / f"client_{client_id}.json").read_text(encoding="utf-8")
        )
        assert tuple(client_metadata["class_counts"]) == CIFAR10_CLASS_NAMES
    assert sorted(all_indices) == list(range(40))
    assert len(set(all_indices)) == 40


def test_generic_config_selects_dataset_model_and_automatic_split_path(tmp_path):
    config = {
        "model": {"name": "AlexNet", "num_classes": 10},
        "dataset": {
            "name": "cifar-10",
            "root": str(tmp_path),
            "split_dir": "auto",
            "split_mode": "dirichlet",
            "dirichlet_alpha": 0.5,
            "input_size": 224,
            "channels": 3,
        },
    }

    validate_classification_config(config)
    assert get_dataset_name(config) == "CIFAR10"
    assert get_model_name(config) == "AlexNet"
    assert get_class_names(config) == CIFAR10_CLASS_NAMES
    assert _split_dir(config, str(tmp_path)).endswith(
        "cifar10_splits/cifar10_dirichlet_alpha_0p5"
    )
