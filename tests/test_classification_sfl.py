from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from model.Alexnet import AlexNet
from src.classification.aggregation import merge_classification_models
from src.classification.metrics import ClassificationMetrics
from src.classification import trainers
from src.task import get_task_name


class TinySplitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(1, 1, bias=False) for _ in range(3)]
        )


def test_task_router_preserves_detection_default():
    assert get_task_name({}) == "detection"
    assert get_task_name({"task": "classification"}) == "classification"
    assert get_task_name({"task": {"type": "detect"}}) == "detection"


def test_classification_metrics_from_confusion_matrix():
    metrics = ClassificationMetrics(num_classes=2)
    metrics.update(torch.tensor([0, 1, 1, 1]), torch.tensor([0, 0, 1, 1]))
    result = metrics.compute()

    assert result["accuracy"] == 0.75
    assert abs(result["precision_macro"] - (5 / 6)) < 1e-12
    assert result["recall_macro"] == 0.75
    assert abs(result["f1_macro"] - (11 / 15)) < 1e-12
    assert result["total"] == 4


def test_dynamic_aggregation_weights_each_layer_by_route(tmp_path):
    edge_0_path = tmp_path / "edge_0.pt"
    edge_1_path = tmp_path / "edge_1.pt"
    server_path = tmp_path / "server.pt"
    torch.save({"layers.0.weight": torch.tensor([[1.0]])}, edge_0_path)
    torch.save(
        {
            "layers.0.weight": torch.tensor([[3.0]]),
            "layers.1.weight": torch.tensor([[5.0]]),
        },
        edge_1_path,
    )
    torch.save(
        {
            # Local server layer 0 is global layer 1 because minimum cut is 0.
            "layers.0.weight": torch.tensor([[7.0]]),
            "layers.1.weight": torch.tensor([[9.0]]),
        },
        server_path,
    )

    merged = merge_classification_models(
        TinySplitModel(),
        edge_models=[
            {"path": edge_0_path, "cut_layer": 0, "num_batches": 2},
            {"path": edge_1_path, "cut_layer": 1, "num_batches": 3},
        ],
        server_models=[
            {
                "path": server_path,
                "route_batch_counts": {0: 2, 1: 3},
            }
        ],
        minimum_cut_layer=0,
    )

    assert torch.allclose(merged.layers[0].weight, torch.tensor([[2.2]]))
    assert torch.allclose(merged.layers[1].weight, torch.tensor([[5.8]]))
    assert torch.allclose(merged.layers[2].weight, torch.tensor([[9.0]]))


def test_alexnet_rejects_a_cut_after_the_classifier():
    assert AlexNet.SUPPORTED_CUT_LAYERS == tuple(range(8))
    try:
        AlexNet.validate_cut_layer(8)
    except ValueError as exc:
        assert "leave at least one stage" in str(exc)
    else:
        raise AssertionError("The final classifier stage cannot be a split point.")


def test_edge_server_trainers_exchange_activation_and_gradient(monkeypatch, tmp_path):
    class TinyEdge(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.layer = nn.Linear(4, 3)

        def forward(self, inputs):
            return self.layer(inputs)

    class TinyServer(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.layer = nn.Linear(3, 10)

        def forward(self, inputs, cut_layer):
            return self.layer(inputs)

    class FakeCommunication:
        def __init__(self):
            self.queues = {}
            self.model_updates = []
            self.metadata = []

        def create_queue(self, name):
            self.queues.setdefault(name, Queue())

        def publish_message(self, queue_name, message):
            self.create_queue(queue_name)
            self.queues[queue_name].put(message)

        def consume_message_sync(self, queue_name):
            self.create_queue(queue_name)
            return self.queues[queue_name].get(timeout=5)

        def send_training_metadata(self, queue_name, client_id, **kwargs):
            self.metadata.append((queue_name, client_id, kwargs))

        def publish_model(self, queue_name, model_path, **kwargs):
            self.model_updates.append((queue_name, model_path, kwargs))

    monkeypatch.setattr(
        trainers, "build_edge_model", lambda *args, **kwargs: TinyEdge()
    )
    monkeypatch.setattr(
        trainers, "build_server_model", lambda *args, **kwargs: TinyServer()
    )
    monkeypatch.setattr(
        trainers,
        "build_client_dataset",
        lambda *args, **kwargs: TensorDataset(
            torch.randn(2, 4), torch.tensor([1, 2])
        ),
    )
    config = {
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "num_epochs": 1,
            "learning_rate": 0.01,
            "optimizer": "SGD",
        },
        "model": {"num_classes": 10, "seed": 42},
        "dataset": {"name": "MNIST"},
        "cut_layer": [0],
    }
    communication = FakeCommunication()
    communication.create_queue("intermediate_queue")
    (tmp_path / "edge").mkdir()
    (tmp_path / "server").mkdir()
    edge = trainers.ClassificationEdgeTrainer(
        config,
        torch.device("cpu"),
        str(tmp_path),
        communication,
        str(tmp_path / "edge"),
        1,
        "edge-client",
        {"client_index": 0, "num_clients": 1},
    )
    server = trainers.ClassificationServerTrainer(
        config,
        torch.device("cpu"),
        communication,
        str(tmp_path / "server"),
        2,
        "server-client",
        num_batches=1,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        server_future = executor.submit(server.run)
        edge_future = executor.submit(edge.run)
        server_future.result(timeout=10)
        edge_future.result(timeout=10)

    assert len(communication.metadata) == 1
    assert len(communication.model_updates) == 2
    server_update = next(
        update for update in communication.model_updates if update[2]["layer_id"] == 2
    )
    assert server_update[2]["route_batch_counts"] == {0: 1}
    assert server_update[2]["metrics"]["total"] == 2
    assert len(server_update[2]["metrics"]["confusion_matrix"]) == 10
