import csv
import os
import pickle

import torch
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.Alexnet import AlexNet
from src.classification.aggregation import merge_classification_models
from src.classification.data import (
    MNIST_CLASS_NAMES,
    build_mnist_client_descriptors,
    build_mnist_validation_dataset,
)
from src.classification.metrics import ClassificationMetrics
from src.communication import Communication
from scripts.draw import draw_graph
from src.utils import (
    create_run_dir,
    get_client_cut_layers,
    get_server_cut_layer,
)


class ClassificationServer:
    """Coordinator for dynamic-cut AlexNet split-federated learning."""

    def __init__(self, config, device, project_root):
        self.config = config
        self.device = device
        self.project_root = project_root
        self.num_clients = list(config["clients"])
        if len(self.num_clients) != 2 or any(count < 1 for count in self.num_clients):
            raise ValueError("clients must contain positive [edge, server] counts.")
        self.client_cut_layers = get_client_cut_layers(
            config, self.num_clients[0]
        )
        for cut in self.client_cut_layers:
            AlexNet.validate_cut_layer(cut)
        self.minimum_cut_layer = get_server_cut_layer(self.client_cut_layers)

        training = config["training"]
        self.batch_size = int(training["batch_size"])
        self.num_workers = int(training.get("num_workers", 0))
        self.num_epochs = int(training["num_epochs"])
        self.num_rounds = int(training["num_rounds"])
        self.num_classes = int(config.get("model", {}).get("num_classes", 10))
        if self.num_classes != len(MNIST_CLASS_NAMES):
            raise ValueError("MNIST classification requires model.num_classes=10.")
        self.model_seed = int(config.get("model", {}).get("seed", 42))

        self.comm = Communication(config)
        self.run_dir = create_run_dir(project_root, layer_id=0)
        self.clients = {}
        self.registered = [0, 0]
        self.metadata_clients = set()
        self.server_workers_started = False
        self.graph_drawn = False
        self.current_epoch = 1
        self.current_round = 1
        self.best_accuracy = -1.0
        self.mlflow_connector = self._build_mlflow_connector()

    def _build_mlflow_connector(self):
        mlflow_config = self.config.get("mlflow", {})
        if not bool(mlflow_config.get("enabled", False)):
            return None
        from src.mlflow import MLflowConnector

        connector = MLflowConnector(
            tracking_uri=mlflow_config["tracking_uri"],
            experiment_name=mlflow_config.get(
                "experiment_name", "SFL_AlexNet_MNIST"
            ),
        )
        connector.start_run(
            run_name=mlflow_config.get("run_name", "AlexNet dynamic-cut SFL")
        )
        connector.log_params(
            {
                "task": "classification",
                "model": "AlexNet",
                "dataset": "MNIST",
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "num_rounds": self.num_rounds,
                "cut_layers": str(self.client_cut_layers),
            }
        )
        return connector

    def run(self):
        print("Classification coordinator initialized.")
        self.comm.connect()
        self.comm.delete_old_queues(
            ["server_queue", "intermediate_queue", "gradient_queue"]
        )
        self.comm.create_queue("intermediate_queue")
        self.comm.create_queue("server_queue")
        # Download/cache MNIST once in the coordinator before edge workers
        # construct their shards from the shared project volume.
        build_mnist_validation_dataset(self.config, self.project_root)
        self.comm.consume_messages("server_queue", self.on_message)

    def on_message(self, ch, method, properties, body):
        payload = pickle.loads(body)
        action = payload.get("action")
        if action == "register":
            self._register_client(payload)
        elif action == "send_number_batch":
            self._receive_metadata(payload)
        elif action == "update_model":
            self._receive_model(payload)
        else:
            print(f"Unknown classification coordinator action: {action}")

    def _register_client(self, payload):
        layer_id = int(payload["layer_id"])
        client_id = payload["client_id"]
        if layer_id not in (1, 2):
            raise ValueError(f"Unsupported layer_id={layer_id}.")
        client_index = self.registered[layer_id - 1]
        if client_index >= self.num_clients[layer_id - 1]:
            raise ValueError(f"Too many layer {layer_id} clients registered.")
        cut_layer = (
            self.client_cut_layers[client_index]
            if layer_id == 1
            else self.minimum_cut_layer
        )
        self.clients[client_id] = {
            "layer_id": layer_id,
            "client_index": client_index,
            "cut_layer": cut_layer,
        }
        self.registered[layer_id - 1] += 1
        print(
            f"Registered classification layer {layer_id} client "
            f"#{client_index + 1} ({client_id}), cut_layer={cut_layer}."
        )

        if self.registered == self.num_clients:
            edge_ids = self.get_client_ids(layer_id=1)
            descriptors_by_index = build_mnist_client_descriptors(
                self.config, self.project_root, self.num_clients[0]
            )
            dataset_descriptors = [
                descriptors_by_index[self.clients[client_id]["client_index"]]
                for client_id in edge_ids
            ]
            for client_id, descriptor in zip(edge_ids, dataset_descriptors):
                self.clients[client_id]["num_samples"] = descriptor.get(
                    "num_samples"
                )
                self.clients[client_id]["class_counts"] = descriptor.get(
                    "class_counts", {}
                )
            self.comm.send_start_message(
                edge_ids,
                datasets=dataset_descriptors,
                cut_layers=[self.clients[client_id]["cut_layer"] for client_id in edge_ids],
            )

    def _receive_metadata(self, payload):
        client_id = payload["client_id"]
        if client_id not in self.clients:
            raise ValueError(f"Metadata came from unknown client {client_id}.")
        self.clients[client_id]["nb_train"] = int(payload["nb_train"])
        if payload.get("num_samples") is not None:
            self.clients[client_id]["num_samples"] = int(payload["num_samples"])
        self.metadata_clients.add(client_id)
        edge_ids = self.get_client_ids(layer_id=1)
        if all(client_id in self.metadata_clients for client_id in edge_ids):
            self.draw_client_graph()
        if self.server_workers_started or not all(
            client_id in self.metadata_clients for client_id in edge_ids
        ):
            return

        total_batches = sum(self.clients[client_id]["nb_train"] for client_id in edge_ids)
        server_ids = self.get_client_ids(layer_id=2)
        if total_batches < len(server_ids):
            raise ValueError("There are fewer batches than classification server workers.")
        base, remainder = divmod(total_batches, len(server_ids))
        allocations = [base + (index < remainder) for index in range(len(server_ids))]
        self.comm.send_start_message(
            server_ids,
            nb=allocations,
            nc=self.num_classes,
            class_names=MNIST_CLASS_NAMES,
            cut_layers=[self.minimum_cut_layer] * len(server_ids),
            supported_cut_layers=self.client_cut_layers,
        )
        self.server_workers_started = True

    def draw_client_graph(self):
        """Print and save the AlexNet split graph once metadata is complete."""
        if self.graph_drawn:
            return
        edge_ids = self.get_client_ids(layer_id=1)
        if len(edge_ids) != self.num_clients[0]:
            return
        if any(self.clients[client_id].get("num_samples") is None for client_id in edge_ids):
            return

        client_data = [
            {
                "name": f"Client {self.clients[client_id]['client_index'] + 1}",
                "cut_layer": self.clients[client_id]["cut_layer"],
                "image_count": self.clients[client_id]["num_samples"],
            }
            for client_id in edge_ids
        ]
        console = Console(record=True)
        draw_graph(
            client_data,
            max_layer=len(AlexNet.LAYER_NAMES) - 1,
            output=console,
            scale=max(
                2,
                max(len(str(client["image_count"])) for client in client_data) + 2,
            ),
        )
        graph_path = os.path.join(self.run_dir, "alexnet_split_graph.txt")
        console.save_text(graph_path, clear=False)
        self.graph_drawn = True
        print(f"AlexNet split graph saved to {graph_path}")

    def _receive_model(self, payload):
        client_id = payload["client_id"]
        if client_id not in self.clients:
            raise ValueError(f"Model came from unknown client {client_id}.")
        layer_id = int(payload["layer_id"])
        model_epoch = int(payload["epoch"]) + 1
        if model_epoch < self.current_epoch:
            print(f"Ignoring stale epoch {model_epoch} from {client_id}.")
            return

        checkpoint_path = os.path.join(
            self.run_dir,
            f"client_{client_id}_layer_{layer_id}_epoch_{model_epoch}.pt",
        )
        with open(checkpoint_path, "wb") as checkpoint_file:
            checkpoint_file.write(payload["model_data"])
        client = self.clients[client_id]
        client[f"model_{model_epoch}"] = checkpoint_path
        client[f"metrics_{model_epoch}"] = payload.get("metrics", {})
        if payload.get("route_batch_counts") is not None:
            client[f"routes_{model_epoch}"] = {
                int(cut): int(count)
                for cut, count in payload["route_batch_counts"].items()
            }

        if layer_id == 1:
            client[f"latency_{model_epoch}"] = {
                name: float(payload.get(name, 0.0))
                for name in (
                    "batch_e2e",
                    "edge_forward",
                    "edge_backward",
                    "server_forward",
                    "server_backward",
                    "inter_delay",
                    "grad_delay",
                )
            }
        if model_epoch != self.current_epoch:
            return

        edge_models = self.get_models(1, self.current_epoch)
        server_models = self.get_models(2, self.current_epoch)
        if (
            len(edge_models) != self.num_clients[0]
            or len(server_models) != self.num_clients[1]
        ):
            return

        full_model = AlexNet(
            num_classes=self.num_classes, seed=self.model_seed
        )
        self.model = merge_classification_models(
            full_model,
            edge_models,
            server_models,
            self.minimum_cut_layer,
        ).to(self.device)
        train_metrics = self._aggregate_train_metrics(server_models)
        validation_metrics = self.validate()
        latency_metrics = self._aggregate_latency(edge_models)
        all_metrics = {
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in validation_metrics.items()},
            **{f"latency/{key}": value for key, value in latency_metrics.items()},
        }
        self._record_epoch(self.current_epoch, all_metrics)
        self._save_checkpoints(validation_metrics, all_metrics)

        if self.mlflow_connector is not None:
            self.mlflow_connector.log_metrics(all_metrics, step=self.current_epoch)
        print(
            f"Epoch {self.current_epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"val_loss={validation_metrics['loss']:.4f}, "
            f"val_acc={validation_metrics['accuracy']:.4f}, "
            f"val_f1={validation_metrics['f1_macro']:.4f}."
        )

        is_round_end = self.current_epoch % self.num_epochs == 0
        is_final_epoch = self.current_epoch >= self.num_epochs * self.num_rounds
        if is_round_end and not is_final_epoch:
            global_path = os.path.join(
                self.run_dir, f"global_model_round_{self.current_round}.pt"
            )
            self._save_global_checkpoint(global_path, all_metrics)
            self.comm.publish_global_model(
                self.get_client_ids(),
                global_model_path=global_path,
                round=self.current_round,
            )
            self.current_round += 1
        elif is_final_epoch:
            self._send_stop()
            if self.mlflow_connector is not None:
                self.mlflow_connector.end_run()

        self.current_epoch += 1

    def get_client_ids(self, layer_id=None):
        return [
            client_id
            for client_id, info in sorted(
                self.clients.items(), key=lambda item: item[1]["client_index"]
            )
            if layer_id is None or info["layer_id"] == layer_id
        ]

    def get_models(self, layer_id, epoch):
        models = []
        for client_id in self.get_client_ids(layer_id):
            info = self.clients[client_id]
            path = info.get(f"model_{epoch}")
            if path is None:
                continue
            models.append(
                {
                    "client_id": client_id,
                    "path": path,
                    "cut_layer": info["cut_layer"],
                    "num_batches": info.get("nb_train", 0),
                    "route_batch_counts": info.get(f"routes_{epoch}", {}),
                    "metrics": info.get(f"metrics_{epoch}", {}),
                    "latency": info.get(f"latency_{epoch}", {}),
                }
            )
        return models

    def _aggregate_train_metrics(self, server_models):
        metrics = ClassificationMetrics(self.num_classes)
        weighted_loss = 0.0
        total = 0
        for model in server_models:
            worker_metrics = model["metrics"]
            worker_total = int(worker_metrics.get("total", 0))
            weighted_loss += float(worker_metrics.get("train_loss", 0.0)) * worker_total
            total += worker_total
            if worker_metrics.get("confusion_matrix") is not None:
                metrics.merge(worker_metrics["confusion_matrix"])
        result = metrics.compute()
        result.pop("total")
        result["loss"] = weighted_loss / max(total, 1)
        return result

    def _aggregate_latency(self, edge_models):
        names = (
            "batch_e2e",
            "edge_forward",
            "edge_backward",
            "server_forward",
            "server_backward",
            "inter_delay",
            "grad_delay",
        )
        total_batches = sum(int(model["num_batches"]) for model in edge_models)
        return {
            name: sum(
                float(model["latency"].get(name, 0.0)) * int(model["num_batches"])
                for model in edge_models
            )
            / max(total_batches, 1)
            for name in names
        }

    def validate(self):
        validation_dataset = build_mnist_validation_dataset(
            self.config, self.project_root
        )
        loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        criterion = nn.CrossEntropyLoss()
        metrics = ClassificationMetrics(self.num_classes)
        loss_sum = 0.0
        sample_count = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="MNIST validation"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss_sum += loss.item() * labels.size(0)
                sample_count += labels.size(0)
                metrics.update(logits, labels)
        result = metrics.compute()
        result.pop("total")
        result["loss"] = loss_sum / max(sample_count, 1)
        return result

    def _checkpoint_payload(self, metrics):
        return {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.current_epoch,
            "num_classes": self.num_classes,
            "class_names": MNIST_CLASS_NAMES,
            "metrics": metrics,
            "task": "classification",
        }

    def _save_global_checkpoint(self, path, metrics):
        torch.save(self._checkpoint_payload(metrics), path)

    def _save_checkpoints(self, validation_metrics, all_metrics):
        last_path = os.path.join(self.run_dir, "last.pt")
        self._save_global_checkpoint(last_path, all_metrics)
        if validation_metrics["accuracy"] > self.best_accuracy:
            self.best_accuracy = validation_metrics["accuracy"]
            self._save_global_checkpoint(
                os.path.join(self.run_dir, "best.pt"), all_metrics
            )

    def _record_epoch(self, epoch, metrics):
        path = os.path.join(self.run_dir, "classification_results.csv")
        row = {"epoch": epoch, **metrics}
        exists = os.path.exists(path)
        with open(path, "a", newline="") as results_file:
            writer = csv.DictWriter(results_file, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _send_stop(self):
        message = pickle.dumps({"action": "stop"})
        for client_id in self.get_client_ids():
            self.comm.publish_message(f"client_queue_{client_id}", message)
