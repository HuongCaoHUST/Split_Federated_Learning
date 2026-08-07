from collections import defaultdict, deque
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification.data import build_client_dataset, get_dataset_name
from src.classification.metrics import ClassificationMetrics
from src.classification.models import (
    build_edge_model,
    build_server_model,
    get_model_name,
)
from src.utils import BatchLogger, get_cut_layer, get_cut_layers


def _build_optimizer(model, config):
    training = config["training"]
    name = str(training.get("optimizer", "Adam")).lower()
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.0))
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(training.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    if name == "adam":
        return optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if name == "adamw":
        return optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    raise ValueError(f"Optimizer '{name}' is not supported for classification.")


class ClassificationEdgeTrainer:
    def __init__(
        self,
        config,
        device,
        project_root,
        comm,
        run_dir,
        layer_id,
        client_id,
        dataset_descriptor,
        global_model_path=None,
        round_index=0,
    ):
        self.config = config
        self.device = device
        self.project_root = project_root
        self.comm = comm
        self.run_dir = run_dir
        self.layer_id = layer_id
        self.client_id = client_id
        self.global_model_path = global_model_path
        self.round_index = int(round_index or 0)

        self.batch_size = int(config["training"]["batch_size"])
        self.num_workers = int(config["training"].get("num_workers", 0))
        self.num_epochs = int(config["training"]["num_epochs"])
        self.cut_layer = get_cut_layer(config)
        self.num_classes = int(config.get("model", {}).get("num_classes", 10))

        if not isinstance(dataset_descriptor, dict):
            raise TypeError("Classification edge requires a dataset descriptor.")
        client_index = int(dataset_descriptor["client_index"])
        num_clients = int(dataset_descriptor["num_clients"])
        self.dataset_name = get_dataset_name(config)
        self.model_name = get_model_name(config)
        train_dataset = build_client_dataset(
            config,
            project_root,
            client_index,
            num_clients,
            dataset_descriptor=dataset_descriptor,
        )
        print(
            f"Client {client_index + 1}/{num_clients} received "
            f"{len(train_dataset)} {self.dataset_name} samples; "
            f"class_counts={dataset_descriptor.get('class_counts', {})}."
        )
        self.num_train_samples = len(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        self.model = build_edge_model(
            config,
            cut_layer=self.cut_layer,
            checkpoint=global_model_path,
        ).to(self.device)
        self.optimizer = _build_optimizer(self.model, config)

        self.gradient_queue_name = f"gradient_queue_{client_id}"
        self.comm.create_queue(self.gradient_queue_name)
        self.batch_logger = BatchLogger(
            self.client_id,
            os.path.join(self.run_dir, "classification_batch_latency.csv"),
        )

    def train_one_epoch(self, local_epoch, global_epoch):
        self.model.train()
        loss_sum = 0.0
        sample_count = 0
        epoch_latency_sum = np.zeros(7, dtype=np.float64)
        progress = tqdm(
            self.train_loader,
            desc=(
                f"Epoch {local_epoch + 1}/{self.num_epochs} "
                f"[{self.model_name} edge]"
            ),
        )

        for images, labels in progress:
            start_batch_time = time.time()
            images = images.to(self.device, non_blocking=True)
            edge_output = self.model(images)
            payload = {
                "client_output": edge_output.detach().cpu().numpy(),
                "labels": labels.cpu().numpy(),
                "reply_to": self.gradient_queue_name,
                "client_id": self.client_id,
                "cut_layer": self.cut_layer,
                "epoch": global_epoch,
            }
            data_bytes = pickle.dumps(payload)
            self.comm.publish_message("intermediate_queue", data_bytes)
            send_intermediate_time = time.time()

            response = pickle.loads(
                self.comm.consume_message_sync(self.gradient_queue_name)
            )
            receive_gradient_time = time.time()
            gradient = torch.as_tensor(
                response["gradient"],
                dtype=edge_output.dtype,
                device=self.device,
            )
            self.optimizer.zero_grad()
            torch.autograd.backward(edge_output, gradient)
            self.optimizer.step()
            end_batch_time = time.time()

            batch_size = labels.size(0)
            loss_sum += float(response["loss"]) * batch_size
            sample_count += batch_size
            timings = np.array(
                [
                    end_batch_time - start_batch_time,
                    send_intermediate_time - start_batch_time,
                    end_batch_time - receive_gradient_time,
                    float(response.get("server_forward_time", 0.0)),
                    float(response.get("server_backward_time", 0.0)),
                    float(response.get("receive_inter_time", 0.0))
                    - send_intermediate_time,
                    receive_gradient_time - float(response.get("send_grad_time", 0.0)),
                ]
            )
            epoch_latency_sum += timings
            self.batch_logger.log_batch(
                global_epoch + 1,
                timings[0],
                data_bytes,
                edge_forward=timings[1],
                edge_backward=timings[2],
                server_forward=timings[3],
                server_backward=timings[4],
                inter_delay=timings[5],
                grad_delay=timings[6],
            )
            progress.set_postfix(loss=f"{float(response['loss']):.4f}")

        if sample_count == 0:
            raise ValueError(f"The {self.dataset_name} client shard is empty.")
        return loss_sum / sample_count, epoch_latency_sum / len(self.train_loader)

    def run(self):
        if self.round_index == 0:
            self.comm.send_training_metadata(
                "server_queue",
                self.client_id,
                nb_train=len(self.train_loader),
                num_samples=self.num_train_samples,
            )
        for local_epoch in range(self.num_epochs):
            global_epoch = local_epoch + self.num_epochs * self.round_index
            train_loss, latencies = self.train_one_epoch(local_epoch, global_epoch)
            checkpoint_path = os.path.join(
                self.run_dir,
                f"classification_edge_{self.client_id}_epoch_{global_epoch + 1}.pt",
            )
            torch.save(self.model.state_dict(), checkpoint_path)
            self.comm.publish_model(
                "server_queue",
                checkpoint_path,
                layer_id=self.layer_id,
                client_id=self.client_id,
                epoch=global_epoch,
                latencies=latencies,
                metrics={"train_loss": train_loss},
            )


class ClassificationServerTrainer:
    def __init__(
        self,
        config,
        device,
        comm,
        run_dir,
        layer_id,
        client_id,
        num_batches,
        global_model_path=None,
        round_index=0,
    ):
        self.config = config
        self.device = device
        self.comm = comm
        self.run_dir = run_dir
        self.layer_id = layer_id
        self.client_id = client_id
        self.num_batches = int(num_batches)
        self.round_index = int(round_index or 0)
        self.num_epochs = int(config["training"]["num_epochs"])
        self.num_classes = int(config.get("model", {}).get("num_classes", 10))
        self.cut_layers = get_cut_layers(config)
        self.model_name = get_model_name(config)
        self.model = build_server_model(
            config,
            supported_cut_layers=self.cut_layers,
            checkpoint=global_model_path,
        ).to(self.device)
        self.optimizer = _build_optimizer(self.model, config)
        self.criterion = nn.CrossEntropyLoss()
        self.pending_payloads = defaultdict(deque)

    def _consume_for_epoch(self, expected_epoch):
        pending = self.pending_payloads[expected_epoch]
        if pending:
            return pending.popleft()
        while True:
            body = self.comm.consume_message_sync("intermediate_queue")
            receive_time = time.time()
            payload = pickle.loads(body)
            payload_epoch = int(payload.get("epoch", -1))
            if payload_epoch < expected_epoch:
                raise ValueError(
                    f"Received stale batch for epoch {payload_epoch}; "
                    f"expected {expected_epoch}."
                )
            if payload_epoch == expected_epoch:
                return payload, receive_time
            self.pending_payloads[payload_epoch].append((payload, receive_time))

    def train_one_epoch(self, local_epoch, global_epoch):
        self.model.train()
        loss_sum = 0.0
        sample_count = 0
        metrics = ClassificationMetrics(self.num_classes)
        route_batch_counts = {cut: 0 for cut in self.cut_layers}
        progress = tqdm(
            range(self.num_batches),
            desc=(
                f"Epoch {local_epoch + 1}/{self.num_epochs} "
                f"[{self.model_name} server]"
            ),
        )

        for _ in progress:
            payload, receive_time = self._consume_for_epoch(global_epoch)
            cut_layer = int(payload["cut_layer"])
            if cut_layer not in route_batch_counts:
                raise ValueError(f"Unexpected cut_layer={cut_layer}.")
            route_batch_counts[cut_layer] += 1
            activation = torch.as_tensor(
                payload["client_output"], dtype=torch.float32, device=self.device
            ).requires_grad_(True)
            labels = torch.as_tensor(
                payload["labels"], dtype=torch.long, device=self.device
            )

            logits = self.model(activation, cut_layer=cut_layer)
            end_forward_time = time.time()
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            end_backward_time = time.time()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
            metrics.update(logits, labels)
            response = {
                "gradient": activation.grad.detach().cpu().numpy(),
                "loss": loss.item(),
                "server_forward_time": end_forward_time - receive_time,
                "server_backward_time": end_backward_time - end_forward_time,
                "receive_inter_time": receive_time,
                "send_grad_time": end_backward_time,
            }
            self.comm.publish_message(payload["reply_to"], pickle.dumps(response))
            progress.set_postfix(loss=f"{loss.item():.4f}")

        result = metrics.compute()
        result["train_loss"] = loss_sum / max(sample_count, 1)
        result["confusion_matrix"] = metrics.confusion_matrix.tolist()
        return result, route_batch_counts

    def run(self):
        for local_epoch in range(self.num_epochs):
            global_epoch = local_epoch + self.num_epochs * self.round_index
            metrics, route_counts = self.train_one_epoch(local_epoch, global_epoch)
            checkpoint_path = os.path.join(
                self.run_dir,
                f"alexnet_server_{self.client_id}_epoch_{global_epoch + 1}.pt",
            )
            torch.save(self.model.state_dict(), checkpoint_path)
            self.comm.publish_model(
                "server_queue",
                checkpoint_path,
                layer_id=self.layer_id,
                client_id=self.client_id,
                epoch=global_epoch,
                route_batch_counts=route_counts,
                metrics=metrics,
            )
