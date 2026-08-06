import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from model.Alexnet import AlexNet
from model.Alexnet_EDGE import AlexNet_EDGE
from model.Alexnet_SERVER import AlexNet_SERVER
from model.Mobilenet import MobileNet
from model.VGG16 import VGG16
from model.VGG16_EDGE import VGG16_EDGE
from model.VGG16_SERVER import VGG16_SERVER
from model.YOLO11n_custom import YOLO11_EDGE, YOLO11_EDGE_5, YOLO11_EDGE_15, YOLO11_EDGE_20, YOLO11_DYNAMIC_SERVER, YOLO11_Full
from src.canonical_gradient import (
    CANONICAL_GRADIENT_QUEUE,
    install_prefix_buffers,
    install_prefix_gradients,
    prefix_buffers,
    prefix_gradients,
    prefix_state_dict,
    validate_canonical_cut5_config,
)
from src.utils import BatchLogger, update_results_csv, save_plots, count_parameters, create_run_dir, clear_memory, get_cut_layer, get_cut_layers, get_server_cut_layer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
import numpy as np
from collections import defaultdict, deque
from src.utils_box import non_max_suppression, scale_boxes, xywh2xyxy, box_iou
from ultralytics.utils.metrics import ap_per_class

MLFLOW_TRACKING_URI = "http://14.225.254.18:5000"
EXPERIMENT_NAME = "Split_Learning"

class TrainerEdge:
    def __init__(self, config, device, project_root, comm, run_dir, layer_id, client_id, datasets, global_model_path = None, round = 0):
        self.config = config
        self.device = device
        self.project_root = project_root
        self.comm = comm
        self.layer_id = layer_id
        self.client_id = client_id
        self.datasets = datasets
        self.global_model_path = global_model_path
        self.round = round
        
        # Set Hyperparameters
        self.run_dir = run_dir
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training'].get('num_workers', 0)
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.momentum = config['training'].get('momentum', 0.9)
        self.model_name = config['model']['edge']
        self.cut_layer = get_cut_layer(config)
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)
        self.pretrained_path = config['model'].get('pretrained_path')
        self.canonical_gradient_mode = config['training'].get(
            'canonical_gradient_mode', False
        )
        if self.canonical_gradient_mode:
            validate_canonical_cut5_config(config)

        # Create gradient queue
        self.gradient_queue_name = f'gradient_queue_{client_id}'
        self.comm.create_queue(self.gradient_queue_name)

        # Initialize batch logger
        self.batch_logger = BatchLogger(self.client_id, "training_log.csv")

        # Initialize model
        self.data_cfg = check_det_dataset(self.datasets)
        self.num_classes = self.data_cfg['nc']

        MODEL_MAP = {
            5: YOLO11_EDGE_5,
            10: YOLO11_EDGE,
            15: YOLO11_EDGE_15,
            20: YOLO11_EDGE_20
        }

        model_class = MODEL_MAP.get(self.cut_layer)
        if model_class is None:
            raise ValueError(
                f"Unsupported edge cut_layer={self.cut_layer}; "
                f"supported values: {sorted(MODEL_MAP)}."
            )

        if self.global_model_path is not None:
            print("Continue Training with global model: ", self.global_model_path)
            self.model = model_class(pretrained = self.global_model_path).to(self.device)
        else:
            self.model = model_class(pretrained = 'yolo11n.pt').to(self.device)

        self.model.names = self.data_cfg['names']
        self.yolo_args = get_cfg(DEFAULT_CFG)
        self.model.args = self.yolo_args

        # In canonical-gradient mode, this edge is only an autograd worker.
        # The sole optimizer state lives in the full model on the server.
        self.optimizer = None
        if not self.canonical_gradient_mode:
            if self.optimizer_name.lower() == 'sgd':
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.937, weight_decay=0.0005)
            elif self.optimizer_name.lower() == 'adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            elif self.optimizer_name.lower() == 'adamw':
                 self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0005)
            else:
                raise ValueError(f"Optimizer {self.optimizer_name} not supported.")

        # Initialize Dataset and DataLoader
        self.train_dataset = YOLODataset(
            img_path=self.data_cfg["train"],
            imgsz=640,
            data=self.data_cfg,
            augment=True,
            hyp=self.yolo_args,
            rect=False,
            stride=32
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn
        )

        # History tracking
        self.history_train_loss = []
        self.history_val_loss = []
        self.history_val_accuracy = []

    def train_one_epoch(self, epoch, global_epoch):
        if self.canonical_gradient_mode:
            return self._train_one_epoch_canonical(epoch, global_epoch)

        self.model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        epoch_latency_sum = np.zeros(7)
        
        for batch in train_progress_bar:
            start_batch_time = time.time()
            images = batch['img'].to(self.device, non_blocking=True).float() / 255.0
            outputs = self.model(images)
            print("Outputs shapes: ", [o.shape for o in outputs])
 
            label_data = {
                "batch_idx": batch["batch_idx"].cpu(),
                "bboxes":    batch["bboxes"].cpu(),
                "cls":       batch["cls"].cpu()
            }
            payload = {
                'client_output': [x.detach().cpu().numpy() for x in outputs],
                'label_data': label_data,
                'reply_to': self.gradient_queue_name,
                'client_id': self.client_id,
                'cut_layer': self.cut_layer,
                'epoch': global_epoch,
            }

            data_bytes = pickle.dumps(payload)
            self.comm.publish_message(queue_name='intermediate_queue', message=data_bytes)
            send_inter_time = time.time()

            response_body = self.comm.consume_message_sync(self.gradient_queue_name)
            received_grad_time = time.time()
            response = pickle.loads(response_body)

            server_grad_numpy = response['gradient']
            batch_loss = response['loss']
            server_forward_time = response.get('server_forward_time', 0)
            server_backward_time = response.get('server_backward_time', 0)
            receive_inter_time = response.get('receive_inter_time', 0)
            send_grad_time = response.get('send_grad_time', 0)

            self.optimizer.zero_grad()

            grad_tensors = []
            for g in server_grad_numpy:
                if isinstance(g, torch.Tensor):
                    grad_tensors.append(g.to(self.device))
                else:
                    grad_tensors.append(torch.from_numpy(g).to(self.device))

            torch.autograd.backward(outputs, grad_tensors)
            self.optimizer.step()

            end_batch_time = time.time()
            latency = end_batch_time - start_batch_time
            self.batch_logger.log_batch(epoch + 1, latency, data_bytes, 
                                        edge_forward = send_inter_time - start_batch_time,
                                        edge_backward = end_batch_time - received_grad_time,
                                        server_forward = server_forward_time,
                                        server_backward = server_backward_time,
                                        inter_delay = receive_inter_time - send_inter_time,
                                        grad_delay = received_grad_time - send_grad_time)
            
            current_batch_times = np.array([
                latency,
                send_inter_time - start_batch_time,
                end_batch_time - received_grad_time,
                server_forward_time,
                server_backward_time,
                receive_inter_time - send_inter_time,
                received_grad_time - send_grad_time
            ])

            epoch_latency_sum += current_batch_times
            # running_loss += batch_loss
            # train_progress_bar.set_postfix({'server_loss': batch_loss})
        clear_memory(device = self.device, threshold=0.85)
        avg_train_loss = running_loss / len(self.train_loader)
        self.history_train_loss.append(avg_train_loss)
        avg_latencies = epoch_latency_sum / len(self.train_loader)
        return avg_train_loss, avg_latencies

    def _train_one_epoch_canonical(self, epoch, global_epoch):
        """Train cut-5 with the only optimizer state held by the server.

        The edge performs forward/backward solely to produce prefix gradients
        and BatchNorm buffers. It never calls ``optimizer.step()``.
        """
        self.model.train()
        running_loss = 0.0
        epoch_latency_sum = np.zeros(7)
        train_progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs} [Canonical Cut-5]",
        )

        for batch_index, batch in enumerate(train_progress_bar):
            start_batch_time = time.time()
            images = batch['img'].to(self.device, non_blocking=True).float() / 255.0
            outputs = self.model(images)
            label_data = {
                "batch_idx": batch["batch_idx"].cpu(),
                "bboxes": batch["bboxes"].cpu(),
                "cls": batch["cls"].cpu(),
            }
            payload = {
                'action': 'canonical_forward',
                'client_output': [x.detach().cpu().numpy() for x in outputs],
                'label_data': label_data,
                'reply_to': self.gradient_queue_name,
                'client_id': self.client_id,
                'cut_layer': self.cut_layer,
                'epoch': global_epoch,
                'batch_index': batch_index,
                'num_samples': int(images.shape[0]),
            }
            data_bytes = pickle.dumps(payload)
            self.comm.publish_message('intermediate_queue', data_bytes)
            send_inter_time = time.time()

            boundary_response = pickle.loads(
                self.comm.consume_message_sync(self.gradient_queue_name)
            )
            if boundary_response.get('action') != 'canonical_boundary_gradient':
                raise ValueError(
                    "Expected canonical_boundary_gradient response, got "
                    f"{boundary_response.get('action')!r}."
                )
            if (
                boundary_response.get('epoch') != global_epoch
                or boundary_response.get('batch_index') != batch_index
            ):
                raise ValueError("Received a boundary gradient for a different canonical step.")
            received_grad_time = time.time()

            self.model.zero_grad(set_to_none=True)
            grad_tensors = [
                grad.to(self.device) if isinstance(grad, torch.Tensor)
                else torch.from_numpy(grad).to(self.device)
                for grad in boundary_response['gradient']
            ]
            torch.autograd.backward(outputs, grad_tensors)

            prefix_payload = {
                'action': 'canonical_prefix_gradient',
                'client_id': self.client_id,
                'epoch': global_epoch,
                'batch_index': batch_index,
                'prefix_gradients': prefix_gradients(self.model),
                'prefix_buffers': prefix_buffers(self.model),
            }
            self.comm.publish_message(
                CANONICAL_GRADIENT_QUEUE,
                pickle.dumps(prefix_payload),
            )

            update_response = pickle.loads(
                self.comm.consume_message_sync(self.gradient_queue_name)
            )
            if update_response.get('action') != 'canonical_prefix_update':
                raise ValueError(
                    "Expected canonical_prefix_update response, got "
                    f"{update_response.get('action')!r}."
                )
            if (
                update_response.get('epoch') != global_epoch
                or update_response.get('batch_index') != batch_index
            ):
                raise ValueError("Received a prefix state for a different canonical step.")
            self.model.load_state_dict(update_response['prefix_state'], strict=True)

            end_batch_time = time.time()
            server_forward_time = boundary_response.get('server_forward_time', 0)
            server_backward_time = boundary_response.get('server_backward_time', 0)
            receive_inter_time = boundary_response.get('receive_inter_time', 0)
            send_grad_time = boundary_response.get('send_grad_time', received_grad_time)
            latency = end_batch_time - start_batch_time
            self.batch_logger.log_batch(
                epoch + 1,
                latency,
                data_bytes,
                edge_forward=send_inter_time - start_batch_time,
                edge_backward=end_batch_time - received_grad_time,
                server_forward=server_forward_time,
                server_backward=server_backward_time,
                inter_delay=receive_inter_time - send_inter_time,
                grad_delay=received_grad_time - send_grad_time,
            )
            epoch_latency_sum += np.array([
                latency,
                send_inter_time - start_batch_time,
                end_batch_time - received_grad_time,
                server_forward_time,
                server_backward_time,
                receive_inter_time - send_inter_time,
                received_grad_time - send_grad_time,
            ])
            batch_loss = float(boundary_response.get('total_loss', 0.0))
            running_loss += batch_loss
            train_progress_bar.set_postfix(total_loss=f'{batch_loss:.4f}')

        clear_memory(device=self.device, threshold=0.85)
        avg_train_loss = running_loss / len(self.train_loader)
        self.history_train_loss.append(avg_train_loss)
        return avg_train_loss, epoch_latency_sum / len(self.train_loader)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_progress_bar = tqdm(self.validation_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
            for images, labels in val_progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(self.validation_loader)
        val_accuracy = 100 * correct / total
        
        self.history_val_loss.append(avg_val_loss)
        self.history_val_accuracy.append(val_accuracy)
        
        return avg_val_loss, val_accuracy
    
    def post_processing(self):
        # Save model
        if self.save_model_enabled:
            save_path = os.path.join(self.run_dir, 'cifar_net_edge.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            # self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, epoch = 'last')
        else:
            print("Model saving skipped as per configuration.")

        print("Plots saved.")
    
    def run(self):
        print("Starting Training...")

        nb_train = len(self.train_loader)
        self.comm.send_training_metadata('server_queue', self.client_id, nb_train)

        for epoch in range(self.num_epochs):
            if self.round >= 1 : global_epoch = epoch + self.num_epochs*self.round
            else: global_epoch = epoch

            avg_train_loss, latencies = self.train_one_epoch(epoch, global_epoch)
            print(f'Epoch [{epoch+1}/{self.num_epochs}] -> Train Loss: {avg_train_loss:.4f}')

            # Save checkpoint
            save_path = os.path.join(self.run_dir, f'cifar_net_server_{global_epoch+1}.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, client_id = self.client_id, epoch = global_epoch, latencies = latencies)
        
        print("Finished Training.")
        self.post_processing()

class TrainerServer:
    def __init__(self, config, device, project_root, comm, run_dir, layer_id, client_id, nb, nc, class_names, global_model_path = None, round = None):
        self.config = config
        self.device = device
        self.project_root = project_root
        self.comm = comm
        self.layer_id = layer_id
        self.client_id = client_id
        self.nb = nb
        self.nc = nc
        self.class_names = class_names
        self.global_model_path = global_model_path
        self.round = round
        
        # Set Hyperparameters
        self.run_dir = run_dir
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training'].get('num_workers', 0)
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.momentum = config['training'].get('momentum', 0.9)
        self.model_name = config['model']['server']
        self.cut_layers = get_cut_layers(config)
        self.cut_layer = get_server_cut_layer(self.cut_layers)
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)
        self.pretrained_path = config['model'].get('pretrained_path', 'yolo11n.pt')
        self.canonical_gradient_mode = config['training'].get(
            'canonical_gradient_mode', False
        )
        if self.canonical_gradient_mode:
            validate_canonical_cut5_config(config)

        # Initialize model
        if self.canonical_gradient_mode and self.global_model_path is not None:
            print("Continue canonical-gradient training with global model: ", self.global_model_path)
            self.model = YOLO11_Full(
                nc=self.nc,
                pretrained=self.global_model_path,
            ).to(self.device)
        elif self.canonical_gradient_mode:
            self.model = YOLO11_Full(
                nc=self.nc,
                pretrained=self.pretrained_path,
            ).to(self.device)
        elif self.global_model_path is not None:
            print("Continue Training with global model: ", self.global_model_path)
            self.model = YOLO11_DYNAMIC_SERVER(
                supported_cut_layers=self.cut_layers,
                pretrained=self.global_model_path,
                nc=self.nc,
            ).to(self.device)
        else:
            self.model = YOLO11_DYNAMIC_SERVER(
                supported_cut_layers=self.cut_layers,
                pretrained='yolo11n.pt',
                nc=self.nc,
            ).to(self.device)
            
        self.model.names = self.class_names
        self.yolo_args = get_cfg(DEFAULT_CFG)
        self.model.args = self.yolo_args
        
        # Init Loss and Optimizer
        self.criterion = v8DetectionLoss(self.model)
        if self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.937, weight_decay=0.0005)
        elif self.optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'adamw':
             self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported.")

        # History tracking
        self.history_train_loss = []
        self.history_val_loss = []
        self.history_val_accuracy = []
        self.pending_intermediate_payloads = defaultdict(deque)
        if self.canonical_gradient_mode:
            self.comm.create_queue(CANONICAL_GRADIENT_QUEUE)

    def _consume_intermediate_for_epoch(self, expected_epoch):
        pending = self.pending_intermediate_payloads[expected_epoch]
        if pending:
            return pending.popleft()

        while True:
            body = self.comm.consume_message_sync('intermediate_queue')
            receive_time = time.time()
            payload = pickle.loads(body)
            payload_epoch = payload.get('epoch')
            if payload_epoch is None:
                raise ValueError("Intermediate payload is missing 'epoch'.")
            payload_epoch = int(payload_epoch)

            if payload_epoch < expected_epoch:
                raise ValueError(
                    f"Received stale intermediate batch for epoch {payload_epoch}; "
                    f"expected epoch {expected_epoch}."
                )
            if payload_epoch == expected_epoch:
                return payload, receive_time

            self.pending_intermediate_payloads[payload_epoch].append(
                (payload, receive_time)
            )

    def _consume_canonical_prefix_gradient(self, client_id, expected_epoch, expected_batch_index):
        """Receive the cut-5 prefix gradients for one synchronous step."""
        body = self.comm.consume_message_sync(CANONICAL_GRADIENT_QUEUE)
        payload = pickle.loads(body)
        if payload.get('action') != 'canonical_prefix_gradient':
            raise ValueError(
                "Expected canonical_prefix_gradient, got "
                f"{payload.get('action')!r}."
            )
        if payload.get('client_id') != client_id:
            raise ValueError("Canonical prefix gradient came from an unexpected client.")
        if payload.get('epoch') != expected_epoch or payload.get('batch_index') != expected_batch_index:
            raise ValueError("Canonical prefix gradient belongs to a different training step.")
        return payload

    def train_one_epoch(self, epoch, global_epoch):
        if self.canonical_gradient_mode:
            return self._train_one_epoch_canonical(epoch, global_epoch)

        self.model.train()
        running_loss = 0.0
        route_batch_counts = {cut_layer: 0 for cut_layer in self.cut_layers}
        train_progress_bar = tqdm(range(self.nb), desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for i in train_progress_bar:
            payload, receive_inter_time = self._consume_intermediate_for_epoch(
                global_epoch
            )
            client_data_numpy = payload['client_output']
            label_data = payload['label_data']
            gradient_queue = payload['reply_to']
            cut_layer = payload.get('cut_layer')
            if cut_layer is None:
                raise ValueError("Intermediate payload is missing 'cut_layer'.")
            cut_layer = int(cut_layer)
            if cut_layer not in route_batch_counts:
                raise ValueError(
                    f"Received cut_layer={cut_layer}, but the server was "
                    f"configured for {sorted(route_batch_counts)}."
                )
            route_batch_counts[cut_layer] += 1

            client_tensors = []
            for client_np in client_data_numpy:
                t = torch.tensor(
                    client_np, 
                    dtype=torch.float32, 
                    device=self.device,
                    requires_grad=True
                )
                client_tensors.append(t)

            outputs = self.model(client_tensors, cut_layer=cut_layer)
            end_server_forward_time = time.time()
            loss, loss_items = self.criterion(outputs, label_data)
            self.optimizer.zero_grad()

            total_loss = loss.sum()
            total_loss.backward()
            self.optimizer.step()
            end_server_backward_time = time.time()

            grads_to_send = [t.grad.cpu() for t in client_tensors]
            response = {
                'gradient': grads_to_send,
                'loss': loss_items,
                'server_forward_time': end_server_forward_time - receive_inter_time,
                'server_backward_time': end_server_backward_time - end_server_forward_time,
                'receive_inter_time': receive_inter_time,
                'send_grad_time': end_server_backward_time
            }
            self.comm.publish_message(gradient_queue, pickle.dumps(response))

            train_progress_bar.set_postfix(
                total_loss=f'{total_loss.item():.4f}',
                box_loss=f'{loss_items[0].item():.4f}',
                cls_loss=f'{loss_items[1].item():.4f}',
                dfl_loss=f'{loss_items[2].item():.4f}'
            )

            running_loss += total_loss.item()
        return running_loss / len(train_progress_bar), loss_items, route_batch_counts

    def _train_one_epoch_canonical(self, epoch, global_epoch):
        """One-edge/one-server canonical-gradient training for uniform cut 5."""
        self.model.train()
        running_loss = 0.0
        route_batch_counts = {5: 0}
        loss_items = None
        train_progress_bar = tqdm(
            range(self.nb),
            desc=f"Epoch {epoch+1}/{self.num_epochs} [Canonical Cut-5]",
        )

        for batch_index in train_progress_bar:
            payload, receive_inter_time = self._consume_intermediate_for_epoch(
                global_epoch
            )
            if payload.get('action') != 'canonical_forward':
                raise ValueError(
                    "canonical_gradient_mode received a non-canonical intermediate payload."
                )
            if int(payload.get('cut_layer', -1)) != 5:
                raise ValueError("canonical_gradient_mode currently supports only cut_layer=5.")

            client_tensors = [
                torch.tensor(
                    client_np,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=True,
                )
                for client_np in payload['client_output']
            ]
            if len(client_tensors) != 2:
                raise ValueError("Canonical cut-5 route expects two boundary tensors.")

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model.forward_from_cut5(client_tensors)
            end_server_forward_time = time.time()
            loss, loss_items = self.criterion(outputs, payload['label_data'])
            total_loss = loss.sum()
            total_loss.backward()
            end_server_backward_time = time.time()

            boundary_response = {
                'action': 'canonical_boundary_gradient',
                'epoch': global_epoch,
                'batch_index': batch_index,
                'gradient': [tensor.grad.detach().cpu() for tensor in client_tensors],
                'loss': loss_items.detach().cpu(),
                'total_loss': total_loss.detach().item(),
                'server_forward_time': end_server_forward_time - receive_inter_time,
                'server_backward_time': end_server_backward_time - end_server_forward_time,
                'receive_inter_time': receive_inter_time,
                'send_grad_time': end_server_backward_time,
            }
            self.comm.publish_message(payload['reply_to'], pickle.dumps(boundary_response))

            prefix_payload = self._consume_canonical_prefix_gradient(
                client_id=payload['client_id'],
                expected_epoch=global_epoch,
                expected_batch_index=batch_index,
            )
            install_prefix_gradients(
                self.model,
                prefix_payload['prefix_gradients'],
                self.device,
            )
            install_prefix_buffers(
                self.model,
                prefix_payload['prefix_buffers'],
                self.device,
            )
            self.optimizer.step()
            end_server_update_time = time.time()

            update_response = {
                'action': 'canonical_prefix_update',
                'epoch': global_epoch,
                'batch_index': batch_index,
                'prefix_state': prefix_state_dict(self.model),
                'server_update_time': end_server_update_time - end_server_backward_time,
            }
            self.comm.publish_message(payload['reply_to'], pickle.dumps(update_response))

            route_batch_counts[5] += 1
            running_loss += total_loss.item()
            train_progress_bar.set_postfix(total_loss=f'{total_loss.item():.4f}')

        return running_loss / len(train_progress_bar), loss_items, route_batch_counts

    def validate_one_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_progress_bar = tqdm(self.validation_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
            for images, labels in val_progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(self.validation_loader)
        val_accuracy = 100 * correct / total
        
        self.history_val_loss.append(avg_val_loss)
        self.history_val_accuracy.append(val_accuracy)
        
        return avg_val_loss, val_accuracy

    def post_processing(self):
        # Save model
        if self.save_model_enabled:
            save_path = os.path.join(self.run_dir, 'cifar_net_server.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            # self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, epoch = 'last')
        else:
            print("Model saving skipped as per configuration.")

        print("Saving plots...")
        save_plots(self.history_train_loss, self.history_val_loss, self.history_val_accuracy, self.run_dir)
        print("Plots saved.")

    def run(self):
        print("Starting Training...")

        for epoch in range(self.num_epochs):
            if self.round >= 1 : global_epoch = epoch + self.num_epochs*self.round
            else: global_epoch = epoch

            avg_train_loss, loss_items, route_batch_counts = self.train_one_epoch(
                epoch,
                global_epoch,
            )

            # avg_val_loss, val_accuracy = self.validate_one_epoch(epoch)

            # Save checkpoint
            save_path = os.path.join(self.run_dir, f'cifar_net_server_{global_epoch+1}.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, client_id = self.client_id,
                                    epoch = global_epoch, loss_items = loss_items,
                                    route_batch_counts = route_batch_counts)
            
            # Log to CSV
            update_results_csv(epoch + 1, avg_train_loss, save_dir = self.run_dir)
        
        print("Finished Training.")
        self.post_processing()
        # self.comm.close()
