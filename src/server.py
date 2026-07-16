from src.communication import Communication
from model.Alexnet import AlexNet
from src.utils import update_results_csv, create_run_dir
from src.mlflow import MLflowConnector
from src.monitoring import DeviceMonitor
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd

MLFLOW_TRACKING_URI = "http://smart-hvac.io.vn:5005/"
EXPERIMENT_NAME = "Split_Learning"

class Server:
    def __init__(self, config, device):
        self.device = device
        config['rabbitmq']['host']='rabbitmq'
        self.num_client = config['clients']
        self.datasets = config['dataset']
        dataset_name = self.datasets.get('data', self.datasets.get('name', 'CIFAR10'))
        if str(dataset_name).upper() in ('CIFAR10', 'MNIST'):
            self.num_classes = 10
            self.class_names = [str(i) for i in range(10)]
        else:
            raise ValueError(f"Unsupported AlexNet classification dataset: {dataset_name}")
        self.client = {}
        self.comm = Communication(config)
        self.registed = [0,0]
        self.nb_count = 0
        self.run_dir = create_run_dir('./', layer_id = 0)
        self.intermediate_model = [0,0]
        self.intermediate_model_layer_1 = []
        self.intermediate_model_layer_2 = []

        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training'].get('num_workers', 0)
        self.num_epochs = config['training']['num_epochs']
        self.num_rounds = config['training']['num_rounds']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.cut_layer = config['model'].get('cut_layer')
        self.epoch = 1
        self.round = 1 
        self.best_fitness = 0.0

        self.box_loss = []
        self.cls_loss = []
        self.dfl_loss = []

        self.client_delays = []
        self.train_losses = {}
        self.monitoring_enabled = config.get('monitoring', {}).get('enabled', False)
        self.val_loader = self._build_validation_loader(dataset_name)
        
        self.mlflow_connector = MLflowConnector(
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=EXPERIMENT_NAME
        )
        self.run_id = self.mlflow_connector.start_run(run_name="New Split Learning").info.run_id

        hyperparams = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_workers": self.num_workers,
            "num_epochs": self.num_epochs,
            "optimizer_name": self.optimizer_name,
            "model_name": "AlexNet"
        }
        self.mlflow_connector.log_params(hyperparams)

    def run(self):
        print("Server class initialized.")
        self.comm.connect()
        self.comm.delete_old_queues(['intermediate_queue', 'gradient_queue'])
        self.comm.create_queue('intermediate_queue')
        self.comm.create_queue('server_queue')
        if self.monitoring_enabled:
            self.monitor = DeviceMonitor(run_id=self.run_id, gateway_url='14.225.254.18:9091')
            self.monitor.start()

        self.comm.consume_messages('server_queue', self.on_message)

    def on_message(self, ch, method, properties, body):
        try:
            payload = pickle.loads(body)
            action = payload.get('action')

            print(f"Received action: {action}")

            if action == 'register':
                layer_id = payload.get('layer_id')
                client_id = payload.get('client_id')
                self.client[client_id] = {"layer_id": layer_id}

                if layer_id == 1:
                    self.registed[0] += 1
                else:
                    self.registed[1] += 1

                if self.registed == self.num_client:
                    self.comm.send_start_message(self.get_client_ids_by_layer(layer_id = 1), datasets = self.datasets)

            elif action == 'send_number_batch':
                nb = payload.get('nb_train')
                client_id = payload.get('client_id')
                self.client[client_id]["nb_train"] = nb
                self.nb_count += 1

                if self.nb_count == self.num_client[0]:
                    nb = self.get_total_nb_by_layer(layer_id = 1)
                    self.comm.send_start_message(self.get_client_ids_by_layer(layer_id = 2), datasets = None, nb = nb, nc = self.num_classes, class_names = self.class_names)

            elif action == 'update_model':
                model_data = payload.get('model_data')
                layer_id = payload.get('layer_id')
                client_id = payload.get('client_id')
                epoch = payload.get('epoch')
                if layer_id == 2 and payload.get('train_loss') is not None:
                    self.train_losses[epoch] = float(payload['train_loss'])
                if layer_id == 1:
                    log_entry = {
                        'epoch': payload.get('epoch'),
                        'client_id': payload.get('client_id'),
                        'batch_e2e': payload.get('batch_e2e'),
                        'edge_forward': payload.get('edge_forward'),
                        'edge_backward': payload.get('edge_backward'),
                        'server_forward': payload.get('server_forward'),
                        'server_backward': payload.get('server_backward'),
                        'inter_delay': payload.get('inter_delay'),
                        'grad_delay': payload.get('grad_delay')
                    }
                    self.client_delays.append(log_entry)
                save_path = f"{self.run_dir}/client_layer_{layer_id}_epoch_{epoch+1}.pt"
                with open(save_path, "wb") as f:
                    f.write(model_data)
                print("Save path: ", save_path)

                idx = layer_id - 1
                self.intermediate_model[idx] += 1
                self.client[client_id][f"model_{epoch+1}"] = save_path
                edge_model = self.get_models_by_layer_and_epoch(layer_id=1, epoch=self.epoch)
                server_model = self.get_models_by_layer_and_epoch(layer_id=2, epoch=self.epoch)

                if len(edge_model) == self.num_client[0] and len(server_model) == self.num_client[1]:
                    print("Edge model: ", edge_model)
                    print("Server model: ", server_model)
                    edge_state = self.aggregate_states(edge_model)
                    server_state = self.aggregate_states(server_model, weighted=False)
                    self.model = AlexNet(num_classes=self.num_classes).to(self.device)
                    self.model.load_state_dict({**edge_state, **server_state}, strict=True)
                    full_path = f"{self.run_dir}/alexnet_epoch_{self.epoch}.pt"
                    torch.save(self.model.state_dict(), full_path)
                    print(f"Merged AlexNet model saved to {full_path}")
                    val_loss, precision, recall, accuracy = self.validate_one_epoch(epoch)

                    print("Delay tables: ", self.client_delays)
                    avg_delays = self.get_epoch_averages(self.epoch - 1)
                    metrics = {
                        "latency/batch_e2e": avg_delays.get("batch_e2e", 0),
                        "latency/edge_forward": avg_delays.get("edge_forward", 0),
                        "latency/edge_backward": avg_delays.get("edge_backward", 0),
                        "latency/server_forward": avg_delays.get("server_forward", 0),
                        "latency/server_backward": avg_delays.get("server_backward", 0),
                        "latency/inter_delay": avg_delays.get("inter_delay", 0),
                        "latency/grad_delay": avg_delays.get("grad_delay", 0),
                    }
                    train_loss = self.train_losses.get(epoch)
                    if train_loss is not None:
                        metrics["train/loss"] = train_loss
                    metrics.update({
                        "val/loss": val_loss,
                        "val/precision": precision,
                        "val/recall": recall,
                        "val/accuracy": accuracy,
                    })
                    self.mlflow_connector.log_metrics(metrics, step=epoch + 1)
                    update_results_csv(
                        epoch + 1, train_loss, val_loss, accuracy * 100, self.run_dir
                    )

                    # Save global model
                    if self.epoch % self.num_epochs == 0 and  self.round < self.num_rounds:
                        edge_path = f"{self.run_dir}/global_edge_{self.round}.pt"
                        server_path = f"{self.run_dir}/global_server_{self.round}.pt"
                        torch.save(edge_state, edge_path)
                        torch.save(server_state, server_path)
                        self.comm.publish_global_model(
                            self.get_client_ids_by_layer(layer_id=1), edge_path, self.round
                        )
                        self.comm.publish_global_model(
                            self.get_client_ids_by_layer(layer_id=2), server_path, self.round
                        )
                        self.round += 1
                    
                    self.intermediate_model = [0,0]
                    self.epoch += 1
            else:
                print(f"Unknown action: {action}")

        except pickle.UnpicklingError:
            print("Error when unpack message.")
        except Exception as e:
            print(f"Error processing message: {e}")

    def get_client_ids_by_layer(self, layer_id=None):
        return [
            client_id for client_id, info in self.client.items() 
            if layer_id is None or info.get("layer_id") == layer_id
        ]
    
    def get_models_by_layer_and_epoch(self, layer_id, epoch):
        key = f"model_{epoch}"
        models = []
        for client_id, info in self.client.items():
            if info.get("layer_id") == layer_id and key in info:
                nb = info.get("nb_train", 0)
                models.append((info[key], nb))
        return models
    
    def get_total_nb_by_layer(self, layer_id):
        return sum(info.get("nb_train", 0) for info in self.client.values() if info.get("layer_id") == layer_id)

    def _build_validation_loader(self, dataset_name):
        transform_steps = [transforms.Resize((227, 227))]
        if str(dataset_name).upper() == 'MNIST':
            transform_steps.append(transforms.Grayscale(num_output_channels=3))
            dataset_class = torchvision.datasets.MNIST
        else:
            dataset_class = torchvision.datasets.CIFAR10
        transform_steps.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = dataset_class(
            root='./data', train=False, download=True,
            transform=transforms.Compose(transform_steps)
        )
        fraction = float(self.datasets.get('validation_fraction', self.datasets.get('subset_fraction', 1.0)))
        if not 0 < fraction <= 1:
            raise ValueError('dataset.validation_fraction must be in the range (0, 1].')
        if fraction < 1.0:
            size = max(1, int(len(dataset) * fraction))
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:size]
            dataset = Subset(dataset, indices.tolist())
        print(f"Server validation samples: {len(dataset)}", flush=True)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.device.type == 'cuda'
        )

    @staticmethod
    def aggregate_states(models, weighted=True):
        """Average compatible AlexNet state dictionaries."""
        if not models:
            raise ValueError("No model checkpoints to aggregate.")
        total_weight = sum(samples for _, samples in models) if weighted else len(models)
        if total_weight <= 0:
            total_weight = len(models)
            weighted = False

        averaged = {}
        for path, samples in models:
            state = torch.load(path, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            weight = (samples / total_weight) if weighted else (1.0 / len(models))
            for key, value in state.items():
                contribution = value * weight
                averaged[key] = contribution if key not in averaged else averaged[key] + contribution
        return averaged
    
    def merged_model(self, full_model, edge_models_list, server_pt_path):
        server_state = torch.load(server_pt_path, map_location='cpu')
        if 'model_state_dict' in server_state: server_state = server_state['model_state_dict']
        elif 'model' in server_state: server_state = server_state['model']

        full_sd = full_model.state_dict()
        merged_sd = {}

        # Edge side model
        print(f"Aggregating {len(edge_models_list)} edge models...")
        total_samples = sum(item[1] for item in edge_models_list)
        if total_samples == 0:
            raise ValueError("Total samples is 0, cannot calculate weighted average.")

        averaged_edge_state = {}

        for path, num_samples in edge_models_list:
            client_state = torch.load(path, map_location='cpu')
            if 'model_state_dict' in client_state: client_state = client_state['model_state_dict']
            elif 'model' in client_state: client_state = client_state['model']
        
            weight_factor = num_samples / total_samples
            
            for key, value in client_state.items():
                clean_key = key.replace('model.', '').replace('layers.', '')
                layer_idx = int(clean_key.split('.')[0])
                if layer_idx <= self.cut_layer:
                    if clean_key not in averaged_edge_state:
                        averaged_edge_state[clean_key] = value * weight_factor
                    else:
                        averaged_edge_state[clean_key] += value * weight_factor
        for clean_key, value in averaged_edge_state.items():
            target_key = f"layers.{clean_key}"
            
            if target_key in full_sd:
                if full_sd[target_key].shape == value.shape:
                    merged_sd[target_key] = value
                else:
                    print(f"Incorrect size at {target_key}: Code {full_sd[target_key].shape} != File {value.shape}")
        
        # Server side model
        SERVER_OFFSET = self.cut_layer + 1
        for key, value in server_state.items():
            clean_key = key.replace('model.', '').replace('layers.', '')
            parts = clean_key.split('.')
            if parts[0].isdigit():
                old_idx = int(parts[0])

                new_idx = old_idx + SERVER_OFFSET
                new_key_parts = [str(new_idx)] + parts[1:]
                target_key = f"layers.{'.'.join(new_key_parts)}"
                
                if target_key in full_sd:
                    if full_sd[target_key].shape == value.shape:
                        merged_sd[target_key] = value
                    else:
                        print(f"Incorrect size at {target_key} (Gốc {old_idx}->Mới {new_idx}): Code {full_sd[target_key].shape} != File {value.shape}")
                else:
                    pass
        full_model.load_state_dict(merged_sd, strict=False)
        print("\nMerged model success.")
        return full_model
    
    def validate_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long, device=self.device
        )
        criterion = nn.CrossEntropyLoss()
        val_progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
        
        with torch.no_grad():
            for images, labels in val_progress_bar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(images)
                loss = criterion(logits, labels)
                running_loss += loss.item()
                predictions = logits.argmax(dim=1)
                indices = labels * self.num_classes + predictions
                confusion += torch.bincount(
                    indices, minlength=self.num_classes ** 2
                ).reshape(self.num_classes, self.num_classes)

        avg_val_loss = running_loss / len(self.val_loader)
        true_positive = confusion.diag().float()
        predicted_positive = confusion.sum(dim=0).float()
        actual_positive = confusion.sum(dim=1).float()
        precision_per_class = true_positive / predicted_positive.clamp_min(1)
        recall_per_class = true_positive / actual_positive.clamp_min(1)
        precision = precision_per_class.mean().item()
        recall = recall_per_class.mean().item()
        accuracy = (true_positive.sum() / confusion.sum().clamp_min(1)).item()
        print(
            f"[Validation][Epoch {epoch + 1}] loss={avg_val_loss:.4f} - "
            f"P={precision:.4f} - R={recall:.4f} - Accuracy={accuracy:.4f}",
            flush=True,
        )
        return avg_val_loss, precision, recall, accuracy
    
    def process_batch(self, detections, labels):
        iou_v = torch.linspace(0.5, 0.95, 10, device=self.device)
        n_iou = iou_v.numel()
        correct = torch.zeros(detections.shape[0], n_iou, dtype=torch.bool, device=self.device)

        if labels.shape[0] == 0:
            return correct
        
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where((iou >= iou_v[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU > 0.5 và cùng class
        
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1).float(), iou[x[0], x[1]][:, None]), 1)
            if x[0].shape[0] > 1:
                # Vectorized greedy matching
                matches_np = matches.cpu().numpy()
                matches_np = matches_np[matches_np[:, 2].argsort()[::-1]]
                matches_np = matches_np[np.unique(matches_np[:, 1], return_index=True)[1]]
                matches_np = matches_np[matches_np[:, 2].argsort()[::-1]]
                matches_np = matches_np[np.unique(matches_np[:, 0], return_index=True)[1]]
                matches = torch.from_numpy(matches_np).to(self.device)
            
            # For the final one-to-one matches, check against all IoU thresholds
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iou_v
            
        return correct
    
    def get_epoch_averages(self, epoch):
        if not self.client_delays:
            return {}

        df = pd.DataFrame(self.client_delays)
        epoch_summary = df[df['epoch'] == epoch].groupby('epoch').mean(numeric_only=True)
        if not epoch_summary.empty:
            return epoch_summary.iloc[0].to_dict()
        return {}
