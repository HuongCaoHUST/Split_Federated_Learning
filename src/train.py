import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time
from model.Alexnet_EDGE import AlexNet_EDGE
from model.Alexnet_SERVER import AlexNet_SERVER
from src.utils import BatchLogger, update_results_csv, save_plots, clear_memory
import numpy as np

MLFLOW_TRACKING_URI = "http://smart-hvac.io.vn:5005/"
EXPERIMENT_NAME = "Split_Learning"


def _load_checkpoint(model, checkpoint_path, device):
    """Load an AlexNet split checkpoint saved by this project."""
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def _classification_dataset(config, project_root, dataset_name=None):
    """Create the classification dataset expected by AlexNet."""
    dataset_config = config.get("dataset", {})
    if isinstance(dataset_config, dict):
        name = dataset_config.get("data", dataset_config.get("name", dataset_name or "CIFAR10"))
    elif isinstance(dataset_name, str) and not dataset_name.endswith((".yaml", ".yml")):
        name = dataset_name
    else:
        raise ValueError(
            "AlexNet performs image classification. Set config['dataset'] to a mapping, "
            "for example: {data: CIFAR10}; a YOLO detection YAML is not compatible."
        )

    transform_steps = [transforms.Resize((227, 227))]
    if str(name).upper() == "MNIST":
        transform_steps.extend([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset_class = torchvision.datasets.MNIST
    elif str(name).upper() == "CIFAR10":
        transform_steps.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset_class = torchvision.datasets.CIFAR10
    else:
        raise ValueError(f"Unsupported AlexNet classification dataset: {name}")

    dataset = dataset_class(
        root=os.path.join(project_root, "data"),
        train=True,
        download=True,
        transform=transforms.Compose(transform_steps),
    )
    class_names = dataset.classes
    subset_fraction = float(dataset_config.get("subset_fraction", 1.0))
    if not 0 < subset_fraction <= 1:
        raise ValueError("dataset.subset_fraction must be in the range (0, 1].")
    if subset_fraction < 1.0:
        subset_size = max(1, int(len(dataset) * subset_fraction))
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
        dataset = Subset(dataset, indices)
        print(
            f"Using {subset_size} samples ({subset_fraction * 100:.2f}% of {name}).",
            flush=True,
        )
    return dataset, len(class_names), class_names

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
        self.log_interval = config['training'].get('log_interval', 10)
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.momentum = config['training'].get('momentum', 0.9)
        self.model_name = config['model']['edge']
        self.cut_layer = config['model']['cut_layer']
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)
        self.pretrained_path = config['model'].get('pretrained_path')

        # Create gradient queue
        self.gradient_queue_name = f'gradient_queue_{client_id}'
        self.comm.create_queue(self.gradient_queue_name)

        # Initialize batch logger
        self.batch_logger = BatchLogger(self.client_id, "training_log.csv")

        # AlexNet is split after its convolutional feature extractor.
        self.train_dataset, self.num_classes, self.class_names = _classification_dataset(
            config, project_root, datasets
        )
        self.model = AlexNet_EDGE(num_classes=self.num_classes).to(self.device)
        if self.global_model_path is not None:
            print("Continue Training with global model: ", self.global_model_path)
            _load_checkpoint(self.model, self.global_model_path, self.device)

        # Init Optimizer
        if self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=0.0005)
        elif self.optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'adamw':
             self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported.")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda"
        )

        # History tracking
        self.history_train_loss = []
        self.history_val_loss = []
        self.history_val_accuracy = []

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        epoch_latency_sum = np.zeros(7)
        
        for batch_idx, (images, labels) in enumerate(train_progress_bar, start=1):
            start_batch_time = time.time()
            images = images.to(self.device, non_blocking=True)
            outputs = self.model(images)
            payload = {
                'client_output': outputs.detach().cpu().numpy(),
                'labels': labels.cpu().numpy(),
                'reply_to': self.gradient_queue_name
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

            server_grad = torch.as_tensor(server_grad_numpy, device=self.device)
            outputs.backward(server_grad)
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
            running_loss += float(batch_loss)
            train_progress_bar.set_postfix(server_loss=f'{float(batch_loss):.4f}')
            if batch_idx == 1 or batch_idx % self.log_interval == 0:
                print(
                    f"[AlexNet][Edge][Epoch {epoch + 1}/{self.num_epochs}] "
                    f"Batch {batch_idx}/{len(self.train_loader)} - "
                    f"loss={float(batch_loss):.4f} - avg_loss={running_loss / batch_idx:.4f}",
                    flush=True,
                )
        clear_memory(device = self.device, threshold=0.85)
        avg_train_loss = running_loss / len(self.train_loader)
        self.history_train_loss.append(avg_train_loss)
        avg_latencies = epoch_latency_sum / len(self.train_loader)
        return avg_train_loss, avg_latencies

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
            avg_train_loss, latencies = self.train_one_epoch(epoch)
            print(f'Epoch [{epoch+1}/{self.num_epochs}] -> Train Loss: {avg_train_loss:.4f}')

            # Save checkpoint
            if self.round >= 1 : global_epoch = epoch + self.num_epochs*self.round 
            else: global_epoch = epoch

            save_path = os.path.join(self.run_dir, f'cifar_net_edge_{global_epoch+1}.pt')
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
        self.log_interval = config['training'].get('log_interval', 10)
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.momentum = config['training'].get('momentum', 0.9)
        self.model_name = config['model']['server']
        self.cut_layer = config['model']['cut_layer']
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)

        # Initialize model
        self.model = AlexNet_SERVER(num_classes=self.nc).to(self.device)
        if self.global_model_path is not None:
            print("Continue Training with global model: ", self.global_model_path)
            _load_checkpoint(self.model, self.global_model_path, self.device)
        
        # Init Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        if self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=0.0005)
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

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(range(self.nb), desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for i in train_progress_bar:
            body = self.comm.consume_message_sync('intermediate_queue')
            receive_inter_time = time.time()
            payload = pickle.loads(body)
            client_data_numpy = payload['client_output']
            labels = torch.as_tensor(payload['labels'], dtype=torch.long, device=self.device)
            gradient_queue = payload['reply_to']

            client_tensor = torch.tensor(
                client_data_numpy, dtype=torch.float32, device=self.device, requires_grad=True
            )
            outputs = self.model(client_tensor)
            end_server_forward_time = time.time()
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()
            end_server_backward_time = time.time()

            response = {
                'gradient': client_tensor.grad.detach().cpu().numpy(),
                'loss': loss.item(),
                'server_forward_time': end_server_forward_time - receive_inter_time,
                'server_backward_time': end_server_backward_time - end_server_forward_time,
                'receive_inter_time': receive_inter_time,
                'send_grad_time': end_server_backward_time
            }
            self.comm.publish_message(gradient_queue, pickle.dumps(response))

            train_progress_bar.set_postfix(loss=f'{loss.item():.4f}')

            running_loss += loss.item()
            batch_idx = i + 1
            if batch_idx == 1 or batch_idx % self.log_interval == 0:
                print(
                    f"[AlexNet][Server][Epoch {epoch + 1}/{self.num_epochs}] "
                    f"Batch {batch_idx}/{self.nb} - loss={loss.item():.4f} - "
                    f"avg_loss={running_loss / batch_idx:.4f}",
                    flush=True,
                )
        return running_loss / self.nb

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
            avg_train_loss = self.train_one_epoch(epoch)

            # avg_val_loss, val_accuracy = self.validate_one_epoch(epoch)

            # Save checkpoint
            if self.round >= 1 : global_epoch = epoch + self.num_epochs*self.round 
            else: global_epoch = epoch

            save_path = os.path.join(self.run_dir, f'cifar_net_server_{global_epoch+1}.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, client_id = self.client_id,
                                    epoch = global_epoch, train_loss=avg_train_loss)
            
            # Log to CSV
            update_results_csv(epoch + 1, avg_train_loss, save_dir = self.run_dir)
        
        print("Finished Training.")
        self.post_processing()
        # self.comm.close()
