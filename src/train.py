import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.Alexnet import AlexNet
from model.Mobilenet import MobileNet
from model.VGG16 import VGG16
from model.VGG16_EDGE import VGG16_EDGE
from model.VGG16_SERVER import VGG16_SERVER
from src.utils import update_results_csv, save_plots, count_parameters, create_run_dir
from src.dataset import Dataset
from src.communication import Communication
from src.server import Server

class TrainerEdge:
    def __init__(self, config, device, num_classes, project_root):
        self.config = config
        self.device = device
        self.num_classes = num_classes
        self.project_root = project_root
        
        # Set Hyperparameters
        self.run_dir = create_run_dir(project_root)
        self.batch_size = config['training']['batch_size']
        self.num_workers = 0
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.momentum = config['training'].get('momentum', 0.9)
        self.model_name = config['model']['edge']
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)

        # Initialize RabbitMQ connection
        self.comm = Communication(config)

        # Load Dataset
        self.dataset_loader = Dataset(config, project_root)
        self.train_dataset, self.val_dataset = self.dataset_loader.prepare_datasets()
        
        # Create Dataloader
        print("Creating DataLoaders...")
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )
        self.validation_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

        # Initialize model
        self.model = self._init_model()
        
        # Init Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        if self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported. Please choose 'SGD' or 'Adam'.")

        # History tracking
        self.history_train_loss = []
        self.history_val_loss = []
        self.history_val_accuracy = []

    def _init_model(self):
        print(f"Initializing model: {self.model_name}...")
        model_map = {
            'AlexNet': AlexNet,
            'MobileNet': MobileNet,
            'VGG16': VGG16,
            'VGG16_EDGE': VGG16_EDGE,
            'VGG16_SERVER': VGG16_SERVER
        }

        if self.model_name not in model_map:
            print(f"Error: Model '{self.model_name}' not recognized. Supported models: {list(model_map.keys())}")
            sys.exit(1)
        
        model = model_map[self.model_name](num_classes=self.num_classes).to(self.device)
        print(f"Model Parameters: {count_parameters(model):,}")
        print("Model initialized.")
        return model

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch_idx, (images, labels) in enumerate(train_progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            payload = {
                'batch_idx': batch_idx,
                'client_output': outputs.detach().cpu().numpy(),
                'labels': labels.cpu().numpy()
            }
            
            data_bytes = pickle.dumps(payload)
            self.comm.publish_message(queue_name='intermediate_queue', message=data_bytes)

            response_body = self.comm.consume_message_sync('edge_queue')
            response = pickle.loads(response_body)

            server_grad_numpy = response['gradient']
            batch_loss = response['loss']

            grad_tensor = torch.from_numpy(server_grad_numpy).to(self.device)
            self.optimizer.zero_grad()
            outputs.backward(grad_tensor)
            self.optimizer.step()
            running_loss += batch_loss
            train_progress_bar.set_postfix({'server_loss': batch_loss})

        avg_train_loss = running_loss / len(self.train_loader)
        self.history_train_loss.append(avg_train_loss)
        return avg_train_loss

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
    
    def run(self):
        print("Starting Training...")
        self.comm.connect()

        for epoch in range(self.num_epochs):
            avg_train_loss = self.train_one_epoch(epoch)
            print(f'Epoch [{epoch+1}/{self.num_epochs}] -> Train Loss: {avg_train_loss:.4f}')
        
        print("Finished Training.")
        self.comm.close()

class TrainerServer:
    def __init__(self, config, device, num_classes, project_root):
        self.config = config
        self.device = device
        self.num_classes = num_classes
        self.project_root = project_root
        
        # Set Hyperparameters
        self.run_dir = create_run_dir(project_root)
        self.batch_size = config['training']['batch_size']
        self.num_workers = 0
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.momentum = config['training'].get('momentum', 0.9)
        self.model_name = config['model']['server']
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)

        # Initialize RabbitMQ connection
        self.comm = Communication(config)

        # Initialize model
        self.model = self._init_model()
        
        # Init Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        if self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported. Please choose 'SGD' or 'Adam'.")

        # History tracking
        self.history_train_loss = []
        self.history_val_loss = []
        self.history_val_accuracy = []

    def _init_model(self):
        print(f"Initializing model: {self.model_name}...")
        model_map = {
            'AlexNet': AlexNet,
            'MobileNet': MobileNet,
            'VGG16': VGG16,
            'VGG16_EDGE': VGG16_EDGE,
            'VGG16_SERVER': VGG16_SERVER
        }

        if self.model_name not in model_map:
            print(f"Error: Model '{self.model_name}' not recognized. Supported models: {list(model_map.keys())}")
            sys.exit(1)
        
        model = model_map[self.model_name](num_classes=self.num_classes).to(self.device)
        print(f"Model Parameters: {count_parameters(model):,}")
        print("Model initialized.")
        return model

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(range(1407), desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for i in train_progress_bar:
            body = self.comm.consume_message_sync('intermediate_queue')
            payload = pickle.loads(body)
            client_data_numpy = payload['client_output']
            labels_numpy = payload['labels']

            client_tensor = torch.tensor(
                client_data_numpy, 
                dtype=torch.float32, 
                device=self.device,
                requires_grad=True
            )
            labels = torch.from_numpy(labels_numpy).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(client_tensor)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            grad_to_send = client_tensor.grad.cpu().numpy()
            response = {
                'gradient': grad_to_send,
                'loss': loss.item()
            }
            self.comm.publish_message('edge_queue', pickle.dumps(response))

            running_loss += loss.item()
        return running_loss / 2

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
            save_path = os.path.join(self.run_dir, self.model_save_path)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            print("Model saving skipped as per configuration.")

        print("Saving plots...")
        save_plots(self.history_train_loss, self.history_val_loss, self.history_val_accuracy, self.run_dir)
        print("Plots saved.")

    def run(self):
        print("Starting Training...")
        self.comm.connect()

        for epoch in range(self.num_epochs):
            avg_train_loss = self.train_one_epoch(epoch)
            # avg_val_loss, val_accuracy = self.validate_one_epoch(epoch)
            
            # Log to CSV
            update_results_csv(epoch + 1, avg_train_loss, save_dir = self.run_dir)
            print(f'Epoch [{epoch+1}/{self.num_epochs}] -> Train Loss: {avg_train_loss:.4f}')
        
        print("Finished Training.")
        self.comm.close()
        self.post_processing()

def train(config, device, num_classes, project_root, layer_id):
    if layer_id == 1:
        trainer = TrainerEdge(config, device, num_classes, project_root)
        trainer.run()
    elif layer_id == 2:
        trainer = TrainerServer(config, device, num_classes, project_root)
        trainer.run()
    elif layer_id == 0:
        server = Server(config)
        server.run()
    else:
        print(f"Layer not supported: {layer_id}")
