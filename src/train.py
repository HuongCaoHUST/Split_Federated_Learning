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
from model.YOLO11n_custom import YOLO11_EDGE, YOLO11_SERVER
from src.utils import update_results_csv, save_plots, count_parameters, create_run_dir, clear_memory
from src.dataset import Dataset
from src.communication import Communication
from src.server import Server
from src.mlflow import MLflowConnector
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
import numpy as np
from src.utils_box import non_max_suppression, scale_boxes, xywh2xyxy, box_iou
from ultralytics.utils.metrics import ap_per_class

MLFLOW_TRACKING_URI = "http://14.225.254.18:5000"
EXPERIMENT_NAME = "Split_Learning"

class TrainerEdge:
    def __init__(self, config, device, project_root, comm, layer_id, client_id):
        self.config = config
        self.device = device
        self.project_root = project_root
        self.comm = comm
        self.layer_id = layer_id
        self.client_id = client_id
        
        # Set Hyperparameters
        self.run_dir = create_run_dir(project_root, layer_id)
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training'].get('num_workers', 0)
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.momentum = config['training'].get('momentum', 0.9)
        self.model_name = config['model']['edge']
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)
        self.pretrained_path = config['model'].get('pretrained_path')

        # Initialize model
        self.data_cfg = check_det_dataset("./datasets/livingroom_4_1.yaml")
        self.num_classes = self.data_cfg['nc']
        self.model = YOLO11_EDGE(pretrained = 'yolo11n.pt').to(self.device)

        self.model.names = self.data_cfg['names']
        self.yolo_args = get_cfg(DEFAULT_CFG)
        self.model.args = self.yolo_args

        # Init Optimizer
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

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch in train_progress_bar:
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
                'label_data': label_data
            }

            data_bytes = pickle.dumps(payload)
            self.comm.publish_message(queue_name='intermediate_queue', message=data_bytes)

            response_body = self.comm.consume_message_sync('edge_queue')
            response = pickle.loads(response_body)

            server_grad_numpy = response['gradient']
            batch_loss = response['loss']

            self.optimizer.zero_grad()

            grad_tensors = []
            for g in server_grad_numpy:
                if isinstance(g, torch.Tensor):
                    grad_tensors.append(g.to(self.device))
                else:
                    grad_tensors.append(torch.from_numpy(g).to(self.device))

            torch.autograd.backward(outputs, grad_tensors)
            self.optimizer.step()
            # running_loss += batch_loss
            # train_progress_bar.set_postfix({'server_loss': batch_loss})
        clear_memory(device = self.device, threshold=0.85)
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
    
    def post_processing(self):
        # Save model
        if self.save_model_enabled:
            save_path = os.path.join(self.run_dir, 'cifar_net_edge.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, epoch = 'last')
        else:
            print("Model saving skipped as per configuration.")

        print("Plots saved.")
    
    def run(self):
        print("Starting Training...")

        nb_train = len(self.train_loader)
        self.comm.send_training_metadata('server_queue', self.client_id, nb_train)

        for epoch in range(self.num_epochs):
            avg_train_loss = self.train_one_epoch(epoch)
            print(f'Epoch [{epoch+1}/{self.num_epochs}] -> Train Loss: {avg_train_loss:.4f}')

            # Save checkpoint
            save_path = os.path.join(self.run_dir, f'cifar_net_server_{epoch+1}.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, epoch = epoch)
        
        print("Finished Training.")
        self.post_processing()
        self.comm.close()

class TrainerServer:
    def __init__(self, config, device, project_root, comm, layer_id, client_id, nb, nc, class_names):
        self.config = config
        self.device = device
        self.project_root = project_root
        self.comm = comm
        self.layer_id = layer_id
        self.client_id = client_id
        self.nb = nb
        self.nc = nc
        self.class_names = class_names
        print("Num class: ", self.nc)
        
        # Set Hyperparameters
        self.run_dir = create_run_dir(project_root, layer_id)
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training'].get('num_workers', 0)
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.momentum = config['training'].get('momentum', 0.9)
        self.model_name = config['model']['server']
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)

        # Initialize model
        self.model = YOLO11_SERVER(pretrained = 'yolo11n.pt', nc = self.nc).to(self.device)
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

        self.mlflow_connector = MLflowConnector(
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=EXPERIMENT_NAME
        )
        self.mlflow_connector.start_run(run_name="New Split Learning")

        hyperparams = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_workers": self.num_workers,
            "num_epochs": self.num_epochs,
            "optimizer_name": self.optimizer_name,
            "model_name": self.model_name
        }
        self.mlflow_connector.log_params(hyperparams)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(range(self.nb), desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for i in train_progress_bar:
            body = self.comm.consume_message_sync('intermediate_queue')
            payload = pickle.loads(body)
            client_data_numpy = payload['client_output']
            label_data = payload['label_data']

            client_tensors = []
            for client_np in client_data_numpy:
                t = torch.tensor(
                    client_np, 
                    dtype=torch.float32, 
                    device=self.device,
                    requires_grad=True
                )
                client_tensors.append(t)

            outputs = self.model(client_tensors)
            loss, loss_items = self.criterion(outputs, label_data)
            self.optimizer.zero_grad()

            total_loss = loss.sum()
            total_loss.backward()
            self.optimizer.step()

            grads_to_send = [t.grad.cpu() for t in client_tensors]
            response = {
                'gradient': grads_to_send,
                'loss': loss_items
            }
            self.comm.publish_message('edge_queue', pickle.dumps(response))

            train_progress_bar.set_postfix(
                total_loss=f'{total_loss.item():.4f}',
                box_loss=f'{loss_items[0].item():.4f}',
                cls_loss=f'{loss_items[1].item():.4f}',
                dfl_loss=f'{loss_items[2].item():.4f}'
            )

            running_loss += total_loss.item()
        return running_loss / len(train_progress_bar)

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
            self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, epoch = 'last')
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
            save_path = os.path.join(self.run_dir, f'cifar_net_server_{epoch+1}.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, epoch = epoch)
            
            # Log to CSV
            update_results_csv(epoch + 1, avg_train_loss, save_dir = self.run_dir)
            print(f'Epoch [{epoch+1}/{self.num_epochs}] -> Train Loss: {avg_train_loss:.4f}')
        
        print("Finished Training.")
        self.mlflow_connector.end_run()
        self.post_processing()
        self.comm.close()
