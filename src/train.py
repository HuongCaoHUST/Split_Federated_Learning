import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
from copy import deepcopy
from datetime import datetime
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
<<<<<<< HEAD
from model.YOLO11 import YOLO11_Full
from src.utils import update_results_csv, save_plots, count_parameters, create_run_dir
from src.dataset import Dataset
from src.communication import Communication
from src.mlflow import MLflowConnector
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
import numpy as np
from src.utils_box import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.ops import xywh2xyxy

MLFLOW_TRACKING_URI = "http://14.225.254.18:5000"
EXPERIMENT_NAME = "Split_Learning"
=======
from model.YOLO11n_custom import YOLO11_EDGE, YOLO11_SERVER, YOLO11_EDGE_5, YOLO11_SERVER_5, YOLO11_EDGE_15, YOLO11_SERVER_15, YOLO11_EDGE_20, YOLO11_SERVER_20
from src.utils import BatchLogger, update_results_csv, save_plots, count_parameters, create_run_dir, clear_memory
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
import numpy as np
from src.utils_box import non_max_suppression, scale_boxes, xywh2xyxy, box_iou
from ultralytics.utils.metrics import ap_per_class
>>>>>>> fed

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
<<<<<<< HEAD
        self.num_workers = config['training'].get('num_workers', 2)
=======
        self.num_workers = config['training'].get('num_workers', 0)
>>>>>>> fed
        self.num_epochs = config['training']['num_epochs']
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

<<<<<<< HEAD
        self.data_cfg = check_det_dataset("./datasets/livingroom_4_1.yaml")
        self.num_classes = self.data_cfg['nc']

        # 2. Initialize model with the correct number of classes
        self.model = YOLO11_Full(pretrained='yolo11n.pt', nc = self.num_classes).to(self.device)

        self.model.names = self.data_cfg['names']

        self.yolo_args = get_cfg(DEFAULT_CFG)
        self.model.args = self.yolo_args
        
        # Init Loss and Optimizer
        self.criterion = v8DetectionLoss(self.model)

=======
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

        if self.global_model_path is not None:
            print("Continue Training with global model: ", self.global_model_path)
            self.model = model_class(pretrained = self.global_model_path).to(self.device)
        else:
            self.model = model_class(pretrained = 'yolo11n.pt').to(self.device)

        self.model.names = self.data_cfg['names']
        self.yolo_args = get_cfg(DEFAULT_CFG)
        self.model.args = self.yolo_args

        # Init Optimizer
>>>>>>> fed
        if self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.937, weight_decay=0.0005)
        elif self.optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'adamw':
             self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported.")
<<<<<<< HEAD
        
        # 3. Initialize Dataset and DataLoader
=======

        # Initialize Dataset and DataLoader
>>>>>>> fed
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
<<<<<<< HEAD

        self.val_dataset = YOLODataset(
            img_path=self.data_cfg["val"],
            imgsz=640,
            data=self.data_cfg,
            augment=False,
            hyp=self.yolo_args,
            rect=False,
            stride=32
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn
        )
=======
>>>>>>> fed

        # History tracking
        self.history_train_loss = []
        self.history_val_loss = []
        self.history_map50 = []

        self.mlflow_connector = MLflowConnector(
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=EXPERIMENT_NAME
        )
        self.mlflow_connector.start_run(run_name="Centralized Learning")

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
        train_progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        epoch_latency_sum = np.zeros(7)
        
        for batch in train_progress_bar:
<<<<<<< HEAD
            images = batch['img'].to(self.device, non_blocking=True).float() / 255.0

            outputs = self.model(images)
            # v8DetectionLoss nhận vào output của model và toàn bộ batch
            # và trả về một tuple (total_loss, loss_components)
            loss, loss_items = self.criterion(outputs, batch)

            self.optimizer.zero_grad()
            total_loss = loss.sum()
            total_loss.backward()
            self.optimizer.step()
            
            running_loss += total_loss.item()

            # Hiển thị các loss thành phần trên thanh tiến trình
            train_progress_bar.set_postfix(
                total_loss=f'{total_loss.item():.4f}',
                box_loss=f'{loss_items[0].item():.4f}',
                cls_loss=f'{loss_items[1].item():.4f}',
                dfl_loss=f'{loss_items[2].item():.4f}'
            )
=======
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
                'reply_to': self.gradient_queue_name
            }

            data_bytes = pickle.dumps(payload)
            self.comm.publish_message(queue_name='intermediate_queue', message=data_bytes)
            send_inter_time = time.time()

            response_body = self.comm.consume_message_sync(self.gradient_queue_name)
            received_grad_time = time.time()
            response = pickle.loads(response_body)
            print("Received response keys: ", response.keys())

            server_grad_numpy = response['gradient']
            batch_loss = response['loss']
            server_forward_time = response.get('server_forward_time', 0)
            server_backward_time = response.get('server_backward_time', 0)
            receive_inter_time = response.get('receive_inter_time', 0)
            send_grad_time = response.get('send_grad_time', 0)

            self.optimizer.zero_grad()
>>>>>>> fed

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
<<<<<<< HEAD
        return avg_train_loss, loss_items
    
=======
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
        self.cut_layer = config['model']['cut_layer']
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)

        MODEL_MAP = {
            5: YOLO11_SERVER_5,
            10: YOLO11_SERVER,
            15: YOLO11_SERVER_15,
            20: YOLO11_SERVER_20
        }

        model_class = MODEL_MAP.get(self.cut_layer)

        # Initialize model
        if self.global_model_path is not None:
            print("Continue Training with global model: ", self.global_model_path)
            self.model = model_class(pretrained = self.global_model_path, nc = self.nc).to(self.device)
        else:
            self.model = model_class(pretrained = 'yolo11n.pt', nc = self.nc).to(self.device)
            
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

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(range(self.nb), desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for i in train_progress_bar:
            body = self.comm.consume_message_sync('intermediate_queue')
            receive_inter_time = time.time()
            payload = pickle.loads(body)
            client_data_numpy = payload['client_output']
            label_data = payload['label_data']
            gradient_queue = payload['reply_to']

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
        return running_loss / len(train_progress_bar), loss_items

>>>>>>> fed
    def validate_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        stats = [] 
        conf_thres = 0.001
        iou_thres = 0.7
        val_progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_progress_bar):
                images = batch['img'].to(self.device, non_blocking=True).float() / 255.0
                
                batch_idx_tensor = batch['batch_idx'].view(-1, 1).to(self.device)
                cls_tensor = batch['cls'].view(-1, 1).to(self.device)
                bboxes_tensor = batch['bboxes'].to(self.device)
                targets = torch.cat((batch_idx_tensor, cls_tensor, bboxes_tensor), 1)
                preds = self.model(images) 
                
                if isinstance(preds, tuple):
                    nms_input = preds[0]
                    loss_input = preds[1]
                else:
                    nms_input = preds
                    loss_input = preds
                loss, loss_items = self.criterion(loss_input, batch)
                running_loss += loss.sum().item()

                preds_nms = non_max_suppression(nms_input, conf_thres=conf_thres, iou_thres=iou_thres)
                for i, pred in enumerate(preds_nms):
                    target_labels = targets[targets[:, 0] == i][:, 1:]
                    nl, npr = target_labels.shape[0], pred.shape[0]
                    correct = torch.zeros(npr, 10, dtype=torch.bool, device=self.device)

                    if npr == 0:
                        if nl:
                            stats.append((correct.cpu(), torch.tensor([], device='cpu'), torch.tensor([], device='cpu'), target_labels[:, 0].cpu()))
                        continue

                    if nl:
                        target_boxes = xywh2xyxy(target_labels[:, 1:]) 
                        target_boxes[:, [0, 2]] *= images.shape[3]
                        target_boxes[:, [1, 3]] *= images.shape[2]
                        labels_pixel = torch.cat((target_labels[:, 0:1], target_boxes), 1)
                        correct = self.process_batch(pred, labels_pixel)

                    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_labels[:, 0].cpu()))

                val_progress_bar.set_postfix(val_loss=f'{loss.sum().item():.4f}')

        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        
        if len(stats) and stats[0].any():
            results = ap_per_class(*stats, plot=False, save_dir=self.run_dir, names=self.model.names)
            
            p, r, ap50, ap = results[2], results[3], results[5][:, 0], results[5].mean(1)
            
            mp = p.mean()       # Mean Precision
            mr = r.mean()       # Mean Recall
            map50 = ap50.mean() # mAP@0.5
            map5095 = ap.mean() # mAP@0.5:0.95
        else:
            mp, mr, map50, map5095 = 0.0, 0.0, 0.0, 0.0

        print(f"Validation Results: Precision: {mp:.4f}, Recall: {mr:.4f}, mAP50: {map50:.4f}, mAP50-95: {map5095:.4f}")
        
        avg_val_loss = running_loss / len(self.val_loader)
        
        return avg_val_loss, loss_items, map50, map5095, mp, mr
    
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

    def post_processing(self, epoch):
        # Save checkpoint
        if self.save_model_enabled:
<<<<<<< HEAD
            save_path = os.path.join(self.run_dir, self.model_save_path)
            model_to_save = deepcopy(self.model).cpu()
            if hasattr(self, 'config'):
                model_to_save.args = self.config 
            if hasattr(self.model, 'names'):
                model_to_save.names = self.model.names
            elif hasattr(self, 'classes'):
                model_to_save.names = self.classes
=======
            save_path = os.path.join(self.run_dir, 'cifar_net_server.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            # self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, epoch = 'last')
        else:
            print("Model saving skipped as per configuration.")
>>>>>>> fed

            checkpoint = {
                'epoch': epoch,
                'best_fitness': None,
                'model': model_to_save,
                'optimizer': self.optimizer.state_dict(),
                'train_loss_history': getattr(self, 'history_train_loss', []),
                'date': datetime.now().isoformat(),
            }

            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved to {save_path}")            
        else:
            print("Checkpoint saving skipped as per configuration.")

    def run(self):
        print("Starting Training...")
<<<<<<< HEAD
        final_epoch = 0
        for epoch in range(self.num_epochs):
            final_epoch = epoch
            avg_train_loss, train_loss_items = self.train_one_epoch(epoch)

            self.history_train_loss.append(avg_train_loss)

            avg_val_loss, val_loss_items, map50, map5095, mp, mr = self.validate_one_epoch(epoch)

            self.history_val_loss.append(avg_val_loss)
            self.history_map50.append(map50)

            self.mlflow_connector.log_metrics({
                "train/box_loss": train_loss_items[0].item(),
                "train/cls_loss": train_loss_items[1].item(),
                "train/dfl_loss": train_loss_items[2].item(),
                "val/box_loss": val_loss_items[0].item(),
                "val/cls_loss": val_loss_items[1].item(),
                "val/dfl_loss": val_loss_items[2].item(),
                "metrics/precision": mp,
                "metrics/recall": mr,
                "metrics/mAP50": map50,
                "metrics/mAP50-95": map5095,             
                }, step=epoch+1)

            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, mAP@.50: {map50:.4f}, mAP@.5-.95: {map5095:.4f}")
            update_results_csv(epoch + 1, avg_train_loss, avg_val_loss, mp, mr, map50, map5095, self.run_dir)

        print("Finished Training.")
        save_plots(self.history_train_loss, self.history_val_loss, self.history_map50, self.run_dir)
        self.mlflow_connector.end_run()
        self.comm.close()
        self.post_processing(final_epoch)

def train(config, device, num_classes, project_root):
    trainer = Trainer(config, device, num_classes, project_root)
    trainer.run()
=======

        for epoch in range(self.num_epochs):
            avg_train_loss, loss_items = self.train_one_epoch(epoch)

            # avg_val_loss, val_accuracy = self.validate_one_epoch(epoch)

            # Save checkpoint
            if self.round >= 1 : global_epoch = epoch + self.num_epochs*self.round 
            else: global_epoch = epoch

            save_path = os.path.join(self.run_dir, f'cifar_net_server_{global_epoch+1}.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            self.comm.publish_model(queue_name='server_queue', model_path = save_path, layer_id = self.layer_id, client_id = self.client_id,
                                    epoch = global_epoch, loss_items = loss_items)
            
            # Log to CSV
            update_results_csv(epoch + 1, avg_train_loss, save_dir = self.run_dir)
        
        print("Finished Training.")
        self.post_processing()
        # self.comm.close()
>>>>>>> fed
