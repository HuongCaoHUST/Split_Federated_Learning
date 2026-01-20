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

from model.Alexnet import AlexNet
from model.Mobilenet import MobileNet
from model.VGG16 import VGG16
from model.VGG16_EDGE import VGG16_EDGE
from model.VGG16_SERVER import VGG16_SERVER
from src.utils import update_results_csv, save_plots, count_parameters, create_run_dir
from src.dataset import Dataset
from src.communication import Communication
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
import numpy as np
from src.utils_box import non_max_suppression, scale_boxes, xywh2xyxy, box_iou
from ultralytics.utils.metrics import ap_per_class

class Trainer:
    def __init__(self, config, device, num_classes, project_root):
        self.config = config
        self.device = device
        self.num_classes = num_classes
        self.project_root = project_root
        
        # Set Hyperparameters
        self.run_dir = create_run_dir(project_root)
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training'].get('num_workers', 2)
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.momentum = config['training'].get('momentum', 0.9)
        self.model_name = config['model']['name']
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)
        self.pretrained_path = config['model'].get('pretrained_path')

        # Initialize RabbitMQ connection
        self.comm = Communication(config)

        self.data_cfg = check_det_dataset("./datasets/livingroom_4_1.yaml")
        print(f"Data configuration: {self.data_cfg}")
        self.num_classes = self.data_cfg['nc']

        # 2. Initialize model with the correct number of classes
        self.model = DetectionModel("yolo11n.yaml", nc=self.num_classes).to(self.device)

        # Load pretrained weights
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print(f"Loading pretrained weights from '{self.pretrained_path}'")
            checkpoint = torch.load(self.pretrained_path, map_location='cpu', weights_only=False)
            self.model.load(checkpoint)
        else:
            if self.pretrained_path:
                print(f"Pretrained weights not found at '{self.pretrained_path}'. Starting from scratch.")
            else:
                print("No pretrained weights specified. Starting from scratch.")

        self.model.names = self.data_cfg['names']

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
        
        # 3. Initialize Dataset and DataLoader
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

        # History tracking
        self.history_train_loss = []
        self.history_val_loss = []
        self.history_map50 = []

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
        
        for batch in train_progress_bar:
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

        avg_train_loss = running_loss / len(self.train_loader)
        return avg_train_loss
    
    def validate_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        stats = [] 
        conf_thres = 0.001
        iou_thres = 0.6
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
                    nms_input = pred
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
        
        return avg_val_loss, map50, map5095, mp, mr
    
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
            save_path = os.path.join(self.run_dir, self.model_save_path)
            model_to_save = deepcopy(self.model).cpu()
            if hasattr(self, 'config'):
                model_to_save.args = self.config 
            if hasattr(self.model, 'names'):
                model_to_save.names = self.model.names
            elif hasattr(self, 'classes'):
                model_to_save.names = self.classes

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
        final_epoch = 0
        for epoch in range(self.num_epochs):
            final_epoch = epoch
            avg_train_loss = self.train_one_epoch(epoch)

            self.history_train_loss.append(avg_train_loss)

            avg_val_loss, map50, map5095, mp, mr = self.validate_one_epoch(epoch)

            self.history_val_loss.append(avg_val_loss)
            self.history_map50.append(map50)

            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, mAP@.50: {map50:.4f}, mAP@.5-.95: {map5095:.4f}")
            update_results_csv(epoch + 1, avg_train_loss, avg_val_loss, mp, mr, map50, map5095, self.run_dir)

        print("Finished Training.")
        save_plots(self.history_train_loss, self.history_val_loss, self.history_map50, self.run_dir)

        self.comm.close()
        self.post_processing(final_epoch)

def train(config, device, num_classes, project_root):
    trainer = Trainer(config, device, num_classes, project_root)
    trainer.run()
