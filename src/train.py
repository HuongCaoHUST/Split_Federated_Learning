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

class Trainer:
    def __init__(self, config, device, num_classes, project_root):
        self.config = config
        self.device = device
        self.num_classes = num_classes
        self.project_root = project_root
        
        # Set Hyperparameters
        self.run_dir = create_run_dir(project_root)
        self.batch_size = config['training']['batch_size']
        self.num_workers = 2
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'Adam')
        self.momentum = config['training'].get('momentum', 0.9)
        self.model_name = config['model']['name']
        self.model_save_path = config['model']['save_path']
        self.save_model_enabled = config['model'].get('save_model', True)

        # Initialize RabbitMQ connection
        self.comm = Communication(config)

        # 1. Load dataset configuration first to get the number of classes
        from ultralytics.data.utils import check_det_dataset
        data_cfg = check_det_dataset("./data/livingroom_4_1.yaml")
        self.num_classes = data_cfg['nc']

        # 2. Initialize model with the correct number of classes
        from ultralytics.nn.tasks import DetectionModel
        self.model = DetectionModel("yolo11n.yaml", nc=self.num_classes)

        # Load pretrained weights
        pretrained_path = './yolo11n.pt'
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from '{pretrained_path}'")
            try:
                # File pretrained 'yolo11n.pt' chứa một đối tượng model được pickle, không chỉ là trọng số.
                # Kể từ PyTorch 2.6, `torch.load` mặc định là `weights_only=True` vì lý do bảo mật.
                # Chúng ta phải đặt nó thành `False` để tải loại checkpoint này.
                ckpt = torch.load(pretrained_path, map_location=self.device, weights_only=False)
                # Handle different checkpoint formats (full model vs. state_dict)
                state_dict = (ckpt.get('model') or ckpt).float().state_dict()
                self.model.load_state_dict(state_dict, strict=False)
                print("Pretrained weights loaded successfully for transfer learning.")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}. Starting from scratch.")
        else:
            print(f"Pretrained weights not found at '{pretrained_path}'. Starting from scratch.")

        self.model.names = data_cfg['names']

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
        from ultralytics.data.dataset import YOLODataset
        self.train_dataset = YOLODataset(
            img_path=data_cfg["train"],
            imgsz=640,
            data=data_cfg,
            augment=True,
            hyp=self.yolo_args,
            rect=False,
            stride=32
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self.train_dataset.collate_fn
        )

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

    def post_processing(self, epoch):
        # Save checkpoint
        if self.save_model_enabled:
            save_path = os.path.join(self.run_dir, self.model_save_path)
            
            # 1. Chuẩn bị model để lưu
            # Deepcopy để không ảnh hưởng model đang train trên GPU
            model_to_save = deepcopy(self.model).cpu()
            
            # --- QUAN TRỌNG: Gắn Metadata vào Model ---
            # Ultralytics cần 'args' và 'names' gắn trực tiếp vào object model 
            # để khi load lại (hoặc xem trên Netron) nó hiểu cấu trúc mạng.
            
            # Gán config (hyperparameters) vào model. 
            # Lưu ý: self.config nên là dict hoặc namespace chứa các tham số như 'model', 'data', etc.
            if hasattr(self, 'config'):
                model_to_save.args = self.config 
            
            # Gán tên class (nếu có trong dataset hoặc config)
            # Ví dụ: {0: 'person', 1: 'bicycle'}
            if hasattr(self.model, 'names'):
                model_to_save.names = self.model.names
            elif hasattr(self, 'classes'): # Trường hợp bạn lưu biến riêng
                model_to_save.names = self.classes

            # 2. Tạo dictionary checkpoint chuẩn Ultralytics
            checkpoint = {
                'epoch': epoch,
                'best_fitness': None, # Ultralytics thường check key này
                'model': model_to_save, # Model object đã gắn args/names
                'optimizer': self.optimizer.state_dict(), # Key chuẩn thường là 'optimizer' thay vì 'optimizer_state_dict'
                'train_loss_history': getattr(self, 'history_train_loss', []),
                'date': datetime.now().isoformat(),
            }
            
            # 3. Lưu file
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved to {save_path}")
            
            # (Tùy chọn) Nếu muốn file nhẹ hơn cho Inference, có thể dùng:
            # torch.save(checkpoint['model'], save_path_stripped) 
            
        else:
            print("Checkpoint saving skipped as per configuration.")

    def run(self):
        print("Starting Training...")
        final_epoch = 0
        for epoch in range(self.num_epochs):
            final_epoch = epoch
            avg_train_loss = self.train_one_epoch(epoch)
            print("avg_train_loss:", avg_train_loss)
            # avg_val_loss, val_accuracy = self.validate_one_epoch(epoch)
            
            # Log to CSV
            # update_results_csv(epoch + 1, avg_train_loss, avg_val_loss, val_accuracy, self.run_dir)
            # print(f'Epoch [{epoch+1}/{self.num_epochs}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        print("Finished Training.")
        self.comm.close()
        self.post_processing(final_epoch)

def train(config, device, num_classes, project_root):
    trainer = Trainer(config, device, num_classes, project_root)
    trainer.run()
