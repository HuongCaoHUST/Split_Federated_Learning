from src.communication import Communication
from ultralytics.data.utils import check_det_dataset
from model.YOLO11n_custom import YOLO11_Full
from ultralytics.data.dataset import YOLODataset
from torch.utils.data import DataLoader
from src.utils import update_results_csv, create_run_dir
from src.utils_box import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss
from src.mlflow import MLflowConnector
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn

MLFLOW_TRACKING_URI = "http://14.225.254.18:5000"
EXPERIMENT_NAME = "Split_Learning"

class Server:
    def __init__(self, config, device):
        self.device = device
        config['rabbitmq']['host']='rabbitmq'
        self.num_client = config['clients']
        self.datasets = config['dataset']
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

        self.box_loss = []
        self.cls_loss = []
        self.dfl_loss = []
        
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
            "model_name": "YOLO11n"
        }
        self.mlflow_connector.log_params(hyperparams)

    def run(self):
        print("Server class initialized.")
        self.comm.connect()
        self.comm.delete_old_queues(['intermediate_queue', 'gradient_queue'])
        self.comm.create_queue('intermediate_queue')
        self.comm.create_queue('server_queue')
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
                    self.data_cfg = check_det_dataset(self.datasets[0])
                    self.num_classes = self.data_cfg['nc']
                    self.class_names = self.data_cfg['names']
                    nb = self.get_total_nb_by_layer(layer_id = 1)
                    self.comm.send_start_message(self.get_client_ids_by_layer(layer_id = 2), datasets = None, nb = nb, nc = self.num_classes, class_names = self.class_names)

            elif action == 'update_model':
                model_data = payload.get('model_data')
                layer_id = payload.get('layer_id')
                client_id = payload.get('client_id')
                epoch = payload.get('epoch')
                if layer_id == 2:
                    self.box_loss.append(payload.get('box_loss'))
                    self.cls_loss.append(payload.get('cls_loss'))
                    self.dfl_loss.append(payload.get('dfl_loss'))

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
                    model_full = YOLO11_Full(nc = self.num_classes)
                    print("Edge model: ", edge_model)
                    print("Server model: ", server_model)
                    
                    self.model = self.merged_model(
                        model_full,
                        edge_models_list=edge_model,
                        server_pt_path=server_model[0][0]
                    ).to(self.device)

                    self.data_cfg = check_det_dataset(self.datasets[0])
                    self.model.names = self.data_cfg['names']
                    self.yolo_args = get_cfg(DEFAULT_CFG)
                    self.model.args = self.yolo_args

                    self.criterion = v8DetectionLoss(self.model)

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

                    avg_val_loss, val_loss_items, map50, map5095, mp, mr = self.validate_one_epoch(epoch)

                    self.mlflow_connector.log_metrics({
                        "train/box_loss": self.box_loss[self.epoch - 1],
                        "train/cls_loss": self.cls_loss[self.epoch - 1],
                        "train/dfl_loss": self.dfl_loss[self.epoch - 1],
                        "val/box_loss": val_loss_items[0].item(),
                        "val/cls_loss": val_loss_items[1].item(),
                        "val/dfl_loss": val_loss_items[2].item(),
                        "metrics/precision": mp,
                        "metrics/recall": mr,
                        "metrics/mAP50": map50,
                        "metrics/mAP50-95": map5095,             
                        }, step=epoch+1)
                    update_results_csv(epoch + 1, avg_val_loss, map50, map5095, self.run_dir)
                    self.intermediate_model = [0,0]

                    if self.epoch % self.num_epochs == 0 and  self.round < self.num_rounds:
                        args_dict = vars(self.yolo_args)
                        save_path = f"{self.run_dir}/global_model_{self.round}.pt"
                        ckpt = {'model': self.model,
                                'args': args_dict,
                                'train_args': args_dict,
                                'epoch': -1}
                        torch.save(ckpt, save_path)
                        print(f"Model saved to {save_path}")
                        self.comm.publish_global_model(self.get_client_ids_by_layer(), global_model_path = save_path, round = self.round)
                        self.round += 1

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
        stats = [] 
        conf_thres = 0.001
        iou_thres = 0.7
        val_progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
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