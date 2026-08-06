from src.communication import Communication
from ultralytics.data.utils import check_det_dataset
from model.YOLO11n_custom import YOLO11_Full
from ultralytics.data.dataset import YOLODataset
from torch.utils.data import DataLoader
from src.utils import (
    update_results_csv,
    create_run_dir,
    get_client_cut_layers,
    get_server_cut_layer,
)
from src.utils_box import non_max_suppression
from src.canonical_gradient import (
    CANONICAL_GRADIENT_QUEUE,
    validate_canonical_cut5_config,
)
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss
from src.mlflow import MLflowConnector
from src.monitoring import DeviceMonitor
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import pandas as pd

MLFLOW_TRACKING_URI = "http://smart-hvac.io.vn:5005/"
EXPERIMENT_NAME = "Split_Learning"

class Server:
    def __init__(self, config, device):
        self.device = device
        config['rabbitmq']['host']='rabbitmq'
        self.num_client = config['clients']
        self.client_cut_layers = get_client_cut_layers(config, self.num_client[0])
        self.cut_layer = get_server_cut_layer(self.client_cut_layers)
        self.canonical_gradient_mode = config['training'].get(
            'canonical_gradient_mode', False
        )
        if self.canonical_gradient_mode:
            validate_canonical_cut5_config(config)
        self.datasets = config['dataset']
        self.client = {}
        self.comm = Communication(config)
        self.registed = [0,0]
        self.metadata_clients = set()
        self.server_workers_started = False
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
        self.epoch = 1
        self.round = 1 
        self.best_fitness = 0.0

        self.box_loss = []
        self.cls_loss = []
        self.dfl_loss = []

        self.client_delays = []
        
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
            "model_name": "YOLO11n"
        }
        self.mlflow_connector.log_params(hyperparams)

    def run(self):
        print("Server class initialized.")
        self.comm.connect()
        self.comm.delete_old_queues([
            'server_queue',
            'intermediate_queue',
            'gradient_queue',
            CANONICAL_GRADIENT_QUEUE,
        ])
        self.comm.create_queue('intermediate_queue')
        self.comm.create_queue('server_queue')
        if self.canonical_gradient_mode:
            self.comm.create_queue(CANONICAL_GRADIENT_QUEUE)
        # self.monitor = DeviceMonitor(run_id=self.run_id, gateway_url='14.225.254.18:9091')
        # self.monitor.start()

        self.comm.consume_messages('server_queue', self.on_message)

    def on_message(self, ch, method, properties, body):
        try:
            payload = pickle.loads(body)
            action = payload.get('action')

            print(f"Received action: {action}")

            if action == 'register':
                layer_id = payload.get('layer_id')
                client_id = payload.get('client_id')

                if layer_id == 1:
                    client_index = self.registed[0]
                    cut_layer = self.client_cut_layers[client_index]
                    self.registed[0] += 1
                else:
                    client_index = self.registed[1]
                    cut_layer = self.cut_layer
                    self.registed[1] += 1

                self.client[client_id] = {
                    "layer_id": layer_id,
                    "client_index": client_index,
                    "cut_layer": cut_layer,
                }
                print(
                    f"Registered layer {layer_id} client #{client_index + 1} "
                    f"({client_id}) with cut_layer={cut_layer}"
                )

                if self.registed == self.num_client:
                    edge_client_ids = self.get_client_ids_by_layer(layer_id=1)
                    self.comm.send_start_message(
                        edge_client_ids,
                        datasets=self.datasets,
                        cut_layers=self.get_cut_layers_by_client_ids(edge_client_ids),
                    )

            elif action == 'send_number_batch':
                nb = payload.get('nb_train')
                client_id = payload.get('client_id')
                if client_id not in self.client:
                    raise ValueError(
                        f"Received batch metadata from unknown client {client_id}."
                    )
                self.client[client_id]["nb_train"] = nb
                if self.client[client_id].get("layer_id") == 1:
                    self.metadata_clients.add(client_id)

                edge_client_ids = self.get_client_ids_by_layer(layer_id=1)
                all_edge_metadata_received = all(
                    client_id in self.metadata_clients
                    for client_id in edge_client_ids
                )
                if all_edge_metadata_received and not self.server_workers_started:
                    self.data_cfg = check_det_dataset(self.datasets[0])
                    self.num_classes = self.data_cfg['nc']
                    self.class_names = self.data_cfg['names']
                    nb = self.get_total_nb_by_layer(layer_id = 1)
                    server_client_ids = self.get_client_ids_by_layer(layer_id=2)
                    worker_count = len(server_client_ids)
                    if worker_count == 0:
                        raise ValueError("No dynamic server worker is registered.")
                    if nb < worker_count:
                        raise ValueError(
                            f"Cannot distribute {nb} batches across "
                            f"{worker_count} server workers."
                        )
                    batches_per_worker, remainder = divmod(nb, worker_count)
                    batch_allocations = [
                        batches_per_worker + (worker_index < remainder)
                        for worker_index in range(worker_count)
                    ]
                    self.comm.send_start_message(
                        server_client_ids,
                        datasets=None,
                        nb=batch_allocations,
                        nc=self.num_classes,
                        class_names=self.class_names,
                        cut_layers=self.get_cut_layers_by_client_ids(server_client_ids),
                        supported_cut_layers=self.client_cut_layers,
                    )
                    self.server_workers_started = True

            elif action == 'update_model':
                model_data = payload.get('model_data')
                layer_id = payload.get('layer_id')
                client_id = payload.get('client_id')
                epoch = payload.get('epoch')
                if client_id not in self.client:
                    raise ValueError(f"Received model from unknown client {client_id}.")
                if epoch is None:
                    raise ValueError("Model update is missing 'epoch'.")
                model_epoch = int(epoch) + 1
                if model_epoch < self.epoch:
                    print(
                        f"Ignoring stale model from client {client_id} for "
                        f"epoch {model_epoch}; next expected epoch is {self.epoch}."
                    )
                    return
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
                elif layer_id == 2:
                    self.box_loss.append(payload.get('box_loss'))
                    self.cls_loss.append(payload.get('cls_loss'))
                    self.dfl_loss.append(payload.get('dfl_loss'))

                save_path = (
                    f"{self.run_dir}/client_{client_id}_layer_{layer_id}_"
                    f"epoch_{model_epoch}.pt"
                )
                with open(save_path, "wb") as f:
                    f.write(model_data)
                print("Save path: ", save_path)

                idx = layer_id - 1
                self.intermediate_model[idx] += 1
                self.client[client_id][f"model_{model_epoch}"] = save_path
                if payload.get('route_batch_counts') is not None:
                    self.client[client_id][f"route_batch_counts_{model_epoch}"] = {
                        int(cut_layer): int(batch_count)
                        for cut_layer, batch_count in payload['route_batch_counts'].items()
                    }

                if model_epoch != self.epoch:
                    print(
                        f"Stored model for future epoch {model_epoch}; "
                        f"waiting for epoch {self.epoch}."
                    )
                    return

                edge_model = self.get_models_by_layer_and_epoch(layer_id=1, epoch=self.epoch)
                server_model = self.get_models_by_layer_and_epoch(layer_id=2, epoch=self.epoch)

                if len(edge_model) == self.num_client[0] and len(server_model) == self.num_client[1]:
                    model_full = YOLO11_Full(nc = self.num_classes)
                    print("Edge model: ", edge_model)
                    print("Server model: ", server_model)
                    
                    if self.canonical_gradient_mode:
                        # In canonical-gradient mode the server checkpoint is
                        # already the complete canonical YOLO11 state. The edge
                        # checkpoint only acts as a completion signal and is
                        # intentionally not averaged a second time.
                        canonical_state = self._load_checkpoint_state(
                            server_model[0]['path']
                        )
                        model_full.load_state_dict(canonical_state, strict=True)
                        self.model = model_full.to(self.device)
                    else:
                        self.model = self.merged_model(
                            model_full,
                            edge_models_list=edge_model,
                            server_models_list=server_model,
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

                    print("Delay tables: ", self.client_delays)
                    avg_delays = self.get_epoch_averages(self.epoch - 1)
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
                        "latency/batch_e2e": avg_delays.get("batch_e2e", 0),
                        "latency/edge_forward": avg_delays.get("edge_forward", 0),
                        "latency/edge_backward": avg_delays.get("edge_backward", 0),
                        "latency/server_forward": avg_delays.get("server_forward", 0),
                        "latency/server_backward": avg_delays.get("server_backward", 0),
                        "latency/inter_delay": avg_delays.get("inter_delay", 0),
                        "latency/grad_delay": avg_delays.get("grad_delay", 0),
                        }, step=epoch+1)
                    update_results_csv(epoch + 1, avg_val_loss, map50, map5095, self.run_dir)

                    # Save best model
                    if map50 > self.best_fitness:
                        self.best_fitness = map50
                        best_path = f"{self.run_dir}/best.pt"
                        args_dict = vars(self.yolo_args)
                        best_ckpt = {
                            'model': self.model.state_dict(),
                            'nc': self.num_classes,
                            'names': self.class_names,
                            'args': args_dict,
                            'train_args': args_dict,
                            'epoch': epoch,
                            'metrics': {'mAP50': map50, 'mAP50-95': map5095}
                        }
                        torch.save(best_ckpt, best_path)
                        print(f"New best model saved to {best_path}")

                    # Save global model
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

    def get_cut_layers_by_client_ids(self, client_ids):
        return [self.client[client_id]["cut_layer"] for client_id in client_ids]
    
    def get_models_by_layer_and_epoch(self, layer_id, epoch):
        key = f"model_{epoch}"
        models = []
        for client_id, info in self.client.items():
            if info.get("layer_id") == layer_id and key in info:
                models.append({
                    "client_id": client_id,
                    "path": info[key],
                    "num_batches": info.get("nb_train", 0),
                    "cut_layer": info.get("cut_layer"),
                    "route_batch_counts": info.get(
                        f"route_batch_counts_{epoch}",
                        {},
                    ),
                })
        return models
    
    def get_total_nb_by_layer(self, layer_id):
        return sum(info.get("nb_train", 0) for info in self.client.values() if info.get("layer_id") == layer_id)
    
    @staticmethod
    def _load_checkpoint_state(path):
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, nn.Module):
            return checkpoint.state_dict()
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint type in {path}: {type(checkpoint)}")

        for key in ('model_state_dict', 'state_dict', 'model'):
            if key not in checkpoint:
                continue
            state = checkpoint[key]
            if isinstance(state, nn.Module):
                return state.state_dict()
            if isinstance(state, dict):
                return state
        return checkpoint

    @staticmethod
    def _canonical_state_items(state, layer_offset=0):
        """Yield each global layer state once, ignoring ModuleList aliases."""
        seen = set()
        for key, value in state.items():
            clean_key = key
            prefix_removed = True
            while prefix_removed:
                prefix_removed = False
                for prefix in ('model.', 'layers.'):
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        prefix_removed = True

            parts = clean_key.split('.')
            if not parts or not parts[0].isdigit():
                continue

            global_layer = int(parts[0]) + layer_offset
            target_key = '.'.join(
                ['layers', str(global_layer)] + parts[1:]
            )
            if target_key in seen:
                continue
            seen.add(target_key)
            yield global_layer, target_key, value

    @staticmethod
    def _weighted_state_average(entries, target_tensor):
        """Average one parameter/buffer using its actual batch exposure."""
        total_batches = sum(entry['num_batches'] for entry in entries)
        if total_batches <= 0:
            raise ValueError("Cannot aggregate a state with zero batch exposure.")

        if target_tensor.is_floating_point() or target_tensor.is_complex():
            accumulator_dtype = (
                torch.complex128 if target_tensor.is_complex() else torch.float64
            )
            accumulator = torch.zeros_like(
                target_tensor,
                dtype=accumulator_dtype,
                device='cpu',
            )
            for entry in entries:
                accumulator.add_(
                    entry['value'].to(dtype=accumulator_dtype, device='cpu'),
                    alpha=entry['num_batches'],
                )
            return (accumulator / total_batches).to(dtype=target_tensor.dtype)

        accumulator = torch.zeros_like(
            target_tensor,
            dtype=torch.float64,
            device='cpu',
        )
        for entry in entries:
            accumulator.add_(
                entry['value'].to(dtype=torch.float64, device='cpu'),
                alpha=entry['num_batches'],
            )
        averaged = accumulator / total_batches
        if target_tensor.dtype == torch.bool:
            return (averaged >= 0.5).to(dtype=target_tensor.dtype)
        return averaged.round().to(dtype=target_tensor.dtype)

    def merged_model(self, full_model, edge_models_list, server_models_list):
        """Aggregate each YOLO layer by the batches that traversed its copy.

        For a global layer ``L``:
        - an edge contributes when its cut is at or after ``L``;
        - a dynamic server contributes batches whose cut is before ``L``.

        This makes overlapping layers (for example layers 6..10 with cuts 5
        and 10) a weighted combination of their edge and server copies.
        """
        if not edge_models_list:
            raise ValueError("At least one edge model is required for aggregation.")
        if not server_models_list:
            raise ValueError("At least one server model is required for aggregation.")

        total_batches = sum(
            int(model['num_batches']) for model in edge_models_list
        )
        if total_batches <= 0:
            raise ValueError("Total edge batch count must be greater than zero.")

        fallback_route_counts = {}
        for model in edge_models_list:
            cut_layer = int(model['cut_layer'])
            num_batches = int(model['num_batches'])
            if num_batches <= 0:
                raise ValueError(
                    f"Edge client {model['client_id']} has invalid batch count "
                    f"{num_batches}."
                )
            fallback_route_counts[cut_layer] = (
                fallback_route_counts.get(cut_layer, 0) + num_batches
            )

        full_state = full_model.state_dict()
        candidates = {}
        source_layer_exposures = {}

        def add_candidates(model, state, layer_offset, exposure_for_layer, source_kind):
            source_name = f"{source_kind}:{model['client_id']}"
            for global_layer, target_key, value in self._canonical_state_items(
                state,
                layer_offset=layer_offset,
            ):
                if target_key not in full_state:
                    continue
                if full_state[target_key].shape != value.shape:
                    raise ValueError(
                        f"Shape mismatch for {target_key} from {source_name}: "
                        f"expected {tuple(full_state[target_key].shape)}, "
                        f"received {tuple(value.shape)}."
                    )

                exposure = int(exposure_for_layer(global_layer))
                if exposure <= 0:
                    continue
                candidates.setdefault(target_key, []).append({
                    'value': value,
                    'num_batches': exposure,
                    'source': source_name,
                })
                source_layer_exposures[(source_name, global_layer)] = exposure

        for model in edge_models_list:
            state = self._load_checkpoint_state(model['path'])
            edge_batches = int(model['num_batches'])
            edge_cut = int(model['cut_layer'])
            add_candidates(
                model,
                state,
                layer_offset=0,
                exposure_for_layer=lambda layer, cut=edge_cut, count=edge_batches: (
                    count if layer <= cut else 0
                ),
                source_kind='edge',
            )

        for model in server_models_list:
            route_counts = {
                int(cut): int(count)
                for cut, count in model.get('route_batch_counts', {}).items()
            }
            if not route_counts:
                if len(server_models_list) != 1:
                    raise ValueError(
                        "route_batch_counts is required when aggregating multiple "
                        "dynamic server workers."
                    )
                route_counts = fallback_route_counts

            state = self._load_checkpoint_state(model['path'])
            add_candidates(
                model,
                state,
                layer_offset=self.cut_layer + 1,
                exposure_for_layer=lambda layer, counts=route_counts: sum(
                    count for cut, count in counts.items() if cut < layer
                ),
                source_kind='server',
            )

        exposure_by_layer = {}
        for (_, global_layer), exposure in source_layer_exposures.items():
            exposure_by_layer[global_layer] = (
                exposure_by_layer.get(global_layer, 0) + exposure
            )
        invalid_exposures = {
            layer: exposure
            for layer, exposure in exposure_by_layer.items()
            if exposure != total_batches
        }
        if invalid_exposures:
            raise ValueError(
                "Layer batch exposure does not match the total edge batches "
                f"({total_batches}): {invalid_exposures}."
            )

        expected_keys = {
            key for key in full_state if key.startswith('layers.')
        }
        missing_keys = sorted(expected_keys - candidates.keys())
        if missing_keys:
            preview = ', '.join(missing_keys[:5])
            raise ValueError(
                f"Aggregation has no source for {len(missing_keys)} model states: "
                f"{preview}."
            )

        merged_state = {
            key: self._weighted_state_average(entries, full_state[key])
            for key, entries in candidates.items()
        }
        full_model.load_state_dict(merged_state, strict=False)

        exposure_summary = ', '.join(
            f"L{layer}={exposure}"
            for layer, exposure in sorted(exposure_by_layer.items())
        )
        print(
            f"Layer-wise aggregation succeeded with {total_batches} batches "
            f"per state path ({exposure_summary})."
        )
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
    
    def get_epoch_averages(self, epoch):
        if not self.client_delays:
            return {}

        df = pd.DataFrame(self.client_delays)
        epoch_summary = df[df['epoch'] == epoch].groupby('epoch').mean(numeric_only=True)
        if not epoch_summary.empty:
            return epoch_summary.iloc[0].to_dict()
        return {}
