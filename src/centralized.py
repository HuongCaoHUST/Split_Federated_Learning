"""Centralized training baseline for the custom YOLO11 model."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy

from model.YOLO11n_custom import YOLO11_Full
from src.mlflow import MLflowConnector


DEFAULT_MLFLOW_TRACKING_URI = "http://smart-hvac.io.vn:5005/"
DEFAULT_EXPERIMENT_NAME = "Centralized_YOLO11"


@dataclass
class EpochMetrics:
    """Metrics emitted after one centralized training epoch."""

    epoch: int
    train_box_loss: float
    train_cls_loss: float
    train_dfl_loss: float
    val_box_loss: float
    val_cls_loss: float
    val_dfl_loss: float
    precision: float
    recall: float
    map50: float
    map50_95: float

    def as_mlflow_metrics(self) -> dict[str, float]:
        return {
            "train/box_loss": self.train_box_loss,
            "train/cls_loss": self.train_cls_loss,
            "train/dfl_loss": self.train_dfl_loss,
            "val/box_loss": self.val_box_loss,
            "val/cls_loss": self.val_cls_loss,
            "val/dfl_loss": self.val_dfl_loss,
            "metrics/precision": self.precision,
            "metrics/recall": self.recall,
            "metrics/mAP50": self.map50,
            "metrics/mAP50-95": self.map50_95,
        }


class CentralizedYOLOTrainer:
    """Train ``YOLO11_Full`` on a single, full YOLO detection dataset.

    The class intentionally uses the same custom model, ``v8DetectionLoss`` and
    AP calculation as the split/federated path, so the resulting baseline is
    directly comparable with SFL runs.
    """

    def __init__(
        self,
        dataset_yaml: str,
        *,
        epochs: int = 100,
        batch_size: int = 8,
        image_size: int = 640,
        learning_rate: float = 1e-3,
        optimizer: str = "SGD",
        momentum: float = 0.937,
        weight_decay: float = 5e-4,
        num_workers: int = 4,
        device: str = "auto",
        pretrained: str | None = "yolo11n.pt",
        output_dir: str | None = None,
        mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
        mlflow_experiment: str = DEFAULT_EXPERIMENT_NAME,
        run_name: str | None = None,
        confidence_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        seed: int = 0,
    ) -> None:
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero.")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        self.dataset_yaml = str(Path(dataset_yaml).expanduser())
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.device = self._resolve_device(device)
        self.pretrained = pretrained
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.seed = seed

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"centralized-yolo11-{timestamp}"
        self.output_dir = Path(
            output_dir or Path("runs") / "centralized" / self.run_name
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "metrics.csv"
        self.best_path = self.output_dir / "best.pt"
        self.last_path = self.output_dir / "last.pt"

        self._set_seed(seed)
        self.data_cfg = check_det_dataset(self.dataset_yaml)
        self.nc = int(self.data_cfg["nc"])
        self.names = self._normalize_names(self.data_cfg["names"])

        self.yolo_args = get_cfg(DEFAULT_CFG)
        self.model = YOLO11_Full(nc=self.nc, pretrained=self.pretrained).to(
            self.device
        )
        self.model.names = self.names
        self.model.args = self.yolo_args
        self.criterion = v8DetectionLoss(self.model)
        self.optimizer = self._build_optimizer()

        self.train_loader = self._build_loader(
            self.data_cfg["train"], augment=True, shuffle=True
        )
        self.val_loader = self._build_loader(
            self.data_cfg["val"], augment=False, shuffle=False
        )
        if len(self.train_loader) == 0:
            raise ValueError("The training dataset contains no batches.")
        if len(self.val_loader) == 0:
            raise ValueError("The validation dataset contains no batches.")

        self.mlflow = MLflowConnector(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment,
        )
        self.history: list[EpochMetrics] = []
        self.best_map50_95 = float("-inf")

    @classmethod
    def from_config(
        cls,
        config_path: str = "config_centralized.yaml",
        **overrides: Any,
    ) -> "CentralizedYOLOTrainer":
        """Create a trainer from a centralized-training YAML file.

        Keyword overrides whose value is not ``None`` take precedence over the
        YAML values. Relative dataset, pretrained and output paths are resolved
        from the directory containing the configuration file.
        """
        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Centralized config was not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError("The centralized config root must be a mapping.")

        dataset = config.get("dataset", {})
        model = config.get("model", {})
        training = config.get("training", {})
        validation = config.get("validation", {})
        mlflow = config.get("mlflow", {})
        output = config.get("output", {})
        sections = {
            "dataset": dataset,
            "model": model,
            "training": training,
            "validation": validation,
            "mlflow": mlflow,
            "output": output,
        }
        invalid_sections = [
            name for name, value in sections.items() if not isinstance(value, dict)
        ]
        if invalid_sections:
            raise ValueError(
                "These centralized config sections must be mappings: "
                + ", ".join(invalid_sections)
            )

        options: dict[str, Any] = {
            "dataset_yaml": dataset.get("yaml"),
            "pretrained": model.get("pretrained", "yolo11n.pt"),
            "epochs": training.get("epochs", 100),
            "batch_size": training.get("batch_size", 8),
            "image_size": training.get("image_size", 640),
            "learning_rate": training.get("learning_rate", 1e-3),
            "optimizer": training.get("optimizer", "SGD"),
            "momentum": training.get("momentum", 0.937),
            "weight_decay": training.get("weight_decay", 5e-4),
            "num_workers": training.get("num_workers", 4),
            "device": training.get("device", "auto"),
            "confidence_threshold": validation.get(
                "confidence_threshold", 0.001
            ),
            "iou_threshold": validation.get("iou_threshold", 0.7),
            "mlflow_tracking_uri": mlflow.get(
                "tracking_uri", DEFAULT_MLFLOW_TRACKING_URI
            ),
            "mlflow_experiment": mlflow.get(
                "experiment_name", DEFAULT_EXPERIMENT_NAME
            ),
            "run_name": mlflow.get("run_name"),
            "output_dir": output.get("directory"),
            "seed": config.get("seed", 0),
        }
        options.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        if not options.get("dataset_yaml"):
            raise ValueError("Missing required config value: dataset.yaml")

        base_dir = path.parent

        def resolve_path(value: Any) -> str | None:
            if value is None:
                return None
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            return str(candidate.resolve())

        options["dataset_yaml"] = resolve_path(options["dataset_yaml"])
        pretrained = options.get("pretrained")
        if pretrained is None or str(pretrained).strip().lower() == "none":
            options["pretrained"] = None
        else:
            options["pretrained"] = resolve_path(pretrained)
        if options.get("output_dir") is not None:
            options["output_dir"] = resolve_path(options["output_dir"])

        return cls(**options)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        normalized = str(device).strip().lower()
        if normalized == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if normalized.isdigit():
            normalized = f"cuda:{normalized}"
        resolved = torch.device(normalized)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device}' was requested, but CUDA is unavailable."
            )
        return resolved

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _normalize_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def _build_loader(self, image_path: Any, *, augment: bool, shuffle: bool):
        dataset = YOLODataset(
            img_path=image_path,
            imgsz=self.image_size,
            data=self.data_cfg,
            augment=augment,
            hyp=self.yolo_args,
            rect=False,
            stride=32,
            prefix="train: " if augment else "val: ",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.num_workers > 0,
            collate_fn=dataset.collate_fn,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.optimizer_name.lower()
        parameters = self.model.parameters()
        if name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        if name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        if name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        raise ValueError("optimizer must be one of: SGD, Adam, AdamW.")

    def _loss_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move the target tensors used by v8DetectionLoss to the model device."""
        loss_batch = dict(batch)
        for key in ("batch_idx", "cls", "bboxes"):
            loss_batch[key] = batch[key].to(self.device, non_blocking=True)
        return loss_batch

    @staticmethod
    def _split_eval_predictions(predictions):
        if isinstance(predictions, tuple):
            return predictions[0], predictions[1]
        return predictions, predictions

    def train_one_epoch(self, epoch: int) -> np.ndarray:
        self.model.train()
        loss_sums = np.zeros(3, dtype=np.float64)
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [centralized train]",
        )

        for batch in progress:
            images = (
                batch["img"].to(self.device, non_blocking=True).float() / 255.0
            )
            loss_batch = self._loss_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(images)
            loss, loss_items = self.criterion(predictions, loss_batch)
            loss.sum().backward()
            self.optimizer.step()

            detached = loss_items.detach().cpu().double().numpy()
            loss_sums += detached
            progress.set_postfix(
                box=f"{detached[0]:.4f}",
                cls=f"{detached[1]:.4f}",
                dfl=f"{detached[2]:.4f}",
            )

        return loss_sums / len(self.train_loader)

    def validate_one_epoch(self, epoch: int) -> tuple[np.ndarray, dict[str, float]]:
        self.model.eval()
        loss_sums = np.zeros(3, dtype=np.float64)
        stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        iou_vector = torch.linspace(0.5, 0.95, 10, device=self.device)
        progress = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.epochs} [centralized val]",
        )

        with torch.inference_mode():
            for batch in progress:
                images = (
                    batch["img"].to(self.device, non_blocking=True).float()
                    / 255.0
                )
                loss_batch = self._loss_batch(batch)
                predictions = self.model(images)
                nms_input, loss_input = self._split_eval_predictions(predictions)
                _, loss_items = self.criterion(loss_input, loss_batch)
                detached = loss_items.detach().cpu().double().numpy()
                loss_sums += detached

                detections = non_max_suppression(
                    nms_input,
                    conf_thres=self.confidence_threshold,
                    iou_thres=self.iou_threshold,
                    nc=self.nc,
                )
                targets = torch.cat(
                    (
                        loss_batch["batch_idx"].view(-1, 1),
                        loss_batch["cls"].view(-1, 1),
                        loss_batch["bboxes"],
                    ),
                    dim=1,
                )

                for image_index, prediction in enumerate(detections):
                    labels = targets[targets[:, 0] == image_index][:, 1:]
                    correct = torch.zeros(
                        prediction.shape[0],
                        iou_vector.numel(),
                        dtype=torch.bool,
                        device=self.device,
                    )
                    if prediction.shape[0] and labels.shape[0]:
                        target_boxes = xywh2xyxy(labels[:, 1:])
                        target_boxes[:, [0, 2]] *= images.shape[3]
                        target_boxes[:, [1, 3]] *= images.shape[2]
                        labels_pixel = torch.cat((labels[:, :1], target_boxes), 1)
                        correct = self._process_batch(
                            prediction, labels_pixel, iou_vector
                        )

                    stats.append(
                        (
                            correct.cpu().numpy(),
                            prediction[:, 4].cpu().numpy(),
                            prediction[:, 5].cpu().numpy(),
                            labels[:, 0].cpu().numpy(),
                        )
                    )

                progress.set_postfix(
                    box=f"{detached[0]:.4f}",
                    cls=f"{detached[1]:.4f}",
                    dfl=f"{detached[2]:.4f}",
                )

        detection_metrics = self._calculate_detection_metrics(stats)
        return loss_sums / len(self.val_loader), detection_metrics

    @staticmethod
    def _process_batch(
        detections: torch.Tensor,
        labels: torch.Tensor,
        iou_vector: torch.Tensor,
    ) -> torch.Tensor:
        correct = torch.zeros(
            detections.shape[0],
            iou_vector.numel(),
            dtype=torch.bool,
            device=detections.device,
        )
        if not labels.shape[0] or not detections.shape[0]:
            return correct

        iou = box_iou(labels[:, 1:], detections[:, :4])
        label_index, detection_index = torch.where(
            (iou >= iou_vector[0])
            & (labels[:, 0:1] == detections[:, 5])
        )
        if not label_index.shape[0]:
            return correct

        matches = torch.cat(
            (
                torch.stack((label_index, detection_index), 1),
                iou[label_index, detection_index, None],
            ),
            1,
        ).cpu().numpy()
        if matches.shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches_tensor = torch.as_tensor(matches, device=detections.device)
        correct[matches_tensor[:, 1].long()] = (
            matches_tensor[:, 2:3] >= iou_vector
        )
        return correct

    def _calculate_detection_metrics(
        self,
        stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict[str, float]:
        empty = {
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
        }
        if not stats:
            return empty

        true_positive, confidence, predicted_class, target_class = (
            np.concatenate(values, axis=0) for values in zip(*stats)
        )
        if not true_positive.shape[0] or not true_positive.any():
            return empty

        results = ap_per_class(
            true_positive,
            confidence,
            predicted_class,
            target_class,
            plot=False,
            save_dir=self.output_dir,
            names=self.names,
        )
        precision, recall, average_precision = results[2], results[3], results[5]
        return {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "map50": float(average_precision[:, 0].mean()),
            "map50_95": float(average_precision.mean()),
        }

    def _save_checkpoint(self, path: Path, epoch: int, metrics: EpochMetrics) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "nc": self.nc,
                "names": self.names,
                "metrics": metrics.as_mlflow_metrics(),
                "dataset_yaml": self.dataset_yaml,
            },
            path,
        )

    def _write_history(self) -> None:
        fieldnames = list(EpochMetrics.__dataclass_fields__)
        with self.history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in self.history:
                writer.writerow(metrics.__dict__)

    def fit(self) -> list[EpochMetrics]:
        """Run training, validation, checkpointing and MLflow logging."""
        self.mlflow.start_run(run_name=self.run_name)
        try:
            self.mlflow.log_params(
                {
                    "mode": "centralized",
                    "model": "YOLO11_Full",
                    "dataset_yaml": self.dataset_yaml,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "image_size": self.image_size,
                    "learning_rate": self.learning_rate,
                    "optimizer": self.optimizer_name,
                    "momentum": self.momentum,
                    "weight_decay": self.weight_decay,
                    "num_workers": self.num_workers,
                    "device": str(self.device),
                    "pretrained": self.pretrained or "none",
                    "seed": self.seed,
                    "num_classes": self.nc,
                    "train_batches": len(self.train_loader),
                    "val_batches": len(self.val_loader),
                }
            )

            for epoch in range(1, self.epochs + 1):
                train_losses = self.train_one_epoch(epoch)
                val_losses, detection = self.validate_one_epoch(epoch)
                metrics = EpochMetrics(
                    epoch=epoch,
                    train_box_loss=float(train_losses[0]),
                    train_cls_loss=float(train_losses[1]),
                    train_dfl_loss=float(train_losses[2]),
                    val_box_loss=float(val_losses[0]),
                    val_cls_loss=float(val_losses[1]),
                    val_dfl_loss=float(val_losses[2]),
                    precision=detection["precision"],
                    recall=detection["recall"],
                    map50=detection["map50"],
                    map50_95=detection["map50_95"],
                )
                self.history.append(metrics)
                self.mlflow.log_metrics(
                    metrics.as_mlflow_metrics(), step=epoch
                )
                self._save_checkpoint(self.last_path, epoch, metrics)
                if metrics.map50_95 > self.best_map50_95:
                    self.best_map50_95 = metrics.map50_95
                    self._save_checkpoint(self.best_path, epoch, metrics)
                self._write_history()

                print(
                    f"Epoch {epoch}: P={metrics.precision:.4f}, "
                    f"R={metrics.recall:.4f}, mAP50={metrics.map50:.4f}, "
                    f"mAP50-95={metrics.map50_95:.4f}"
                )

            self.mlflow.log_artifact(str(self.history_path), "centralized")
            self.mlflow.log_artifact(str(self.best_path), "centralized")
            self.mlflow.log_artifact(str(self.last_path), "centralized")
            dataset_path = Path(self.dataset_yaml)
            if dataset_path.is_file():
                self.mlflow.log_artifact(str(dataset_path), "centralized/config")
            return self.history
        finally:
            self.mlflow.end_run()
