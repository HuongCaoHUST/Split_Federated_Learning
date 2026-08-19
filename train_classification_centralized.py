"""Centralized classification baseline for AlexNet, ResNet18, or MobileNetV2."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification.data import build_training_dataset, build_validation_dataset
from src.classification.metrics import ClassificationMetrics
from src.classification.models import build_full_model, get_dataset_name, get_model_name, validate_classification_config


def _device(value: str) -> torch.device:
    value = str(value).lower().strip()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(value)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimizer(model, training):
    name = str(training.get("optimizer", "Adam")).lower()
    kwargs = {
        "lr": float(training.get("learning_rate", 1e-4)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=float(training.get("momentum", 0.9)), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def _evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    metrics = ClassificationMetrics(num_classes)
    loss_sum = 0.0
    samples = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            samples += labels.size(0)
            metrics.update(logits, labels)
    result = metrics.compute()
    result.pop("total")
    result["loss"] = loss_sum / max(samples, 1)
    return result


def train(config_path: str, overrides: dict | None = None):
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    overrides = overrides or {}
    training = config.setdefault("training", {})
    training.update({key: value for key, value in overrides.items() if value is not None})
    project_root = str(path.parent)
    validate_classification_config(config)
    device = _device(training.get("device", "auto"))
    seed = int(config.get("model", {}).get("seed", 42))
    _seed(seed)

    model = build_full_model(config).to(device)
    num_classes = int(config["model"].get("num_classes", 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = _optimizer(model, training)
    batch_size = int(training.get("batch_size", 64))
    workers = int(training.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(build_training_dataset(config, project_root), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(build_validation_dataset(config, project_root), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)

    output_dir = Path(training.get("output_dir", "results/classification_centralized")).expanduser()
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "classification_centralized_results.csv"
    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
    best_accuracy = -1.0
    print(f"Centralized {get_model_name(config)} on {get_dataset_name(config)} using {device}")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = None
        for epoch in range(1, int(training.get("num_epochs", 10)) + 1):
            model.train()
            train_metrics = ClassificationMetrics(num_classes)
            loss_sum = 0.0
            samples = 0
            for images, labels in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * labels.size(0)
                samples += labels.size(0)
                train_metrics.update(logits.detach(), labels)
            train_result = train_metrics.compute()
            train_result.pop("total")
            train_result["loss"] = loss_sum / max(samples, 1)
            val_result = _evaluate(model, val_loader, criterion, device, num_classes)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_result.items()}, **{f"val_{k}": v for k, v in val_result.items()}}
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            csv_file.flush()
            payload = {"model_state_dict": model.state_dict(), "epoch": epoch, "metrics": row, "task": "classification", "dataset": get_dataset_name(config), "model_name": get_model_name(config), "num_classes": num_classes}
            torch.save(payload, last_path)
            if val_result["accuracy"] > best_accuracy:
                best_accuracy = val_result["accuracy"]
                torch.save(payload, best_path)
            print(f"Epoch {epoch}: train_acc={train_result['accuracy']:.4f}, val_acc={val_result['accuracy']:.4f}, val_f1={val_result['f1_macro']:.4f}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train a centralized classification baseline.")
    parser.add_argument("--config", default="config_classification_centralized.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = train(args.config, {"num_epochs": args.epochs, "batch_size": args.batch_size, "device": args.device, "output_dir": args.output})
    print(f"Centralized classification results saved to: {output}")


if __name__ == "__main__":
    main()
