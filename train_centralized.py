"""Command-line entry point for the centralized YOLO11 baseline."""

import argparse
from src.centralized import CentralizedYOLOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model.YOLO11_Full on a centralized YOLO dataset."
    )
    parser.add_argument("--config", default="config_centralized.yaml")
    parser.add_argument("--data", default=None, help="Override dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--optimizer", choices=("SGD", "Adam", "AdamW"), default=None
    )
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--device", default=None, help="Override with auto, cpu, cuda, cuda:0, or 0."
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Ultralytics checkpoint, or 'none' for random initialization.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mlflow-uri", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = CentralizedYOLOTrainer.from_config(
        args.config,
        dataset_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.imgsz,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.workers,
        device=args.device,
        pretrained=args.pretrained,
        output_dir=args.output,
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_experiment=args.experiment,
        run_name=args.run_name,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        seed=args.seed,
    )
    trainer.fit()
    print(f"Centralized results saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
