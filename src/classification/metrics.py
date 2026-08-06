import torch


class ClassificationMetrics:
    """Streaming multiclass confusion matrix and macro classification metrics."""

    def __init__(self, num_classes):
        if isinstance(num_classes, bool) or int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2.")
        self.num_classes = int(num_classes)
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, logits_or_predictions, targets):
        predictions = logits_or_predictions
        if predictions.ndim > 1:
            predictions = predictions.argmax(dim=1)
        predictions = predictions.detach().to(dtype=torch.int64, device="cpu")
        targets = targets.detach().to(dtype=torch.int64, device="cpu")
        if predictions.shape != targets.shape:
            raise ValueError(
                "Predictions and targets must have the same batch shape; "
                f"received {tuple(predictions.shape)} and {tuple(targets.shape)}."
            )
        if targets.numel() == 0:
            return
        if targets.min() < 0 or targets.max() >= self.num_classes:
            raise ValueError("Targets contain a class outside the configured range.")
        if predictions.min() < 0 or predictions.max() >= self.num_classes:
            raise ValueError("Predictions contain a class outside the configured range.")

        encoded = targets * self.num_classes + predictions
        counts = torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        )
        self.confusion_matrix += counts.reshape(
            self.num_classes, self.num_classes
        )

    def merge(self, confusion_matrix):
        matrix = torch.as_tensor(confusion_matrix, dtype=torch.int64)
        if matrix.shape != self.confusion_matrix.shape:
            raise ValueError(
                f"Expected confusion matrix shape {tuple(self.confusion_matrix.shape)}, "
                f"received {tuple(matrix.shape)}."
            )
        self.confusion_matrix += matrix.cpu()

    def compute(self):
        matrix = self.confusion_matrix.to(torch.float64)
        true_positive = matrix.diag()
        predicted_count = matrix.sum(dim=0)
        target_count = matrix.sum(dim=1)

        precision = torch.where(
            predicted_count > 0, true_positive / predicted_count, 0.0
        )
        recall = torch.where(target_count > 0, true_positive / target_count, 0.0)
        f1 = torch.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
        total = matrix.sum()
        accuracy = true_positive.sum() / total if total > 0 else torch.tensor(0.0)
        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()),
            "f1_macro": float(f1.mean()),
            "total": int(total),
        }

    def state_dict(self):
        return {"confusion_matrix": self.confusion_matrix.clone()}
