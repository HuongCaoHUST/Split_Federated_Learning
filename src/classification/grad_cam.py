"""Grad-CAM for visualising the activation transmitted at a split point.

The class is intended for the staged classification models in ``model/``.
Gradients are calculated with respect to the output of the edge partition,
while the remaining stages of the global model provide the class score.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from model.staged_classification import load_checkpoint_state


class SplitGradCAM:
    """Compute Grad-CAM at an inclusive edge/server cut.

    Parameters
    ----------
    full_model_class, edge_model_class:
        Matching full and edge model classes, e.g. ``AlexNet`` and
        ``AlexNetEdge``.
    best_weights:
        Path to a global checkpoint (``best.pt``, a state dict, or a checkpoint
        containing ``model_state_dict``/``state_dict``/``model``).
    cut_layer:
        Global stage index. The feature map is the output of this stage.
    preprocess:
        Callable converting a PIL image to a tensor. If omitted, images are
        resized to 224x224 and converted to a [0, 1] RGB tensor.
    """

    def __init__(
        self,
        full_model_class,
        edge_model_class,
        best_weights,
        cut_layer,
        num_classes=10,
        device=None,
        seed=42,
        preprocess=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cut_layer = full_model_class.validate_cut_layer(int(cut_layer))
        self.preprocess = preprocess or self._default_preprocess

        # Both partitions receive exactly the same global checkpoint weights.
        state = load_checkpoint_state(best_weights)
        self.full_model = full_model_class(num_classes=num_classes, seed=seed)
        try:
            self.full_model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "best_weights must be a complete global-model checkpoint. "
                f"The supplied file appears to be an edge/server partition: {best_weights}"
            ) from exc
        self.edge_model = edge_model_class(
            cut_layer=self.cut_layer, num_classes=num_classes, seed=seed
        )
        # Read from the already-loaded full model so legacy layouts (for
        # example old AlexNet ``features.*`` checkpoints) are normalized by
        # the full model before the edge subset is selected.
        normalized_state = self.full_model.state_dict()
        edge_keys = set(self.edge_model.state_dict())
        edge_state = {
            key: value for key, value in normalized_state.items() if key in edge_keys
        }
        self.edge_model.load_state_dict(edge_state, strict=True)
        self.full_model.to(self.device).eval()
        self.edge_model.to(self.device).eval()

    @staticmethod
    def _default_preprocess(image):
        image = image.convert("RGB").resize((224, 224))
        values = torch.from_numpy(np.asarray(image).copy()).float() / 255.0
        return values.permute(2, 0, 1)

    def _image_tensor(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            tensor = self.preprocess(image)
        elif torch.is_tensor(image):
            tensor = image
        else:
            raise TypeError("image must be a path, PIL.Image, or torch.Tensor")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError("image tensor must have shape [C,H,W] or [N,C,H,W]")
        return tensor.to(self.device).float()

    @torch.enable_grad()
    def explain(self, image, target_class=None, save_path=None):
        """Return ``(heatmap, overlay, class_index)`` and optionally save it.

        ``heatmap`` and ``overlay`` are uint8 RGB numpy arrays. ``target_class``
        defaults to the class predicted by the global model.
        """
        x = self._image_tensor(image)
        activation = self.edge_model(x)
        if activation.ndim != 4:
            raise ValueError(
                f"cut_layer={self.cut_layer} outputs {tuple(activation.shape)}; "
                "choose a cut before flattening/pooling to a vector."
            )
        activation.retain_grad()
        logits = self.full_model.forward_from(activation, self.cut_layer)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim != 2:
            raise ValueError("Grad-CAM expects classification logits shaped [N, classes].")
        class_index = int(logits[0].argmax().item()) if target_class is None else int(target_class)
        if not 0 <= class_index < logits.shape[1]:
            raise ValueError(f"target_class must be in [0, {logits.shape[1] - 1}]")
        self.full_model.zero_grad(set_to_none=True)
        self.edge_model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()

        weights = activation.grad[0].mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * activation[0]).sum(dim=0, keepdim=True))
        cam = F.interpolate(cam[None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu()
        cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-8)
        heatmap = (cam.numpy() * 255).astype(np.uint8)

        base = x[0].detach().cpu().permute(1, 2, 0).numpy()
        base = (base - base.min()) / max(base.max() - base.min(), 1e-8)
        base = (base * 255).astype(np.uint8)
        import matplotlib.cm as cm
        color = (cm.jet(cam.numpy())[..., :3] * 255).astype(np.uint8)
        overlay = (0.45 * base + 0.55 * color).clip(0, 255).astype(np.uint8)
        if save_path is not None:
            Image.fromarray(overlay).save(save_path)
        return heatmap, overlay, class_index
