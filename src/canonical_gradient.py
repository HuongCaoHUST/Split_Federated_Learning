"""Utilities for the cut-invariant, canonical-gradient split-training mode.

The first implementation intentionally supports one uniform YOLO11 cut at
layer 5.  It is a correctness baseline: the server owns the only optimizer
state for the complete canonical model; the edge is an autograd worker that
returns prefix gradients and BatchNorm buffers.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


CANONICAL_GRADIENT_QUEUE = "canonical_gradient_queue"
CANONICAL_CUT = 5
PREFIX_LAYER_COUNT = CANONICAL_CUT + 1


def validate_canonical_cut5_config(config: Mapping) -> None:
    """Reject configurations outside the deliberately narrow baseline scope."""
    cut_layers = config.get("cut_layer", [])
    clients = config.get("clients", [])
    if list(cut_layers) != [CANONICAL_CUT]:
        raise ValueError(
            "canonical_gradient_mode currently supports only uniform cut_layer: [5]."
        )
    if list(clients) != [1, 1]:
        raise ValueError(
            "canonical_gradient_mode baseline requires clients: [1, 1]."
        )


def is_prefix_state_name(name: str) -> bool:
    """Whether a state/parameter name belongs to YOLO11 layers 0..5."""
    for layer_index in range(PREFIX_LAYER_COUNT):
        if name.startswith(f"layers.{layer_index}."):
            return True
    return False


def cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    """Detach a tensor for a trusted pickle message without aliasing storage."""
    return tensor.detach().cpu().clone()


def prefix_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract a strict-loadable state dict for ``YOLO11_EDGE_5``."""
    return {
        name: cpu_clone(value)
        for name, value in model.state_dict().items()
        if is_prefix_state_name(name)
    }


def prefix_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return every prefix parameter gradient, failing closed if one is absent."""
    gradients = {}
    for name, parameter in model.named_parameters():
        if not is_prefix_state_name(name):
            continue
        if parameter.grad is None:
            raise RuntimeError(
                f"Canonical split backward produced no gradient for prefix parameter {name}."
            )
        gradients[name] = cpu_clone(parameter.grad)
    if not gradients:
        raise RuntimeError("No prefix gradients were collected from the edge model.")
    return gradients


def prefix_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return BatchNorm and other prefix buffers for canonical-state sync."""
    return {
        name: cpu_clone(buffer)
        for name, buffer in model.named_buffers()
        if is_prefix_state_name(name)
    }


def install_prefix_gradients(
    canonical_model: torch.nn.Module,
    gradients: Mapping[str, torch.Tensor],
    device: torch.device,
) -> None:
    """Install edge-computed prefix grads into the canonical full model."""
    parameters = dict(canonical_model.named_parameters())
    expected = {
        name for name in parameters if is_prefix_state_name(name)
    }
    received = set(gradients)
    if received != expected:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise ValueError(
            "Prefix gradient keys do not match canonical layers 0..5; "
            f"missing={missing}, unexpected={unexpected}."
        )

    for name, gradient in gradients.items():
        parameter = parameters[name]
        if parameter.shape != gradient.shape:
            raise ValueError(
                f"Gradient shape mismatch for {name}: expected "
                f"{tuple(parameter.shape)}, got {tuple(gradient.shape)}."
            )
        parameter.grad = gradient.to(device=device, dtype=parameter.dtype)


def install_prefix_buffers(
    canonical_model: torch.nn.Module,
    buffers: Mapping[str, torch.Tensor],
    device: torch.device,
) -> None:
    """Copy edge BatchNorm buffers into the canonical full model."""
    canonical_buffers = dict(canonical_model.named_buffers())
    expected = {
        name for name in canonical_buffers if is_prefix_state_name(name)
    }
    received = set(buffers)
    if received != expected:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise ValueError(
            "Prefix buffer keys do not match canonical layers 0..5; "
            f"missing={missing}, unexpected={unexpected}."
        )

    with torch.no_grad():
        for name, value in buffers.items():
            target = canonical_buffers[name]
            if target.shape != value.shape:
                raise ValueError(
                    f"Buffer shape mismatch for {name}: expected "
                    f"{tuple(target.shape)}, got {tuple(value.shape)}."
                )
            target.copy_(value.to(device=device, dtype=target.dtype))
