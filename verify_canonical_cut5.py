"""Numerically verify one canonical-gradient cut-5 update against full YOLO11.

This does not require RabbitMQ or a dataset. It executes a real YOLO detection
loss on one synthetic batch, compares loss and every available parameter
gradient, takes one SGD step, then compares the complete full-model state.
"""

import argparse

import torch
import torch.optim as optim
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

from model.YOLO11n_custom import YOLO11_EDGE_5, YOLO11_Full
from src.canonical_gradient import (
    install_prefix_buffers,
    install_prefix_gradients,
    prefix_buffers,
    prefix_gradients,
    prefix_state_dict,
)


def setup_model(model):
    model.args = get_cfg(DEFAULT_CFG)
    model.names = {index: str(index) for index in range(model.nc)}
    return model.train()


def max_model_state_difference(reference, candidate):
    differences = []
    nonfloating_mismatches = []
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    for name, reference_value in reference_state.items():
        if not name.startswith('layers.'):
            continue  # Ignore ModuleList aliases named model.*.
        candidate_value = candidate_state[name]
        if reference_value.is_floating_point() or reference_value.is_complex():
            differences.append((reference_value - candidate_value).abs().max().item())
        elif not torch.equal(reference_value, candidate_value):
            nonfloating_mismatches.append(name)
    return max(differences, default=0.0), nonfloating_mismatches


def max_gradient_difference(reference, candidate):
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    differences = []
    missing = []
    for name, reference_parameter in reference_parameters.items():
        candidate_parameter = candidate_parameters[name]
        if reference_parameter.grad is None or candidate_parameter.grad is None:
            if (reference_parameter.grad is None) != (candidate_parameter.grad is None):
                missing.append(name)
            continue
        differences.append(
            (reference_parameter.grad - candidate_parameter.grad).abs().max().item()
        )
    return max(differences, default=0.0), missing


def run_verification(imgsz, learning_rate, tolerance, device):
    torch.manual_seed(7)
    full = setup_model(YOLO11_Full(nc=80, pretrained='yolo11n.pt').to(device))
    canonical = setup_model(YOLO11_Full(nc=80, pretrained='yolo11n.pt').to(device))
    edge = YOLO11_EDGE_5(pretrained='yolo11n.pt').to(device).train()
    edge.load_state_dict(prefix_state_dict(canonical), strict=True)

    full_criterion = v8DetectionLoss(full)
    canonical_criterion = v8DetectionLoss(canonical)
    full_optimizer = optim.SGD(full.parameters(), lr=learning_rate, momentum=0.0)
    canonical_optimizer = optim.SGD(
        canonical.parameters(), lr=learning_rate, momentum=0.0
    )

    images = torch.rand(1, 3, imgsz, imgsz, device=device)
    batch = {
        'img': images,
        'batch_idx': torch.tensor([0], device=device),
        'cls': torch.tensor([[0.0]], device=device),
        'bboxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
    }

    # Reference: unsplit full-model forward, loss, backward, and one update.
    full_optimizer.zero_grad(set_to_none=True)
    full_loss, _ = full_criterion(full(images), batch)
    full_total_loss = full_loss.sum()
    full_total_loss.backward()

    # Proposed method: split forward/backward, then install edge prefix grads
    # into the server's canonical full model before its only optimizer step.
    canonical_optimizer.zero_grad(set_to_none=True)
    edge.zero_grad(set_to_none=True)
    edge_outputs = edge(images)
    boundary_leaves = [value.detach().requires_grad_(True) for value in edge_outputs]
    split_loss, _ = canonical_criterion(
        canonical.forward_from_cut5(boundary_leaves), batch
    )
    split_total_loss = split_loss.sum()
    split_total_loss.backward()
    torch.autograd.backward(edge_outputs, [value.grad for value in boundary_leaves])
    install_prefix_gradients(canonical, prefix_gradients(edge), device)
    install_prefix_buffers(canonical, prefix_buffers(edge), device)

    loss_difference = (full_total_loss - split_total_loss).abs().item()
    gradient_difference, gradient_presence_mismatches = max_gradient_difference(
        full, canonical
    )
    full_optimizer.step()
    canonical_optimizer.step()
    state_difference, buffer_mismatches = max_model_state_difference(full, canonical)

    print(f'loss_difference={loss_difference:.8g}')
    print(f'max_gradient_difference={gradient_difference:.8g}')
    print(f'max_state_difference_after_step={state_difference:.8g}')
    print(f'gradient_presence_mismatches={gradient_presence_mismatches}')
    print(f'nonfloating_buffer_mismatches={buffer_mismatches}')

    passed = (
        loss_difference <= tolerance
        and gradient_difference <= tolerance
        and state_difference <= tolerance
        and not gradient_presence_mismatches
        and not buffer_mismatches
    )
    if not passed:
        raise SystemExit('Canonical split verification FAILED.')
    print('Canonical cut-5 verification PASSED.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--device', default='cpu')
    arguments = parser.parse_args()
    run_verification(
        imgsz=arguments.imgsz,
        learning_rate=arguments.learning_rate,
        tolerance=arguments.tolerance,
        device=torch.device(arguments.device),
    )
