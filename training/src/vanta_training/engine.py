"""Focused PyTorch training and validation loops."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    """Loss and accuracy accumulated over one complete data loader."""

    loss: float
    accuracy: float
    sample_count: int


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    gradient_scaler: torch.amp.GradScaler | None = None,
) -> EpochMetrics:
    """Train for one epoch and return sample-weighted metrics."""

    model.train()
    _keep_frozen_batch_norm_in_evaluation_mode(model)
    total_loss = 0.0
    correct_predictions = 0
    sample_count = 0
    use_mixed_precision = gradient_scaler is not None

    for images, targets in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_mixed_precision,
        ):
            logits = model(images)
            loss = loss_function(logits, targets)

        if gradient_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.detach().item() * batch_size
        correct_predictions += (logits.argmax(dim=1) == targets).sum().item()
        sample_count += batch_size

    return _finalize_metrics(total_loss, correct_predictions, sample_count)


def _keep_frozen_batch_norm_in_evaluation_mode(model: nn.Module) -> None:
    """Prevent frozen feature statistics from drifting during transfer learning."""

    batch_norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for module in model.modules():
        if isinstance(module, batch_norm_types) and not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.eval()


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate one epoch without retaining gradients."""

    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    sample_count = 0

    for images, targets in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        loss = loss_function(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct_predictions += (logits.argmax(dim=1) == targets).sum().item()
        sample_count += batch_size

    return _finalize_metrics(total_loss, correct_predictions, sample_count)


def _finalize_metrics(
    total_loss: float,
    correct_predictions: int,
    sample_count: int,
) -> EpochMetrics:
    if sample_count == 0:
        raise ValueError("Cannot calculate metrics for an empty data loader.")
    return EpochMetrics(
        loss=total_loss / sample_count,
        accuracy=correct_predictions / sample_count,
        sample_count=sample_count,
    )
