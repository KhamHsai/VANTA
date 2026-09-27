"""Portable model checkpoint persistence and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from vanta_training.constants import (
    CHECKPOINT_FORMAT_VERSION,
    CLASS_TO_INDEX,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_ARCHITECTURE,
)
from vanta_training.model import build_model


def build_checkpoint_metadata(**experiment_details: Any) -> dict[str, Any]:
    """Create the metadata required to reproduce inference."""

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": MODEL_ARCHITECTURE,
        "class_to_index": dict(CLASS_TO_INDEX),
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        **experiment_details,
    }


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    metadata: dict[str, Any],
) -> None:
    """Save a CPU-portable checkpoint atomically."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "metadata": metadata}, temporary_path)
    temporary_path.replace(output_path)


def _read_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Unable to load checkpoint '{checkpoint_path}': {error}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary.")
    return checkpoint


def validate_checkpoint_metadata(metadata: object) -> dict[str, Any]:
    """Reject checkpoints that cannot reproduce VANTA preprocessing."""

    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is missing or invalid.")
    if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format version.")
    if metadata.get("architecture") != MODEL_ARCHITECTURE:
        raise ValueError(f"Checkpoint architecture must be '{MODEL_ARCHITECTURE}'.")
    if metadata.get("class_to_index") != CLASS_TO_INDEX:
        raise ValueError("Checkpoint class mapping does not match VANTA's five classes.")
    return metadata


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load one validated checkpoint and move its model to the requested device."""

    checkpoint = _read_checkpoint(checkpoint_path)
    metadata = validate_checkpoint_metadata(checkpoint.get("metadata"))
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model state is missing or invalid.")

    model = build_model(pretrained=False, number_of_classes=len(CLASS_TO_INDEX))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint model weights are incompatible: {error}") from error
    model.to(device)
    model.eval()
    return model, metadata
