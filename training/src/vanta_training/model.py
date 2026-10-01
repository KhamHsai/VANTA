"""MobileNetV3-Small construction and transfer-learning controls."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from vanta_training.constants import CLASS_NAMES


def build_model(
    *,
    pretrained: bool = True,
    pretrained_weights_path: Path | None = None,
    number_of_classes: int = len(CLASS_NAMES),
) -> nn.Module:
    """Build MobileNetV3-Small with a task-specific classification output."""

    if pretrained_weights_path is not None:
        model = _build_from_local_imagenet_weights(pretrained_weights_path)
    else:
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        try:
            model = mobilenet_v3_small(weights=weights)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                "Unable to load ImageNet weights. If internet access is disabled, attach "
                "the TorchVision MobileNetV3-Small weight file and pass its local path."
            ) from error

    output_layer = model.classifier[-1]
    if not isinstance(output_layer, nn.Linear):
        raise TypeError("Unexpected MobileNetV3-Small classifier structure.")
    model.classifier[-1] = nn.Linear(output_layer.in_features, number_of_classes)
    return model


def _build_from_local_imagenet_weights(weights_path: Path) -> nn.Module:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pretrained weight file does not exist: {weights_path}")
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Unable to load pretrained weights '{weights_path}': {error}") from error
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Pretrained weight file must contain a PyTorch state dictionary.")

    model = mobilenet_v3_small(weights=None)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(
            "Pretrained weights are not compatible with TorchVision MobileNetV3-Small."
        ) from error
    return model


def freeze_feature_extractor(model: nn.Module) -> None:
    """Freeze ImageNet features while leaving the new classifier trainable."""

    for parameter in model.features.parameters():  # type: ignore[attr-defined]
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():  # type: ignore[attr-defined]
        parameter.requires_grad = True


def unfreeze_last_feature_blocks(model: nn.Module, number_of_blocks: int) -> None:
    """Unfreeze the final feature blocks for lower-rate fine-tuning."""

    if number_of_blocks < 1:
        raise ValueError("number_of_blocks must be at least 1.")

    features = model.features  # type: ignore[attr-defined]
    for parameter in features.parameters():
        parameter.requires_grad = False
    for block in features[-number_of_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():  # type: ignore[attr-defined]
        parameter.requires_grad = True
