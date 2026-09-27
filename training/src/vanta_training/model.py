"""MobileNetV3-Small construction and transfer-learning controls."""

from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from vanta_training.constants import CLASS_NAMES


def build_model(*, pretrained: bool = True, number_of_classes: int = len(CLASS_NAMES)) -> nn.Module:
    """Build MobileNetV3-Small with a task-specific classification output."""

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    output_layer = model.classifier[-1]
    if not isinstance(output_layer, nn.Linear):
        raise TypeError("Unexpected MobileNetV3-Small classifier structure.")
    model.classifier[-1] = nn.Linear(output_layer.in_features, number_of_classes)
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
