from pathlib import Path

import torch
from torchvision.models import mobilenet_v3_small

from vanta_training.constants import CLASS_NAMES
from vanta_training.model import build_model, freeze_feature_extractor


def test_model_outputs_one_logit_per_class() -> None:
    model = build_model(pretrained=False)
    model.eval()

    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 224, 224))

    assert output.shape == (2, len(CLASS_NAMES))


def test_freeze_feature_extractor_only_leaves_classifier_trainable() -> None:
    model = build_model(pretrained=False)
    freeze_feature_extractor(model)

    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_model_loads_local_torchvision_weights(tmp_path: Path) -> None:
    weights_path = tmp_path / "mobilenet_v3_small.pth"
    torch.save(mobilenet_v3_small(weights=None).state_dict(), weights_path)

    model = build_model(pretrained_weights_path=weights_path)

    assert model.classifier[-1].out_features == len(CLASS_NAMES)
