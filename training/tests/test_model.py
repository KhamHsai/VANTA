import torch

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
