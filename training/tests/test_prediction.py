from pathlib import Path

import pytest
import torch
from PIL import Image

from vanta_training.checkpoint import (
    build_checkpoint_metadata,
    load_checkpoint,
    save_checkpoint,
)
from vanta_training.constants import CLASS_NAMES, CLASS_TO_INDEX
from vanta_training.model import build_model
from vanta_training.prediction import ImagePredictor


def test_checkpoint_round_trip_preserves_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    metadata = build_checkpoint_metadata(best_epoch=3, validation_accuracy=0.75)
    save_checkpoint(checkpoint_path, build_model(pretrained=False), metadata)

    loaded_model, loaded_metadata = load_checkpoint(checkpoint_path, torch.device("cpu"))

    assert loaded_metadata["class_to_index"] == CLASS_TO_INDEX
    assert loaded_metadata["best_epoch"] == 3
    assert loaded_model.classifier[-1].out_features == len(CLASS_NAMES)


def test_prediction_returns_all_class_scores(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    image_path = tmp_path / "example.png"
    save_checkpoint(
        checkpoint_path,
        build_model(pretrained=False),
        build_checkpoint_metadata(best_epoch=1),
    )
    Image.new("RGB", (80, 60), color=(20, 100, 180)).save(image_path)

    result = ImagePredictor(checkpoint_path, "cpu").predict(image_path)

    assert result["predicted_class"] in CLASS_NAMES
    assert set(result["scores"]) == set(CLASS_NAMES)
    assert sum(result["scores"].values()) == pytest.approx(1.0)


def test_prediction_reports_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        ImagePredictor(tmp_path / "missing.pt", "cpu")


def test_prediction_reports_invalid_image(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    invalid_image_path = tmp_path / "broken.jpg"
    save_checkpoint(
        checkpoint_path,
        build_model(pretrained=False),
        build_checkpoint_metadata(best_epoch=1),
    )
    invalid_image_path.write_text("not an image", encoding="utf-8")

    predictor = ImagePredictor(checkpoint_path, "cpu")
    with pytest.raises(ValueError, match="not a readable image"):
        predictor.predict(invalid_image_path)
