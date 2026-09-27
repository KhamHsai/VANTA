"""Single-image prediction with one reusable model instance."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image, UnidentifiedImageError

from vanta_training.checkpoint import load_checkpoint
from vanta_training.device import select_device
from vanta_training.transforms import build_inference_transform


class PredictionResult(TypedDict):
    predicted_class: str
    scores: dict[str, float]


class ImagePredictor:
    """Load a checkpoint once and reuse it for multiple predictions."""

    def __init__(self, checkpoint_path: Path, device_name: str = "auto") -> None:
        self.device = select_device(device_name)
        self.model, metadata = load_checkpoint(checkpoint_path, self.device)
        class_to_index = metadata["class_to_index"]
        self.class_names = tuple(
            class_name for class_name, _ in sorted(class_to_index.items(), key=lambda item: item[1])
        )
        self.transform = build_inference_transform()

    @torch.inference_mode()
    def predict(self, image_path: Path) -> PredictionResult:
        """Classify an image and return normalized scores for all five classes."""

        if not image_path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")
        try:
            with Image.open(image_path) as image:
                image_tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Input file is not a readable image: {image_path}") from error

        logits = self.model(image_tensor.to(self.device))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
        scores = {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(self.class_names)
        }
        predicted_index = int(probabilities.argmax().item())
        return {
            "predicted_class": self.class_names[predicted_index],
            "scores": scores,
        }
