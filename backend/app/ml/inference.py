"""MobileNetV3-Small loading, validation, and image inference."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from app.core.config import get_settings


MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"
DEFAULT_CHECKPOINT_PATH = MODEL_DIRECTORY / "best_model.pt"
DEFAULT_CLASS_MAPPING_PATH = MODEL_DIRECTORY / "class_mapping.json"

EXPECTED_ARCHITECTURE = "mobilenet_v3_small"
EXPECTED_CLASSES = ("general", "metal", "organic", "paper", "plastic")
EXPECTED_IMAGE_SIZE = 224
INFERENCE_RESIZE_SIZE = 256
EXPECTED_IMAGENET_MEAN = (0.485, 0.456, 0.406)
EXPECTED_IMAGENET_STD = (0.229, 0.224, 0.225)


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as a safe image."""


@dataclass(frozen=True)
class PredictionResult:
    """Internal prediction output, including timing for diagnostics."""

    label: str
    confidence: float
    is_uncertain: bool
    scores: dict[str, float]
    inference_time_ms: float


class ImageInferenceService:
    """Load one production checkpoint and reuse it for image predictions."""

    def __init__(
        self,
        *,
        confidence_threshold: float,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        class_mapping_path: Path = DEFAULT_CLASS_MAPPING_PATH,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        self.confidence_threshold = confidence_threshold
        self.device = _select_local_device()
        self.class_to_index, mapping_manifest_id = _load_class_mapping(class_mapping_path)
        self.class_names = tuple(
            name for name, _ in sorted(self.class_to_index.items(), key=lambda item: item[1])
        )
        self.model, self.metadata = _load_model(checkpoint_path, self.device)
        _validate_metadata(self.metadata, self.class_to_index, mapping_manifest_id)
        self.transform = _build_inference_transform(self.metadata)

    def predict_bytes(self, image_bytes: bytes) -> PredictionResult:
        """Decode one image and return ordered probabilities for every class."""

        image = _decode_image(image_bytes)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        _synchronize_device(self.device)
        started_at = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(input_tensor), dim=1).squeeze(0)
        _synchronize_device(self.device)
        inference_time_ms = (time.perf_counter() - started_at) * 1_000

        probabilities_cpu = probabilities.cpu()
        predicted_index = int(probabilities_cpu.argmax().item())
        confidence = float(probabilities_cpu[predicted_index].item())
        scores = {
            class_name: float(probabilities_cpu[index].item())
            for index, class_name in enumerate(self.class_names)
        }
        return PredictionResult(
            label=self.class_names[predicted_index],
            confidence=confidence,
            is_uncertain=confidence < self.confidence_threshold,
            scores=scores,
            inference_time_ms=inference_time_ms,
        )


def _select_local_device() -> torch.device:
    """Prefer Apple Silicon acceleration and otherwise use portable CPU inference."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_class_mapping(mapping_path: Path) -> tuple[dict[str, int], str]:
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping does not exist: {mapping_path}")
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read class mapping: {error}") from error

    class_to_index = payload.get("class_to_index") if isinstance(payload, dict) else None
    manifest_id = payload.get("manifest_id") if isinstance(payload, dict) else None
    if not isinstance(class_to_index, dict) or not all(
        isinstance(name, str) and isinstance(index, int)
        for name, index in class_to_index.items()
    ):
        raise ValueError("Class mapping must contain string labels and integer indices.")
    ordered_classes = tuple(
        name for name, _ in sorted(class_to_index.items(), key=lambda item: item[1])
    )
    if ordered_classes != EXPECTED_CLASSES or set(class_to_index.values()) != set(range(5)):
        raise ValueError("Class mapping does not match VANTA's five ordered classes.")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError("Class mapping is missing its manifest ID.")
    return class_to_index, manifest_id


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Unable to load model checkpoint: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("Model checkpoint must contain a dictionary.")

    state_dict = checkpoint.get("state_dict")
    metadata = checkpoint.get("metadata")
    if not isinstance(state_dict, dict) or not isinstance(metadata, dict):
        raise ValueError("Model checkpoint is missing weights or metadata.")

    model = mobilenet_v3_small(weights=None)
    output_layer = model.classifier[-1]
    if not isinstance(output_layer, nn.Linear):
        raise TypeError("Unexpected MobileNetV3-Small classifier structure.")
    model.classifier[-1] = nn.Linear(output_layer.in_features, len(EXPECTED_CLASSES))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(f"Model checkpoint weights are incompatible: {error}") from error
    model.to(device)
    model.eval()
    return model, metadata


def _validate_metadata(
    metadata: dict[str, Any],
    class_to_index: dict[str, int],
    mapping_manifest_id: str,
) -> None:
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported model checkpoint format version.")
    if metadata.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError(f"Model architecture must be '{EXPECTED_ARCHITECTURE}'.")
    if metadata.get("class_to_index") != class_to_index:
        raise ValueError("Checkpoint and class mapping labels do not match.")
    if metadata.get("split_manifest_id") != mapping_manifest_id:
        raise ValueError("Checkpoint and class mapping manifest IDs do not match.")
    if metadata.get("image_size") != EXPECTED_IMAGE_SIZE:
        raise ValueError(f"Model image size must be {EXPECTED_IMAGE_SIZE} pixels.")

    normalization = metadata.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Checkpoint normalization metadata is missing.")
    if tuple(normalization.get("mean", ())) != EXPECTED_IMAGENET_MEAN:
        raise ValueError("Checkpoint normalization mean is not the expected ImageNet mean.")
    if tuple(normalization.get("std", ())) != EXPECTED_IMAGENET_STD:
        raise ValueError("Checkpoint normalization standard deviation is invalid.")


def _build_inference_transform(metadata: dict[str, Any]) -> transforms.Compose:
    normalization = metadata["normalization"]
    return transforms.Compose(
        [
            transforms.Resize(INFERENCE_RESIZE_SIZE),
            transforms.CenterCrop(metadata["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(normalization["mean"], normalization["std"]),
        ]
    )


def _decode_image(image_bytes: bytes) -> Image.Image:
    try:
        # Treat decompression-bomb warnings as invalid input before large pixel buffers
        # can consume excessive server memory.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as source_image:
                source_image.load()
                return source_image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidImageError("The uploaded file is not a valid image.") from error
    except Image.DecompressionBombWarning as error:
        raise InvalidImageError("The uploaded image dimensions are too large.") from error


def _synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


@lru_cache(maxsize=1)
def get_inference_service() -> ImageInferenceService:
    """Return the process-wide inference service, loading the model only once."""

    settings = get_settings()
    return ImageInferenceService(
        confidence_threshold=settings.prediction_confidence_threshold,
    )
