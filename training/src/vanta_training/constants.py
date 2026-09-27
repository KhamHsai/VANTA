"""Shared constants for the five-class VANTA classifier."""

from typing import Final

CLASS_NAMES: Final[tuple[str, ...]] = (
    "general",
    "metal",
    "organic",
    "paper",
    "plastic",
)
CLASS_TO_INDEX: Final[dict[str, int]] = {
    class_name: index for index, class_name in enumerate(CLASS_NAMES)
}

# AlphaTrash has used both spellings at different points in its history.
SOURCE_LABEL_ALIASES: Final[dict[str, str]] = {
    "general": "general",
    "metal": "metal",
    "organic": "organic",
    "paper": "paper",
    "plastic": "plastic",
    "plasic": "plastic",
}

DATASET_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
IMAGE_SIZE: Final[int] = 224
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)
MODEL_ARCHITECTURE: Final[str] = "mobilenet_v3_small"
CHECKPOINT_FORMAT_VERSION: Final[int] = 1
