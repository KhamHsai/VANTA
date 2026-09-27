from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from vanta_training.constants import CLASS_NAMES, DATASET_SPLITS


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Iterator[Path]:
    """Create a tiny split-preserving dataset without using project data."""

    dataset_root = tmp_path / "trash_dataset"
    for split_index, split in enumerate(DATASET_SPLITS):
        for class_index, class_name in enumerate(CLASS_NAMES):
            source_label = "plasic" if class_name == "plastic" else class_name
            class_directory = dataset_root / split / source_label
            class_directory.mkdir(parents=True)
            color = (20 + split_index * 40, 20 + class_index * 35, 40 + split_index + class_index)
            Image.new("RGB", (32, 24), color=color).save(class_directory / "sample.png")
    yield dataset_root
