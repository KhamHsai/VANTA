from pathlib import Path
from shutil import copyfile

import pytest
from torchvision import transforms

from vanta_training.constants import CLASS_TO_INDEX
from vanta_training.dataset import (
    AlphaTrashDataset,
    canonicalize_source_label,
    raise_for_dataset_issues,
    validate_dataset,
)


def test_source_label_aliases_include_original_misspelling() -> None:
    assert canonicalize_source_label("plastic") == "plastic"
    assert canonicalize_source_label("plasic") == "plastic"


def test_dataset_loads_all_classes_with_stable_mapping(synthetic_dataset: Path) -> None:
    dataset = AlphaTrashDataset(synthetic_dataset, "train", transforms.ToTensor())

    assert len(dataset) == 5
    assert dataset.class_to_index == CLASS_TO_INDEX
    assert sorted(sample.class_index for sample in dataset.samples) == list(range(5))
    image, class_index = dataset[0]
    assert image.shape == (3, 24, 32)
    assert class_index in CLASS_TO_INDEX.values()


def test_validation_counts_images_and_detects_no_initial_issues(
    synthetic_dataset: Path,
) -> None:
    report = validate_dataset(synthetic_dataset)

    assert report.total_images == 15
    assert all(count == 1 for split in report.image_counts.values() for count in split.values())
    assert report.missing_classes == []
    assert report.corrupt_images == []
    assert report.cross_split_duplicates == []


def test_validation_detects_corruption_and_cross_split_duplicate(
    synthetic_dataset: Path,
) -> None:
    duplicate_path = synthetic_dataset / "test" / "metal" / "duplicate.png"
    copyfile(synthetic_dataset / "train" / "general" / "sample.png", duplicate_path)
    corrupt_path = synthetic_dataset / "val" / "paper" / "broken.jpg"
    corrupt_path.write_bytes(b"not an image")

    report = validate_dataset(synthetic_dataset)

    assert corrupt_path.relative_to(synthetic_dataset).as_posix() in report.corrupt_images
    assert len(report.cross_split_duplicates) == 1
    with pytest.raises(ValueError, match="Dataset validation failed"):
        raise_for_dataset_issues(report)
