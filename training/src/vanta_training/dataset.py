"""AlphaTrash dataset loading and integrity validation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset

from vanta_training.constants import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    DATASET_SPLITS,
    IMAGE_EXTENSIONS,
    SOURCE_LABEL_ALIASES,
)


@dataclass(frozen=True)
class ImageSample:
    """One image path and its canonical VANTA class index."""

    path: Path
    class_index: int


@dataclass(frozen=True)
class DuplicateGroup:
    """Files in different splits that contain exactly the same bytes."""

    sha256: str
    files: list[str]
    splits: list[str]


@dataclass
class DatasetValidationReport:
    """Serializable results from validating an AlphaTrash checkout."""

    dataset_root: str
    image_counts: dict[str, dict[str, int]]
    missing_classes: list[str]
    corrupt_images: list[str]
    unsupported_files: list[str]
    cross_split_duplicates: list[DuplicateGroup]

    @property
    def total_images(self) -> int:
        return sum(sum(counts.values()) for counts in self.image_counts.values())

    @property
    def has_blocking_errors(self) -> bool:
        return bool(self.missing_classes or self.corrupt_images or self.unsupported_files)

    def to_dict(self) -> dict[str, Any]:
        report = asdict(self)
        report["total_images"] = self.total_images
        report["has_blocking_errors"] = self.has_blocking_errors
        return report


def canonicalize_source_label(source_label: str) -> str:
    """Map a verified AlphaTrash folder name to a VANTA class name."""

    normalized_label = source_label.strip().lower()
    try:
        return SOURCE_LABEL_ALIASES[normalized_label]
    except KeyError as error:
        supported = ", ".join(sorted(SOURCE_LABEL_ALIASES))
        raise ValueError(
            f"Unsupported source label '{source_label}'. Expected one of: {supported}."
        ) from error


def _label_directories(split_directory: Path) -> list[tuple[Path, str]]:
    if not split_directory.is_dir():
        raise FileNotFoundError(f"Dataset split directory does not exist: {split_directory}")

    label_directories: list[tuple[Path, str]] = []
    for source_directory in sorted(split_directory.iterdir()):
        if source_directory.name.startswith(".") or not source_directory.is_dir():
            continue
        canonical_label = canonicalize_source_label(source_directory.name)
        label_directories.append((source_directory, canonical_label))
    return label_directories


def collect_split_samples(dataset_root: Path, split: str) -> list[ImageSample]:
    """Collect image paths while retaining the source repository's split."""

    if split not in DATASET_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {', '.join(DATASET_SPLITS)}.")

    split_directory = dataset_root / split
    samples: list[ImageSample] = []
    discovered_classes: set[str] = set()

    for source_directory, canonical_label in _label_directories(split_directory):
        discovered_classes.add(canonical_label)
        class_index = CLASS_TO_INDEX[canonical_label]
        image_paths = sorted(
            path
            for path in source_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        samples.extend(ImageSample(path, class_index) for path in image_paths)

    missing_classes = set(CLASS_NAMES) - discovered_classes
    if missing_classes:
        missing = ", ".join(sorted(missing_classes))
        raise ValueError(f"Split '{split}' is missing class directories: {missing}.")
    if not samples:
        raise ValueError(f"Split '{split}' contains no supported image files.")
    return samples


class AlphaTrashDataset(Dataset[tuple[Tensor, int]]):
    """PyTorch dataset that maps AlphaTrash labels to VANTA class indices."""

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform
        self.samples = collect_split_samples(dataset_root, split)
        self.class_to_index = dict(CLASS_TO_INDEX)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                image_tensor = self.transform(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Unable to read image: {sample.path}") from error
        return image_tensor, sample.class_index


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(dataset_root: Path) -> DatasetValidationReport:
    """Check class coverage, readability, and exact leakage across splits."""

    dataset_root = dataset_root.resolve()
    image_counts = {
        split: {class_name: 0 for class_name in CLASS_NAMES} for split in DATASET_SPLITS
    }
    missing_classes: list[str] = []
    corrupt_images: list[str] = []
    unsupported_files: list[str] = []
    files_by_digest: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split in DATASET_SPLITS:
        split_directory = dataset_root / split
        if not split_directory.is_dir():
            missing_classes.extend(f"{split}/{class_name}" for class_name in CLASS_NAMES)
            continue

        discovered_classes: set[str] = set()
        for source_directory, canonical_label in _label_directories(split_directory):
            discovered_classes.add(canonical_label)
            for path in sorted(source_directory.rglob("*")):
                if not path.is_file() or path.name.startswith("."):
                    continue
                relative_path = path.relative_to(dataset_root).as_posix()
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    unsupported_files.append(relative_path)
                    continue

                image_counts[split][canonical_label] += 1
                try:
                    with Image.open(path) as image:
                        image.verify()
                    files_by_digest[_sha256_file(path)].append((split, relative_path))
                except (OSError, UnidentifiedImageError):
                    corrupt_images.append(relative_path)

        for class_name in CLASS_NAMES:
            if class_name not in discovered_classes or image_counts[split][class_name] == 0:
                missing_classes.append(f"{split}/{class_name}")

    duplicate_groups = []
    for digest, occurrences in files_by_digest.items():
        splits = sorted({split for split, _ in occurrences})
        if len(splits) > 1:
            duplicate_groups.append(
                DuplicateGroup(
                    sha256=digest,
                    files=sorted(path for _, path in occurrences),
                    splits=splits,
                )
            )

    duplicate_groups.sort(key=lambda group: (group.splits, group.files))
    return DatasetValidationReport(
        dataset_root=str(dataset_root),
        image_counts=image_counts,
        missing_classes=sorted(missing_classes),
        corrupt_images=sorted(corrupt_images),
        unsupported_files=sorted(unsupported_files),
        cross_split_duplicates=duplicate_groups,
    )


def save_validation_report(report: DatasetValidationReport, output_path: Path) -> None:
    """Write a dataset report without modifying any source images."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


def raise_for_dataset_issues(
    report: DatasetValidationReport,
    *,
    allow_cross_split_duplicates: bool = False,
) -> None:
    """Stop training when data issues could invalidate an experiment."""

    problems: list[str] = []
    if report.missing_classes:
        problems.append(f"missing or empty classes: {len(report.missing_classes)}")
    if report.corrupt_images:
        problems.append(f"corrupt images: {len(report.corrupt_images)}")
    if report.unsupported_files:
        problems.append(f"unsupported files: {len(report.unsupported_files)}")
    if report.cross_split_duplicates and not allow_cross_split_duplicates:
        problems.append(
            f"exact duplicate groups across splits: {len(report.cross_split_duplicates)}"
        )

    if problems:
        joined_problems = "; ".join(problems)
        raise ValueError(
            "Dataset validation failed ("
            f"{joined_problems}). Review the validation report; no files were changed."
        )
