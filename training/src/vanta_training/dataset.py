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

MANIFEST_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ImageSample:
    """One image path and its canonical VANTA class index."""

    path: Path
    class_index: int


@dataclass(frozen=True)
class InspectedImage:
    """A verified source image with its content digest."""

    relative_path: str
    split: str
    label: str
    sha256: str


@dataclass(frozen=True)
class DuplicateGroup:
    """Files in different splits that contain exactly the same bytes."""

    sha256: str
    files: list[str]
    splits: list[str]


@dataclass
class DatasetValidationReport:
    """Serializable results from validating source folders or a manifest."""

    dataset_root: str
    image_counts: dict[str, dict[str, int]]
    missing_classes: list[str]
    missing_images: list[str]
    corrupt_images: list[str]
    unsupported_files: list[str]
    hash_mismatches: list[str]
    cross_split_duplicates: list[DuplicateGroup]

    @property
    def total_images(self) -> int:
        return sum(sum(counts.values()) for counts in self.image_counts.values())

    @property
    def has_blocking_errors(self) -> bool:
        return bool(
            self.missing_classes
            or self.missing_images
            or self.corrupt_images
            or self.unsupported_files
            or self.hash_mismatches
            or self.cross_split_duplicates
        )

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


def _empty_image_counts() -> dict[str, dict[str, int]]:
    return {split: {class_name: 0 for class_name in CLASS_NAMES} for split in DATASET_SPLITS}


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
    """Collect an original split for auditing, not model training."""

    if split not in DATASET_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {', '.join(DATASET_SPLITS)}.")

    samples: list[ImageSample] = []
    discovered_classes: set[str] = set()
    for source_directory, canonical_label in _label_directories(dataset_root / split):
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


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so large datasets do not increase memory use."""

    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_source_dataset(
    dataset_root: Path,
) -> tuple[list[InspectedImage], DatasetValidationReport]:
    """Verify and hash each source image exactly once."""

    dataset_root = dataset_root.resolve()
    image_counts = _empty_image_counts()
    missing_classes: list[str] = []
    corrupt_images: list[str] = []
    unsupported_files: list[str] = []
    inspected_images: list[InspectedImage] = []

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
                    digest = sha256_file(path)
                except (OSError, UnidentifiedImageError):
                    corrupt_images.append(relative_path)
                    continue
                inspected_images.append(
                    InspectedImage(relative_path, split, canonical_label, digest)
                )

        for class_name in CLASS_NAMES:
            if class_name not in discovered_classes or image_counts[split][class_name] == 0:
                missing_classes.append(f"{split}/{class_name}")

    report = _build_validation_report(
        dataset_root=dataset_root,
        image_counts=image_counts,
        inspected_images=inspected_images,
        missing_classes=missing_classes,
        missing_images=[],
        corrupt_images=corrupt_images,
        unsupported_files=unsupported_files,
        hash_mismatches=[],
    )
    return inspected_images, report


def _build_validation_report(
    *,
    dataset_root: Path,
    image_counts: dict[str, dict[str, int]],
    inspected_images: list[InspectedImage],
    missing_classes: list[str],
    missing_images: list[str],
    corrupt_images: list[str],
    unsupported_files: list[str],
    hash_mismatches: list[str],
) -> DatasetValidationReport:
    files_by_digest: dict[str, list[InspectedImage]] = defaultdict(list)
    for image in inspected_images:
        files_by_digest[image.sha256].append(image)

    duplicate_groups: list[DuplicateGroup] = []
    for digest, occurrences in files_by_digest.items():
        splits = sorted({image.split for image in occurrences})
        if len(splits) > 1:
            duplicate_groups.append(
                DuplicateGroup(
                    sha256=digest,
                    files=sorted(image.relative_path for image in occurrences),
                    splits=splits,
                )
            )
    duplicate_groups.sort(key=lambda group: (group.splits, group.files))

    return DatasetValidationReport(
        dataset_root=str(dataset_root),
        image_counts=image_counts,
        missing_classes=sorted(missing_classes),
        missing_images=sorted(missing_images),
        corrupt_images=sorted(corrupt_images),
        unsupported_files=sorted(unsupported_files),
        hash_mismatches=sorted(hash_mismatches),
        cross_split_duplicates=duplicate_groups,
    )


def validate_dataset(dataset_root: Path) -> DatasetValidationReport:
    """Check original folders for readability and exact split leakage."""

    _, report = inspect_source_dataset(dataset_root)
    return report


def _load_manifest_entries(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Clean split manifest does not exist: {manifest_path}. "
            "Run scripts/create_clean_manifest.py first."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read split manifest '{manifest_path}': {error}") from error

    if not isinstance(manifest, dict) or manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError("Unsupported or missing split manifest format version.")
    if manifest.get("class_to_index") != CLASS_TO_INDEX:
        raise ValueError("Split manifest class mapping does not match VANTA's classes.")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Split manifest entries are missing or invalid.")
    return entries


def _resolve_manifest_image(dataset_root: Path, relative_path: str) -> Path:
    candidate = (dataset_root / relative_path).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes the dataset root: {relative_path}") from error
    return candidate


def _validate_manifest_identity(relative_path: str, split: str, label: str) -> None:
    path_parts = Path(relative_path).parts
    if len(path_parts) < 3 or path_parts[0] != split:
        raise ValueError(
            f"Manifest path '{relative_path}' does not match assigned split '{split}'."
        )
    source_label = canonicalize_source_label(path_parts[1])
    if source_label != label:
        raise ValueError(
            f"Manifest path '{relative_path}' does not match assigned label '{label}'."
        )


def collect_manifest_samples(
    dataset_root: Path,
    manifest_path: Path,
    split: str,
) -> list[ImageSample]:
    """Collect one leakage-free split exclusively from a clean manifest."""

    if split not in DATASET_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {', '.join(DATASET_SPLITS)}.")
    dataset_root = dataset_root.resolve()
    entries = _load_manifest_entries(manifest_path)
    samples: list[ImageSample] = []
    represented_classes: set[str] = set()

    for entry in entries:
        if not isinstance(entry, dict) or entry.get("split") != split:
            continue
        label = entry.get("label")
        relative_path = entry.get("path")
        if label not in CLASS_TO_INDEX or not isinstance(relative_path, str):
            raise ValueError("Split manifest contains an invalid path or class label.")
        _validate_manifest_identity(relative_path, split, label)
        image_path = _resolve_manifest_image(dataset_root, relative_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
        represented_classes.add(label)
        samples.append(ImageSample(image_path, CLASS_TO_INDEX[label]))

    missing_classes = set(CLASS_NAMES) - represented_classes
    if missing_classes:
        missing = ", ".join(sorted(missing_classes))
        raise ValueError(f"Manifest split '{split}' is missing classes: {missing}.")
    return samples


def validate_manifest(dataset_root: Path, manifest_path: Path) -> DatasetValidationReport:
    """Verify every manifest reference, digest, class, and split assignment."""

    dataset_root = dataset_root.resolve()
    entries = _load_manifest_entries(manifest_path)
    image_counts = _empty_image_counts()
    missing_images: list[str] = []
    corrupt_images: list[str] = []
    unsupported_files: list[str] = []
    hash_mismatches: list[str] = []
    inspected_images: list[InspectedImage] = []
    seen_paths: set[str] = set()

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Split manifest contains a non-object entry.")
        relative_path = entry.get("path")
        split = entry.get("split")
        label = entry.get("label")
        expected_digest = entry.get("sha256")
        if (
            not isinstance(relative_path, str)
            or split not in DATASET_SPLITS
            or label not in CLASS_TO_INDEX
            or not isinstance(expected_digest, str)
        ):
            raise ValueError("Split manifest contains an invalid entry.")
        if relative_path in seen_paths:
            raise ValueError(f"Split manifest references a path more than once: {relative_path}")
        seen_paths.add(relative_path)
        _validate_manifest_identity(relative_path, split, label)
        image_counts[split][label] += 1

        image_path = _resolve_manifest_image(dataset_root, relative_path)
        if not image_path.is_file():
            missing_images.append(relative_path)
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            unsupported_files.append(relative_path)
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            actual_digest = sha256_file(image_path)
        except (OSError, UnidentifiedImageError):
            corrupt_images.append(relative_path)
            continue
        if actual_digest != expected_digest:
            hash_mismatches.append(relative_path)
        inspected_images.append(InspectedImage(relative_path, split, label, actual_digest))

    missing_classes = [
        f"{split}/{class_name}"
        for split in DATASET_SPLITS
        for class_name in CLASS_NAMES
        if image_counts[split][class_name] == 0
    ]
    return _build_validation_report(
        dataset_root=dataset_root,
        image_counts=image_counts,
        inspected_images=inspected_images,
        missing_classes=missing_classes,
        missing_images=missing_images,
        corrupt_images=corrupt_images,
        unsupported_files=unsupported_files,
        hash_mismatches=hash_mismatches,
    )


class AlphaTrashDataset(Dataset[tuple[Tensor, int]]):
    """Load a leakage-free AlphaTrash split from a required manifest."""

    def __init__(
        self,
        dataset_root: Path,
        manifest_path: Path,
        split: str,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.split = split
        self.transform = transform
        self.samples = collect_manifest_samples(self.dataset_root, self.manifest_path, split)
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


def save_validation_report(report: DatasetValidationReport, output_path: Path) -> None:
    """Write a dataset report without modifying any source images."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


def raise_for_dataset_issues(report: DatasetValidationReport) -> None:
    """Stop training when manifest integrity or split isolation is invalid."""

    problems: list[str] = []
    if report.missing_classes:
        problems.append(f"missing or empty classes: {len(report.missing_classes)}")
    if report.missing_images:
        problems.append(f"missing images: {len(report.missing_images)}")
    if report.corrupt_images:
        problems.append(f"corrupt images: {len(report.corrupt_images)}")
    if report.unsupported_files:
        problems.append(f"unsupported files: {len(report.unsupported_files)}")
    if report.hash_mismatches:
        problems.append(f"content hash mismatches: {len(report.hash_mismatches)}")
    if report.cross_split_duplicates:
        problems.append(
            f"exact duplicate groups across splits: {len(report.cross_split_duplicates)}"
        )

    if problems:
        raise ValueError(
            f"Dataset validation failed ({'; '.join(problems)}). "
            "Review the validation report; no files were changed."
        )
