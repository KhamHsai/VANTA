"""Deterministic leakage-free split manifest generation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vanta_training.constants import CLASS_NAMES, CLASS_TO_INDEX, DATASET_SPLITS
from vanta_training.dataset import (
    MANIFEST_FORMAT_VERSION,
    DatasetValidationReport,
    InspectedImage,
    inspect_source_dataset,
)

# Keeping test first preserves the original final evaluation set wherever possible.
SPLIT_PRIORITY = ("test", "val", "train")
SPLIT_RANK = {split: rank for rank, split in enumerate(SPLIT_PRIORITY)}


@dataclass(frozen=True)
class ConflictingLabelGroup:
    """Identical bytes assigned to different canonical classes."""

    sha256: str
    labels: list[str]
    files: list[dict[str, str]]


@dataclass(frozen=True)
class ManifestBuildResult:
    """Manifest output plus conflicts that require human judgment."""

    manifest: dict[str, Any]
    conflicts: list[ConflictingLabelGroup]
    source_report: DatasetValidationReport


def _count_images(images: list[InspectedImage]) -> dict[str, dict[str, int]]:
    counts = {split: {class_name: 0 for class_name in CLASS_NAMES} for split in DATASET_SPLITS}
    for image in images:
        counts[image.split][image.label] += 1
    return counts


def _manifest_identifier(entries: list[dict[str, str]]) -> str:
    canonical_json = json.dumps(entries, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def build_clean_manifest(
    dataset_root: Path,
    *,
    source_revision: str,
) -> ManifestBuildResult:
    """Choose one deterministic representative for every unique image hash."""

    inspected_images, source_report = inspect_source_dataset(dataset_root)
    non_duplicate_problems = (
        source_report.missing_classes
        or source_report.corrupt_images
        or source_report.unsupported_files
    )
    if non_duplicate_problems:
        raise ValueError(
            "Source dataset contains missing classes, corrupt images, or unsupported files. "
            "Review the source validation report before generating a manifest."
        )

    images_by_digest: dict[str, list[InspectedImage]] = defaultdict(list)
    for image in inspected_images:
        images_by_digest[image.sha256].append(image)

    conflicts: list[ConflictingLabelGroup] = []
    selected_images: list[InspectedImage] = []
    excluded_duplicates: list[dict[str, str]] = []

    for digest in sorted(images_by_digest):
        occurrences = images_by_digest[digest]
        labels = sorted({image.label for image in occurrences})
        if len(labels) > 1:
            conflicts.append(
                ConflictingLabelGroup(
                    sha256=digest,
                    labels=labels,
                    files=[
                        {
                            "path": image.relative_path,
                            "label": image.label,
                            "split": image.split,
                        }
                        for image in sorted(occurrences, key=lambda item: item.relative_path)
                    ],
                )
            )
            continue

        selected = min(
            occurrences,
            key=lambda image: (SPLIT_RANK[image.split], image.relative_path),
        )
        selected_images.append(selected)
        for duplicate in sorted(occurrences, key=lambda image: image.relative_path):
            if duplicate == selected:
                continue
            excluded_duplicates.append(
                {
                    "path": duplicate.relative_path,
                    "label": duplicate.label,
                    "original_split": duplicate.split,
                    "kept_path": selected.relative_path,
                    "kept_split": selected.split,
                    "sha256": digest,
                }
            )

    conflicts.sort(key=lambda group: group.sha256)
    selected_images.sort(
        key=lambda image: (DATASET_SPLITS.index(image.split), image.label, image.relative_path)
    )
    entries = [
        {
            "path": image.relative_path,
            "label": image.label,
            "split": image.split,
            "sha256": image.sha256,
        }
        for image in selected_images
    ]
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "source_dataset": "Patipol-BKK/alphatrash-dataset",
        "source_revision": source_revision,
        "hash_algorithm": "sha256",
        "split_priority": list(SPLIT_PRIORITY),
        "class_to_index": dict(CLASS_TO_INDEX),
        "manifest_id": _manifest_identifier(entries),
        "original_counts": source_report.image_counts,
        "cleaned_counts": _count_images(selected_images),
        "entries": entries,
        "excluded_duplicates": excluded_duplicates,
        "excluded_label_conflicts": [asdict(conflict) for conflict in conflicts],
    }
    return ManifestBuildResult(manifest, conflicts, source_report)


def save_clean_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Write deterministic JSON containing only dataset-relative paths."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def save_conflict_report(
    conflicts: list[ConflictingLabelGroup],
    output_path: Path,
) -> None:
    """Write label conflicts for manual review without choosing a label."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conflict_count": len(conflicts),
        "conflicts": [asdict(conflict) for conflict in conflicts],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
