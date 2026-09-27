from pathlib import Path
from shutil import copyfile

from PIL import Image

from vanta_training.manifest import SPLIT_PRIORITY, build_clean_manifest


def _add_unique_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)


def test_duplicate_is_removed_and_test_split_has_priority(synthetic_dataset: Path) -> None:
    # Keep another training example so de-duplication does not empty the class.
    _add_unique_image(synthetic_dataset / "train" / "general" / "unique.png", (1, 2, 3))
    duplicate_path = synthetic_dataset / "test" / "general" / "duplicate.png"
    copyfile(synthetic_dataset / "train" / "general" / "sample.png", duplicate_path)

    result = build_clean_manifest(synthetic_dataset, source_revision="test-revision")

    assert result.conflicts == []
    assert result.manifest is not None
    manifest = result.manifest
    assert manifest["split_priority"] == list(SPLIT_PRIORITY)
    paths = {entry["path"] for entry in manifest["entries"]}
    assert "test/general/duplicate.png" in paths
    assert "train/general/sample.png" not in paths
    assert len({entry["sha256"] for entry in manifest["entries"]}) == len(manifest["entries"])


def test_manifest_generation_is_deterministic(synthetic_dataset: Path) -> None:
    first = build_clean_manifest(synthetic_dataset, source_revision="same-revision")
    second = build_clean_manifest(synthetic_dataset, source_revision="same-revision")

    assert first.manifest == second.manifest


def test_conflicting_duplicate_labels_require_manual_review(synthetic_dataset: Path) -> None:
    conflicting_path = synthetic_dataset / "val" / "metal" / "conflicting.png"
    copyfile(synthetic_dataset / "train" / "general" / "sample.png", conflicting_path)

    result = build_clean_manifest(synthetic_dataset, source_revision="test-revision")

    assert result.manifest is not None
    assert len(result.conflicts) == 1
    assert result.conflicts[0].labels == ["general", "metal"]
    manifest_paths = {entry["path"] for entry in result.manifest["entries"]}
    assert "train/general/sample.png" not in manifest_paths
    assert "val/metal/conflicting.png" not in manifest_paths
    assert result.manifest["excluded_label_conflicts"]


def test_duplicate_spanning_all_splits_is_kept_only_in_test(synthetic_dataset: Path) -> None:
    _add_unique_image(synthetic_dataset / "train" / "general" / "unique.png", (1, 2, 3))
    source_path = synthetic_dataset / "train" / "general" / "sample.png"
    validation_copy = synthetic_dataset / "val" / "general" / "three-way-copy.png"
    test_copy = synthetic_dataset / "test" / "general" / "three-way-copy.png"
    copyfile(source_path, validation_copy)
    copyfile(source_path, test_copy)

    result = build_clean_manifest(synthetic_dataset, source_revision="test-revision")

    assert result.manifest is not None
    matching_entries = [
        entry
        for entry in result.manifest["entries"]
        if entry["path"].endswith("three-way-copy.png")
    ]
    assert matching_entries == [
        {
            "path": "test/general/three-way-copy.png",
            "label": "general",
            "split": "test",
            "sha256": matching_entries[0]["sha256"],
        }
    ]
    excluded_paths = {entry["path"] for entry in result.manifest["excluded_duplicates"]}
    assert "train/general/sample.png" in excluded_paths
    assert "val/general/three-way-copy.png" in excluded_paths
