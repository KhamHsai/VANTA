"""Generate deterministic leakage-free splits without copying source images."""

import argparse
import subprocess
from pathlib import Path

from vanta_training.manifest import (
    build_clean_manifest,
    save_clean_manifest,
    save_conflict_report,
)


def _source_revision(dataset_root: Path, supplied_revision: str | None) -> str:
    if supplied_revision:
        return supplied_revision
    repository_directory = dataset_root.resolve().parent
    try:
        return subprocess.run(
            ["git", "-C", str(repository_directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"Unable to determine dataset Git revision from {repository_directory}."
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/alphatrash-dataset/trash_dataset"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/alphatrash_clean_splits.json"),
    )
    parser.add_argument(
        "--conflict-report",
        type=Path,
        default=Path("runs/duplicate_label_conflicts.json"),
    )
    parser.add_argument(
        "--source-revision",
        help="Dataset Git commit; detected automatically when its .git directory is available.",
    )
    arguments = parser.parse_args()

    try:
        result = build_clean_manifest(
            arguments.data_dir,
            source_revision=_source_revision(arguments.data_dir, arguments.source_revision),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Manifest generation failed: {error}") from error

    save_conflict_report(result.conflicts, arguments.conflict_report)
    save_clean_manifest(result.manifest, arguments.output)
    excluded_count = len(result.manifest["excluded_duplicates"])
    entry_count = len(result.manifest["entries"])
    print(f"Clean manifest: {arguments.output.resolve()}")
    print(f"Referenced images: {entry_count}")
    print(f"Excluded duplicate references: {excluded_count}")
    print(f"Excluded conflicting-label groups: {len(result.conflicts)}")
    if result.conflicts:
        print(f"Manual review required: {arguments.conflict_report.resolve()}")
    print(f"Manifest ID: {result.manifest['manifest_id']}")


if __name__ == "__main__":
    main()
