"""Validate AlphaTrash structure, images, and exact cross-split duplicates."""

import argparse
from pathlib import Path

from vanta_training.dataset import save_validation_report, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/alphatrash-dataset/trash_dataset"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/dataset_validation.json"))
    arguments = parser.parse_args()

    try:
        report = validate_dataset(arguments.data_dir)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Dataset validation failed: {error}") from error
    save_validation_report(report, arguments.output)
    print(f"Validated {report.total_images} images at {report.dataset_root}")
    for split, counts in report.image_counts.items():
        count_summary = ", ".join(f"{name}={count}" for name, count in counts.items())
        print(f"{split}: {count_summary}")
    print(f"Missing or empty classes: {len(report.missing_classes)}")
    print(f"Corrupt images: {len(report.corrupt_images)}")
    print(f"Unsupported files: {len(report.unsupported_files)}")
    print(f"Exact duplicate groups across splits: {len(report.cross_split_duplicates)}")
    print(f"Full report: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
