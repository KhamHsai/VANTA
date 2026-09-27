"""Evaluate a trained VANTA checkpoint on AlphaTrash's untouched test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from vanta_training.checkpoint import load_checkpoint
from vanta_training.constants import CLASS_TO_INDEX
from vanta_training.dataset import AlphaTrashDataset
from vanta_training.device import select_device
from vanta_training.evaluation import evaluate_classifier, save_evaluation_results
from vanta_training.transforms import build_inference_transform


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/alphatrash-dataset/trash_dataset"),
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def run_evaluation(arguments: argparse.Namespace) -> dict[str, object]:
    """Load the checkpoint once and evaluate only the original test split."""

    if arguments.batch_size <= 0 or arguments.workers < 0:
        raise ValueError("batch size must be positive and workers cannot be negative.")
    device = select_device(arguments.device)
    model, metadata = load_checkpoint(arguments.checkpoint, device)
    if metadata["class_to_index"] != CLASS_TO_INDEX:
        raise ValueError("Dataset and checkpoint class mappings do not match.")

    test_dataset = AlphaTrashDataset(
        arguments.data_dir.resolve(),
        "test",
        transform=build_inference_transform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=arguments.workers > 0,
    )
    results = evaluate_classifier(model, test_loader, device)
    save_evaluation_results(results, arguments.output_dir)
    return results


def main() -> None:
    arguments = build_argument_parser().parse_args()
    try:
        results = run_evaluation(arguments)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Evaluation failed: {error}") from error
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
