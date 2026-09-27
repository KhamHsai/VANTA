"""Test-set evaluation metrics and inference latency measurement."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vanta_training.constants import CLASS_NAMES, IMAGE_SIZE
from vanta_training.device import synchronize_device


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Calculate classification metrics and model-forward latency."""

    number_of_classes = len(CLASS_NAMES)
    confusion_matrix = torch.zeros((number_of_classes, number_of_classes), dtype=torch.int64)
    total_inference_seconds = 0.0
    sample_count = 0
    model.eval()

    # Warm-up avoids charging one-time accelerator initialization to the model.
    warmup_input = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device)
    model(warmup_input)
    synchronize_device(device)

    for images, targets in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")

        synchronize_device(device)
        started_at = time.perf_counter()
        logits = model(images)
        synchronize_device(device)
        total_inference_seconds += time.perf_counter() - started_at

        predictions = logits.argmax(dim=1).cpu()
        flattened_indices = targets.cpu() * number_of_classes + predictions
        confusion_matrix += torch.bincount(
            flattened_indices, minlength=number_of_classes**2
        ).reshape(number_of_classes, number_of_classes)
        sample_count += targets.size(0)

    if sample_count == 0:
        raise ValueError("The test data loader is empty.")
    return metrics_from_confusion_matrix(confusion_matrix, total_inference_seconds, sample_count)


def metrics_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
    total_inference_seconds: float,
    sample_count: int,
) -> dict[str, Any]:
    """Derive per-class and aggregate metrics with zero-division safety."""

    matrix = confusion_matrix.to(torch.float64)
    true_positives = matrix.diag()
    predicted_counts = matrix.sum(dim=0)
    actual_counts = matrix.sum(dim=1)

    precision = torch.where(predicted_counts > 0, true_positives / predicted_counts, 0.0)
    recall = torch.where(actual_counts > 0, true_positives / actual_counts, 0.0)
    f1_denominator = precision + recall
    f1 = torch.where(f1_denominator > 0, 2 * precision * recall / f1_denominator, 0.0)

    per_class = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(actual_counts[index].item()),
        }
        for index, class_name in enumerate(CLASS_NAMES)
    }
    return {
        "accuracy": float(true_positives.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix.tolist(),
        "confusion_matrix_labels": list(CLASS_NAMES),
        "average_inference_latency_ms": total_inference_seconds * 1000 / sample_count,
        "evaluated_images": sample_count,
    }


def save_evaluation_results(results: dict[str, Any], output_directory: Path) -> None:
    """Save machine-readable metrics and a labeled confusion-matrix CSV."""

    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    matrix_path = output_directory / "confusion_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["actual/predicted", *results["confusion_matrix_labels"]])
        for class_name, row in zip(
            results["confusion_matrix_labels"],
            results["confusion_matrix"],
            strict=True,
        ):
            writer.writerow([class_name, *row])
