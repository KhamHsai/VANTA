"""Run a one-batch CUDA preflight before a full Kaggle training job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from vanta_training.constants import CLASS_TO_INDEX
from vanta_training.dataset import (
    AlphaTrashDataset,
    raise_for_dataset_issues,
    validate_manifest,
)
from vanta_training.model import build_model, freeze_feature_extractor
from vanta_training.reproducibility import seed_data_loader_worker, set_random_seed
from vanta_training.transforms import build_inference_transform, build_training_transform


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _one_batch_loader(
    dataset: AlphaTrashDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_data_loader_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def _load_manifest_id(manifest_path: Path) -> str:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_id = manifest["manifest_id"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"Unable to read manifest ID from {manifest_path}.") from error
    if not isinstance(manifest_id, str):
        raise ValueError("Manifest ID must be a string.")
    return manifest_id


def run_smoke_test(arguments: argparse.Namespace) -> dict[str, Any]:
    """Verify the exact Kaggle prerequisites with one train and validation batch."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. In Kaggle Settings, select a GPU accelerator and restart."
        )
    if arguments.batch_size <= 0 or arguments.workers < 0:
        raise ValueError("Batch size must be positive and workers cannot be negative.")

    set_random_seed(arguments.seed)
    device = torch.device("cuda")
    dataset_root = arguments.data_dir.resolve()
    manifest_path = arguments.manifest.resolve()
    manifest_id = _load_manifest_id(manifest_path)
    if manifest_id != arguments.expected_manifest_id:
        raise ValueError(
            f"Manifest ID mismatch: expected {arguments.expected_manifest_id}, got {manifest_id}."
        )

    validation_report = validate_manifest(dataset_root, manifest_path)
    raise_for_dataset_issues(validation_report)
    training_dataset = AlphaTrashDataset(
        dataset_root, manifest_path, "train", build_training_transform()
    )
    validation_dataset = AlphaTrashDataset(
        dataset_root, manifest_path, "val", build_inference_transform()
    )
    training_loader = _one_batch_loader(
        training_dataset,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        shuffle=True,
        seed=arguments.seed,
    )
    validation_loader = _one_batch_loader(
        validation_dataset,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        shuffle=False,
        seed=arguments.seed,
    )

    model = build_model(
        pretrained=arguments.pretrained_weights is None,
        pretrained_weights_path=arguments.pretrained_weights,
    )
    freeze_feature_extractor(model)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
        weight_decay=1e-4,
    )
    gradient_scaler = torch.amp.GradScaler("cuda")

    training_images, training_targets = next(iter(training_loader))
    training_images = training_images.to(device, non_blocking=True)
    training_targets = training_targets.to(device, non_blocking=True)
    model.train()
    model.features.eval()  # type: ignore[attr-defined]
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        training_logits = model(training_images)
        training_loss = loss_function(training_logits, training_targets)
    gradient_scaler.scale(training_loss).backward()
    gradient_scaler.step(optimizer)
    gradient_scaler.update()

    validation_images, validation_targets = next(iter(validation_loader))
    validation_images = validation_images.to(device, non_blocking=True)
    validation_targets = validation_targets.to(device, non_blocking=True)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        validation_logits = model(validation_images)
        validation_loss = loss_function(validation_logits, validation_targets)

    if not torch.isfinite(training_loss) or not torch.isfinite(validation_loss):
        raise RuntimeError("Smoke test produced a non-finite loss.")

    result = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "manifest_id": manifest_id,
        "validated_images": validation_report.total_images,
        "train_samples": len(training_dataset),
        "validation_samples": len(validation_dataset),
        "batch_size": arguments.batch_size,
        "workers": arguments.workers,
        "training_batch_shape": list(training_images.shape),
        "validation_batch_shape": list(validation_images.shape),
        "training_loss": float(training_loss.detach().cpu()),
        "validation_loss": float(validation_loss.detach().cpu()),
        "class_to_index": dict(CLASS_TO_INDEX),
        "pretrained_weights": (
            str(arguments.pretrained_weights.resolve())
            if arguments.pretrained_weights
            else "torchvision-default"
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    arguments = build_argument_parser().parse_args()
    try:
        result = run_smoke_test(arguments)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Smoke test failed: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
