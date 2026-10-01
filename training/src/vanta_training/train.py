"""Command-line transfer learning for MobileNetV3-Small."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from vanta_training.checkpoint import build_checkpoint_metadata, load_checkpoint, save_checkpoint
from vanta_training.constants import CLASS_TO_INDEX
from vanta_training.dataset import (
    AlphaTrashDataset,
    raise_for_dataset_issues,
    save_validation_report,
    validate_manifest,
)
from vanta_training.device import select_device
from vanta_training.engine import EpochMetrics, train_one_epoch, validate_one_epoch
from vanta_training.model import (
    build_model,
    freeze_feature_extractor,
    unfreeze_last_feature_blocks,
)
from vanta_training.reproducibility import seed_data_loader_worker, set_random_seed
from vanta_training.transforms import build_inference_transform, build_training_transform


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/alphatrash-dataset/trash_dataset"),
        help="Directory containing the unchanged AlphaTrash image folders.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/alphatrash_clean_splits.json"),
        help="Required leakage-free split manifest.",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("runs/training_metrics.json"))
    parser.add_argument("--class-mapping", type=Path, default=Path("runs/class_mapping.json"))
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("runs/cleaned_dataset_validation.json"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage-a-epochs", type=int, default=10)
    parser.add_argument("--stage-a-learning-rate", type=float, default=1e-3)
    parser.add_argument("--stage-b-epochs", type=int, default=0)
    parser.add_argument("--stage-b-learning-rate", type=float, default=1e-4)
    parser.add_argument("--unfreeze-blocks", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop a stage after this many non-improving epochs; 0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        help="Optional local ImageNet weight file for offline environments.",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    positive_values = {
        "batch size": arguments.batch_size,
        "stage A epochs": arguments.stage_a_epochs,
        "stage A learning rate": arguments.stage_a_learning_rate,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
    if arguments.stage_b_epochs < 0:
        raise ValueError("stage B epochs cannot be negative.")
    if arguments.stage_b_epochs > 0 and arguments.stage_b_learning_rate <= 0:
        raise ValueError("stage B learning rate must be greater than zero.")
    if arguments.workers < 0 or arguments.early_stopping_patience < 0:
        raise ValueError("workers and early-stopping patience cannot be negative.")


def _build_data_loader(
    dataset: AlphaTrashDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
    )


def _is_better_checkpoint(
    validation_metrics: EpochMetrics,
    best_accuracy: float,
    best_loss: float,
) -> bool:
    return validation_metrics.accuracy > best_accuracy or (
        validation_metrics.accuracy == best_accuracy and validation_metrics.loss < best_loss
    )


def _write_metrics(metrics_path: Path, payload: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_training(arguments: argparse.Namespace) -> None:
    """Validate data and execute the configured transfer-learning stages."""

    _validate_arguments(arguments)
    set_random_seed(arguments.seed)
    device = select_device(arguments.device)
    dataset_root = arguments.data_dir.resolve()

    manifest_path = arguments.manifest.resolve()
    print(f"Validating clean manifest {manifest_path} ...")
    validation_report = validate_manifest(dataset_root, manifest_path)
    save_validation_report(validation_report, arguments.validation_report)
    raise_for_dataset_issues(validation_report)
    print(f"Validated {validation_report.total_images} images.")
    manifest_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    _write_metrics(
        arguments.class_mapping,
        {
            "class_to_index": dict(CLASS_TO_INDEX),
            "manifest_id": manifest_metadata["manifest_id"],
        },
    )

    training_dataset = AlphaTrashDataset(
        dataset_root,
        manifest_path,
        "train",
        transform=build_training_transform(),
    )
    validation_dataset = AlphaTrashDataset(
        dataset_root,
        manifest_path,
        "val",
        transform=build_inference_transform(),
    )
    data_loader_generator = torch.Generator().manual_seed(arguments.seed)
    training_loader = _build_data_loader(
        training_dataset,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        shuffle=True,
        device=device,
        generator=data_loader_generator,
    )
    validation_loader = _build_data_loader(
        validation_dataset,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        shuffle=False,
        device=device,
        generator=data_loader_generator,
    )

    print(f"Using device: {device}")
    model = build_model(
        pretrained=arguments.pretrained_weights is None,
        pretrained_weights_path=arguments.pretrained_weights,
    )
    freeze_feature_extractor(model)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    gradient_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    global_epoch = 0
    run_started_at = datetime.now(UTC).isoformat()
    run_configuration = {
        "data_dir": str(dataset_root),
        "manifest": str(manifest_path),
        "manifest_id": manifest_metadata["manifest_id"],
        "batch_size": arguments.batch_size,
        "stage_a_epochs": arguments.stage_a_epochs,
        "stage_a_learning_rate": arguments.stage_a_learning_rate,
        "stage_b_epochs": arguments.stage_b_epochs,
        "stage_b_learning_rate": arguments.stage_b_learning_rate,
        "unfreeze_blocks": arguments.unfreeze_blocks,
        "weight_decay": arguments.weight_decay,
        "early_stopping_patience": arguments.early_stopping_patience,
        "seed": arguments.seed,
        "workers": arguments.workers,
        "device": str(device),
        "pretrained_weights": (
            str(arguments.pretrained_weights.resolve())
            if arguments.pretrained_weights
            else "torchvision"
        ),
    }

    stages = [("classifier", arguments.stage_a_epochs, arguments.stage_a_learning_rate)]
    if arguments.stage_b_epochs > 0:
        stages.append(("fine_tune", arguments.stage_b_epochs, arguments.stage_b_learning_rate))

    for stage_name, stage_epochs, learning_rate in stages:
        if stage_name == "fine_tune":
            model, _ = load_checkpoint(arguments.checkpoint, device)
            unfreeze_last_feature_blocks(model, arguments.unfreeze_blocks)

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=arguments.weight_decay,
        )
        epochs_without_improvement = 0

        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            training_metrics = train_one_epoch(
                model,
                training_loader,
                loss_function,
                optimizer,
                device,
                gradient_scaler,
            )
            validation_metrics = validate_one_epoch(model, validation_loader, loss_function, device)
            epoch_record = {
                "epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": stage_epoch,
                "learning_rate": learning_rate,
                "train": asdict(training_metrics),
                "validation": asdict(validation_metrics),
            }
            history.append(epoch_record)
            print(
                f"epoch={global_epoch} stage={stage_name} "
                f"train_loss={training_metrics.loss:.4f} "
                f"train_accuracy={training_metrics.accuracy:.4f} "
                f"val_loss={validation_metrics.loss:.4f} "
                f"val_accuracy={validation_metrics.accuracy:.4f}"
            )

            if _is_better_checkpoint(validation_metrics, best_accuracy, best_loss):
                best_accuracy = validation_metrics.accuracy
                best_loss = validation_metrics.loss
                epochs_without_improvement = 0
                checkpoint_metadata = build_checkpoint_metadata(
                    source_dataset="Patipol-BKK/alphatrash-dataset",
                    source_revision=manifest_metadata["source_revision"],
                    split_manifest_id=manifest_metadata["manifest_id"],
                    best_epoch=global_epoch,
                    best_stage=stage_name,
                    validation_accuracy=best_accuracy,
                    validation_loss=best_loss,
                    training_configuration=run_configuration,
                )
                save_checkpoint(arguments.checkpoint, model, checkpoint_metadata)
            else:
                epochs_without_improvement += 1

            _write_metrics(
                arguments.metrics,
                {
                    "started_at": run_started_at,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "configuration": run_configuration,
                    "class_to_index": dict(CLASS_TO_INDEX),
                    "best_validation_accuracy": best_accuracy,
                    "best_validation_loss": best_loss,
                    "history": history,
                },
            )

            patience = arguments.early_stopping_patience
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping stage '{stage_name}' after {stage_epoch} epochs.")
                break

    print(f"Best checkpoint: {arguments.checkpoint}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")


def main() -> None:
    arguments = build_argument_parser().parse_args()
    try:
        run_training(arguments)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Training failed: {error}") from error


if __name__ == "__main__":
    main()
