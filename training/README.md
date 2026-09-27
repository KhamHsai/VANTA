# VANTA model training

This directory is reserved for future PyTorch training and evaluation code using MobileNetV3-Small.

No datasets, model weights, or large machine-learning dependencies are included in the foundation. When training work begins, keep downloaded data under `data/`, checkpoints under `checkpoints/`, and run outputs under `runs/`; those paths are ignored by Git.

Suggested future layout:

- `configs/` — training configuration
- `src/vanta_training/` — reusable dataset, model, and training modules
- `scripts/` — command-line entry points
- `tests/` — focused unit tests

