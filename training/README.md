# VANTA MobileNetV3-Small training pipeline

This directory contains the reproducible training, evaluation, and single-image inference pipeline for VANTA. It uses ImageNet-pretrained MobileNetV3-Small and keeps machine-learning dependencies separate from the FastAPI backend.

No model has been trained by these scripts yet. Checkpoints, metrics, evaluation outputs, and datasets are ignored by Git.

## Verified AlphaTrash dataset

Source: [Patipol-BKK/alphatrash-dataset](https://github.com/Patipol-BKK/alphatrash-dataset), local revision `33801d876c161b1e35073ebd401a6e2a0cdece3b`.

The existing checkout was inspected at `../data/alphatrash-dataset/trash_dataset`. It retains the source `train`, `val`, and `test` splits. This checkout uses `plastic`; the loader also recognizes the `plasic` spelling documented in older upstream versions and maps either spelling to VANTA's canonical `plastic` class without renaming files.

| Split | general | metal | organic | paper | plastic | Total |
|---|---:|---:|---:|---:|---:|---:|
| train | 833 | 746 | 574 | 627 | 1,206 | 3,986 |
| val | 179 | 160 | 123 | 135 | 259 | 856 |
| test | 178 | 159 | 122 | 133 | 257 | 849 |
| total | 1,190 | 1,065 | 819 | 895 | 1,722 | 5,691 |

All files in the inspected checkout use the `.jpeg` extension. Training reads only `train/` and `val/`; `test/` is loaded only by the separate evaluation command.

If the dataset is absent on another machine, clone it manually from the repository root:

```bash
git clone --depth 1 https://github.com/Patipol-BKK/alphatrash-dataset.git data/alphatrash-dataset
```

Do not re-split or shuffle the source dataset.

## Install

Python 3.11 or 3.12 is recommended. From the repository root:

```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

PyTorch automatically uses the appropriate CPU/MPS build on macOS. On CUDA systems, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) first if a specific CUDA wheel is required, then install this project.

## Validate the dataset

Validation checks every image, counts each class, reports unreadable/corrupt files, and hashes image bytes to find exact duplicates crossing split boundaries. It never deletes, moves, or relabels an image.

```bash
cd training
source .venv/bin/activate
python scripts/validate_dataset.py \
  --data-dir ../data/alphatrash-dataset/trash_dataset \
  --output runs/dataset_validation.json
```

Review `runs/dataset_validation.json`. Training stops on corrupt files, missing/empty classes, unsupported files, or cross-split duplicates. If duplicate groups are expected and have been reviewed, training can be explicitly allowed with `--allow-cross-split-duplicates`; the files remain unchanged.

The completed local audit found no missing classes, corrupt images, or unsupported files. It did find **119 exact duplicate pairs across splits**: 57 train/validation, 51 train/test, and 11 validation/test. These are source files with names such as `pic1230.jpeg` and `pic1230 - Copy.jpeg`; the pipeline did not remove or move them. Review the JSON report before deciding whether to train with the original splits. Any metrics produced with these duplicates retained should disclose the resulting data-leakage limitation.

## Train

Stage A freezes pretrained features and learns the five-class classifier. Stage B is optional and unfreezes the final feature blocks with a smaller learning rate.

Classifier stage only:

```bash
python scripts/train.py \
  --data-dir ../data/alphatrash-dataset/trash_dataset \
  --batch-size 32 \
  --stage-a-epochs 10 \
  --stage-a-learning-rate 0.001 \
  --early-stopping-patience 3 \
  --seed 42
```

Classifier training followed by fine-tuning:

```bash
python scripts/train.py \
  --data-dir ../data/alphatrash-dataset/trash_dataset \
  --batch-size 32 \
  --stage-a-epochs 10 \
  --stage-a-learning-rate 0.001 \
  --stage-b-epochs 8 \
  --stage-b-learning-rate 0.0001 \
  --unfreeze-blocks 3 \
  --early-stopping-patience 3 \
  --seed 42
```

The best validation checkpoint is written to `checkpoints/best_model.pt`, and per-epoch metrics are written to `runs/training_metrics.json`. The test split is not used during training or checkpoint selection. Use `--device cpu`, `--device mps`, or `--device cuda` to override automatic device selection.

With the currently verified checkout, the commands above intentionally stop at the duplicate safety check. To preserve the upstream splits and proceed after accepting and documenting the leakage risk, append:

```bash
--allow-cross-split-duplicates
```

For an M1 Mac with 8 GB RAM, begin with `--device mps --batch-size 16 --workers 2`. Reduce the batch size to 8 if memory pressure is high. The saved checkpoint contains CPU tensors and loads portably on macOS even when trained on CUDA.

## Evaluate once on the test set

Run this after model selection is complete:

```bash
python scripts/evaluate.py \
  --data-dir ../data/alphatrash-dataset/trash_dataset \
  --checkpoint checkpoints/best_model.pt \
  --output-dir outputs/evaluation \
  --device auto
```

The command writes `evaluation_metrics.json` and `confusion_matrix.csv`. Metrics include accuracy, per-class precision/recall/F1, macro F1, and average model-forward latency per image. Data loading and preprocessing are intentionally excluded from latency.

## Predict one image

Use any real image path; for example:

```bash
python scripts/predict.py ../data/alphatrash-dataset/trash_dataset/test/plastic/10.jpeg \
  --checkpoint checkpoints/best_model.pt \
  --device mps
```

The JSON response contains the predicted category and scores for all five categories. Applications making repeated predictions should instantiate `ImagePredictor` once and reuse it so the checkpoint is not repeatedly loaded.

## Run tests and lint

Tests create tiny synthetic images and never read or modify the real dataset:

```bash
python -m pytest
ruff check src tests scripts
```

## Kaggle free-GPU workflow

Create a Kaggle notebook, enable **Settings → Accelerator → GPU**, and add the VANTA repository plus AlphaTrash as notebook inputs or upload them as private Kaggle datasets. Then use these cells, adjusting the two input paths shown by Kaggle:

```python
%cd /kaggle/input/vanta/VANTA/training
!python -m pip install -q -e .
```

```python
!python scripts/validate_dataset.py \
    --data-dir /kaggle/input/alphatrash-dataset/trash_dataset \
    --output /kaggle/working/dataset_validation.json
```

```python
!python scripts/train.py \
    --data-dir /kaggle/input/alphatrash-dataset/trash_dataset \
    --checkpoint /kaggle/working/best_model.pt \
    --metrics /kaggle/working/training_metrics.json \
    --validation-report /kaggle/working/dataset_validation.json \
    --device cuda \
    --batch-size 64 \
    --stage-a-epochs 10 \
    --stage-b-epochs 8 \
    --early-stopping-patience 3
```

Download `best_model.pt`, `training_metrics.json`, and `dataset_validation.json` from Kaggle's `/kaggle/working` output panel. Place the checkpoint at `training/checkpoints/best_model.pt` locally, then use the evaluation or prediction commands above. Do not evaluate repeatedly on the test set while tuning the model.

## Package layout

```text
training/
├── pyproject.toml
├── scripts/               # Thin command-line entry points
├── src/vanta_training/    # Dataset, model, training, evaluation, prediction
└── tests/                 # Lightweight synthetic-data tests
```
