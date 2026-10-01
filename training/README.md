# VANTA MobileNetV3-Small training pipeline

This directory contains the reproducible training, evaluation, and single-image inference pipeline for VANTA. It uses ImageNet-pretrained MobileNetV3-Small and keeps machine-learning dependencies separate from the FastAPI backend.

No model has been trained by these scripts yet. Checkpoints, metrics, evaluation outputs, and datasets are ignored by Git.

## Verified AlphaTrash dataset

Source: [Patipol-BKK/alphatrash-dataset](https://github.com/Patipol-BKK/alphatrash-dataset), local revision `33801d876c161b1e35073ebd401a6e2a0cdece3b`.

The existing checkout was inspected at `../data/alphatrash-dataset/trash_dataset`. It retains the source `train`, `val`, and `test` splits. This checkout uses `plastic`; the loader also recognizes the `plasic` spelling documented in older upstream versions and maps either spelling to VANTA's canonical `plastic` class without renaming files.

| Split | Version | general | metal | organic | paper | plastic | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| train | original | 833 | 746 | 574 | 627 | 1,206 | 3,986 |
| train | clean | 833 | 639 | 574 | 627 | 1,065 | 3,738 |
| val | original | 179 | 160 | 123 | 135 | 259 | 856 |
| val | clean | 179 | 151 | 123 | 135 | 251 | 839 |
| test | original | 178 | 159 | 122 | 133 | 257 | 849 |
| test | clean | 177 | 157 | 122 | 132 | 251 | 839 |
| total | original | 1,190 | 1,065 | 819 | 895 | 1,722 | 5,691 |
| total | clean | 1,189 | 947 | 819 | 894 | 1,567 | 5,416 |

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

## Generate leakage-free splits

Training and evaluation require `manifests/alphatrash_clean_splits.json`; they cannot silently fall back to the contaminated folder splits. The generator verifies and hashes every source image, then keeps one path per SHA-256 digest. For a digest found in multiple splits, the deterministic priority is **test, then validation, then training**. This preserves unique test examples wherever possible while preventing the same content from influencing model fitting or selection.

The generator never modifies or copies an image. Paths in the manifest are relative to the dataset root, so the same file is portable to Kaggle. Regenerate it with:

```bash
cd training
source .venv/bin/activate
python scripts/create_clean_manifest.py \
  --data-dir ../data/alphatrash-dataset/trash_dataset \
  --output manifests/alphatrash_clean_splits.json \
  --conflict-report runs/duplicate_label_conflicts.json
```

The current source contains one conflicting-label group: the exact same bytes appear as `test/general/28.jpeg` and `test/paper/28.jpeg`. Both references are excluded from the clean manifest and recorded for manual review; the generator never chooses a label. It also removes 273 redundant same-label references. All original files remain untouched.

## Validate the clean manifest

Validation resolves every manifest path, opens every image, recomputes its SHA-256 digest, checks label/path consistency, counts every class, and verifies split isolation:

```bash
python scripts/validate_dataset.py \
  --data-dir ../data/alphatrash-dataset/trash_dataset \
  --manifest manifests/alphatrash_clean_splits.json \
  --output runs/cleaned_dataset_validation.json
```

The verified clean manifest contains 5,416 images, all five classes in all three splits, zero missing or corrupt references, zero hash mismatches, and zero exact duplicates across splits. Its deterministic ID is `5fe461138fdbc07c2b591209d9865a50db14420fca9f34e2ec45e55914586464`.

To audit the unchanged original folders again, add `--original-splits` and choose a separate report path.

## Train

Stage A freezes pretrained features and learns the five-class classifier. Stage B is optional and unfreezes the final feature blocks with a smaller learning rate.

Classifier stage only:

```bash
python scripts/train.py \
  --data-dir ../data/alphatrash-dataset/trash_dataset \
  --manifest manifests/alphatrash_clean_splits.json \
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
  --manifest manifests/alphatrash_clean_splits.json \
  --batch-size 32 \
  --stage-a-epochs 10 \
  --stage-a-learning-rate 0.001 \
  --stage-b-epochs 8 \
  --stage-b-learning-rate 0.0001 \
  --unfreeze-blocks 3 \
  --early-stopping-patience 3 \
  --seed 42
```

The best validation checkpoint is written to `checkpoints/best_model.pt`, and per-epoch metrics are written to `runs/training_metrics.json`. The checkpoint records the manifest ID, and evaluation rejects a checkpoint created from a different manifest. The test split is not used during training or checkpoint selection. Use `--device cpu`, `--device mps`, or `--device cuda` to override automatic device selection.

For an M1 Mac with 8 GB RAM, begin with `--device mps --batch-size 16 --workers 2`. Reduce the batch size to 8 if memory pressure is high. The saved checkpoint contains CPU tensors and loads portably on macOS even when trained on CUDA.

## Evaluate once on the test set

Run this after model selection is complete:

```bash
python scripts/evaluate.py \
  --data-dir ../data/alphatrash-dataset/trash_dataset \
  --manifest manifests/alphatrash_clean_splits.json \
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

Use the ready-to-import notebook at `notebooks/vanta_kaggle_training.ipynb`. Full training is disabled in the notebook until its smoke test passes.

### 1. Prepare Kaggle inputs

Create two private Kaggle Datasets through **Datasets → New Dataset**:

1. Upload the VANTA `training/` directory, including `pyproject.toml`, `src/`, `scripts/`, and `manifests/`. This is small and contains no credentials.
2. Upload the unchanged `data/alphatrash-dataset/trash_dataset/` directory. Do not flatten or rename its `train`, `val`, and `test` folders.

Import `vanta_kaggle_training.ipynb` into Kaggle, attach both datasets through **Add Input**, and select **Settings → Accelerator → GPU**. Update only `PROJECT_INPUT` and `DATASET_ROOT` in the first code cell to match the paths displayed in Kaggle's Input panel.

The notebook copies only the training code to `/kaggle/working`; it reads all 4.7 GB of images directly from the read-only dataset input. It regenerates the manifest and stops unless its ID is exactly:

```text
5fe461138fdbc07c2b591209d9865a50db14420fca9f34e2ec45e55914586464
```

### 2. Pretrained weights with internet disabled

With Kaggle internet enabled, TorchVision downloads and caches `MobileNet_V3_Small_Weights.DEFAULT` automatically during the smoke test. With internet disabled, download the official `mobilenet_v3_small-047dcff4.pth` file in an internet-enabled environment, upload it as a third private Kaggle Dataset, attach it to the notebook, and set:

```python
PRETRAINED_WEIGHTS = Path(
    "/kaggle/input/mobilenet-v3-small-weights/mobilenet_v3_small-047dcff4.pth"
)
```

Both smoke testing and full training use this same path. The loader verifies that it is a compatible TorchVision MobileNetV3-Small state dictionary before replacing the classifier.

### 3. Smoke test

The notebook runs `scripts/kaggle_smoke_test.py` before training. It requires CUDA, verifies all 5,416 manifest paths and SHA-256 hashes, checks the manifest ID, loads pretrained weights once, loads training and validation batches with batch size 16 and two workers, then performs one mixed-precision training step and one validation forward pass. It writes `/kaggle/working/vanta-output/smoke_test.json` and does not save the smoke-test model.

### 4. Full training command

The notebook prints this command and runs it only after `RUN_FULL_TRAINING = True`:

```bash
python scripts/train.py \
  --data-dir /kaggle/input/alphatrash-dataset/trash_dataset \
  --manifest /kaggle/working/vanta-output/alphatrash_clean_splits.json \
  --checkpoint /kaggle/working/vanta-output/best_model.pt \
  --metrics /kaggle/working/vanta-output/training_metrics.json \
  --class-mapping /kaggle/working/vanta-output/class_mapping.json \
  --validation-report /kaggle/working/vanta-output/cleaned_dataset_validation.json \
  --device cuda \
  --batch-size 16 \
  --workers 2 \
  --stage-a-epochs 10 \
  --stage-a-learning-rate 0.001 \
  --stage-b-epochs 8 \
  --stage-b-learning-rate 0.0001 \
  --unfreeze-blocks 3 \
  --early-stopping-patience 3 \
  --seed 42
```

Append `--pretrained-weights /kaggle/input/.../mobilenet_v3_small-047dcff4.pth` when using the offline weight input.

This command does not evaluate the final test set. It writes the best checkpoint, class mapping, manifest ID, full epoch history, best validation metrics, clean validation report, and smoke report under `/kaggle/working/vanta-output/`.

### 5. Download and use on an M1 Mac

After training, run the notebook's packaging cell and save a notebook version. Download `vanta-training-artifacts.zip` from Kaggle's Output panel, then locally:

```bash
cd /path/to/VANTA/training
unzip ~/Downloads/vanta-training-artifacts.zip -d kaggle-output
mkdir -p checkpoints
cp kaggle-output/best_model.pt checkpoints/best_model.pt

# Apple Metal
python scripts/predict.py /path/to/image.jpeg \
  --checkpoint checkpoints/best_model.pt \
  --device mps

# CPU fallback
python scripts/predict.py /path/to/image.jpeg \
  --checkpoint checkpoints/best_model.pt \
  --device cpu
```

The checkpoint stores CPU tensors, the five-class mapping, preprocessing metadata, source revision, manifest ID, and best validation metrics, so it is portable from Kaggle CUDA to macOS CPU or MPS.

## Package layout

```text
training/
├── manifests/             # Portable, versioned clean split assignments
├── notebooks/             # Kaggle workflow with training disabled by default
├── pyproject.toml
├── scripts/               # Thin command-line entry points
├── src/vanta_training/    # Dataset, model, training, evaluation, prediction
└── tests/                 # Lightweight synthetic-data tests
```
