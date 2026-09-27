"""Reproducibility settings for model training."""

import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These cuDNN settings make common convolution paths reproducible while
    # avoiding the broad performance cost of deterministic algorithms globally.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_data_loader_worker(worker_id: int) -> None:
    """Give each data-loader worker a deterministic NumPy/Python seed."""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
