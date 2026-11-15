"""Small helpers: reproducibility, device selection, timing."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """``"auto"`` picks CUDA when available, else CPU."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Print the wall-clock time of a block."""
    start = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - start:.1f} s")
