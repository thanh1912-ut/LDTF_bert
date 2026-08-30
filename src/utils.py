"""Reproducibility, serialization, and small runtime utilities."""

from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, including CUDA when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torchrun binds each process to one current device. Seeding only that
        # device avoids creating CUDA contexts on sibling GPUs.
        torch.cuda.manual_seed(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        # This is an explicit throughput mode. Undo deterministic settings in
        # case an earlier run in the same interpreter enabled them.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python workers from PyTorch's per-worker initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    """Write JSON, converting paths and NumPy scalar values where needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent, default=default)


def load_json(path: str | Path) -> Any:
    """Read UTF-8 JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append one JSON object to a JSONL file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Print elapsed wall-clock time for a block."""
    import time

    start = time.perf_counter()
    yield
    print(f"[timer] {label}: {time.perf_counter() - start:.2f}s", flush=True)


def get_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def capture_rng_state() -> dict[str, Any]:
    """Capture all process RNG states needed for a resumable checkpoint."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        # In one-process-per-GPU DDP, touching every visible CUDA device would
        # create an unnecessary context on the sibling GPU. Save only this
        # process's current device; restore_rng_state still accepts legacy lists.
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "torch_cuda_device": torch.cuda.current_device()
        if torch.cuda.is_available()
        else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(tuple(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        if isinstance(cuda_state, (list, tuple)):  # legacy single-process checkpoints
            torch.cuda.set_rng_state_all(cuda_state)
        else:
            torch.cuda.set_rng_state(cuda_state)
