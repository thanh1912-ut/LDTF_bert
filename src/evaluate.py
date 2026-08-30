"""Inference and checkpoint evaluation.

This module evaluates a model on whatever loader it is given. It never
constructs the official test loader itself; only
:func:`src.guard.official_test_loader` can do that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .metrics import compute_classification_metrics
from .registry import build_model_from_architecture
from .train import forward_kwargs, move_batch_to_device
from .utils import get_device, save_json


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint dictionary from disk."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint not found at {target}.")
    state = torch.load(target, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        raise KeyError(f"Checkpoint at {target} has no 'model_state_dict' entry.")
    return state


def load_model_from_checkpoint(
    path: str | Path, *, cache_dir: Optional[str] = None
) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model described by a checkpoint and load its weights."""
    state = load_checkpoint(path)
    architecture = state.get("architecture")
    if not architecture:
        raise KeyError(
            f"Checkpoint at {path} has no 'architecture' block; it cannot be rebuilt."
        )
    model = build_model_from_architecture(architecture, cache_dir=cache_dir)
    model.load_state_dict(state["model_state_dict"])
    return model, state


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw ``(logits, labels)`` arrays for an entire loader."""
    device = device or get_device()
    model.to(device)
    model.eval()
    logits_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        logits = model(**forward_kwargs(batch))["logits"]
        logits_chunks.append(logits.float().cpu().numpy())
        label_chunks.append(batch["labels"].cpu().numpy())
    return (
        np.concatenate(logits_chunks, axis=0),
        np.concatenate(label_chunks, axis=0),
    )


def evaluate_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device | None = None,
    output_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    label_names: tuple[str, ...] = config.LABEL_NAMES,
) -> dict[str, object]:
    """Evaluate *model* on *dataloader* and optionally persist the results."""
    logits, labels = predict_logits(model, dataloader, device=device)
    metrics = compute_classification_metrics(logits, labels, label_names=label_names)
    if predictions_path is not None:
        target = Path(predictions_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target, logits=logits, labels=labels, predictions=np.argmax(logits, axis=-1)
        )
        metrics["predictions_file"] = str(target)
    if output_path is not None:
        save_json(metrics, output_path)
    return metrics
