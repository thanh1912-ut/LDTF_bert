"""Training engine: optimizer groups, AMP, accumulation, checkpoints, resume.

Protocol summary
----------------
* Frozen backbone: ``requires_grad=False``, absent from the optimizer, and kept
  in ``eval()`` after ``model.train()`` so its dropout stays disabled.
* Differential learning rates: backbone and randomly-initialised head groups,
  each split into decay / no-decay by parameter name.
* Gradient accumulation normalises by the true group size and flushes the final
  partial group; the scheduler steps only when the optimizer steps.
* Checkpoint selection: validation macro F1, tie-broken by lower validation loss
  and then by earlier epoch.
* ``best.pt`` is slim; ``last.pt`` is fully resumable.
* The official test split is never constructed or touched here.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import config, guard
from .dataset import DistributedEvalSampler
from .distributed import (
    DistributedContext,
    all_gather_objects,
    barrier,
    broadcast_object,
    current_context,
    reduce_maxima,
    reduce_sums,
    unwrap_model,
    wrap_ddp,
)
from .metrics import compute_classification_metrics
from .progress import TrainingReporter
from .utils import (
    append_jsonl,
    capture_rng_state,
    restore_rng_state,
    save_json,
    set_seed,
)

NO_DECAY_SUFFIXES = ("bias", "layernorm.weight", "layer_norm.weight", "ln.weight")
PROGRESS_SYNC_INTERVAL = 20


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters and switches for one training run."""

    output_dir: Path
    run_id: str = "run"
    epochs: int = config.EPOCHS
    backbone_learning_rate: float = config.BACKBONE_LEARNING_RATE
    head_learning_rate: float = config.HEAD_LEARNING_RATE
    weight_decay: float = config.WEIGHT_DECAY
    warmup_ratio: float = config.WARMUP_RATIO
    max_grad_norm: float = config.MAX_GRAD_NORM
    grad_accum_steps: int = config.GRAD_ACCUM_STEPS
    label_smoothing: float = config.LABEL_SMOOTHING
    freeze_backbone: bool = False
    seed: int = config.SEED
    patience: int = config.EARLY_STOPPING_PATIENCE
    use_amp: bool = True
    deterministic: bool = True
    use_fused_optimizer: bool = True
    global_batch_size: int = config.BATCH_SIZE
    per_device_batch_size: int = config.BATCH_SIZE
    global_eval_batch_size: int = config.EVAL_BATCH_SIZE
    per_device_eval_batch_size: int = config.EVAL_BATCH_SIZE
    world_size: int = 1
    num_workers: int = config.NUM_WORKERS
    pad_to_multiple_of: int | None = None
    amp_dtype: str = "disabled"
    distributed_backend: str = "none"
    device_names: tuple[str, ...] = ()
    data_signature: dict[str, str] = field(default_factory=dict)
    architecture: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainResult:
    """Outcome of a training run."""

    best_val_f1_macro: float
    best_val_accuracy: float
    best_val_loss: float
    best_epoch: int
    history: list[dict[str, float]]
    resumed_from_epoch: int = 0
    total_train_seconds: float = 0.0
    session_train_seconds: float = 0.0
    peak_vram_gb: float = 0.0


# ---------------------------------------------------------------------------
# Optimizer and schedule
# ---------------------------------------------------------------------------
def _is_no_decay(name: str, parameter: nn.Parameter) -> bool:
    """Return whether a parameter must be excluded from weight decay.

    Excluded: biases and LayerNorm weights (the standard BERT convention), plus
    any 1-D parameter. The rank rule catches normalisation weights registered
    under a non-standard attribute name, and it also protects quantities where
    decay would be semantically wrong, such as the scalar-mix layer logits
    (decay would pull the mixture toward uniform) and per-class scorer biases.
    """
    lowered = name.lower()
    if any(lowered.endswith(suffix) for suffix in NO_DECAY_SUFFIXES):
        return True
    return parameter.ndim <= 1


def build_optimizer(model: nn.Module, train_config: TrainConfig) -> AdamW:
    """Build AdamW with four groups: {backbone, head} x {decay, no-decay}.

    Frozen parameters are excluded. Every trainable parameter appears exactly
    once; the invariant is asserted before the optimizer is returned.
    """
    backbone = getattr(model, "backbone", None)
    backbone_ids = (
        {id(p) for p in backbone.parameters()} if backbone is not None else set()
    )

    groups: dict[str, list[nn.Parameter]] = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"Parameter {name} appears twice in named_parameters().")
        seen.add(id(parameter))
        scope = "backbone" if id(parameter) in backbone_ids else "head"
        decay = "no_decay" if _is_no_decay(name, parameter) else "decay"
        groups[f"{scope}_{decay}"].append(parameter)

    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected:
        raise ValueError(
            "Optimizer coverage mismatch: some trainable parameters were skipped."
        )

    param_groups = [
        {
            "params": groups["backbone_decay"],
            "lr": train_config.backbone_learning_rate,
            "weight_decay": train_config.weight_decay,
            "name": "backbone_decay",
        },
        {
            "params": groups["backbone_no_decay"],
            "lr": train_config.backbone_learning_rate,
            "weight_decay": 0.0,
            "name": "backbone_no_decay",
        },
        {
            "params": groups["head_decay"],
            "lr": train_config.head_learning_rate,
            "weight_decay": train_config.weight_decay,
            "name": "head_decay",
        },
        {
            "params": groups["head_no_decay"],
            "lr": train_config.head_learning_rate,
            "weight_decay": 0.0,
            "name": "head_no_decay",
        },
    ]
    active_groups = [group for group in param_groups if group["params"]]
    parameters = [parameter for group in active_groups for parameter in group["params"]]
    fused = bool(
        train_config.use_fused_optimizer
        and parameters
        and all(parameter.device.type == "cuda" for parameter in parameters)
    )
    return AdamW(active_groups, fused=fused)


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_config: TrainConfig, total_steps: int
) -> LambdaLR:
    """Linear warmup then linear decay, applied to every group's own base LR."""
    total_steps = max(1, total_steps)
    warmup_steps = max(1, int(total_steps * train_config.warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        remaining = total_steps - current_step
        return max(0.0, float(remaining) / float(max(1, total_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def optimizer_coverage_report(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict:
    """Return a report proving the optimizer covers exactly the trainable set."""
    in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    frozen = {id(p) for p in model.parameters() if not p.requires_grad}
    return {
        "num_trainable": len(trainable),
        "num_in_optimizer": len(in_optimizer),
        "missing_trainable": len(trainable - in_optimizer),
        "frozen_in_optimizer": len(frozen & in_optimizer),
        "covered": trainable == in_optimizer and not (frozen & in_optimizer),
    }


def _scope_learning_rates(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    """Return current ``(backbone, head)`` LRs without relying on group order."""

    named = {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    backbone = next(
        (lr for name, lr in named.items() if name.startswith("backbone")), 0.0
    )
    head = next((lr for name, lr in named.items() if name.startswith("head")), 0.0)
    return backbone, head


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move every tensor in the batch to *device*."""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_kwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract the model input arguments from a collated batch."""
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    if "token_type_ids" in batch:
        kwargs["token_type_ids"] = batch["token_type_ids"]
    if "special_tokens_mask" in batch:
        kwargs["special_tokens_mask"] = batch["special_tokens_mask"]
    return kwargs


def _amp_settings(
    device: torch.device, enabled: bool
) -> tuple[torch.dtype, bool, bool]:
    """Return (dtype, autocast_enabled, needs_grad_scaler) for this device."""
    if not enabled or device.type != "cuda":
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False  # BF16 needs no loss scaling
    return torch.float16, True, True


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    distributed_context: DistributedContext | None = None,
    use_amp: bool = False,
) -> dict[str, object]:
    """Return exact global metrics on a held-out loader.

    In DDP, validation loaders use unequal, non-padding shards. Forward passes
    therefore bypass the DDP wrapper (whose collectives expect equal step
    counts), then the small logits/label arrays are gathered across ranks.
    """
    context = distributed_context or current_context()
    was_training = model.training
    model.eval()
    evaluation_model = unwrap_model(model)
    amp_dtype, autocast_enabled, _ = _amp_settings(device, use_amp)
    total_loss = 0.0
    total_examples = 0
    logits_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled
        ):
            logits = evaluation_model(**forward_kwargs(batch))["logits"]
        loss = criterion(logits.float(), batch["labels"])
        count = int(batch["labels"].size(0))
        total_loss += float(loss.item()) * count
        total_examples += count
        logits_chunks.append(logits.float().cpu().numpy())
        label_chunks.append(batch["labels"].cpu().numpy())
    if was_training:
        model.train()
    local_payload = {
        "loss_sum": total_loss,
        "count": total_examples,
        "logits": np.concatenate(logits_chunks, axis=0) if logits_chunks else None,
        "labels": np.concatenate(label_chunks, axis=0) if label_chunks else None,
    }
    gathered = all_gather_objects(local_payload, context)
    non_empty = [payload for payload in gathered if payload["count"]]
    if not non_empty:
        raise ValueError("Cannot evaluate an empty dataloader.")
    logits_array = np.concatenate([payload["logits"] for payload in non_empty], axis=0)
    labels_array = np.concatenate([payload["labels"] for payload in non_empty], axis=0)
    metrics = compute_classification_metrics(logits_array, labels_array)
    global_loss = sum(float(payload["loss_sum"]) for payload in gathered)
    global_count = sum(int(payload["count"]) for payload in gathered)
    metrics["loss"] = global_loss / max(1, global_count)
    return metrics


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def _to_cpu(value):
    """Recursively move tensors to CPU so checkpoints are device-independent.

    ``clone()`` forces a fully materialised, contiguous CPU copy. On Apple MPS a
    lazily evaluated tensor can otherwise change size between the moment
    ``torch.save`` writes its header and the moment it writes the payload, which
    surfaces as ``RuntimeError: unexpected pos N vs N-k``.
    """
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True).contiguous()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_to_cpu(item) for item in value]
        return type(value)(converted) if isinstance(value, tuple) else converted
    return value


def _atomic_save(payload: dict, path: Path, *, attempts: int = 3) -> None:
    """Write a checkpoint atomically, on CPU, verifying it is readable.

    The payload is written to a temporary file and only moved into place once
    ``torch.load`` can reopen it, so an interrupted or corrupt write can never
    replace a good checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    portable = _to_cpu(payload)

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            torch.save(portable, temporary)
            torch.load(temporary, map_location="cpu", weights_only=False)
            temporary.replace(path)
            return
        except Exception as error:  # noqa: BLE001 - retried, then re-raised
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < attempts:
                print(
                    f"[train] checkpoint write to {path.name} failed "
                    f"({type(error).__name__}); retry {attempt}/{attempts - 1}",
                    flush=True,
                )
    raise RuntimeError(
        f"Could not write checkpoint {path}: {last_error}"
    ) from last_error


def build_best_payload(
    model: nn.Module,
    train_config: TrainConfig,
    *,
    epoch: int,
    metrics: dict[str, float],
    fused_optimizer: bool | None = None,
) -> dict:
    """Return the slim publication checkpoint (no optimizer state)."""
    model = unwrap_model(model)
    architecture = train_config.architecture or (
        model.architecture_config() if hasattr(model, "architecture_config") else {}
    )
    return {
        "model_state_dict": model.state_dict(),
        "architecture": architecture,
        "data_signature": train_config.data_signature,
        "protocol": {
            "selection_metric": config.SELECTION_METRIC,
            "epochs": train_config.epochs,
            "backbone_learning_rate": train_config.backbone_learning_rate,
            "head_learning_rate": train_config.head_learning_rate,
            "weight_decay": train_config.weight_decay,
            "warmup_ratio": train_config.warmup_ratio,
            "grad_accum_steps": train_config.grad_accum_steps,
            "freeze_backbone": train_config.freeze_backbone,
            "global_batch_size": train_config.global_batch_size,
            "per_device_batch_size": train_config.per_device_batch_size,
            "global_eval_batch_size": train_config.global_eval_batch_size,
            "per_device_eval_batch_size": train_config.per_device_eval_batch_size,
            "effective_global_batch_size": (
                train_config.global_batch_size * train_config.grad_accum_steps
            ),
            "world_size": train_config.world_size,
            "distributed_backend": train_config.distributed_backend,
            "device_names": train_config.device_names,
            "num_workers_per_rank": train_config.num_workers,
            "amp_requested": train_config.use_amp,
            "amp_dtype": train_config.amp_dtype,
            "deterministic": train_config.deterministic,
            "pad_to_multiple_of": train_config.pad_to_multiple_of,
            "fused_optimizer": (
                train_config.use_fused_optimizer
                if fused_optimizer is None
                else fused_optimizer
            ),
        },
        "best_metrics": metrics,
        "epoch": epoch,
        "seed": train_config.seed,
        "run_id": train_config.run_id,
        "torch_version": torch.__version__,
    }


def build_last_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    train_config: TrainConfig,
    *,
    epoch: int,
    global_step: int,
    best_metrics: dict[str, float],
    best_epoch: int,
    patience_counter: int,
    history: list[dict[str, float]],
    loader_generator_state: torch.Tensor | None,
    total_train_seconds: float,
    distributed_runtime_states: list[dict] | None = None,
) -> dict:
    """Return the fully resumable checkpoint."""
    model = unwrap_model(model)
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "history": history,
        "rng_state": capture_rng_state(),
        "loader_generator_state": loader_generator_state,
        "total_train_seconds": total_train_seconds,
        "distributed_runtime_states": distributed_runtime_states,
        "train_config": {
            **asdict(train_config),
            "output_dir": str(train_config.output_dir),
        },
        "data_signature": train_config.data_signature,
        "seed": train_config.seed,
    }


def _selection_key(metrics: dict[str, float]) -> tuple[float, float]:
    """Higher is better: macro F1 first, then negative validation loss."""
    return (float(metrics["f1_macro"]), -float(metrics["loss"]))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_config: TrainConfig,
    *,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
    resume: bool = False,
    reporter: TrainingReporter | None = None,
    show_progress: bool = False,
    distributed_context: DistributedContext | None = None,
) -> TrainResult:
    """Train *model*, selecting the checkpoint by validation macro F1.

    Set ``show_progress=True`` for the YOLO-style live display, or pass a
    pre-built ``reporter``. Reporting never affects optimisation.
    """
    for loader, expected in ((train_loader, "train"), (val_loader, "validation")):
        split = getattr(loader, "split_name", None)
        if split is not None and split != expected:
            raise ValueError(
                f"Expected the {expected} split but received {split!r}. "
                "The official test split must never enter training."
            )

    context = distributed_context or current_context()
    if train_config.world_size != context.world_size:
        raise ValueError(
            f"TrainConfig world_size={train_config.world_size} but runtime world_size="
            f"{context.world_size}."
        )
    if context.enabled:
        if getattr(train_loader, "world_size", None) != context.world_size:
            raise ValueError("The training loader lacks matching distributed metadata.")
        if getattr(val_loader, "world_size", None) != context.world_size:
            raise ValueError(
                "The validation loader lacks matching distributed metadata."
            )
        if not isinstance(train_loader.sampler, DistributedSampler):
            raise ValueError("DDP training requires a torch DistributedSampler.")
        if not isinstance(val_loader.sampler, DistributedEvalSampler):
            raise ValueError(
                "DDP validation requires the exact non-padding DistributedEvalSampler."
            )
        if (
            getattr(train_loader, "global_batch_size", None)
            != train_config.global_batch_size
            or getattr(train_loader, "per_device_batch_size", None)
            != train_config.per_device_batch_size
        ):
            raise ValueError(
                "The distributed loader batch contract differs from TrainConfig."
            )
        if (
            getattr(val_loader, "global_batch_size", None)
            != train_config.global_eval_batch_size
            or getattr(val_loader, "per_device_batch_size", None)
            != train_config.per_device_eval_batch_size
        ):
            raise ValueError(
                "The distributed validation batch contract differs from TrainConfig."
            )

    # The model has already been constructed from the common seed. Offset the
    # runtime RNG so dropout masks differ across ranks while remaining
    # reproducible for a given rank/world-size configuration.
    set_seed(train_config.seed + context.rank, deterministic=train_config.deterministic)
    output_dir = Path(train_config.output_dir)
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(context)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    log_path = output_dir / "train_log.jsonl"
    official_test_paths = (
        output_dir / "test_metrics.json",
        output_dir / "test_predictions.npz",
    )

    device = context.device
    base_model = unwrap_model(model)
    base_model.to(device)

    # Freeze BEFORE the optimizer is built, so frozen tensors never enter it.
    if train_config.freeze_backbone and hasattr(base_model, "freeze_backbone"):
        base_model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(base_model, train_config)

    if train_config.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1.")
    accum = train_config.grad_accum_steps
    num_batches = len(train_loader)
    steps_per_epoch = max(1, math.ceil(num_batches / accum))
    total_steps = steps_per_epoch * train_config.epochs
    scheduler = build_scheduler(optimizer, train_config, total_steps)

    amp_dtype, autocast_enabled, needs_scaler = _amp_settings(
        device, train_config.use_amp
    )
    scaler = torch.amp.GradScaler(device.type, enabled=needs_scaler)

    if not context.is_main:
        reporter = None
    if reporter is None and show_progress and context.is_main:
        reporter = TrainingReporter(
            train_config.run_id,
            total_epochs=train_config.epochs,
            total_batches=len(train_loader),
        )

    history: list[dict[str, float]] = []
    best_metrics: dict[str, float] = {
        "f1_macro": -1.0,
        "accuracy": -1.0,
        "loss": float("inf"),
    }
    best_epoch = -1
    patience_counter = 0
    global_step = 0
    start_epoch = 0
    prior_train_seconds = 0.0
    generator = getattr(train_loader, "generator", None)

    resume_available = bool(last_path.exists()) if context.is_main else False
    resume_available = bool(broadcast_object(resume_available, context))
    will_resume = resume and resume_available
    official_test_exists = (
        any(path.exists() for path in official_test_paths) if context.is_main else False
    )
    official_test_exists = bool(broadcast_object(official_test_exists, context))
    if official_test_exists:
        raise RuntimeError(
            "Refusing to train a run that already has official-test artifacts. "
            "Use a new output directory/run id so test results cannot be mixed with "
            "a changed checkpoint."
        )
    if context.is_main and not will_resume:
        # Preserve a prior attempt while preventing stale checkpoints/log rows
        # from being mistaken for the fresh run if it fails before epoch one.
        stale_paths = [
            path
            for path in (
                best_path,
                last_path,
                log_path,
                output_dir / "val_metrics.json",
                output_dir / "run_summary.json",
            )
            if path.exists()
        ]
        if stale_paths:
            archive_dir = (
                output_dir
                / "previous_runs"
                / (
                    f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
                )
            )
            archive_dir.mkdir(parents=True, exist_ok=False)
            for stale_path in stale_paths:
                stale_path.replace(archive_dir / stale_path.name)
    barrier(context)
    if will_resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        signature = checkpoint.get("data_signature", {})
        if (
            signature
            and train_config.data_signature
            and signature != train_config.data_signature
        ):
            raise RuntimeError(
                "Refusing to resume: the checkpoint's data signature differs from the "
                "current data."
            )
        saved_runtime_states = checkpoint.get("distributed_runtime_states")
        if (
            saved_runtime_states is not None
            and len(saved_runtime_states) != context.world_size
        ):
            raise RuntimeError(
                "Refusing exact resume: checkpoint world size differs from the current "
                f"runtime ({len(saved_runtime_states)} != {context.world_size})."
            )
        saved_config = checkpoint.get("train_config", {})
        saved_world_size = int(saved_config.get("world_size", 1))
        if saved_runtime_states is None and saved_world_size != context.world_size:
            raise RuntimeError(
                "Refusing exact resume: the legacy checkpoint was created with a "
                "different world size."
            )
        exact_resume_fields = (
            "seed",
            "epochs",
            "backbone_learning_rate",
            "head_learning_rate",
            "weight_decay",
            "warmup_ratio",
            "max_grad_norm",
            "world_size",
            "global_batch_size",
            "per_device_batch_size",
            "global_eval_batch_size",
            "per_device_eval_batch_size",
            "grad_accum_steps",
            "label_smoothing",
            "freeze_backbone",
            "patience",
            "use_amp",
            "deterministic",
            "use_fused_optimizer",
            "num_workers",
            "pad_to_multiple_of",
            "amp_dtype",
            "distributed_backend",
            "device_names",
            "architecture",
        )
        for field_name in exact_resume_fields:
            if field_name in saved_config and saved_config[field_name] != getattr(
                train_config, field_name
            ):
                raise RuntimeError(
                    "Refusing exact resume: training protocol field "
                    f"{field_name!r} changed from {saved_config[field_name]!r} to "
                    f"{getattr(train_config, field_name)!r}."
                )
        base_model.load_state_dict(checkpoint["model_state_dict"])
        base_model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_metrics = dict(checkpoint["best_metrics"])
        best_epoch = int(checkpoint["best_epoch"])
        patience_counter = int(checkpoint["patience_counter"])
        history = list(checkpoint["history"])
        prior_train_seconds = float(
            checkpoint.get(
                "total_train_seconds",
                sum(float(record.get("epoch_time_sec", 0.0)) for record in history),
            )
        )
        runtime_state = (
            saved_runtime_states[context.rank]
            if saved_runtime_states is not None
            else {
                "rng_state": checkpoint["rng_state"],
                "loader_generator_state": checkpoint.get("loader_generator_state"),
            }
        )
        restore_rng_state(runtime_state["rng_state"])
        if (
            generator is not None
            and runtime_state.get("loader_generator_state") is not None
        ):
            generator.set_state(runtime_state["loader_generator_state"])
        if context.is_main:
            print(
                f"[train] resumed {train_config.run_id} from epoch {start_epoch}",
                flush=True,
            )

    training_model = wrap_ddp(base_model, context)

    resumed_from = start_epoch
    resume_reached_early_stop = bool(
        will_resume
        and train_config.patience > 0
        and patience_counter >= train_config.patience
    )
    if resume_reached_early_stop and context.is_main:
        print(
            f"[train] {train_config.run_id} had already met early stopping at "
            f"epoch {start_epoch}; no additional epoch will run",
            flush=True,
        )
    run_start = time.perf_counter()
    peak_vram_gb = 0.0
    last_confusion: list[list[int]] | None = None
    if reporter is not None:
        reporter.start(
            parameter_counts=count_model_parameters(base_model),
            extra=(
                f"{'frozen' if train_config.freeze_backbone else 'fine-tuned'} backbone, "
                f"seed {train_config.seed}, {device.type}, world {context.world_size}"
            ),
        )
    guard.begin_training()
    try:
        epoch_range = (
            range(0)
            if resume_reached_early_stop
            else range(start_epoch + 1, train_config.epochs + 1)
        )
        for epoch in epoch_range:
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.perf_counter()

            training_model.train()
            if train_config.freeze_backbone and hasattr(base_model, "backbone"):
                base_model.backbone.eval()  # keep frozen-backbone dropout disabled

            optimizer.zero_grad(set_to_none=True)
            running_loss = torch.zeros((), dtype=torch.float32, device=device)
            running_examples = 0
            skipped_steps = 0

            batch_iterable = (
                reporter.epoch_iter(train_loader, epoch)
                if reporter is not None
                else train_loader
            )
            for index, batch in enumerate(batch_iterable):
                batch = move_batch_to_device(batch, device)
                group_start = (index // accum) * accum
                group_size = min(accum, num_batches - group_start)
                is_boundary = ((index + 1) % accum == 0) or ((index + 1) == num_batches)

                sync_context = (
                    training_model.no_sync()
                    if context.enabled and not is_boundary
                    else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=autocast_enabled,
                    ):
                        logits = training_model(**forward_kwargs(batch))["logits"]
                        loss = criterion(logits, batch["labels"])
                    scaler.scale(loss / group_size).backward()

                count = int(batch["labels"].size(0))
                detached_loss = loss.detach().float()
                running_loss.add_(detached_loss * count)
                running_examples += count

                if reporter is not None:
                    backbone_lr, head_lr = _scope_learning_rates(optimizer)
                    should_sync_loss = (
                        index == 0
                        or (index + 1) % PROGRESS_SYNC_INTERVAL == 0
                        or (index + 1) == num_batches
                    )
                    reporter.batch(
                        float(detached_loss.item()) if should_sync_loss else None,
                        backbone_lr,
                        head_lr,
                    )

                if not is_boundary:
                    continue

                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    max_norm=train_config.max_grad_norm,
                )
                if not torch.isfinite(total_norm):
                    if context.is_main:
                        print(
                            f"[train] non-finite gradient at epoch {epoch} step {index + 1}; "
                            "skipping optimizer and scheduler step",
                            flush=True,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    skipped_steps += 1
                    continue

                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)

            validation = evaluate_model(
                training_model,
                val_loader,
                eval_criterion,
                device,
                distributed_context=context,
                # Keep validation/model-selection numerically aligned with the
                # original protocol and with final evaluation (both FP32).
                use_amp=False,
            )
            last_confusion = validation.get("confusion_matrix")  # type: ignore[assignment]
            if device.type == "cuda":
                torch.cuda.synchronize()
                peak_vram_gb = max(
                    peak_vram_gb, torch.cuda.max_memory_allocated(device) / 2**30
                )

            global_loss_sum, global_examples = reduce_sums(
                [float(running_loss.item()), float(running_examples)], context
            )
            (
                epoch_seconds,
                epoch_peak_vram,
                global_skipped_steps,
                session_elapsed,
            ) = reduce_maxima(
                [
                    time.perf_counter() - epoch_start,
                    peak_vram_gb,
                    float(skipped_steps),
                    time.perf_counter() - run_start,
                ],
                context,
            )
            backbone_lr, head_lr = _scope_learning_rates(optimizer)
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": global_loss_sum / max(1.0, global_examples),
                "val_loss": float(validation["loss"]),
                "val_accuracy": float(validation["accuracy"]),
                "val_f1_macro": float(validation["f1_macro"]),
                "learning_rate_backbone": backbone_lr,
                "learning_rate_head": head_lr,
                "epoch_time_sec": round(epoch_seconds, 2),
                "peak_vram_gb": round(epoch_peak_vram, 4),
                "skipped_steps": int(global_skipped_steps),
            }
            history.append(record)
            if context.is_main:
                append_jsonl(record, log_path)

            candidate = {
                "f1_macro": record["val_f1_macro"],
                "accuracy": record["val_accuracy"],
                "loss": record["val_loss"],
            }
            improved = _selection_key(candidate) > _selection_key(best_metrics)
            if improved:
                best_metrics = candidate
                best_epoch = epoch
                patience_counter = 0
                if context.is_main:
                    _atomic_save(
                        build_best_payload(
                            base_model,
                            train_config,
                            epoch=epoch,
                            metrics=best_metrics,
                            fused_optimizer=bool(
                                optimizer.defaults.get("fused", False)
                            ),
                        ),
                        best_path,
                    )
            else:
                patience_counter += 1

            local_runtime_state = {
                "rank": context.rank,
                "rng_state": capture_rng_state(),
                "loader_generator_state": (
                    generator.get_state() if generator is not None else None
                ),
            }
            runtime_states = all_gather_objects(local_runtime_state, context)
            if context.is_main:
                _atomic_save(
                    build_last_payload(
                        base_model,
                        optimizer,
                        scheduler,
                        scaler,
                        train_config,
                        epoch=epoch,
                        global_step=global_step,
                        best_metrics=best_metrics,
                        best_epoch=best_epoch,
                        patience_counter=patience_counter,
                        history=history,
                        loader_generator_state=(
                            generator.get_state() if generator is not None else None
                        ),
                        total_train_seconds=prior_train_seconds + session_elapsed,
                        distributed_runtime_states=runtime_states,
                    ),
                    last_path,
                )

                save_json(
                    {
                        "run_id": train_config.run_id,
                        "best_epoch": best_epoch,
                        "best_metrics": best_metrics,
                        "history": history,
                    },
                    output_dir / "val_metrics.json",
                )
            barrier(context)
            if reporter is not None:
                reporter.epoch_end(validation, is_best=improved)
            elif context.is_main:
                print(
                    f"[{train_config.run_id}] epoch {epoch}/{train_config.epochs} "
                    f"train_loss={record['train_loss']:.4f} "
                    f"val_loss={record['val_loss']:.4f} "
                    f"val_acc={record['val_accuracy']:.4f} "
                    f"val_f1={record['val_f1_macro']:.4f}"
                    f"{'  * best' if improved else ''}",
                    flush=True,
                )
            if progress_callback is not None and context.is_main:
                progress_callback(epoch, record)
            if train_config.patience and patience_counter >= train_config.patience:
                if context.is_main:
                    print(f"[train] early stopping after epoch {epoch}", flush=True)
                break
    finally:
        guard.end_training()
        if reporter is not None:
            reporter.bars.close()

    if reporter is not None:
        reporter.finish(
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            confusion_matrix=last_confusion,
        )

    session_train_seconds, peak_vram_gb = reduce_maxima(
        [time.perf_counter() - run_start, peak_vram_gb], context
    )
    total_train_seconds = prior_train_seconds + session_train_seconds
    return TrainResult(
        best_val_f1_macro=float(best_metrics["f1_macro"]),
        best_val_accuracy=float(best_metrics["accuracy"]),
        best_val_loss=float(best_metrics["loss"]),
        best_epoch=best_epoch,
        history=history,
        resumed_from_epoch=resumed_from,
        total_train_seconds=round(total_train_seconds, 2),
        session_train_seconds=round(session_train_seconds, 2),
        peak_vram_gb=round(peak_vram_gb, 4),
    )


def gradient_audit(model: nn.Module) -> dict[str, object]:
    """Report trainable parameters that received no gradient or a bad gradient.

    Call after ``loss.backward()``.
    """
    missing: list[str] = []
    non_finite: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            non_finite.append(name)
    return {
        "missing_grad": missing,
        "non_finite_grad": non_finite,
        "ok": not missing and not non_finite,
    }


def count_model_parameters(model: nn.Module) -> dict[str, object]:
    """Return a module-level parameter breakdown for any project model."""
    if hasattr(model, "count_parameters"):
        return model.count_parameters()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": {"total": total, "trainable": trainable, "frozen": total - trainable}
    }
