"""Small, explicit helpers for single-process and DDP execution.

The training entrypoints initialise this module from the environment created by
``torchrun``.  With no distributed environment variables the same code remains
a normal single-device program.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .utils import get_device


@dataclass(frozen=True)
class DistributedContext:
    """Runtime identity for one process."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def backend(self) -> str:
        if self.enabled and dist.is_available() and dist.is_initialized():
            return str(dist.get_backend())
        return "none"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {value!r}.") from error


def initialize_distributed() -> DistributedContext:
    """Initialise a process group when launched by ``torchrun``.

    CUDA runs use NCCL and bind each process to exactly one GPU. CPU-only
    distributed tests use Gloo. Calling this function without ``torchrun`` is a
    no-op and returns the project's usual CUDA/MPS/CPU device.
    """

    world_size = _env_int("WORLD_SIZE", 1)
    if world_size <= 1:
        return DistributedContext(0, 0, 1, get_device(), False)
    if not dist.is_available():
        raise RuntimeError("WORLD_SIZE > 1 but torch.distributed is unavailable.")

    rank = _env_int("RANK", -1)
    local_rank = _env_int("LOCAL_RANK", -1)
    if rank < 0 or local_rank < 0:
        raise RuntimeError(
            "WORLD_SIZE > 1 requires RANK and LOCAL_RANK. Launch with torchrun."
        )

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA "
                "device(s) are visible."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    if dist.get_world_size() != world_size or dist.get_rank() != rank:
        raise RuntimeError(
            "The initialized process group disagrees with torchrun's environment."
        )
    return DistributedContext(rank, local_rank, world_size, device, initialized_here)


def current_context() -> DistributedContext:
    """Describe the current process without creating a process group."""

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = _env_int("LOCAL_RANK", rank)
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return DistributedContext(rank, local_rank, world_size, device, False)
    return DistributedContext(0, 0, 1, get_device(), False)


def cleanup_distributed(context: DistributedContext) -> None:
    """Destroy a process group owned by *context*. Safe in ``finally`` blocks."""

    if context.initialized_here and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext | None = None) -> None:
    """Synchronise ranks when distributed execution is active."""

    context = context or current_context()
    if context.enabled and dist.is_initialized():
        dist.barrier()


def all_gather_objects(
    value: Any, context: DistributedContext | None = None
) -> list[Any]:
    """Gather a small picklable value from every rank, in rank order."""

    context = context or current_context()
    if not context.enabled:
        return [value]
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, value)
    return gathered


def broadcast_object(
    value: Any, context: DistributedContext | None = None, *, source: int = 0
) -> Any:
    """Broadcast one small picklable value from *source* to every rank."""

    context = context or current_context()
    if not context.enabled:
        return value
    payload = [value if context.rank == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def reduce_sums(
    values: list[float], context: DistributedContext | None = None
) -> list[float]:
    """All-reduce numeric sums and return the result on every rank."""

    context = context or current_context()
    if not context.enabled:
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def reduce_maxima(
    values: list[float], context: DistributedContext | None = None
) -> list[float]:
    """All-reduce numeric maxima and return the result on every rank."""

    context = context or current_context()
    if not context.enabled:
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.cpu().tolist()


def wrap_ddp(model: nn.Module, context: DistributedContext) -> nn.Module:
    """Wrap *model* in efficient one-process-per-GPU DDP when needed."""

    if not context.enabled:
        return model
    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "static_graph": True,
    }
    if context.device.type == "cuda":
        kwargs.update(device_ids=[context.local_rank], output_device=context.local_rank)
    return DistributedDataParallel(model, **kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model without a DDP prefix in its state dict."""

    return model.module if isinstance(model, DistributedDataParallel) else model
