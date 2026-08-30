"""CPU/Gloo coverage for the DDP runtime and exact distributed validation."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.dataset import DistributedEvalSampler
from src.distributed import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
)
from src.train import TrainConfig, evaluate_model, train_model
from src.utils import set_seed
from tests.test_train import TinyDataset, TinyModel


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _ddp_worker(rank: int, world_size: int, port: int, output_dir: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    context = initialize_distributed()
    try:
        set_seed(123)
        model = TinyModel()
        train_dataset = TinyDataset(size=9)
        val_dataset = TinyDataset(size=5)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=123,
            drop_last=False,
        )
        val_sampler = DistributedEvalSampler(
            val_dataset, rank=rank, world_size=world_size
        )
        generator = torch.Generator().manual_seed(123 + rank)
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            sampler=train_sampler,
            generator=generator,
        )
        val_loader = DataLoader(val_dataset, batch_size=2, sampler=val_sampler)
        train_loader.split_name = "train"
        val_loader.split_name = "validation"
        train_loader.world_size = world_size
        val_loader.world_size = world_size
        train_loader.global_batch_size = 4
        train_loader.per_device_batch_size = 2
        val_loader.global_batch_size = 4
        val_loader.per_device_batch_size = 2

        result = train_model(
            model,
            train_loader,
            val_loader,
            TrainConfig(
                output_dir=Path(output_dir),
                run_id="ddp_smoke",
                epochs=1,
                use_amp=False,
                use_fused_optimizer=False,
                global_batch_size=4,
                per_device_batch_size=2,
                global_eval_batch_size=4,
                per_device_eval_batch_size=2,
                world_size=world_size,
                amp_dtype="disabled",
                distributed_backend=context.backend,
                device_names=("cpu", "cpu"),
            ),
            distributed_context=context,
        )
        Path(output_dir, f"rank_{rank}.json").write_text(
            json.dumps(
                {
                    "best_epoch": result.best_epoch,
                    "f1": result.best_val_f1_macro,
                    "loss": result.best_val_loss,
                }
            ),
            encoding="utf-8",
        )
    finally:
        cleanup_distributed(context)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_training_writes_one_loadable_checkpoint_and_agrees(tmp_path):
    torch.multiprocessing.spawn(
        _ddp_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank_zero = json.loads((tmp_path / "rank_0.json").read_text(encoding="utf-8"))
    rank_one = json.loads((tmp_path / "rank_1.json").read_text(encoding="utf-8"))
    assert rank_zero == rank_one

    checkpoint = torch.load(
        tmp_path / "best.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["protocol"]["world_size"] == 2
    assert checkpoint["protocol"]["distributed_backend"] == "gloo"
    assert not any(key.startswith("module.") for key in checkpoint["model_state_dict"])
    assert (
        len((tmp_path / "train_log.jsonl").read_text(encoding="utf-8").splitlines())
        == 1
    )

    restored = TinyModel()
    restored.load_state_dict(checkpoint["model_state_dict"])
    full_validation = DataLoader(TinyDataset(size=5), batch_size=2)
    metrics = evaluate_model(
        restored,
        full_validation,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        use_amp=False,
    )
    assert rank_zero["f1"] == pytest.approx(metrics["f1_macro"])
    assert rank_zero["loss"] == pytest.approx(metrics["loss"])


def test_ddp_rejects_untagged_ordinary_loaders(tmp_path):
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="distributed metadata"):
        train_model(
            TinyModel(),
            DataLoader(TinyDataset(size=4), batch_size=2),
            DataLoader(TinyDataset(size=4), batch_size=2),
            TrainConfig(
                output_dir=tmp_path,
                epochs=1,
                use_amp=False,
                world_size=2,
                global_batch_size=4,
                per_device_batch_size=2,
            ),
            distributed_context=context,
        )
