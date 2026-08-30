"""Compare the legacy per-example tokenizer with the batched loader path.

This is an input-only benchmark: it does not construct a model or access the
official test split. Run it on Kaggle with the same batch/workers planned for
training to tune CPU feeding independently from GPU compute.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import (
    AgNewsDataset,
    AgNewsTextDataset,
    BatchedTokenizingCollator,
    DynamicPaddingCollator,
    load_split,
)
from src.models import LdtfBert
from src.utils import seed_worker


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AG News input tokenization."
    )
    parser.add_argument("--data", type=Path, default=config.PROCESSED_VAL)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args(argv)


def _loader(dataset, collator, *, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        worker_init_fn=seed_worker if num_workers else None,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers else None,
    )


def _measure(loader: DataLoader, repeats: int) -> tuple[float, int, int]:
    best_seconds = float("inf")
    total_examples = 0
    total_tokens = 0
    for _ in range(repeats):
        examples = 0
        tokens = 0
        started = time.perf_counter()
        for batch in loader:
            examples += int(batch["labels"].size(0))
            tokens += int(batch["attention_mask"].sum().item())
        best_seconds = min(best_seconds, time.perf_counter() - started)
        total_examples, total_tokens = examples, tokens
    return best_seconds, total_examples, total_tokens


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.repeats < 1:
        raise ValueError(
            "batch size/repeats must be positive and workers non-negative."
        )
    columns = [config.TEXT_COLUMN, config.LABEL_COLUMN]
    frame = load_split(args.data, columns=columns)
    tokenizer = LdtfBert.build_tokenizer(config.MODEL_NAME)

    legacy = _loader(
        AgNewsDataset(frame, tokenizer),
        DynamicPaddingCollator(tokenizer.pad_token_id),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    batched = _loader(
        AgNewsTextDataset(frame),
        BatchedTokenizingCollator(tokenizer),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    legacy_first = next(iter(legacy))
    batched_first = next(iter(batched))
    if set(legacy_first) != set(batched_first) or any(
        not torch.equal(legacy_first[key], batched_first[key]) for key in legacy_first
    ):
        raise RuntimeError("Batched tokenizer output differs from the legacy pipeline.")

    legacy_seconds, examples, tokens = _measure(legacy, args.repeats)
    batched_seconds, batched_examples, batched_tokens = _measure(batched, args.repeats)
    if (examples, tokens) != (batched_examples, batched_tokens):
        raise RuntimeError("The compared input paths did not cover identical data.")

    print(f"rows={examples:,} non_pad_tokens={tokens:,} batch={args.batch_size}")
    print(
        f"legacy_per_example {legacy_seconds:8.3f}s  {examples / legacy_seconds:9.1f} rows/s"
    )
    print(
        f"batched_tokenizer  {batched_seconds:8.3f}s  {examples / batched_seconds:9.1f} rows/s"
    )
    print(f"speedup {legacy_seconds / batched_seconds:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
