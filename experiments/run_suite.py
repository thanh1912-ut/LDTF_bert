"""Train several runs in sequence with a live leaderboard.

Usage::

    python -m experiments.run_suite --preset ablations
    python -m experiments.run_suite --run A0 --run A1 --run A3 --seed 42
    python -m experiments.run_suite --preset all --seed 42 --seed 1337 --seed 2024

Each run trains through the same engine as ``run_experiment``. After every run
the leaderboard is redrawn, sorted by validation macro F1. A run whose
``run_summary.json`` already exists is skipped unless ``--force`` is given, so an
interrupted Colab session can simply be restarted.

Validation metrics only. The official test split is never touched here.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback

import torch

from src import config
from src.dataset import build_dataloaders, data_signature, per_device_batch_size
from src.distributed import (
    all_gather_objects,
    barrier,
    broadcast_object,
    cleanup_distributed,
    initialize_distributed,
)
from src.models import LdtfBert
from src.progress import SuiteBoard, TrainingReporter
from src.registry import ABLATION_RUNS, build_model, is_frozen_run
from src.train import TrainConfig, train_model
from src.utils import load_json, save_json, set_seed

PRESETS = {
    "baselines": [
        "B1_bert_frozen_cls",
        "B2_bert_finetuned_cls",
        "B3_bert_frozen_mean_pool",
        "B4_bert_scalar_mix",
        "B5_token_attention_only",
        "B6_full_ldtf_frozen",
        "B7_full_ldtf_finetuned",
    ],
    "ablations": list(ABLATION_RUNS),
    # The comparisons that actually decide the scientific question.
    "core": ["A0", "A1", "A3", "A4", "A11"],
    "all": [
        "B1_bert_frozen_cls",
        "B2_bert_finetuned_cls",
        "B3_bert_frozen_mean_pool",
        "B4_bert_scalar_mix",
        "B5_token_attention_only",
        "B6_full_ldtf_frozen",
        "B7_full_ldtf_finetuned",
        *ABLATION_RUNS,
    ],
}


def cached_summary_matches(
    existing: dict,
    *,
    base_run: str,
    seed: int,
    frozen: bool,
    epochs: int,
    is_debug_subset: bool,
    expected_runtime: dict,
    expected_signature: dict,
) -> bool:
    """Return whether a cached result belongs to this exact suite protocol."""

    existing_runtime = existing.get("runtime", {})
    existing_signature = existing.get("data_signature", {})
    return bool(
        existing.get("base_run") == base_run
        and existing.get("seed") == seed
        and existing.get("backbone_frozen") == frozen
        and existing.get("epochs") == epochs
        and existing.get("is_debug_subset") == is_debug_subset
        and all(
            existing_runtime.get(key) == value
            for key, value in expected_runtime.items()
        )
        and all(
            existing_signature.get(key) == value
            for key, value in expected_signature.items()
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a suite of runs with a leaderboard."
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument(
        "--run", action="append", default=[], help="Explicit run id. Repeatable."
    )
    parser.add_argument(
        "--seed", action="append", type=int, default=[], help="Repeatable."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Global micro-batch across all ranks.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=config.GRAD_ACCUM_STEPS)
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Workers per rank."
    )
    parser.add_argument("--limit-train-rows", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fast-nondeterministic", action="store_true")
    parser.add_argument("--no-fused-optimizer", action="store_true")
    parser.add_argument(
        "--pad-to-multiple-of", type=int, choices=(8, 16, 32), default=None
    )
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false", default=True
    )
    parser.add_argument(
        "--force", action="store_true", help="Retrain runs that already have a summary."
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="Ignore an incomplete last.pt instead of automatically resuming it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed run and carry on instead of aborting the suite.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = initialize_distributed()
    try:
        if args.continue_on_error and context.enabled:
            raise ValueError(
                "--continue-on-error is unsafe with DDP because a failed collective "
                "cannot be reused. Remove the flag when launching with torchrun."
            )
        if context.is_main:
            config.ensure_output_dirs()
        barrier(context)

        workers = args.num_workers
        if workers is None:
            workers = 1 if context.enabled else config.NUM_WORKERS
        if workers < 0:
            raise ValueError("--num-workers must be non-negative.")
        if args.grad_accum_steps < 1:
            raise ValueError("--grad-accum-steps must be at least 1.")
        if workers:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        run_ids = list(args.run) or PRESETS[args.preset or "core"]
        seeds = args.seed or [config.SEED]
        jobs = [(run_id, seed) for seed in seeds for run_id in run_ids]
        debug_suffix = (
            f"_debug{args.limit_train_rows}"
            if args.limit_train_rows is not None
            else ""
        )
        labels = [f"{run_id}{debug_suffix}_seed{seed}" for run_id, seed in jobs]

        board = SuiteBoard(labels, enabled=args.progress and context.is_main)
        if context.is_main:
            print(f"[suite] {len(jobs)} run(s): {', '.join(labels)}\n", flush=True)

        tokenizer = LdtfBert.build_tokenizer(config.MODEL_NAME)
        base_signature = data_signature()
        local_device_name = (
            str(context.device)
            if context.device.type != "cuda"
            else str(torch.cuda.get_device_name(context.device))
        )
        device_names = tuple(all_gather_objects(local_device_name, context))
        amp_dtype = (
            "disabled"
            if args.no_amp or context.device.type != "cuda"
            else ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
        )
        local_batch_size = per_device_batch_size(args.batch_size, context.world_size)
        local_eval_batch_size = per_device_batch_size(
            args.eval_batch_size, context.world_size
        )
        failures: list[str] = []

        for run_id, seed in jobs:
            label = f"{run_id}{debug_suffix}_seed{seed}"
            output_dir = config.experiment_output_dir(label)
            summary_path = output_dir / "run_summary.json"

            frozen = is_frozen_run(run_id)
            epochs = args.epochs or (config.FROZEN_EPOCHS if frozen else config.EPOCHS)
            expected_signature = {
                **base_signature,
                "limit_train_rows": (
                    "none"
                    if args.limit_train_rows is None
                    else str(args.limit_train_rows)
                ),
                "subsample_seed": (
                    "none" if args.limit_train_rows is None else str(seed)
                ),
            }
            expected_runtime = {
                "world_size": context.world_size,
                "distributed_backend": context.backend,
                "devices": list(device_names),
                "global_batch_size": args.batch_size,
                "per_device_batch_size": local_batch_size,
                "global_eval_batch_size": args.eval_batch_size,
                "per_device_eval_batch_size": local_eval_batch_size,
                "effective_global_batch_size": (
                    args.batch_size * args.grad_accum_steps
                ),
                "num_workers_per_rank": workers,
                "amp": not args.no_amp and context.device.type == "cuda",
                "amp_dtype": amp_dtype,
                "deterministic": not args.fast_nondeterministic,
                "fused_optimizer": (
                    not args.no_fused_optimizer and context.device.type == "cuda"
                ),
                "pad_to_multiple_of": args.pad_to_multiple_of,
            }

            decision = "train"
            existing = None
            if context.is_main and summary_path.exists() and not args.force:
                existing = load_json(summary_path)
                compatible = cached_summary_matches(
                    existing,
                    base_run=run_id,
                    seed=seed,
                    frozen=frozen,
                    epochs=epochs,
                    is_debug_subset=args.limit_train_rows is not None,
                    expected_runtime=expected_runtime,
                    expected_signature=expected_signature,
                )
                decision = "cached" if compatible else "incompatible"
            decision = str(broadcast_object(decision, context))
            if decision == "incompatible":
                raise RuntimeError(
                    f"Cached run {label} was produced by a different data/runtime "
                    "protocol. Use --force to retrain it explicitly."
                )
            if decision == "cached":
                if context.is_main:
                    assert existing is not None
                    board.update(
                        label,
                        status="cached",
                        epochs=existing.get("best_epoch", "-"),
                        trainable=existing.get("parameter_counts", {})
                        .get("total", {})
                        .get("trainable", "-"),
                        val_f1=existing.get("best_val_f1_macro"),
                        val_accuracy=existing.get("best_val_accuracy"),
                        seconds=existing.get("total_train_seconds"),
                    )
                barrier(context)
                continue

            resume_incomplete = False
            if context.is_main:
                resume_incomplete = bool(
                    not args.force
                    and not args.restart_incomplete
                    and not summary_path.exists()
                    and (output_dir / "last.pt").exists()
                )
            resume_incomplete = bool(broadcast_object(resume_incomplete, context))
            if resume_incomplete and context.is_main:
                print(
                    f"[suite] resuming incomplete run {label} from last.pt", flush=True
                )

            if context.is_main:
                board.mark_running(label)
            started = time.perf_counter()
            try:
                set_seed(seed, deterministic=not args.fast_nondeterministic)
                loaders = build_dataloaders(
                    tokenizer,
                    batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size,
                    seed=seed,
                    num_workers=workers,
                    limit_train_rows=args.limit_train_rows,
                    rank=context.rank,
                    world_size=context.world_size,
                    pad_to_multiple_of=args.pad_to_multiple_of,
                )
                model = build_model(run_id)
                run_signature = {
                    **expected_signature,
                    "train_rows": str(loaders["train_size"]),
                }

                train_config = TrainConfig(
                    output_dir=output_dir,
                    run_id=label,
                    epochs=epochs,
                    grad_accum_steps=args.grad_accum_steps,
                    freeze_backbone=frozen,
                    seed=seed,
                    use_amp=not args.no_amp,
                    deterministic=not args.fast_nondeterministic,
                    use_fused_optimizer=not args.no_fused_optimizer,
                    global_batch_size=args.batch_size,
                    per_device_batch_size=loaders["train"].per_device_batch_size,
                    global_eval_batch_size=args.eval_batch_size,
                    per_device_eval_batch_size=loaders[
                        "validation"
                    ].per_device_batch_size,
                    world_size=context.world_size,
                    num_workers=workers,
                    pad_to_multiple_of=args.pad_to_multiple_of,
                    amp_dtype=amp_dtype,
                    distributed_backend=context.backend,
                    device_names=device_names,
                    data_signature=run_signature,
                    architecture=model.architecture_config(),
                )
                reporter = (
                    TrainingReporter(
                        label, total_epochs=epochs, total_batches=len(loaders["train"])
                    )
                    if args.progress and context.is_main
                    else None
                )
                result = train_model(
                    model,
                    loaders["train"],
                    loaders["validation"],
                    train_config,
                    reporter=reporter,
                    resume=resume_incomplete,
                    distributed_context=context,
                )
                counts = model.count_parameters()
                if context.is_main:
                    save_json(
                        {
                            "run_id": label,
                            "base_run": run_id,
                            "seed": seed,
                            "backbone_frozen": frozen,
                            "epochs": epochs,
                            "train_rows": loaders["train_size"],
                            "is_debug_subset": args.limit_train_rows is not None,
                            "parameter_counts": counts,
                            "best_epoch": result.best_epoch,
                            "best_val_f1_macro": result.best_val_f1_macro,
                            "best_val_accuracy": result.best_val_accuracy,
                            "best_val_loss": result.best_val_loss,
                            "total_train_seconds": result.total_train_seconds,
                            "session_train_seconds": result.session_train_seconds,
                            "peak_vram_gb": result.peak_vram_gb,
                            "seconds_per_epoch": round(
                                result.total_train_seconds
                                / max(1, len(result.history)),
                                2,
                            ),
                            "runtime": {
                                "world_size": context.world_size,
                                "distributed_backend": context.backend,
                                "devices": device_names,
                                "global_batch_size": args.batch_size,
                                "per_device_batch_size": (
                                    loaders["train"].per_device_batch_size
                                ),
                                "global_eval_batch_size": args.eval_batch_size,
                                "per_device_eval_batch_size": loaders[
                                    "validation"
                                ].per_device_batch_size,
                                "effective_global_batch_size": (
                                    args.batch_size * args.grad_accum_steps
                                ),
                                "num_workers_per_rank": workers,
                                "amp": not args.no_amp
                                and context.device.type == "cuda",
                                "amp_dtype": amp_dtype,
                                "deterministic": not args.fast_nondeterministic,
                                "fused_optimizer": (
                                    not args.no_fused_optimizer
                                    and context.device.type == "cuda"
                                ),
                                "pad_to_multiple_of": args.pad_to_multiple_of,
                                "train_sampler_padding": loaders[
                                    "train"
                                ].sampler_padding,
                            },
                            "data_signature": run_signature,
                            "architecture": train_config.architecture,
                            "history": result.history,
                        },
                        summary_path,
                    )
                    board.update(
                        label,
                        status="done",
                        epochs=result.best_epoch,
                        trainable=counts["total"]["trainable"],
                        val_f1=result.best_val_f1_macro,
                        val_accuracy=result.best_val_accuracy,
                        seconds=result.total_train_seconds,
                    )
                barrier(context)
            except Exception:
                failures.append(label)
                if context.is_main:
                    board.update(
                        label, status="FAILED", seconds=time.perf_counter() - started
                    )
                    traceback.print_exc()
                if not args.continue_on_error:
                    return 1

        if context.is_main:
            leaderboard = board.render()
            save_json(
                {
                    "runs": labels,
                    "failures": failures,
                    "note": (
                        "Validation metrics only. Official test metrics require "
                        "experiments.final_eval."
                    ),
                },
                config.REPORTS_DIR / "suite_summary.json",
            )
            (config.TABLES_DIR / "suite_leaderboard.txt").write_text(
                leaderboard, encoding="utf-8"
            )

            if failures:
                print(
                    f"[suite] {len(failures)} run(s) failed: {', '.join(failures)}",
                    flush=True,
                )
            else:
                print("[suite] all runs completed", flush=True)
        return 1 if failures else 0
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    raise SystemExit(main())
