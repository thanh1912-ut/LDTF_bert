"""Train one baseline or ablation run.

Usage::

    python -m experiments.run_experiment --run A0
    python -m experiments.run_experiment --run B2_bert_finetuned_cls --seed 1337
    python -m experiments.run_experiment --run A0 --frozen-backbone   # A15 pair

This script trains and validates only. It never constructs the official test
loader; final numbers come from ``python -m experiments.final_eval``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from src import config
from src.dataset import build_dataloaders, data_signature
from src.distributed import (
    all_gather_objects,
    barrier,
    broadcast_object,
    cleanup_distributed,
    initialize_distributed,
)
from src.models import LdtfBert
from src.registry import ABLATION_RUNS, BASELINE_RUNS, build_model, is_frozen_run
from src.train import (
    TrainConfig,
    build_optimizer,
    optimizer_coverage_report,
    train_model,
)
from src.utils import save_json, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one LDTF-BERT run.")
    parser.add_argument(
        "--run",
        required=True,
        help=f"Run id. Baselines: {sorted(BASELINE_RUNS)}. Ablations: {list(ABLATION_RUNS)}.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Global micro-batch across all ranks (32 becomes 16 per GPU on T4 x2).",
    )
    parser.add_argument("--eval-batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=config.GRAD_ACCUM_STEPS)
    parser.add_argument(
        "--backbone-lr", type=float, default=config.BACKBONE_LEARNING_RATE
    )
    parser.add_argument("--head-lr", type=float, default=config.HEAD_LEARNING_RATE)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers per rank (default: 1 with DDP, otherwise config value).",
    )
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start fresh and archive any existing run artifacts under previous_runs/.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--fast-nondeterministic",
        action="store_true",
        help="Allow cuDNN autotuning for speed; record the protocol as non-deterministic.",
    )
    parser.add_argument(
        "--no-fused-optimizer",
        action="store_true",
        help="Disable CUDA fused AdamW (enabled automatically on CUDA by default).",
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        choices=(8, 16, 32),
        default=None,
        help="Optionally pad each dynamic batch to a Tensor-Core-friendly multiple.",
    )
    parser.add_argument(
        "--frozen-backbone",
        dest="frozen_backbone",
        action="store_true",
        default=None,
        help="Force a frozen backbone (used for the A15 regime pair).",
    )
    parser.add_argument(
        "--finetune-backbone",
        dest="frozen_backbone",
        action="store_false",
        help="Force a fine-tuned backbone.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        default=True,
        help="Disable the YOLO-style live display and print one line per epoch.",
    )
    parser.add_argument(
        "--limit-train-rows",
        type=int,
        default=None,
        help=(
            "Debug only: train on a stratified subset of this many rows. "
            "Recorded in run_summary.json so such runs cannot be mistaken for "
            "full results."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = initialize_distributed()
    try:
        if context.is_main:
            config.ensure_output_dirs()
        barrier(context)
        set_seed(args.seed, deterministic=not args.fast_nondeterministic)

        workers = args.num_workers
        if workers is None:
            workers = 1 if context.enabled else config.NUM_WORKERS
        if workers < 0:
            raise ValueError("--num-workers must be non-negative.")
        if args.grad_accum_steps < 1:
            raise ValueError("--grad-accum-steps must be at least 1.")
        if workers:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        frozen = (
            is_frozen_run(args.run)
            if args.frozen_backbone is None
            else args.frozen_backbone
        )
        suffix = "_frozen" if args.frozen_backbone is True else ""
        if args.limit_train_rows is not None:
            suffix += f"_debug{args.limit_train_rows}"
        run_id = f"{args.run}{suffix}_seed{args.seed}"
        output_dir = args.output_dir or config.experiment_output_dir(run_id)
        if args.resume and args.force:
            raise ValueError("--resume and --force are mutually exclusive.")
        last_exists = (
            bool((output_dir / "last.pt").exists()) if context.is_main else False
        )
        has_artifacts = (
            any(
                (output_dir / name).exists()
                for name in (
                    "best.pt",
                    "last.pt",
                    "train_log.jsonl",
                    "run_summary.json",
                )
            )
            if context.is_main
            else False
        )
        last_exists = bool(broadcast_object(last_exists, context))
        has_artifacts = bool(broadcast_object(has_artifacts, context))
        if args.resume and not last_exists:
            raise FileNotFoundError(
                f"--resume requested but no last.pt exists in {output_dir}."
            )
        if has_artifacts and not args.resume and not args.force:
            raise RuntimeError(
                f"Run artifacts already exist in {output_dir}. Use --resume to continue "
                "or --force to archive them and start fresh."
            )

        tokenizer = LdtfBert.build_tokenizer(config.MODEL_NAME)
        loaders = build_dataloaders(
            tokenizer,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            num_workers=workers,
            limit_train_rows=args.limit_train_rows,
            rank=context.rank,
            world_size=context.world_size,
            pad_to_multiple_of=args.pad_to_multiple_of,
        )
        if context.is_main and args.limit_train_rows is not None:
            print(
                f"[{run_id}] WARNING: training on a {loaders['train_size']}-row debug "
                "subset; these results are not full results.",
                flush=True,
            )
        if context.is_main:
            print(
                f"[{run_id}] train={loaders['train_size']} validation={loaders['val_size']} "
                f"world={context.world_size} global_batch={args.batch_size} "
                f"per_gpu={loaders['train'].per_device_batch_size} "
                "(official test not loaded)",
                flush=True,
            )

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
        signature = {
            **data_signature(),
            "train_rows": str(loaders["train_size"]),
            "limit_train_rows": (
                "none" if args.limit_train_rows is None else str(args.limit_train_rows)
            ),
            "subsample_seed": (
                "none" if args.limit_train_rows is None else str(args.seed)
            ),
        }

        model = build_model(args.run, frozen_backbone=args.frozen_backbone)
        counts = model.count_parameters()
        if context.is_main:
            print(f"[{run_id}] parameters: {counts['total']}", flush=True)

        epochs = args.epochs
        if epochs is None:
            epochs = config.FROZEN_EPOCHS if frozen else config.EPOCHS

        train_config = TrainConfig(
            output_dir=output_dir,
            run_id=run_id,
            epochs=epochs,
            backbone_learning_rate=args.backbone_lr,
            head_learning_rate=args.head_lr,
            grad_accum_steps=args.grad_accum_steps,
            freeze_backbone=frozen,
            seed=args.seed,
            patience=args.patience,
            use_amp=not args.no_amp,
            deterministic=not args.fast_nondeterministic,
            use_fused_optimizer=not args.no_fused_optimizer,
            global_batch_size=args.batch_size,
            per_device_batch_size=loaders["train"].per_device_batch_size,
            global_eval_batch_size=args.eval_batch_size,
            per_device_eval_batch_size=loaders["validation"].per_device_batch_size,
            world_size=context.world_size,
            num_workers=workers,
            pad_to_multiple_of=args.pad_to_multiple_of,
            amp_dtype=amp_dtype,
            distributed_backend=context.backend,
            device_names=device_names,
            data_signature=signature,
            architecture=model.architecture_config(),
        )

        coverage = optimizer_coverage_report(
            model, build_optimizer(model, train_config)
        )
        if not coverage["covered"]:
            raise RuntimeError(
                f"Optimizer does not cover the trainable set: {coverage}"
            )

        result = train_model(
            model,
            loaders["train"],
            loaders["validation"],
            train_config,
            resume=args.resume,
            show_progress=args.progress and context.is_main,
            distributed_context=context,
        )

        if context.is_main:
            save_json(
                {
                    "run_id": run_id,
                    "base_run": args.run,
                    "seed": args.seed,
                    "backbone_frozen": frozen,
                    "epochs": epochs,
                    "train_rows": loaders["train_size"],
                    "is_debug_subset": args.limit_train_rows is not None,
                    "parameter_counts": counts,
                    "optimizer_coverage": coverage,
                    "best_epoch": result.best_epoch,
                    "best_val_f1_macro": result.best_val_f1_macro,
                    "best_val_accuracy": result.best_val_accuracy,
                    "best_val_loss": result.best_val_loss,
                    "total_train_seconds": result.total_train_seconds,
                    "session_train_seconds": result.session_train_seconds,
                    "peak_vram_gb": result.peak_vram_gb,
                    "seconds_per_epoch": (
                        round(
                            result.total_train_seconds / max(1, len(result.history)), 2
                        )
                    ),
                    "runtime": {
                        "world_size": context.world_size,
                        "distributed_backend": context.backend,
                        "devices": device_names,
                        "global_batch_size": args.batch_size,
                        "per_device_batch_size": loaders["train"].per_device_batch_size,
                        "global_eval_batch_size": args.eval_batch_size,
                        "per_device_eval_batch_size": loaders[
                            "validation"
                        ].per_device_batch_size,
                        "effective_global_batch_size": (
                            args.batch_size * args.grad_accum_steps
                        ),
                        "num_workers_per_rank": workers,
                        "amp": not args.no_amp and context.device.type == "cuda",
                        "amp_dtype": amp_dtype,
                        "deterministic": not args.fast_nondeterministic,
                        "fused_optimizer": (
                            not args.no_fused_optimizer
                            and context.device.type == "cuda"
                        ),
                        "pad_to_multiple_of": args.pad_to_multiple_of,
                        "train_sampler_padding": loaders["train"].sampler_padding,
                    },
                    "data_signature": train_config.data_signature,
                    "architecture": train_config.architecture,
                    "history": result.history,
                },
                output_dir / "run_summary.json",
            )
            print(
                f"[{run_id}] best val macro F1={result.best_val_f1_macro:.4f} "
                f"accuracy={result.best_val_accuracy:.4f} at epoch {result.best_epoch}",
                flush=True,
            )
        barrier(context)
        return 0
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    raise SystemExit(main())
