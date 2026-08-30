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
from pathlib import Path

from src import config
from src.dataset import build_dataloaders, data_signature
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
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=config.GRAD_ACCUM_STEPS)
    parser.add_argument("--backbone-lr", type=float, default=config.BACKBONE_LEARNING_RATE)
    parser.add_argument("--head-lr", type=float, default=config.HEAD_LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
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
    config.ensure_output_dirs()
    set_seed(args.seed)

    frozen = is_frozen_run(args.run) if args.frozen_backbone is None else args.frozen_backbone
    suffix = "_frozen" if args.frozen_backbone is True else ""
    run_id = f"{args.run}{suffix}_seed{args.seed}"
    output_dir = args.output_dir or config.experiment_output_dir(run_id)

    tokenizer = LdtfBert.build_tokenizer(config.MODEL_NAME)
    loaders = build_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_train_rows=args.limit_train_rows,
    )
    if args.limit_train_rows is not None:
        print(
            f"[{run_id}] WARNING: training on a {loaders['train_size']}-row debug "
            "subset; these results are not full results.",
            flush=True,
        )
    print(
        f"[{run_id}] train={loaders['train_size']} validation={loaders['val_size']} "
        f"(official test not loaded)",
        flush=True,
    )

    model = build_model(args.run, frozen_backbone=args.frozen_backbone)
    counts = model.count_parameters()
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
        data_signature=data_signature(),
        architecture=model.architecture_config(),
    )

    coverage = optimizer_coverage_report(model, build_optimizer(model, train_config))
    if not coverage["covered"]:
        raise RuntimeError(f"Optimizer does not cover the trainable set: {coverage}")

    result = train_model(
        model,
        loaders["train"],
        loaders["validation"],
        train_config,
        resume=args.resume,
        show_progress=args.progress,
    )

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
            "peak_vram_gb": result.peak_vram_gb,
            "seconds_per_epoch": (
                round(result.total_train_seconds / max(1, len(result.history)), 2)
            ),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
