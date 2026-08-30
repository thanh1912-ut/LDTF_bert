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
import time
import traceback

from src import config
from src.dataset import build_dataloaders, data_signature
from src.models import LdtfBert
from src.progress import SuiteBoard, TrainingReporter
from src.registry import ABLATION_RUNS, BASELINE_RUNS, build_model, is_frozen_run
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a suite of runs with a leaderboard.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--run", action="append", default=[], help="Explicit run id. Repeatable.")
    parser.add_argument("--seed", action="append", type=int, default=[], help="Repeatable.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=config.GRAD_ACCUM_STEPS)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--limit-train-rows", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false", default=True)
    parser.add_argument(
        "--force", action="store_true", help="Retrain runs that already have a summary."
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed run and carry on instead of aborting the suite.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_output_dirs()

    run_ids = list(args.run) or PRESETS[args.preset or "core"]
    seeds = args.seed or [config.SEED]
    jobs = [(run_id, seed) for seed in seeds for run_id in run_ids]
    labels = [f"{run_id}_seed{seed}" for run_id, seed in jobs]

    board = SuiteBoard(labels, enabled=args.progress)
    print(f"[suite] {len(jobs)} run(s): {', '.join(labels)}\n", flush=True)

    tokenizer = LdtfBert.build_tokenizer(config.MODEL_NAME)
    signature = data_signature()
    failures: list[str] = []

    for run_id, seed in jobs:
        label = f"{run_id}_seed{seed}"
        output_dir = config.experiment_output_dir(label)
        summary_path = output_dir / "run_summary.json"

        if summary_path.exists() and not args.force:
            existing = load_json(summary_path)
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
            continue

        board.mark_running(label)
        started = time.perf_counter()
        try:
            set_seed(seed)
            loaders = build_dataloaders(
                tokenizer,
                batch_size=args.batch_size,
                seed=seed,
                num_workers=args.num_workers,
                limit_train_rows=args.limit_train_rows,
            )
            model = build_model(run_id)
            frozen = is_frozen_run(run_id)
            epochs = args.epochs or (config.FROZEN_EPOCHS if frozen else config.EPOCHS)

            train_config = TrainConfig(
                output_dir=output_dir,
                run_id=label,
                epochs=epochs,
                grad_accum_steps=args.grad_accum_steps,
                freeze_backbone=frozen,
                seed=seed,
                use_amp=not args.no_amp,
                data_signature=signature,
                architecture=model.architecture_config(),
            )
            reporter = (
                TrainingReporter(
                    label, total_epochs=epochs, total_batches=len(loaders["train"])
                )
                if args.progress
                else None
            )
            result = train_model(
                model,
                loaders["train"],
                loaders["validation"],
                train_config,
                reporter=reporter,
            )
            counts = model.count_parameters()
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
                    "peak_vram_gb": result.peak_vram_gb,
                    "seconds_per_epoch": round(
                        result.total_train_seconds / max(1, len(result.history)), 2
                    ),
                    "data_signature": signature,
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
        except Exception:
            failures.append(label)
            board.update(label, status="FAILED", seconds=time.perf_counter() - started)
            traceback.print_exc()
            if not args.continue_on_error:
                return 1

    leaderboard = board.render()
    save_json(
        {
            "runs": labels,
            "failures": failures,
            "note": "Validation metrics only. Official test metrics require experiments.final_eval.",
        },
        config.REPORTS_DIR / "suite_summary.json",
    )
    (config.TABLES_DIR / "suite_leaderboard.txt").write_text(leaderboard, encoding="utf-8")

    if failures:
        print(f"[suite] {len(failures)} run(s) failed: {', '.join(failures)}", flush=True)
        return 1
    print("[suite] all runs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
