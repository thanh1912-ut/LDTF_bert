"""Locked final evaluation on the official AG News test split.

This is the only script permitted to read ``research_test.parquet``. It refuses
to run unless the environment carries the unlock token::

    export LDTF_ALLOW_OFFICIAL_TEST=I_AM_REPORTING_FINAL_RESULTS
    python -m experiments.final_eval --run A0_seed42 --reason "final camera-ready numbers"

Every access is appended to ``reports/official_test_access.jsonl``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config, guard
from src.evaluate import evaluate_dataloader, load_model_from_checkpoint
from src.models import LdtfBert
from src.utils import get_device, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen checkpoints on the official test split."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run directory name under outputs/. Repeatable.",
    )
    parser.add_argument("--reason", required=True, help="Why the seal is being broken.")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not guard.is_unlocked():
        raise guard.OfficialTestAccessError(
            "The official test split is sealed. Export "
            f"{guard.UNLOCK_ENV_VAR}={guard.UNLOCK_TOKEN} to unseal it, and only "
            "after all model selection is complete."
        )
    config.ensure_output_dirs()

    tokenizer = LdtfBert.build_tokenizer(config.MODEL_NAME)
    device = get_device()
    results: dict[str, object] = {}

    for run_id in args.run:
        output_dir = config.experiment_output_dir(run_id)
        checkpoint_path = output_dir / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No best.pt found for run {run_id} at {checkpoint_path}.")

        model, state = load_model_from_checkpoint(checkpoint_path)
        loader = guard.official_test_loader(
            tokenizer,
            reason=args.reason,
            run_id=run_id,
            batch_size=args.batch_size,
            checkpoint_path=checkpoint_path,
        )
        metrics = evaluate_dataloader(
            model,
            loader,
            device=device,
            output_path=output_dir / "test_metrics.json",
            predictions_path=output_dir / "test_predictions.npz",
        )
        metrics["checkpoint_epoch"] = state.get("epoch")
        metrics["selection_metrics"] = state.get("best_metrics")
        results[run_id] = metrics
        print(
            f"[final_eval] {run_id}: accuracy={metrics['accuracy']:.4f} "
            f"macro F1={metrics['f1_macro']:.4f}",
            flush=True,
        )

    save_json(
        {"reason": args.reason, "results": results},
        config.REPORTS_DIR / "final_test_metrics.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
