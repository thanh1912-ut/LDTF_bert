"""Export the baseline and ablation result tables.

Rows for runs that have not been executed are emitted as ``PENDING``. Metrics
are read exclusively from ``outputs/<run>/run_summary.json`` (validation) and
``outputs/<run>/test_metrics.json`` (official test, written only by
``experiments.final_eval``). No value in these tables is ever fabricated.
"""

from __future__ import annotations

import argparse

from src import config
from src.registry import build_model, is_frozen_run
from src.utils import load_json, save_json

PENDING = "PENDING"

BASELINE_ROWS = [
    ("B0", "TF-IDF + Logistic Regression", "none", "bag-of-ngrams", "none", "class-specific LR"),
    ("B1", "BERT Frozen CLS", "frozen", "final-layer [CLS]", "final layer", "Linear(D,C)"),
    ("B2", "BERT Fine-Tuned CLS", "fine-tuned", "final-layer [CLS]", "final layer", "Linear(D,C)"),
    ("B3", "BERT Frozen Mean Pool", "frozen", "masked mean pool", "final layer", "Linear(D,C)"),
    ("B4", "BERT Scalar Mix", "fine-tuned", "masked mean pool", "global scalar mix", "Linear(D,C)"),
    (
        "B5",
        "Label Token Attention Only",
        "fine-tuned",
        "label-conditioned",
        "uniform layer mean",
        "shared linear",
    ),
    (
        "B6",
        "Full LDTF Frozen",
        "frozen",
        "label-conditioned",
        "direct label-conditioned",
        "shared linear",
    ),
    (
        "B7",
        "Full LDTF Fine-Tuned",
        "fine-tuned",
        "label-conditioned",
        "direct label-conditioned",
        "shared linear",
    ),
]

BASELINE_RUN_IDS = {
    "B0": "B0_tfidf_logreg",
    "B1": "B1_bert_frozen_cls",
    "B2": "B2_bert_finetuned_cls",
    "B3": "B3_bert_frozen_mean_pool",
    "B4": "B4_bert_scalar_mix",
    "B5": "B5_token_attention_only",
    "B6": "B6_full_ldtf_frozen",
    "B7": "B7_full_ldtf_finetuned",
}

ABLATION_ROWS = [
    ("A0", "Full Direct LDTF", "label-conditioned", "direct label-conditioned", "all layers", "trainable", "shared linear"),
    ("A1", "No Depth Router", "label-conditioned", "uniform layer mean", "all layers", "trainable", "shared linear"),
    ("A2", "Final Layer Only", "label-conditioned", "none", "final layer", "trainable", "shared linear"),
    ("A3", "Global Scalar Mix", "label-conditioned", "global scalar mix", "all layers", "trainable", "shared linear"),
    ("A4", "Indirect Depth Gating", "label-conditioned", "indirect gating", "all layers", "trainable", "shared linear"),
    ("A5", "Frozen Label Queries", "label-conditioned", "direct label-conditioned", "all layers", "frozen", "shared linear"),
    ("A6", "Fixed Random Label Queries", "label-conditioned", "direct label-conditioned", "all layers", "fixed random", "shared linear"),
    ("A7", "Shared Linear Scorer", "label-conditioned", "direct label-conditioned", "all layers", "trainable", "shared linear"),
    ("A8", "Shared MLP Scorer", "label-conditioned", "direct label-conditioned", "all layers", "trainable", "shared MLP"),
    ("A9", "Class-Specific Linear Scorer", "label-conditioned", "direct label-conditioned", "all layers", "trainable", "class-specific"),
    ("A10", "Router Dimension 64", "label-conditioned (R=64)", "direct (R=64)", "all layers", "trainable", "shared linear"),
    ("A11", "Router Dimension 128", "label-conditioned (R=128)", "direct (R=128)", "all layers", "trainable", "shared linear"),
    ("A12", "Router Dimension 256", "label-conditioned (R=256)", "direct (R=256)", "all layers", "trainable", "shared linear"),
    ("A13", "Include Special Tokens", "label-conditioned (+special)", "direct label-conditioned", "all layers", "trainable", "shared linear"),
    ("A14", "Exclude Special Tokens", "label-conditioned (-special)", "direct label-conditioned", "all layers", "trainable", "shared linear"),
    ("A15", "Frozen vs Fine-Tuned", "label-conditioned", "direct label-conditioned", "all layers", "trainable", "shared linear"),
]


def _load_run(run_id: str, seed: int) -> tuple[dict | None, dict | None]:
    directory = config.experiment_output_dir(f"{run_id}_seed{seed}")
    summary_path = directory / "run_summary.json"
    test_path = directory / "test_metrics.json"
    summary = load_json(summary_path) if summary_path.exists() else None
    test = load_json(test_path) if test_path.exists() else None
    return summary, test


def _parameter_counts(row_id: str) -> tuple[str, str]:
    """Return (total, trainable) parameter strings, computed without training."""
    if row_id == "B0":
        return PENDING, PENDING
    try:
        run_key = BASELINE_RUN_IDS.get(row_id, row_id)
        if row_id == "A15":
            run_key = "A0"
        model = build_model(run_key)
        counts = model.count_parameters()["total"]
        return f"{counts['total']:,}", f"{counts['trainable']:,}"
    except Exception:  # pragma: no cover - offline or unsupported id
        return PENDING, PENDING


def _metric(value: object) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else PENDING


def build_baseline_table(seed: int, *, with_params: bool) -> str:
    header = (
        "| ID | Model | Backbone regime | Token mechanism | Depth mechanism | Scorer | "
        "Total params | Trainable params | Accuracy | Macro F1 | Peak VRAM | Time/epoch |\n"
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for row_id, name, regime, token, depth, scorer in BASELINE_ROWS:
        run_id = BASELINE_RUN_IDS[row_id]
        summary, test = _load_run(run_id, seed)
        total, trainable = (
            _parameter_counts(row_id) if with_params else (PENDING, PENDING)
        )
        if summary:
            total = f"{summary.get('parameter_counts', {}).get('total', {}).get('total', 0):,}" if summary.get("parameter_counts") else total
            trainable = f"{summary.get('parameter_counts', {}).get('total', {}).get('trainable', 0):,}" if summary.get("parameter_counts") else trainable
        accuracy = _metric(test.get("accuracy")) if test else PENDING
        macro_f1 = _metric(test.get("f1_macro")) if test else PENDING
        vram = (
            f"{summary['peak_vram_gb']:.2f} GB"
            if summary and summary.get("peak_vram_gb")
            else PENDING
        )
        per_epoch = (
            f"{summary['seconds_per_epoch']:.1f} s"
            if summary and summary.get("seconds_per_epoch")
            else PENDING
        )
        lines.append(
            f"| {row_id} | {name} | {regime} | {token} | {depth} | {scorer} | "
            f"{total} | {trainable} | {accuracy} | {macro_f1} | {vram} | {per_epoch} |\n"
        )
    return "".join(lines)


def build_ablation_table(seed: int, *, with_params: bool) -> str:
    header = (
        "| ID | Variant | Token routing | Depth routing | Layer policy | Query policy | "
        "Scorer | Params | Macro F1 | Delta F1 vs A0 |\n"
        "|---|---|---|---|---|---|---|---:|---:|---:|\n"
    )
    reference_summary, reference_test = _load_run("A0", seed)
    reference_f1 = reference_test.get("f1_macro") if reference_test else None

    lines = [header]
    for row_id, name, token, depth, layers, query, scorer in ABLATION_ROWS:
        summary, test = _load_run(row_id, seed)
        params = _parameter_counts(row_id)[1] if with_params else PENDING
        if summary and summary.get("parameter_counts"):
            params = f"{summary['parameter_counts']['total']['trainable']:,}"
        macro_f1 = _metric(test.get("f1_macro")) if test else PENDING
        if test and reference_f1 is not None:
            delta = f"{float(test['f1_macro']) - float(reference_f1):+.4f}"
        else:
            delta = PENDING
        lines.append(
            f"| {row_id} | {name} | {token} | {depth} | {layers} | {query} | {scorer} | "
            f"{params} | {macro_f1} | {delta} |\n"
        )
    return "".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export result table schemas.")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--with-params",
        action="store_true",
        help="Instantiate each model to fill exact parameter counts (downloads BERT).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_output_dirs()

    baseline = build_baseline_table(args.seed, with_params=args.with_params)
    ablation = build_ablation_table(args.seed, with_params=args.with_params)

    note = (
        "> Rows marked PENDING have not been executed. No metric in these tables is\n"
        "> estimated, fabricated, or copied from a fixture. Official-test metrics are\n"
        "> produced solely by `python -m experiments.final_eval`.\n\n"
    )
    baseline_path = config.TABLES_DIR / "baseline_table.md"
    ablation_path = config.TABLES_DIR / "ablation_table.md"
    baseline_path.write_text(
        f"# Baseline Results (seed {args.seed})\n\n{note}{baseline}", encoding="utf-8"
    )
    ablation_path.write_text(
        f"# Ablation Results (seed {args.seed})\n\n{note}"
        "> A0 is the reference. Delta F1 is computed only when both A0 and the row\n"
        "> have real official-test metrics.\n\n" + ablation,
        encoding="utf-8",
    )
    save_json(
        {"seed": args.seed, "baseline_table": str(baseline_path), "ablation_table": str(ablation_path)},
        config.TABLES_DIR / "tables_index.json",
    )
    print(f"[tables] wrote {baseline_path}")
    print(f"[tables] wrote {ablation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
