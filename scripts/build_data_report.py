"""Produce the data report: counts, balance, duplicates, overlap, token lengths."""

from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.dataset import describe_split, file_sha256, load_split
from src.models import LdtfBert
from src.utils import save_json


def _overlap(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    if column not in left.columns or column not in right.columns:
        return -1
    return int(len(set(left[column]) & set(right[column])))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the data report.")
    parser.add_argument(
        "--token-lengths",
        action="store_true",
        help="Also compute token-length statistics (loads the tokenizer).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_output_dirs()

    tokenizer = LdtfBert.build_tokenizer(config.MODEL_NAME) if args.token_lengths else None
    train = load_split(config.PROCESSED_TRAIN)
    validation = load_split(config.PROCESSED_VAL)

    report: dict[str, object] = {
        "cleaning_version": "prepare_agnews v2 (src/data/scripts/prepare_agnews.py)",
        "label_mapping": {
            str(index): name for index, name in enumerate(config.LABEL_NAMES)
        },
        "truncation_policy": (
            f"tokenizer(truncation=True, max_length={config.MAX_LENGTH}); "
            "longest_first truncation from the right; dynamic per-batch padding"
        ),
        "text_encoding": config.TEXT_ENCODING,
        "checksums": {
            "train": file_sha256(config.PROCESSED_TRAIN),
            "validation": file_sha256(config.PROCESSED_VAL),
        },
        "splits": {
            "train": describe_split(train, tokenizer, name="train"),
            "validation": describe_split(validation, tokenizer, name="validation"),
        },
        "train_validation_overlap": {
            "row_id": _overlap(train, validation, "row_id"),
            "text_hash": _overlap(train, validation, "text_hash"),
            "near_duplicate_group_id": _overlap(
                train, validation, "near_duplicate_group_id"
            ),
        },
        "official_test": (
            "Not inspected. The official test split is sealed by src.guard and is "
            "excluded from this report by design. Train/test overlap statistics were "
            "produced by the upstream data pipeline; see "
            "src/data/reports/split_overlap_report.json."
        ),
    }
    output_path = config.REPORTS_DIR / "data_report.json"
    save_json(report, output_path)
    print(f"[data_report] wrote {output_path}")
    print(
        f"[data_report] train={report['splits']['train']['num_rows']} "
        f"validation={report['splits']['validation']['num_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
