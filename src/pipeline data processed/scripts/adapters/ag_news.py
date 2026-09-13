"""Adapter for the fixed AG News train, validation, and test Parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import LABEL_NAMES, standard_frame, text_value


def _first_existing(row: pd.Series, names: tuple[str, ...]) -> str:
    for name in names:
        if name in row.index:
            value = text_value(row[name])
            if value:
                return value
    return ""


def _adapt_split(path: Path, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"AG News {split} split not found: {path}")

    source = pd.read_parquet(path)
    required = {"label"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"AG News {split} is missing columns: {sorted(missing)}")

    rows = []
    for position, row in source.iterrows():
        label = int(row["label"])
        source_id = _first_existing(row, ("row_id",)) or f"{split}_{position:06d}"
        title = _first_existing(row, ("title_raw", "title_clean", "Title"))
        description = _first_existing(
            row, ("description_raw", "description_clean", "Description")
        )
        text = _first_existing(row, ("text",)) or " ".join(
            part for part in (title, description) if part
        )
        original_label = (
            int(row["original_class_index"])
            if "original_class_index" in row.index and pd.notna(row["original_class_index"])
            else label + 1
        )
        rows.append(
            {
                "row_id": f"ag_news:{source_id}",
                "source_dataset": "ag_news",
                "source_id": source_id,
                "source_split": split,
                "original_label": original_label,
                "source_category": LABEL_NAMES[label],
                "title_raw": title,
                "description_raw": description,
                "text_raw": text,
                "label": label,
                "label_name": LABEL_NAMES[label],
            }
        )
    return standard_frame(rows)


def load_ag_news_splits(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Load the fixed baseline splits without repartitioning them."""
    return {
        split: _adapt_split(Path(path), split)
        for split, path in paths.items()
        if split in {"train", "validation", "test"}
    }
