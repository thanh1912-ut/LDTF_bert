"""Adapter for the Kaggle BBC full-text topic dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import LABEL_NAMES, standard_frame, text_value


def _split_title(text: str, word_count: int) -> tuple[str, str]:
    """Create deterministic pair fields while retaining the untouched full text."""
    words = text.split()
    title = " ".join(words[:word_count])
    description = " ".join(words[word_count:])
    return title, description or text


def load_bbc_news(
    path: Path, label_mapping: dict[str, int], title_word_count: int = 16
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"BBC News source not found: {path}")

    source = pd.read_csv(path)
    required = {"category", "text"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"BBC News is missing columns: {sorted(missing)}")

    normalized_mapping = {str(key).strip().casefold(): int(value) for key, value in label_mapping.items()}
    categories = source["category"].astype(str).str.strip().str.casefold()
    accepted = source.loc[categories.isin(normalized_mapping)].copy()

    rows = []
    empty_source_rows = 0
    for position, row in accepted.iterrows():
        category = text_value(row["category"]).casefold()
        text = text_value(row["text"])
        if not text:
            empty_source_rows += 1
            continue
        title, description = _split_title(text, title_word_count)
        label = normalized_mapping[category]
        rows.append(
            {
                "row_id": f"bbc_news:{position:06d}",
                "source_dataset": "bbc_news",
                "source_id": str(position),
                "source_split": "external",
                "original_label": category,
                "source_category": category,
                "title_raw": title,
                "description_raw": description,
                "text_raw": text,
                "label": label,
                "label_name": LABEL_NAMES[label],
            }
        )

    report = {
        "input_rows": int(len(source)),
        "accepted_rows": int(len(rows)),
        "excluded_category_rows": int(len(source) - len(accepted)),
        "empty_source_rows": int(empty_source_rows),
    }
    return standard_frame(rows), report
