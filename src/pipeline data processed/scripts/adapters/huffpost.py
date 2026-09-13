"""Adapter for the HuffPost News Category Dataset JSON Lines file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import LABEL_NAMES, standard_frame, text_value


def load_huffpost_world_news(
    path: Path, accepted_categories: dict[str, int]
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"HuffPost source not found: {path}")

    source = pd.read_json(path, lines=True)
    required = {"category", "headline", "short_description"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"HuffPost is missing columns: {sorted(missing)}")

    normalized_mapping = {
        str(key).strip().casefold(): int(value) for key, value in accepted_categories.items()
    }
    categories = source["category"].astype(str).str.strip().str.casefold()
    accepted = source.loc[categories.isin(normalized_mapping)].copy()

    rows = []
    empty_source_rows = 0
    for position, row in accepted.iterrows():
        category = text_value(row["category"])
        title = text_value(row["headline"])
        description = text_value(row["short_description"])
        text = " ".join(part for part in (title, description) if part)
        if not text:
            empty_source_rows += 1
            continue
        label = normalized_mapping[category.casefold()]
        source_id = text_value(row["link"]) if "link" in row.index else ""
        source_id = source_id or str(position)
        rows.append(
            {
                "row_id": f"huffpost:{position:06d}",
                "source_dataset": "huffpost",
                "source_id": source_id,
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
