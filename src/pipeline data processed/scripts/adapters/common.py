"""Shared schema helpers for source adapters."""

from __future__ import annotations

from typing import Any

import pandas as pd


LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

STANDARD_COLUMNS = [
    "row_id",
    "source_dataset",
    "source_id",
    "source_split",
    "original_label",
    "source_category",
    "title_raw",
    "description_raw",
    "text_raw",
    "label",
    "label_name",
]


def text_value(value: Any) -> str:
    """Return a usable string without turning missing values into 'nan'."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def standard_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build and validate the common pre-cleaning source schema."""
    frame = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    if frame.empty:
        return frame

    frame["label"] = frame["label"].astype("int64")
    frame["original_label"] = frame["original_label"].astype(str)
    frame["label_name"] = frame["label"].map(LABEL_NAMES)
    if frame["label_name"].isna().any():
        invalid = sorted(frame.loc[frame["label_name"].isna(), "label"].unique())
        raise ValueError(f"Invalid normalized labels: {invalid}")
    if frame["row_id"].duplicated().any():
        raise ValueError("Source adapter generated duplicate row_id values.")
    return frame
