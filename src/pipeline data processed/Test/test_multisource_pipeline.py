"""Focused tests for multi-source cleaning, adapters, and balancing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PIPELINE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location("multisource_pipeline", SCRIPTS / "run_pipeline.py")
assert SPEC and SPEC.loader
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


def test_clean_text_decodes_entities_without_corrupting_valid_entities() -> None:
    assert PIPELINE.clean_text("A #38; B &#39;ok&#39; <br> now") == "A & B 'ok' now"


def test_balanced_sample_uses_smallest_available_class() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [f"row-{index}" for index in range(10)],
            "label": [0, 0, 1, 1, 1, 2, 2, 3, 3, 3],
        }
    )
    selected, report = PIPELINE.balanced_sample(frame, maximum_per_label=3, seed=42)
    assert selected.groupby("label").size().to_dict() == {0: 2, 1: 2, 2: 2, 3: 2}
    assert report["selected_per_label"] == 2


def test_bbc_adapter_keeps_only_mapped_categories(tmp_path: Path) -> None:
    source = tmp_path / "bbc.csv"
    pd.DataFrame(
        {
            "category": ["sport", "business", "politics"],
            "text": ["one two three four", "five six seven eight", "excluded"],
        }
    ).to_csv(source, index=False)
    frame, report = PIPELINE.load_bbc_news(
        source, {"sport": 1, "business": 2, "tech": 3}, title_word_count=2
    )
    assert frame["label"].tolist() == [1, 2]
    assert report["excluded_category_rows"] == 1


def test_huffpost_adapter_keeps_world_news_only(tmp_path: Path) -> None:
    source = tmp_path / "huffpost.json"
    pd.DataFrame(
        [
            {
                "category": "WORLD NEWS",
                "headline": "Global headline",
                "short_description": "International report",
                "link": "https://example.test/world",
            },
            {
                "category": "POLITICS",
                "headline": "Domestic headline",
                "short_description": "Domestic report",
                "link": "https://example.test/politics",
            },
        ]
    ).to_json(source, orient="records", lines=True)
    frame, report = PIPELINE.load_huffpost_world_news(source, {"WORLD NEWS": 0})
    assert frame["label"].tolist() == [0]
    assert report["excluded_category_rows"] == 1


def test_exact_overlap_with_evaluation_is_removed() -> None:
    def make_frame(row_id: str, split: str, label: int, text: str) -> pd.DataFrame:
        raw = pd.DataFrame(
            [
                {
                    "row_id": row_id,
                    "source_dataset": "fixture",
                    "source_id": row_id,
                    "source_split": split,
                    "original_label": str(label),
                    "source_category": str(label),
                    "title_raw": text,
                    "description_raw": "",
                    "text_raw": text,
                    "label": label,
                    "label_name": PIPELINE.LABEL_NAMES[label],
                }
            ]
        )
        return PIPELINE.clean_frame(raw)[0]

    candidate = make_frame("external:1", "external", 0, "same world story")
    references = {
        "train": make_frame("ag:train", "train", 1, "different sports story"),
        "validation": make_frame("ag:validation", "validation", 0, "same world story"),
        "test": make_frame("ag:test", "test", 2, "different business story"),
    }
    remaining, removed, duplicates = PIPELINE.exact_deduplicate_candidates(
        candidate, references
    )
    assert remaining.empty
    assert removed["reason"].tolist() == ["external_exact_overlap_with_evaluation"]
    assert duplicates["matched_row_id"].str.startswith("validation:").all()
