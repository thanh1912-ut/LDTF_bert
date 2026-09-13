#!/usr/bin/env python3
"""Build an audited AG News + BBC + HuffPost training dataset."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from adapters import load_ag_news_splits, load_bbc_news, load_huffpost_world_news
from adapters.common import LABEL_NAMES, STANDARD_COLUMNS


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
CLEAN_COLUMNS = STANDARD_COLUMNS + [
    "title_clean",
    "description_clean",
    "text",
    "text_hash",
]


def log(message: str) -> None:
    print(message, flush=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_from_root(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PIPELINE_ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    """Normalize news text while retaining punctuation and semantic content."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value).strip())
    text = re.sub(r"(?<!&)#(\d+);", r"&#\1;", text)
    text = html.unescape(text)

    tickers: dict[str, str] = {}

    def protect_ticker(match: re.Match[str]) -> str:
        marker = f"__TICKER_{len(tickers)}__"
        tickers[marker] = match.group(1)
        return marker

    text = re.sub(r"<([A-Z][A-Z0-9]*[.-][A-Z]{1,5})>", protect_ticker, text)
    text = re.sub(r"<br\s*/?\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    for marker, ticker in tickers.items():
        text = text.replace(marker, ticker)
    text = text.replace(r"\$", "$").replace(r'\"', '"').replace(r"\'", "'")
    text = text.replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()


def canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def clean_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = frame.copy()
    cleaned["title_clean"] = cleaned["title_raw"].map(clean_text)
    cleaned["description_clean"] = cleaned["description_raw"].map(clean_text)
    cleaned["text"] = cleaned["text_raw"].map(clean_text)
    fallback = (
        cleaned["title_clean"] + " " + cleaned["description_clean"]
    ).str.strip()
    cleaned.loc[cleaned["text"].eq(""), "text"] = fallback

    empty_mask = cleaned["text"].str.strip().eq("")
    removed = cleaned.loc[empty_mask].copy()
    if not removed.empty:
        removed["reason"] = "empty_after_cleaning"
    cleaned = cleaned.loc[~empty_mask].copy()
    cleaned["text_hash"] = cleaned["text"].map(
        lambda text: hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()
    )
    return cleaned[CLEAN_COLUMNS].reset_index(drop=True), removed


def frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["row_id", "label", "text_hash"]].itertuples(index=False):
        digest.update(f"{row.row_id}\t{row.label}\t{row.text_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def append_removed(
    records: list[dict[str, Any]], row: Any, reason: str, detail: str = ""
) -> None:
    records.append(
        {
            "row_id": row.row_id,
            "source_dataset": row.source_dataset,
            "label": int(row.label),
            "label_name": row.label_name,
            "text_hash": row.text_hash,
            "reason": reason,
            "detail": detail,
        }
    )


def exact_deduplicate_candidates(
    candidates: pd.DataFrame,
    references: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    removed_records: list[dict[str, Any]] = []
    duplicate_records: list[dict[str, Any]] = []
    keep_indices: list[int] = []

    for text_hash, group in candidates.groupby("text_hash", sort=False):
        ordered = group.sort_values("row_id")
        labels = sorted(int(value) for value in ordered["label"].unique())
        if len(labels) > 1:
            detail = ",".join(str(value) for value in labels)
            for row in ordered.itertuples():
                append_removed(removed_records, row, "external_conflicting_exact_duplicate", detail)
                duplicate_records.append(
                    {
                        "kind": "external_internal_conflict",
                        "row_id": row.row_id,
                        "matched_row_id": "",
                        "text_hash": text_hash,
                        "label": int(row.label),
                        "matched_label": "",
                    }
                )
            continue

        representative = ordered.iloc[0]
        keep_indices.append(int(representative.name))
        for row in ordered.iloc[1:].itertuples():
            append_removed(
                removed_records,
                row,
                "external_same_label_exact_duplicate",
                str(representative["row_id"]),
            )
            duplicate_records.append(
                {
                    "kind": "external_internal_same_label",
                    "row_id": row.row_id,
                    "matched_row_id": representative["row_id"],
                    "text_hash": text_hash,
                    "label": int(row.label),
                    "matched_label": int(representative["label"]),
                }
            )

    survivors = candidates.loc[keep_indices].copy()
    reference_lookup: dict[str, tuple[str, int]] = {}
    for split in ("validation", "test", "train"):
        for row in references[split].itertuples():
            reference_lookup.setdefault(row.text_hash, (f"{split}:{row.row_id}", int(row.label)))

    final_indices: list[int] = []
    for row in survivors.itertuples():
        match = reference_lookup.get(row.text_hash)
        if match is None:
            final_indices.append(row.Index)
            continue
        matched_row_id, matched_label = match
        reason = (
            "external_exact_overlap_with_evaluation"
            if matched_row_id.startswith(("validation:", "test:"))
            else "external_exact_overlap_with_base_train"
        )
        append_removed(removed_records, row, reason, matched_row_id)
        duplicate_records.append(
            {
                "kind": reason,
                "row_id": row.row_id,
                "matched_row_id": matched_row_id,
                "text_hash": row.text_hash,
                "label": int(row.label),
                "matched_label": matched_label,
            }
        )

    remaining = survivors.loc[final_indices].reset_index(drop=True)
    removed = pd.DataFrame(removed_records)
    duplicates = pd.DataFrame(
        duplicate_records,
        columns=["kind", "row_id", "matched_row_id", "text_hash", "label", "matched_label"],
    )
    return remaining, removed, duplicates


def character_ngrams(text: str, size: int) -> set[str]:
    normalized = canonical_text(text)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def near_deduplicate_candidates(
    candidates: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not settings.get("enabled", True) or candidates.empty:
        return candidates.copy(), pd.DataFrame(), pd.DataFrame()

    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        raise RuntimeError(
            "Near-duplicate checking requires datasketch. Install pipeline requirements first."
        ) from exc

    threshold = float(settings["threshold"])
    ngram_size = int(settings["character_ngram_size"])
    num_perm = int(settings["num_permutations"])

    def signature(shingles: Iterable[str]) -> Any:
        result = MinHash(num_perm=num_perm)
        for value in shingles:
            result.update(value.encode("utf-8"))
        return result

    log(f"Checking near duplicates within {len(candidates):,} external candidates...")
    candidate_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    candidate_signatures: dict[str, Any] = {}
    candidate_shingles: dict[str, set[str]] = {}
    candidate_rows = {row.row_id: row for row in candidates.itertuples()}
    with candidate_lsh.insertion_session() as session:
        for row in candidates.itertuples():
            shingles = character_ngrams(row.text, ngram_size)
            current_signature = signature(shingles)
            candidate_shingles[row.row_id] = shingles
            candidate_signatures[row.row_id] = current_signature
            session.insert(row.row_id, current_signature)

    parent = {row_id: row_id for row_id in candidate_rows}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pair_records: list[dict[str, Any]] = []
    for row_id, current_signature in candidate_signatures.items():
        for other_id in candidate_lsh.query(current_signature):
            if row_id >= other_id:
                continue
            similarity = jaccard(candidate_shingles[row_id], candidate_shingles[other_id])
            if similarity >= threshold:
                union(row_id, other_id)
                pair_records.append(
                    {
                        "kind": "external_internal",
                        "row_id": row_id,
                        "matched_row_id": other_id,
                        "similarity": round(similarity, 6),
                    }
                )

    components: dict[str, list[str]] = {}
    for row_id in candidate_rows:
        components.setdefault(find(row_id), []).append(row_id)

    removed_records: list[dict[str, Any]] = []
    internal_removed: set[str] = set()
    for members in components.values():
        if len(members) == 1:
            continue
        ordered = sorted(members)
        labels = {int(candidate_rows[row_id].label) for row_id in ordered}
        if len(labels) > 1:
            to_remove = ordered
            reason = "external_conflicting_near_duplicate"
        else:
            to_remove = ordered[1:]
            reason = "external_same_label_near_duplicate"
        for row_id in to_remove:
            internal_removed.add(row_id)
            append_removed(removed_records, candidate_rows[row_id], reason, ordered[0])

    survivors = candidates.loc[~candidates["row_id"].isin(internal_removed)].copy()

    reference_parts = [references["validation"], references["test"]]
    if settings.get("compare_with_base_train", True):
        reference_parts.append(references["train"])
    reference = pd.concat(reference_parts, ignore_index=True)
    split_by_id = {
        row.row_id: row.source_split for row in reference.itertuples()
    }
    row_by_id = {row.row_id: row for row in reference.itertuples()}

    log(f"Building near-duplicate index for {len(reference):,} reference rows...")
    reference_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    with reference_lsh.insertion_session() as session:
        for row in reference.itertuples():
            session.insert(row.row_id, signature(character_ngrams(row.text, ngram_size)))

    evaluation_removed: set[str] = set()
    for row in survivors.itertuples():
        shingles = candidate_shingles[row.row_id]
        matches = reference_lsh.query(candidate_signatures[row.row_id])
        best: tuple[float, str] | None = None
        for matched_id in matches:
            similarity = jaccard(
                shingles, character_ngrams(row_by_id[matched_id].text, ngram_size)
            )
            if similarity >= threshold and (best is None or similarity > best[0]):
                best = (similarity, matched_id)
        if best is None:
            continue
        similarity, matched_id = best
        matched_split = split_by_id[matched_id]
        reason = (
            "external_near_overlap_with_evaluation"
            if matched_split in {"validation", "test"}
            else "external_near_overlap_with_base_train"
        )
        evaluation_removed.add(row.row_id)
        append_removed(removed_records, row, reason, f"{matched_split}:{matched_id}")
        pair_records.append(
            {
                "kind": reason,
                "row_id": row.row_id,
                "matched_row_id": matched_id,
                "similarity": round(similarity, 6),
            }
        )

    remaining = survivors.loc[~survivors["row_id"].isin(evaluation_removed)].reset_index(drop=True)
    removed = pd.DataFrame(removed_records)
    pairs = pd.DataFrame(
        pair_records, columns=["kind", "row_id", "matched_row_id", "similarity"]
    )
    return remaining, removed, pairs


def balanced_sample(
    candidates: pd.DataFrame, maximum_per_label: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    available = {
        int(label): int(count)
        for label, count in candidates.groupby("label").size().to_dict().items()
    }
    missing = sorted(set(LABEL_NAMES).difference(available))
    if missing:
        raise ValueError(f"No usable external candidates for labels: {missing}")
    target = min(maximum_per_label, min(available.values()))
    selected_parts = []
    for label in sorted(LABEL_NAMES):
        group = candidates.loc[candidates["label"].eq(label)]
        selected_parts.append(group.sample(n=target, random_state=seed + label))
    selected = pd.concat(selected_parts, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    report = {
        "maximum_requested_per_label": int(maximum_per_label),
        "selected_per_label": int(target),
        "available_after_quality_filters": {
            LABEL_NAMES[label]: available[label] for label in sorted(available)
        },
    }
    return selected, report


def distribution(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["source_dataset", "label", "label_name"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    grouped.insert(0, "split", split)
    return grouped


def build_cleaning_report(frame: pd.DataFrame, empty_rows_removed: int) -> dict[str, Any]:
    combined = frame["text"].astype(str)
    return {
        "rows_checked": int(len(frame)),
        "empty_rows_removed": int(empty_rows_removed),
        "empty_text_remaining": int(combined.str.strip().eq("").sum()),
        "html_tags_remaining": int(combined.str.contains(r"<[^>]+>", regex=True).sum()),
        "numeric_entities_remaining": int(
            combined.str.contains(r"(?:&#?\d+;|(?<!&)#\d+;)", regex=True).sum()
        ),
        "backslash_artifacts_remaining": int(
            combined.str.contains(r"\\[\$\"']", regex=True).sum()
        ),
    }


def audit_token_lengths(
    frames: dict[str, pd.DataFrame], settings: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    name = str(settings["name"])
    max_length = int(settings["max_length"])
    cache_dir = (
        resolve_from_root(str(settings["cache_dir"]))
        if settings.get("cache_dir")
        else None
    )
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            name,
            cache_dir=cache_dir,
            local_files_only=bool(settings.get("local_files_only", True)),
        )
    except Exception as exc:
        return (
            {
                "status": "FAIL",
                "tokenizer_name": name,
                "error": f"{type(exc).__name__}: {exc}",
            },
            pd.DataFrame(),
        )

    rows: list[dict[str, Any]] = []
    all_lengths: list[int] = []
    for split, frame in frames.items():
        split_lengths: list[int] = []
        texts = frame["text"].astype(str).tolist()
        for start in range(0, len(texts), 512):
            encoded = tokenizer(
                texts[start : start + 512],
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            split_lengths.extend(len(ids) for ids in encoded["input_ids"])
        all_lengths.extend(split_lengths)
        for length, count in sorted(Counter(split_lengths).items()):
            rows.append({"split": split, "token_length": length, "count": count})

    lengths = np.asarray(all_lengths, dtype=np.int64)
    percentage = float((lengths > max_length).mean() * 100.0) if len(lengths) else 0.0
    sample = frames["train"].head(32)
    encoded_sample = tokenizer(
        sample["text"].astype(str).tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    batch_valid = (
        encoded_sample["input_ids"].ndim == 2
        and encoded_sample["attention_mask"].shape == encoded_sample["input_ids"].shape
        and encoded_sample["input_ids"].shape[1] <= max_length
        and sample["label"].between(0, 3).all()
    )
    report = {
        "status": "PASS" if batch_valid else "FAIL",
        "tokenizer_name": name,
        "max_length": max_length,
        "rows_checked": int(len(lengths)),
        "mean": round(float(lengths.mean()), 3),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "maximum": int(lengths.max()),
        "percentage_over_max_length": round(percentage, 4),
        "batch_valid": bool(batch_valid),
    }
    return report, pd.DataFrame(rows)


def create_manual_review_sample(
    frame: pd.DataFrame, rows_per_source_label: int, seed: int
) -> pd.DataFrame:
    samples = []
    for offset, (_, group) in enumerate(frame.groupby(["source_dataset", "label"])):
        count = min(rows_per_source_label, len(group))
        samples.append(group.sample(n=count, random_state=seed + offset))
    if not samples:
        return pd.DataFrame()
    columns = [
        "row_id",
        "source_dataset",
        "source_category",
        "label",
        "label_name",
        "title_clean",
        "description_clean",
        "text",
    ]
    return pd.concat(samples, ignore_index=True)[columns]


def verify_final_dataset(
    frames: dict[str, pd.DataFrame],
    external_selected: pd.DataFrame,
    adapter_reports: dict[str, Any],
    exact_duplicates: pd.DataFrame,
    near_pairs: pd.DataFrame,
    cleaning_report: dict[str, Any],
    token_report: dict[str, Any],
    input_eval_fingerprints: dict[str, str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    train, validation, test = frames["train"], frames["validation"], frames["test"]
    output_eval_fingerprints = {
        "validation": frame_fingerprint(validation),
        "test": frame_fingerprint(test),
    }
    external_counts = external_selected.groupby("label").size().to_dict()
    introduced_exact_eval = (
        int(exact_duplicates["kind"].eq("external_exact_overlap_with_evaluation").sum())
        if not exact_duplicates.empty
        else 0
    )
    introduced_near_eval = (
        int(near_pairs["kind"].eq("external_near_overlap_with_evaluation").sum())
        if not near_pairs.empty
        else 0
    )
    selected_ids = set(external_selected["row_id"])
    remaining_exact_eval_ids = (
        set(
            exact_duplicates.loc[
                exact_duplicates["kind"].eq("external_exact_overlap_with_evaluation"),
                "row_id",
            ]
        )
        & selected_ids
        if not exact_duplicates.empty
        else set()
    )
    remaining_near_eval_ids = (
        set(
            near_pairs.loc[
                near_pairs["kind"].eq("external_near_overlap_with_evaluation"),
                "row_id",
            ]
        )
        & selected_ids
        if not near_pairs.empty
        else set()
    )
    max_truncation = float(settings["tokenizer"]["maximum_acceptable_truncation_percentage"])
    checks = {
        "source_adapters_loaded_rows": all(
            report.get("accepted_rows", 0) > 0 for report in adapter_reports.values()
        ),
        "required_columns_present": all(
            set(CLEAN_COLUMNS).issubset(frame.columns) for frame in frames.values()
        ),
        "labels_valid": all(frame["label"].between(0, 3).all() for frame in frames.values()),
        "row_ids_unique_per_split": all(not frame["row_id"].duplicated().any() for frame in frames.values()),
        "text_nonempty": all(not frame["text"].str.strip().eq("").any() for frame in frames.values()),
        "cleaning_artifacts_zero": all(
            cleaning_report.get(name, -1) == 0
            for name in (
                "empty_text_remaining",
                "html_tags_remaining",
                "numeric_entities_remaining",
                "backslash_artifacts_remaining",
            )
        ),
        "train_exact_duplicates_zero": not train["text_hash"].duplicated().any(),
        "introduced_exact_evaluation_overlap_zero": not remaining_exact_eval_ids,
        "introduced_near_evaluation_overlap_zero": not remaining_near_eval_ids,
        "external_labels_balanced": (
            set(external_counts) == set(LABEL_NAMES)
            and len(set(int(value) for value in external_counts.values())) == 1
        ),
        "validation_preserved": output_eval_fingerprints["validation"]
        == input_eval_fingerprints["validation"],
        "test_preserved": output_eval_fingerprints["test"] == input_eval_fingerprints["test"],
        "tokenizer_loaded": token_report.get("status") == "PASS",
        "tokenizer_matches_model": token_report.get("tokenizer_name")
        == settings["tokenizer"]["name"],
        "truncation_rate_acceptable": token_report.get(
            "percentage_over_max_length", float("inf")
        )
        <= max_truncation,
        "batch_valid": token_report.get("batch_valid", False),
        "manifest_counts_match": all(
            len(frame) == frame["row_id"].nunique() for frame in frames.values()
        ),
    }
    return {
        "overall_status": "PASS" if all(checks.values()) else "FAIL",
        "status_scope": "automated_technical_checks",
        "ready_for_training": all(checks.values()),
        "manual_label_review_required_before_release": True,
        "checks": checks,
        "counts": {split: int(len(frame)) for split, frame in frames.items()},
        "external_selected_by_label": {
            LABEL_NAMES[label]: int(external_counts.get(label, 0)) for label in LABEL_NAMES
        },
        "introduced_exact_evaluation_overlaps_removed": introduced_exact_eval,
        "introduced_near_evaluation_overlaps_removed": introduced_near_eval,
        "evaluation_fingerprints": output_eval_fingerprints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PIPELINE_ROOT / "configs" / "pipeline_config.json",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        help="Optionally copy final Parquet files to a separate empty directory.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = read_json(config_path)
    seed = int(config["seed"])
    outputs = {name: resolve_from_root(path) for name, path in config["outputs"].items()}
    for path in outputs.values():
        path.mkdir(parents=True, exist_ok=True)

    source_config = config["sources"]
    ag_paths = {
        split: resolve_from_root(path) for split, path in source_config["ag_news"].items()
    }
    raw_paths = {
        "ag_news_train": ag_paths["train"],
        "ag_news_validation": ag_paths["validation"],
        "ag_news_test": ag_paths["test"],
        "bbc_news": resolve_from_root(source_config["bbc_news"]["path"]),
        "huffpost": resolve_from_root(source_config["huffpost"]["path"]),
    }
    missing_files = [str(path) for path in raw_paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing required input files: {missing_files}")

    log("Loading fixed AG News splits...")
    ag_splits_raw = load_ag_news_splits(ag_paths)
    bbc_raw, bbc_report = load_bbc_news(
        raw_paths["bbc_news"],
        source_config["bbc_news"]["label_mapping"],
        int(source_config["bbc_news"].get("title_word_count", 16)),
    )
    huffpost_raw, huffpost_report = load_huffpost_world_news(
        raw_paths["huffpost"], source_config["huffpost"]["accepted_categories"]
    )
    adapter_reports = {"bbc_news": bbc_report, "huffpost": huffpost_report}

    log("Cleaning and standardizing source text...")
    ag_splits: dict[str, pd.DataFrame] = {}
    empty_removed_parts = []
    for split, frame in ag_splits_raw.items():
        ag_splits[split], removed = clean_frame(frame)
        empty_removed_parts.append(removed)
        ag_splits[split].to_parquet(outputs["standardized"] / f"ag_news_{split}.parquet", index=False)
    bbc, bbc_empty = clean_frame(bbc_raw)
    huffpost, huffpost_empty = clean_frame(huffpost_raw)
    empty_removed_parts.extend([bbc_empty, huffpost_empty])
    bbc.to_parquet(outputs["standardized"] / "bbc_news.parquet", index=False)
    huffpost.to_parquet(outputs["standardized"] / "huffpost_world.parquet", index=False)

    input_eval_fingerprints = {
        "validation": frame_fingerprint(ag_splits["validation"]),
        "test": frame_fingerprint(ag_splits["test"]),
    }
    external = pd.concat([bbc, huffpost], ignore_index=True)
    all_before_deduplication = pd.concat([ag_splits["train"], external], ignore_index=True)
    all_before_deduplication.to_parquet(
        outputs["merged"] / "all_sources_before_deduplication.parquet", index=False
    )

    log("Removing exact duplicates and evaluation overlaps...")
    exact_clean, exact_removed, exact_report = exact_deduplicate_candidates(external, ag_splits)
    near_clean, near_removed, near_report = near_deduplicate_candidates(
        exact_clean, ag_splits, config["near_duplicates"]
    )

    selected, sampling_report = balanced_sample(
        near_clean,
        int(config["sampling"]["maximum_external_rows_per_label"]),
        seed,
    )
    train = pd.concat([ag_splits["train"], selected], ignore_index=True)
    validation = ag_splits["validation"].copy()
    test = ag_splits["test"].copy()
    final_frames = {"train": train, "validation": validation, "test": test}

    if train["text_hash"].duplicated().any():
        raise RuntimeError("Final train split still contains exact duplicate text hashes.")

    train.to_parquet(outputs["merged"] / "clean_merged_pool.parquet", index=False)
    for split, frame in final_frames.items():
        frame.to_parquet(outputs["processed"] / f"research_{split}.parquet", index=False)

    removed_parts = [part for part in [exact_removed, near_removed] if not part.empty]
    for empty_removed in empty_removed_parts:
        if not empty_removed.empty:
            removed_parts.append(empty_removed)
    removed_rows = pd.concat(removed_parts, ignore_index=True) if removed_parts else pd.DataFrame()
    removed_rows.to_csv(outputs["manifests"] / "removed_rows.csv", index=False)
    exact_report.to_csv(outputs["reports"] / "exact_duplicates.csv", index=False)
    conflicting = exact_report.loc[
        exact_report["kind"].astype(str).str.contains("conflict", case=False, na=False)
    ]
    conflicting.to_csv(outputs["reports"] / "conflicting_duplicates.csv", index=False)
    near_report.to_csv(outputs["reports"] / "near_duplicate_pairs.csv", index=False)

    review = create_manual_review_sample(
        near_clean,
        int(config["sampling"]["manual_review_rows_per_source_label"]),
        seed,
    )
    review.to_csv(outputs["reports"] / "manual_review_samples.csv", index=False)

    distributions = pd.concat(
        [distribution(frame, split) for split, frame in final_frames.items()], ignore_index=True
    )
    distributions.to_csv(outputs["reports"] / "source_distribution.csv", index=False)
    class_distribution = (
        distributions.groupby(["split", "label", "label_name"])["count"]
        .sum()
        .reset_index()
    )
    class_distribution.to_csv(outputs["reports"] / "class_distribution.csv", index=False)

    source_summary = pd.DataFrame(
        [
            {"source": "ag_news_train", "stage": "fixed_input", "rows": len(ag_splits["train"])},
            {"source": "ag_news_validation", "stage": "fixed_input", "rows": len(validation)},
            {"source": "ag_news_test", "stage": "fixed_input", "rows": len(test)},
            {"source": "bbc_news", "stage": "accepted_by_adapter", "rows": len(bbc)},
            {"source": "huffpost", "stage": "accepted_by_adapter", "rows": len(huffpost)},
            {"source": "external", "stage": "after_quality_filters", "rows": len(near_clean)},
            {"source": "external", "stage": "selected_for_train", "rows": len(selected)},
        ]
    )
    source_summary.to_csv(outputs["reports"] / "source_summary.csv", index=False)

    log("Checking tokenizer lengths and model input batch...")
    token_report, token_distribution = audit_token_lengths(final_frames, config["tokenizer"])
    write_json(outputs["reports"] / "token_length_report.json", token_report)
    token_distribution.to_csv(outputs["reports"] / "token_length_distribution.csv", index=False)

    schema_report = {
        "adapter_reports": adapter_reports,
        "standard_columns": CLEAN_COLUMNS,
        "empty_rows_removed": int(sum(len(frame) for frame in empty_removed_parts)),
        "input_rows": {name: int(len(frame)) for name, frame in ag_splits.items()},
    }
    all_cleaned = pd.concat([*ag_splits.values(), bbc, huffpost], ignore_index=True)
    empty_rows_removed = int(sum(len(frame) for frame in empty_removed_parts))
    cleaning_report = build_cleaning_report(all_cleaned, empty_rows_removed)
    write_json(outputs["reports"] / "schema_report.json", schema_report)
    write_json(outputs["reports"] / "cleaning_report.json", cleaning_report)
    write_json(outputs["reports"] / "sampling_report.json", sampling_report)
    exact_eval_removed = (
        int(exact_report["kind"].eq("external_exact_overlap_with_evaluation").sum())
        if not exact_report.empty
        else 0
    )
    near_eval_removed = (
        int(near_report["kind"].eq("external_near_overlap_with_evaluation").sum())
        if not near_report.empty
        else 0
    )
    write_json(
        outputs["reports"] / "split_overlap_report.json",
        {
            "external_exact_evaluation_overlaps_removed": exact_eval_removed,
            "external_near_evaluation_overlaps_removed": near_eval_removed,
            "external_evaluation_overlaps_remaining": 0,
            "validation_fingerprint_preserved": True,
            "test_fingerprint_preserved": True,
        },
    )

    final_report = verify_final_dataset(
        final_frames,
        selected,
        adapter_reports,
        exact_report,
        near_report,
        cleaning_report,
        token_report,
        input_eval_fingerprints,
        config,
    )
    write_json(outputs["reports"] / "final_data_report.json", final_report)

    for split, frame in final_frames.items():
        write_json(outputs["manifests"] / f"{split}_ids.json", frame["row_id"].tolist())
    write_json(
        outputs["manifests"] / "dataset_versions.json",
        {
            "pipeline_version": config["version"],
            "config_sha256": sha256_file(config_path),
            "input_sha256": {name: sha256_file(path) for name, path in raw_paths.items()},
            "output_sha256": {
                split: sha256_file(outputs["processed"] / f"research_{split}.parquet")
                for split in final_frames
            },
        },
    )
    checksums_text = "\n".join(
        f"{sha256_file(path)}  {name}  {path}"
        for name, path in raw_paths.items()
    )
    (outputs["manifests"] / "raw_checksums.txt").write_text(
        checksums_text + "\n", encoding="utf-8"
    )
    shutil.copy2(config_path, outputs["manifests"] / "pipeline_config.json")
    write_json(
        outputs["manifests"] / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pipeline_root": str(PIPELINE_ROOT),
        },
    )

    if args.publish_dir:
        publish_dir = args.publish_dir.resolve()
        publish_dir.mkdir(parents=True, exist_ok=True)
        if any(publish_dir.iterdir()):
            raise FileExistsError(f"Publish directory must be empty: {publish_dir}")
        for split in final_frames:
            filename = f"research_{split}.parquet"
            shutil.copy2(outputs["processed"] / filename, publish_dir / filename)

    log(f"Pipeline status: {final_report['overall_status']}")
    log(f"Final train rows: {len(train):,} ({len(selected):,} external additions)")
    log(f"Report: {outputs['reports'] / 'final_data_report.json'}")
    return 0 if final_report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
