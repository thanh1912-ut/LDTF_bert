"""AG News data loading: CSV/Parquet input, dynamic padding, seeded loaders.

The official test split is deliberately unreachable from this module's public
loader factory. See :mod:`src.guard`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import config
from .utils import seed_worker


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_split(path: str | Path, *, _guard_ok: bool = False) -> pd.DataFrame:
    """Read a split from Parquet or CSV.

    Reading the official test parquet through this function is refused unless
    the caller is :mod:`src.guard`.
    """
    target = Path(path)
    if not _guard_ok:
        from .guard import assert_not_official_test

        assert_not_official_test(target)
    if not target.exists():
        raise FileNotFoundError(f"Expected a data file at {target}, but it does not exist.")

    suffix = target.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(target)
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(target)
    else:
        raise ValueError(f"Unsupported data format {suffix!r}; expected .parquet or .csv.")
    return frame.reset_index(drop=True)


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a data file, for checkpoint provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Encoded:
    """One tokenized example. Sequence tensors are unpadded (variable length)."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    special_tokens_mask: torch.Tensor
    label: torch.Tensor


class AgNewsDataset(Dataset):
    """Tokenizes AG News rows on access, without padding.

    Padding is applied per batch by :class:`DynamicPaddingCollator`, so the
    truncation policy is the only length policy applied here.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        *,
        max_length: int = config.MAX_LENGTH,
        text_encoding: str = config.TEXT_ENCODING,
        split_name: str = "unknown",
    ) -> None:
        super().__init__()
        if text_encoding not in {"single", "pair"}:
            raise ValueError(f"text_encoding must be 'single' or 'pair', got {text_encoding!r}.")
        if config.LABEL_COLUMN not in frame.columns:
            raise ValueError(
                f"Missing '{config.LABEL_COLUMN}' column; got {list(frame.columns)}."
            )

        self.text_encoding = text_encoding
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.max_length = max_length

        if text_encoding == "single":
            if config.TEXT_COLUMN not in frame.columns:
                raise ValueError(
                    f"Missing '{config.TEXT_COLUMN}' column; got {list(frame.columns)}."
                )
            self.first_texts: list[str] = frame[config.TEXT_COLUMN].astype(str).tolist()
            self.second_texts: list[str] | None = None
        else:
            missing = [
                column
                for column in (config.TITLE_COLUMN, config.DESCRIPTION_COLUMN)
                if column not in frame.columns
            ]
            if missing:
                raise ValueError(f"Pair encoding requires columns {missing}, which are absent.")
            self.first_texts = frame[config.TITLE_COLUMN].astype(str).tolist()
            self.second_texts = frame[config.DESCRIPTION_COLUMN].astype(str).tolist()

        labels = frame[config.LABEL_COLUMN].astype(int)
        if not labels.between(0, config.NUM_CLASSES - 1).all():
            raise ValueError(
                f"Labels must lie in [0, {config.NUM_CLASSES - 1}]; found "
                f"[{int(labels.min())}, {int(labels.max())}]."
            )
        self.labels: list[int] = labels.tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Encoded:
        if self.second_texts is None:
            encoded = self.tokenizer(
                self.first_texts[index],
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_special_tokens_mask=True,
            )
        else:
            encoded = self.tokenizer(
                self.first_texts[index],
                self.second_texts[index],
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_special_tokens_mask=True,
            )
        token_type_ids = encoded.get("token_type_ids")
        return Encoded(
            input_ids=torch.tensor(encoded["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(encoded["attention_mask"], dtype=torch.long),
            token_type_ids=(
                torch.tensor(token_type_ids, dtype=torch.long)
                if token_type_ids is not None
                else None
            ),
            special_tokens_mask=torch.tensor(
                encoded["special_tokens_mask"], dtype=torch.long
            ),
            label=torch.tensor(self.labels[index], dtype=torch.long),
        )


class DynamicPaddingCollator:
    """Pad a batch to its own longest sequence."""

    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: list[Encoded]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate an empty batch.")
        max_length = max(int(item.input_ids.numel()) for item in batch)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        def pad(tensor: torch.Tensor, value: int) -> torch.Tensor:
            missing = max_length - int(tensor.numel())
            if missing == 0:
                return tensor
            return torch.cat(
                [tensor, torch.full((missing,), value, dtype=tensor.dtype)], dim=0
            )

        collated = {
            "input_ids": torch.stack([pad(item.input_ids, self.pad_token_id) for item in batch]),
            "attention_mask": torch.stack([pad(item.attention_mask, 0) for item in batch]),
            # Padding positions count as "special" so they are never routable.
            "special_tokens_mask": torch.stack(
                [pad(item.special_tokens_mask, 1) for item in batch]
            ),
            "labels": torch.stack([item.label for item in batch]),
        }
        if batch[0].token_type_ids is not None:
            collated["token_type_ids"] = torch.stack(
                [pad(item.token_type_ids, 0) for item in batch]
            )
        return collated


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------
def _build_loader(
    frame: pd.DataFrame,
    tokenizer,
    *,
    batch_size: int,
    shuffle: bool,
    split_name: str,
    seed: int,
    num_workers: int,
    text_encoding: str,
    max_length: int,
) -> DataLoader:
    dataset = AgNewsDataset(
        frame,
        tokenizer,
        max_length=max_length,
        text_encoding=text_encoding,
        split_name=split_name,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id),
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    # Used by the training engine to assert it was not handed the test split.
    loader.split_name = split_name  # type: ignore[attr-defined]
    return loader


def build_train_dataloader(
    frame: pd.DataFrame,
    tokenizer,
    *,
    batch_size: int | None = None,
    seed: int = config.SEED,
    num_workers: int = config.NUM_WORKERS,
    text_encoding: str = config.TEXT_ENCODING,
    max_length: int = config.MAX_LENGTH,
) -> DataLoader:
    """Return a shuffled training loader with a seeded generator."""
    return _build_loader(
        frame,
        tokenizer,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=True,
        split_name="train",
        seed=seed,
        num_workers=num_workers,
        text_encoding=text_encoding,
        max_length=max_length,
    )


def build_eval_dataloader(
    frame: pd.DataFrame,
    tokenizer,
    *,
    batch_size: int | None = None,
    split_name: str = "validation",
    seed: int = config.SEED,
    num_workers: int = config.NUM_WORKERS,
    text_encoding: str = config.TEXT_ENCODING,
    max_length: int = config.MAX_LENGTH,
) -> DataLoader:
    """Return a non-shuffled evaluation loader."""
    return _build_loader(
        frame,
        tokenizer,
        batch_size=batch_size or config.EVAL_BATCH_SIZE,
        shuffle=False,
        split_name=split_name,
        seed=seed,
        num_workers=num_workers,
        text_encoding=text_encoding,
        max_length=max_length,
    )


def subsample_stratified(
    frame: pd.DataFrame, num_rows: int, *, seed: int = config.SEED
) -> pd.DataFrame:
    """Return a class-balanced subset of *frame* with about *num_rows* rows.

    Intended for smoke runs and debugging only. Callers must record that a
    subset was used so such runs are never mistaken for full results.
    """
    if num_rows >= len(frame):
        return frame
    per_class = max(1, num_rows // config.NUM_CLASSES)
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in frame.groupby(config.LABEL_COLUMN)
    ]
    return (
        pd.concat(parts)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def build_dataloaders(
    tokenizer,
    *,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    seed: int = config.SEED,
    num_workers: int = config.NUM_WORKERS,
    text_encoding: str = config.TEXT_ENCODING,
    max_length: int = config.MAX_LENGTH,
    limit_train_rows: int | None = None,
) -> dict[str, object]:
    """Return the train and validation loaders. Never returns a test loader."""
    train_frame = load_split(config.PROCESSED_TRAIN)
    val_frame = load_split(config.PROCESSED_VAL)
    if limit_train_rows is not None:
        train_frame = subsample_stratified(train_frame, limit_train_rows, seed=seed)
    return {
        "train": build_train_dataloader(
            train_frame,
            tokenizer,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            text_encoding=text_encoding,
            max_length=max_length,
        ),
        "validation": build_eval_dataloader(
            val_frame,
            tokenizer,
            batch_size=eval_batch_size,
            split_name="validation",
            seed=seed,
            num_workers=num_workers,
            text_encoding=text_encoding,
            max_length=max_length,
        ),
        "train_size": len(train_frame),
        "val_size": len(val_frame),
    }


def data_signature() -> dict[str, str]:
    """Return checksums identifying the train/validation data used by a run."""
    return {
        "train_sha256": file_sha256(config.PROCESSED_TRAIN),
        "validation_sha256": file_sha256(config.PROCESSED_VAL),
        "text_encoding": config.TEXT_ENCODING,
        "max_length": str(config.MAX_LENGTH),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def describe_split(frame: pd.DataFrame, tokenizer=None, *, name: str = "split") -> dict[str, object]:
    """Return counts, class balance, missing values and duplicate statistics."""
    report: dict[str, object] = {
        "name": name,
        "num_rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    if config.LABEL_COLUMN in frame.columns:
        counts = Counter(frame[config.LABEL_COLUMN].astype(int).tolist())
        report["class_distribution"] = {
            config.LABEL_NAMES[label]: int(counts.get(label, 0))
            for label in range(config.NUM_CLASSES)
        }
    if config.TEXT_COLUMN in frame.columns:
        text = frame[config.TEXT_COLUMN].astype(str)
        report["missing_text"] = int(text.str.strip().eq("").sum())
        report["exact_duplicate_texts"] = int(text.duplicated().sum())
        if config.LABEL_COLUMN in frame.columns:
            pairs = frame[[config.TEXT_COLUMN, config.LABEL_COLUMN]]
            grouped = pairs.groupby(config.TEXT_COLUMN)[config.LABEL_COLUMN].nunique()
            report["conflicting_duplicate_texts"] = int((grouped > 1).sum())
        if tokenizer is not None:
            lengths = np.array(
                [
                    len(tokenizer(value, truncation=False)["input_ids"])
                    for value in text.tolist()
                ]
            )
            report["token_lengths"] = {
                "mean": float(lengths.mean()),
                "median": float(np.median(lengths)),
                "p95": float(np.percentile(lengths, 95)),
                "p99": float(np.percentile(lengths, 99)),
                "max": int(lengths.max()),
                "truncation_rate_at_max_length": float(
                    (lengths > config.MAX_LENGTH).mean()
                ),
            }
    return report
