"""Tests for the data pipeline: encoding, dynamic padding, seeding, guards."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src import config, guard
from src.dataset import (
    AgNewsDataset,
    AgNewsTextDataset,
    BatchedTokenizingCollator,
    DistributedEvalSampler,
    DynamicPaddingCollator,
    build_dataloaders,
    build_eval_dataloader,
    build_train_dataloader,
    describe_split,
    load_split,
    per_device_batch_size,
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "Stocks climb after the central bank holds rates steady.",
                "Short.",
                "A much longer sports report describing the final minutes of the game "
                "in considerable and frankly excessive detail for a test fixture.",
                "New chip architecture announced.",
            ],
            "title_clean": ["Stocks climb", "Short", "Sports report", "New chip"],
            "description_clean": [
                "after the central bank holds rates steady",
                "text",
                "describing the final minutes",
                "architecture announced",
            ],
            "label": [2, 1, 1, 3],
        }
    )


def test_dataset_returns_special_tokens_mask(frame, tokenizer):
    dataset = AgNewsDataset(frame, tokenizer, split_name="train")
    sample = dataset[0]
    assert sample.special_tokens_mask.shape == sample.input_ids.shape
    assert int(sample.special_tokens_mask[0]) == 1  # [CLS]
    assert int(sample.special_tokens_mask[-1]) == 1  # [SEP]


def test_dataset_is_unpadded(frame, tokenizer):
    dataset = AgNewsDataset(frame, tokenizer)
    lengths = {int(dataset[index].input_ids.numel()) for index in range(len(dataset))}
    assert len(lengths) > 1, "samples must keep their natural lengths"


def test_pair_encoding_produces_token_type_ids(frame, tokenizer):
    dataset = AgNewsDataset(frame, tokenizer, text_encoding="pair")
    sample = dataset[0]
    assert sample.token_type_ids is not None
    assert int(sample.token_type_ids.max()) == 1, "second segment must be marked"


def test_single_encoding_has_one_segment(frame, tokenizer):
    dataset = AgNewsDataset(frame, tokenizer, text_encoding="single")
    assert int(dataset[0].token_type_ids.max()) == 0


def test_dynamic_padding_pads_to_batch_longest(frame, tokenizer):
    dataset = AgNewsDataset(frame, tokenizer)
    collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator([dataset[0], dataset[1]])
    expected = max(int(dataset[0].input_ids.numel()), int(dataset[1].input_ids.numel()))
    assert batch["input_ids"].shape == (2, expected)
    assert batch["input_ids"].shape[1] < config.MAX_LENGTH


@pytest.mark.parametrize("text_encoding", ["single", "pair"])
def test_batched_tokenization_matches_legacy_path(frame, tokenizer, text_encoding):
    legacy_dataset = AgNewsDataset(frame, tokenizer, text_encoding=text_encoding)
    legacy = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)(
        [legacy_dataset[index] for index in range(len(legacy_dataset))]
    )
    raw_dataset = AgNewsTextDataset(frame, text_encoding=text_encoding)
    batched = BatchedTokenizingCollator(tokenizer)(
        [raw_dataset[index] for index in range(len(raw_dataset))]
    )
    assert set(batched) == set(legacy)
    for key in legacy:
        assert torch.equal(batched[key], legacy[key]), key


def test_batched_tokenizer_can_pad_to_a_multiple(frame, tokenizer):
    dataset = AgNewsTextDataset(frame)
    batch = BatchedTokenizingCollator(tokenizer, pad_to_multiple_of=8)(
        [dataset[index] for index in range(len(dataset))]
    )
    assert batch["input_ids"].shape[1] % 8 == 0


def test_padding_positions_are_masked_and_marked_special(frame, tokenizer):
    dataset = AgNewsDataset(frame, tokenizer)
    collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator([dataset[2], dataset[1]])
    short_length = int(dataset[1].input_ids.numel())
    assert int(batch["attention_mask"][1, short_length:].sum()) == 0
    assert int(batch["special_tokens_mask"][1, short_length:].min()) == 1


def test_truncation_respects_max_length(tokenizer):
    long_frame = pd.DataFrame({"text": ["word " * 500], "label": [0]})
    dataset = AgNewsDataset(long_frame, tokenizer, max_length=config.MAX_LENGTH)
    assert int(dataset[0].input_ids.numel()) == config.MAX_LENGTH


def test_invalid_labels_are_rejected(tokenizer):
    bad = pd.DataFrame({"text": ["a"], "label": [9]})
    with pytest.raises(ValueError, match="Labels must lie"):
        AgNewsDataset(bad, tokenizer)


def test_missing_columns_are_rejected(tokenizer):
    with pytest.raises(ValueError, match="label"):
        AgNewsDataset(pd.DataFrame({"text": ["a"]}), tokenizer)


def test_train_loader_shuffling_is_reproducible(frame, tokenizer):
    def first_batch(seed: int):
        loader = build_train_dataloader(frame, tokenizer, batch_size=2, seed=seed)
        return next(iter(loader))["labels"]

    assert torch.equal(first_batch(11), first_batch(11))


def test_eval_loader_is_not_shuffled(frame, tokenizer):
    loader = build_eval_dataloader(frame, tokenizer, batch_size=4)
    labels = next(iter(loader))["labels"]
    assert torch.equal(labels, torch.tensor(frame["label"].tolist()))


def test_loaders_are_tagged_with_their_split(frame, tokenizer):
    assert build_train_dataloader(frame, tokenizer).split_name == "train"
    assert build_eval_dataloader(frame, tokenizer).split_name == "validation"


def test_csv_and_parquet_are_both_supported(frame, tmp_path):
    csv_path = tmp_path / "split.csv"
    parquet_path = tmp_path / "split.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path)
    assert len(load_split(csv_path)) == len(frame)
    assert len(load_split(parquet_path)) == len(frame)


def test_load_split_projects_only_requested_columns(frame, tmp_path):
    parquet_path = tmp_path / "split.parquet"
    frame.to_parquet(parquet_path)
    projected = load_split(parquet_path, columns=["text", "label"])
    assert list(projected.columns) == ["text", "label"]


def test_global_batch_is_partitioned_exactly():
    assert per_device_batch_size(32, 1) == 32
    assert per_device_batch_size(32, 2) == 16
    with pytest.raises(ValueError, match="not divisible"):
        per_device_batch_size(31, 2)


def test_distributed_eval_sampler_has_exact_disjoint_coverage(frame):
    dataset = AgNewsTextDataset(frame)
    left = list(DistributedEvalSampler(dataset, rank=0, world_size=2))
    right = list(DistributedEvalSampler(dataset, rank=1, world_size=2))
    assert set(left).isdisjoint(right)
    assert sorted(left + right) == list(range(len(dataset)))


def test_distributed_train_loader_has_equal_shards_and_records_padding(
    frame, tokenizer
):
    odd_frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    rank_zero = build_train_dataloader(
        odd_frame, tokenizer, batch_size=4, seed=11, rank=0, world_size=2
    )
    rank_one = build_train_dataloader(
        odd_frame, tokenizer, batch_size=4, seed=11, rank=1, world_size=2
    )
    assert len(rank_zero.sampler) == len(rank_one.sampler)
    assert rank_zero.per_device_batch_size == 2
    assert rank_zero.sampler_padding == 1


def test_distributed_loader_workers_use_spawn(frame, tokenizer):
    loader = build_train_dataloader(
        frame,
        tokenizer,
        batch_size=4,
        seed=11,
        num_workers=1,
        rank=0,
        world_size=2,
    )
    assert loader.multiprocessing_context.get_start_method() == "spawn"
    assert next(iter(loader))["labels"].numel() == 2


def test_unsupported_format_is_rejected(tmp_path):
    path = tmp_path / "split.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported data format"):
        load_split(path)


def test_describe_split_reports_balance_and_duplicates(frame):
    report = describe_split(frame, name="fixture")
    assert report["num_rows"] == 4
    assert report["class_distribution"]["Sports"] == 2
    assert report["exact_duplicate_texts"] == 0


# -- official test guard ----------------------------------------------------
def test_load_split_refuses_the_official_test_file():
    with pytest.raises(guard.OfficialTestAccessError):
        load_split(config.PROCESSED_TEST)


def test_build_dataloaders_never_returns_a_test_loader(tokenizer, monkeypatch, frame):
    monkeypatch.setattr("src.dataset.load_split", lambda *a, **k: frame)
    loaders = build_dataloaders(tokenizer, batch_size=2)
    assert set(loaders) == {"train", "validation", "train_size", "val_size"}
    assert "test" not in loaders


def test_official_test_loader_requires_the_unlock_token(tokenizer, monkeypatch):
    monkeypatch.delenv(guard.UNLOCK_ENV_VAR, raising=False)
    with pytest.raises(guard.OfficialTestAccessError, match="sealed"):
        guard.official_test_loader(tokenizer, reason="test", run_id="unit")


def test_official_test_loader_is_blocked_during_training(tokenizer, monkeypatch):
    monkeypatch.setenv(guard.UNLOCK_ENV_VAR, guard.UNLOCK_TOKEN)
    guard.begin_training()
    try:
        with pytest.raises(
            guard.OfficialTestAccessError, match="training loop is active"
        ):
            guard.official_test_loader(tokenizer, reason="test", run_id="unit")
    finally:
        guard.end_training()
