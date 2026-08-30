"""CLI contract tests for distributed launch behavior."""

from __future__ import annotations

import pytest

from experiments.final_eval import main as final_eval_main
from experiments.run_experiment import parse_args as parse_experiment_args
from experiments.run_suite import cached_summary_matches


def test_experiment_batch_size_is_global_cli_value():
    args = parse_experiment_args(["--run", "A0", "--batch-size", "32"])
    assert args.batch_size == 32
    assert args.eval_batch_size == 64


def test_final_eval_rejects_torchrun_before_touching_the_seal(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="plain Python"):
        final_eval_main(["--run", "A0_seed42", "--reason", "unit test"])


def test_suite_cache_requires_the_same_runtime_and_data_protocol():
    runtime = {"world_size": 2, "global_batch_size": 32}
    signature = {"train_sha256": "abc", "limit_train_rows": "none"}
    existing = {
        "base_run": "A0",
        "seed": 42,
        "backbone_frozen": False,
        "epochs": 3,
        "is_debug_subset": False,
        "runtime": dict(runtime),
        "data_signature": dict(signature),
    }
    kwargs = {
        "base_run": "A0",
        "seed": 42,
        "frozen": False,
        "epochs": 3,
        "is_debug_subset": False,
        "expected_runtime": runtime,
        "expected_signature": signature,
    }
    assert cached_summary_matches(existing, **kwargs)
    incompatible_runtime = {**runtime, "world_size": 1}
    assert not cached_summary_matches(
        existing, **{**kwargs, "expected_runtime": incompatible_runtime}
    )
    assert not cached_summary_matches(existing, **{**kwargs, "is_debug_subset": True})
