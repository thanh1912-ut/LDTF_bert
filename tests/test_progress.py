"""Tests for the YOLO-style progress reporters.

Reporting must be purely presentational: it may never change a training result.
"""

from __future__ import annotations

import pytest

from src import config
from src.progress import (
    BatchProgress,
    ConfusionView,
    EpochTable,
    SuiteBoard,
    TrainingReporter,
    format_duration,
)


@pytest.fixture
def metrics() -> dict[str, object]:
    return {
        "loss": 0.2431,
        "accuracy": 0.9412,
        "f1_macro": 0.9408,
        "confusion_matrix": [
            [1800, 30, 40, 30],
            [20, 1850, 15, 15],
            [45, 10, 1790, 55],
            [35, 10, 60, 1795],
        ],
        "per_class": {
            "World": {"precision": 0.94, "recall": 0.947, "f1": 0.943, "support": 1900},
            "Sports": {"precision": 0.97, "recall": 0.974, "f1": 0.972, "support": 1900},
            "Business": {"precision": 0.94, "recall": 0.942, "f1": 0.941, "support": 1900},
            "Sci/Tech": {"precision": 0.95, "recall": 0.945, "f1": 0.947, "support": 1900},
        },
    }


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0s"), (45, "45s"), (61, "1:01"), (245, "4:05"), (3600, "1:00:00"), (5025, "1:23:45")],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_format_duration_handles_invalid():
    assert format_duration(float("nan")) == "--"
    assert format_duration(-1) == "--"


def test_epoch_table_lists_every_class_and_a_summary(metrics):
    text = EpochTable(enabled=False).render(metrics)
    for name in config.LABEL_NAMES:
        assert name in text
    assert "all" in text
    assert "0.9408" in text  # macro F1 in the summary row
    assert "0.9412" in text  # accuracy in the tail


def test_epoch_table_marks_the_best_epoch(metrics):
    assert "best" in EpochTable(enabled=False).render(metrics, is_best=True)
    assert "best" not in EpochTable(enabled=False).render(metrics, is_best=False)


def test_epoch_table_tolerates_missing_per_class(metrics):
    text = EpochTable(enabled=False).render({"loss": 1.0, "accuracy": 0.25, "f1_macro": 0.1})
    assert "all" in text


def test_epoch_table_macro_average_is_the_mean_of_per_class(metrics):
    text = EpochTable(enabled=False).render(metrics)
    summary = [line for line in text.splitlines() if line.strip().startswith("all")][0]
    expected = sum(
        metrics["per_class"][name]["precision"] for name in config.LABEL_NAMES
    ) / len(config.LABEL_NAMES)
    assert f"{expected:.4f}" in summary


def test_confusion_view_renders_every_row(metrics):
    text = ConfusionView(enabled=False).render(metrics["confusion_matrix"])
    assert "1800" in text and "1850" in text
    body = [line for line in text.splitlines() if "1800" in line or "1850" in line]
    assert len(body) == 2


def test_suite_board_sorts_by_macro_f1():
    board = SuiteBoard(["A0", "A1", "A3"], enabled=False)
    board.update("A0", status="done", val_f1=0.941, val_accuracy=0.942)
    board.update("A1", status="done", val_f1=0.950, val_accuracy=0.951)
    board.update("A3", status="done", val_f1=0.930, val_accuracy=0.931)
    rows = [line.split()[0] for line in board.render().splitlines()[2:]]
    assert rows == ["A1", "A0", "A3"]


def test_suite_board_places_unfinished_runs_last():
    board = SuiteBoard(["A0", "A1"], enabled=False)
    board.update("A0", status="done", val_f1=0.90, val_accuracy=0.91)
    rows = [line.split()[0] for line in board.render().splitlines()[2:]]
    assert rows == ["A0", "A1"]


def test_suite_board_reports_statuses():
    board = SuiteBoard(["A0"], enabled=False)
    assert "queued" in board.render()
    board.mark_running("A0")
    assert "running" in board.render()
    board.update("A0", status="done", val_f1=0.9, val_accuracy=0.9)
    assert "done" in board.render()


def test_batch_progress_running_loss_is_a_running_mean():
    bars = BatchProgress(total_epochs=1, total_batches=4, enabled=False)
    for value in (1.0, 2.0, 3.0, 4.0):
        bars.update(value, 2e-5, 1e-3)
    assert bars.running_loss == pytest.approx(2.5)


def test_batch_progress_disabled_returns_a_plain_iterator():
    bars = BatchProgress(total_epochs=1, total_batches=3, enabled=False)
    assert list(bars.epoch([1, 2, 3], epoch=1)) == [1, 2, 3]
    bars.close()


def test_reporter_disabled_is_silent(capsys, metrics):
    reporter = TrainingReporter("quiet", total_epochs=1, total_batches=2, enabled=False)
    reporter.start(parameter_counts={"total": {"total": 1, "trainable": 1}})
    list(reporter.epoch_iter([1, 2], epoch=1))
    reporter.batch(0.5, 2e-5, 1e-3)
    reporter.epoch_end(metrics, is_best=True)
    reporter.finish(best_epoch=1, best_metrics={"f1_macro": 0.9, "accuracy": 0.9})
    assert capsys.readouterr().out == ""


def test_reporter_enabled_prints_the_expected_sections(capsys, metrics):
    reporter = TrainingReporter("loud", total_epochs=1, total_batches=2, enabled=True)
    reporter.start(
        parameter_counts={"total": {"total": 109_681_921, "trainable": 790_273, "non_backbone": 790_273}}
    )
    reporter.epoch_end(metrics, is_best=True)
    reporter.finish(
        best_epoch=1,
        best_metrics={"f1_macro": 0.9408, "accuracy": 0.9412},
        confusion_matrix=metrics["confusion_matrix"],
    )
    output = capsys.readouterr().out
    assert "run loud" in output
    assert "109,681,921" in output
    assert "Sci/Tech" in output
    assert "Confusion matrix" in output
    assert "best epoch 1" in output


def test_reporter_does_not_change_training_results(tmp_path):
    """A run with reporting must reproduce a run without it, exactly.

    Regression guard: wrapping a DataLoader in tqdm perturbs the shuffling
    order, so BatchProgress must never wrap the loader.
    """
    import torch

    from src.train import TrainConfig, train_model
    from tests.test_train import TinyModel, make_loader

    def run(show: bool, tag: str):
        torch.manual_seed(0)
        model = TinyModel()
        return train_model(
            model,
            make_loader(),
            make_loader(split_name="validation"),
            TrainConfig(output_dir=tmp_path / tag, run_id="r", epochs=1, use_amp=False),
            show_progress=show,
        )

    quiet, loud = run(False, "quiet"), run(True, "loud")
    # Training is the deterministic part and must match exactly: identical seed,
    # identical data order, identical updates.
    assert quiet.history[0]["train_loss"] == pytest.approx(loud.history[0]["train_loss"])
    # Validation is only compared where the backend is reproducible. Apple MPS
    # is not bitwise reproducible even for repeated evaluation of fixed weights,
    # which is pinned separately by test_backend_evaluation_determinism.
    from src.utils import get_device

    if get_device().type != "mps":
        assert quiet.history[0]["val_loss"] == pytest.approx(loud.history[0]["val_loss"])
        assert quiet.best_val_f1_macro == pytest.approx(loud.best_val_f1_macro)
        assert quiet.best_val_accuracy == pytest.approx(loud.best_val_accuracy)


def test_backend_evaluation_determinism():
    """Document whether repeated evaluation of fixed weights is reproducible.

    On CPU and CUDA this must hold exactly. On Apple MPS it is currently not
    guaranteed, so the test records the fact rather than asserting a false
    invariant; runs reported in the paper should use CPU or CUDA.
    """
    import torch
    import torch.nn as nn

    from src.train import evaluate_model
    from src.utils import get_device
    from tests.test_train import TinyModel, make_loader

    torch.manual_seed(0)
    model = TinyModel().to(get_device())
    loader = make_loader(split_name="validation")
    losses = {
        evaluate_model(model, loader, nn.CrossEntropyLoss(), get_device())["loss"]
        for _ in range(3)
    }
    if get_device().type == "mps":
        pytest.skip(f"MPS evaluation is not bitwise reproducible; observed {len(losses)} values")
    assert len(losses) == 1, f"evaluation of fixed weights must be deterministic, got {losses}"


def test_progress_bar_does_not_perturb_data_order():
    """Directly pin the tqdm-wrapping hazard that motivated the design."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.arange(24))

    def order(use_bar: bool) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(42)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)
        bars = BatchProgress(1, len(loader), enabled=use_bar)
        batches = []
        for batch in bars.epoch(loader, epoch=1):
            batches.append(batch[0].tolist())
            bars.update(0.0, 0.0, 0.0)
        bars.close()
        return batches

    assert order(False) == order(True)
