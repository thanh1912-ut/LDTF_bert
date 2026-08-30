"""Tests for the training engine: optimizer, accumulation, checkpoints, resume."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src import config, guard
from src.train import (
    TrainConfig,
    build_optimizer,
    build_scheduler,
    evaluate_model,
    gradient_audit,
    optimizer_coverage_report,
    train_model,
)


# ---------------------------------------------------------------------------
# Tiny offline model with the same contract as the real ones
# ---------------------------------------------------------------------------
class TinyBackbone(nn.Module):
    def __init__(self, vocab: int = 50, hidden: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.dense = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.5)
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            for module in self.modules():
                module.training = False
        return self

    def forward(self, input_ids, attention_mask):
        hidden = self.norm(self.dense(self.dropout(self.embedding(input_ids))))
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


class TinyModel(nn.Module):
    def __init__(self, hidden: int = 8, classes: int = config.NUM_CLASSES) -> None:
        super().__init__()
        self.backbone = TinyBackbone(hidden=hidden)
        self.classifier = nn.Linear(hidden, classes)

    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    def architecture_config(self) -> dict:
        return {"class_name": "TinyModel"}

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": {
                "total": total,
                "trainable": trainable,
                "frozen": total - trainable,
            }
        }

    def forward(
        self, input_ids, attention_mask, token_type_ids=None, special_tokens_mask=None
    ):
        return {"logits": self.classifier(self.backbone(input_ids, attention_mask))}


class TinyDataset(Dataset):
    def __init__(self, size: int = 20, tokens: int = 6) -> None:
        generator = torch.Generator().manual_seed(0)
        self.input_ids = torch.randint(0, 50, (size, tokens), generator=generator)
        self.labels = torch.randint(0, config.NUM_CLASSES, (size,), generator=generator)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": torch.ones_like(self.input_ids[index]),
            "special_tokens_mask": torch.zeros_like(self.input_ids[index]),
            "labels": self.labels[index],
        }


def make_loader(
    size: int = 20, batch_size: int = 4, split_name: str = "train"
) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        TinyDataset(size),
        batch_size=batch_size,
        shuffle=split_name == "train",
        collate_fn=lambda items: {
            key: torch.stack([item[key] for item in items]) for key in items[0]
        },
        generator=generator,
    )
    loader.split_name = split_name
    return loader


@pytest.fixture
def train_config(tmp_path) -> TrainConfig:
    return TrainConfig(output_dir=tmp_path, run_id="unit", epochs=2, use_amp=False)


# ---------------------------------------------------------------------------
# Optimizer groups
# ---------------------------------------------------------------------------
def test_optimizer_has_four_groups_with_differential_learning_rates(train_config):
    model = TinyModel()
    optimizer = build_optimizer(model, train_config)
    names = [group["name"] for group in optimizer.param_groups]
    assert names == [
        "backbone_decay",
        "backbone_no_decay",
        "head_decay",
        "head_no_decay",
    ]
    by_name = {group["name"]: group for group in optimizer.param_groups}
    assert by_name["backbone_decay"]["lr"] == train_config.backbone_learning_rate
    assert by_name["head_decay"]["lr"] == train_config.head_learning_rate
    assert by_name["backbone_no_decay"]["weight_decay"] == 0.0
    assert by_name["head_no_decay"]["weight_decay"] == 0.0


def test_bias_and_layernorm_weights_are_excluded_from_decay(train_config):
    model = TinyModel()
    optimizer = build_optimizer(model, train_config)
    no_decay_ids = {
        id(p)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.0
        for p in group["params"]
    }
    decay_ids = {
        id(p)
        for group in optimizer.param_groups
        if group["weight_decay"] > 0.0
        for p in group["params"]
    }
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if lowered.endswith("bias") or lowered.endswith("norm.weight"):
            assert id(parameter) in no_decay_ids, f"{name} must not be weight-decayed"
        elif parameter.ndim >= 2:
            assert id(parameter) in decay_ids, f"{name} must be weight-decayed"


def test_one_dimensional_parameters_are_never_decayed(train_config):
    """Scalar-mix logits and per-class biases must not be pulled toward zero."""
    from src.models import ClassSpecificLinearScorer, GlobalScalarMix

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mix = GlobalScalarMix(num_layers=12)
            self.scorer = ClassSpecificLinearScorer(8, 4)

    model = Wrapper()
    optimizer = build_optimizer(model, train_config)
    no_decay_ids = {
        id(p)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.0
        for p in group["params"]
    }
    assert id(model.mix.layer_logits) in no_decay_ids
    assert id(model.mix.gamma) in no_decay_ids
    assert id(model.scorer.class_bias) in no_decay_ids


def test_optimizer_covers_every_trainable_parameter_exactly_once(train_config):
    model = TinyModel()
    optimizer = build_optimizer(model, train_config)
    report = optimizer_coverage_report(model, optimizer)
    assert report["covered"]
    assert report["missing_trainable"] == 0
    assert report["frozen_in_optimizer"] == 0
    seen = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in seen}) == len(seen), "no parameter may be duplicated"


def test_frozen_parameters_never_enter_the_optimizer(train_config):
    model = TinyModel()
    model.freeze_backbone()
    optimizer = build_optimizer(model, train_config)
    in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in in_optimizer for p in model.backbone.parameters())
    assert optimizer_coverage_report(model, optimizer)["covered"]


# ---------------------------------------------------------------------------
# Frozen regime
# ---------------------------------------------------------------------------
def test_frozen_backbone_stays_in_eval_mode_and_gets_no_gradient(
    train_config, tmp_path
):
    model = TinyModel()
    config_frozen = TrainConfig(
        output_dir=tmp_path,
        run_id="frozen",
        epochs=1,
        freeze_backbone=True,
        use_amp=False,
    )
    result = train_model(
        model, make_loader(), make_loader(split_name="validation"), config_frozen
    )
    assert result.best_epoch == 1
    assert model.backbone.training is False
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())

    # The loop ends with zero_grad, so re-run one step to inspect head gradients.
    batch = next(iter(make_loader()))
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    logits = model(batch["input_ids"], batch["attention_mask"])["logits"]
    nn.CrossEntropyLoss()(logits, batch["labels"]).backward()
    assert model.classifier.weight.grad is not None
    assert all(p.grad is None for p in model.backbone.parameters())


def test_frozen_backbone_weights_do_not_change(tmp_path):
    model = TinyModel()
    before = model.backbone.dense.weight.detach().cpu().clone()
    train_model(
        model,
        make_loader(),
        make_loader(split_name="validation"),
        TrainConfig(
            output_dir=tmp_path,
            run_id="frozen",
            epochs=1,
            freeze_backbone=True,
            use_amp=False,
        ),
    )
    assert torch.equal(before, model.backbone.dense.weight.detach().cpu())


def test_finetuning_updates_the_backbone(tmp_path):
    model = TinyModel()
    before = model.backbone.dense.weight.detach().cpu().clone()
    train_model(
        model,
        make_loader(),
        make_loader(split_name="validation"),
        TrainConfig(
            output_dir=tmp_path,
            run_id="ft",
            epochs=1,
            freeze_backbone=False,
            use_amp=False,
        ),
    )
    assert not torch.equal(before, model.backbone.dense.weight.detach().cpu())


# ---------------------------------------------------------------------------
# Scheduler and accumulation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("accum", [1, 2, 3, 4])
def test_scheduler_steps_match_ceil_of_batches_over_accumulation(tmp_path, accum):
    """The final partial accumulation group must be flushed, not dropped."""
    loader = make_loader(size=20, batch_size=3)  # 7 batches
    model = TinyModel()
    train_config = TrainConfig(
        output_dir=tmp_path,
        run_id="accum",
        epochs=1,
        grad_accum_steps=accum,
        use_amp=False,
    )
    result = train_model(
        model, loader, make_loader(split_name="validation"), train_config
    )
    expected = math.ceil(len(loader) / accum)
    assert result.history[-1]["global_step"] == expected


def test_non_positive_accumulation_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        train_model(
            TinyModel(),
            make_loader(),
            make_loader(split_name="validation"),
            TrainConfig(
                output_dir=tmp_path,
                run_id="bad_accum",
                epochs=1,
                grad_accum_steps=0,
                use_amp=False,
            ),
        )


def test_scheduler_warmup_then_decay(train_config):
    model = TinyModel()
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config, total_steps=100)
    # LambdaLR records each group's base LR and applies lr_lambda(0) = 0 at init.
    base = optimizer.param_groups[0]["initial_lr"]
    assert base == pytest.approx(train_config.backbone_learning_rate)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    for _ in range(10):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base, rel=1e-6)
    for _ in range(90):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_scheduler_preserves_differential_group_learning_rates(train_config):
    model = TinyModel()
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config, total_steps=10)
    scheduler.step()
    by_name = {group["name"]: group["lr"] for group in optimizer.param_groups}
    ratio = by_name["head_decay"] / max(by_name["backbone_decay"], 1e-12)
    expected = train_config.head_learning_rate / train_config.backbone_learning_rate
    assert ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def test_best_checkpoint_is_slim_and_carries_provenance(tmp_path):
    model = TinyModel()
    train_config = TrainConfig(
        output_dir=tmp_path,
        run_id="ckpt",
        epochs=2,
        use_amp=False,
        data_signature={"train_sha256": "abc"},
        architecture={"class_name": "TinyModel"},
    )
    train_model(
        model, make_loader(), make_loader(split_name="validation"), train_config
    )
    payload = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)

    assert "optimizer_state_dict" not in payload
    assert "scheduler_state_dict" not in payload
    assert "rng_state" not in payload
    for key in (
        "model_state_dict",
        "architecture",
        "data_signature",
        "protocol",
        "best_metrics",
        "epoch",
        "seed",
    ):
        assert key in payload, f"best.pt must contain {key}"
    assert payload["protocol"]["selection_metric"] == config.SELECTION_METRIC


def test_last_checkpoint_is_fully_resumable(tmp_path):
    model = TinyModel()
    train_config = TrainConfig(
        output_dir=tmp_path, run_id="ckpt", epochs=2, use_amp=False
    )
    train_model(
        model, make_loader(), make_loader(split_name="validation"), train_config
    )
    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "epoch",
        "global_step",
        "best_metrics",
        "patience_counter",
        "history",
        "rng_state",
        "loader_generator_state",
        "total_train_seconds",
    ):
        assert key in payload, f"last.pt must contain {key}"


def test_fresh_run_archives_stale_artifacts_instead_of_mixing_logs(tmp_path):
    for name in ("best.pt", "last.pt", "val_metrics.json", "run_summary.json"):
        (tmp_path / name).write_text(f"stale {name}", encoding="utf-8")
    (tmp_path / "train_log.jsonl").write_text("stale log\n", encoding="utf-8")

    train_model(
        TinyModel(),
        make_loader(),
        make_loader(split_name="validation"),
        TrainConfig(output_dir=tmp_path, run_id="fresh", epochs=1, use_amp=False),
    )

    archives = list((tmp_path / "previous_runs").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "train_log.jsonl").read_text(
        encoding="utf-8"
    ) == "stale log\n"
    assert "stale" not in (tmp_path / "train_log.jsonl").read_text(encoding="utf-8")


def test_training_refuses_a_run_with_official_test_artifacts(tmp_path):
    (tmp_path / "test_metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="official-test artifacts"):
        train_model(
            TinyModel(),
            make_loader(),
            make_loader(split_name="validation"),
            TrainConfig(output_dir=tmp_path, run_id="tested", epochs=1, use_amp=False),
        )


def test_checkpoint_selection_uses_macro_f1_then_loss_then_epoch(tmp_path):
    from src.train import _selection_key

    best = {"f1_macro": 0.80, "accuracy": 0.8, "loss": 0.50}
    higher_f1 = {"f1_macro": 0.81, "accuracy": 0.7, "loss": 0.90}
    same_f1_lower_loss = {"f1_macro": 0.80, "accuracy": 0.8, "loss": 0.40}
    same_f1_higher_loss = {"f1_macro": 0.80, "accuracy": 0.9, "loss": 0.60}

    assert _selection_key(higher_f1) > _selection_key(best)
    assert _selection_key(same_f1_lower_loss) > _selection_key(best)
    assert _selection_key(same_f1_higher_loss) < _selection_key(best)
    # An exact tie does not beat the incumbent, so the earlier epoch is kept.
    assert not _selection_key(dict(best)) > _selection_key(best)


def test_resume_restores_state_and_continues(tmp_path):
    torch.manual_seed(0)
    model = TinyModel()
    config_ = TrainConfig(output_dir=tmp_path, run_id="resume", epochs=3, use_amp=False)

    def interrupt_after_epoch_one(epoch, _record):
        if epoch == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train_model(
            model,
            make_loader(),
            make_loader(split_name="validation"),
            config_,
            progress_callback=interrupt_after_epoch_one,
        )
    first_payload = torch.load(
        tmp_path / "last.pt", map_location="cpu", weights_only=False
    )
    assert len(first_payload["history"]) == 1

    resumed_model = TinyModel()
    second = train_model(
        resumed_model,
        make_loader(),
        make_loader(split_name="validation"),
        config_,
        resume=True,
    )
    assert second.resumed_from_epoch == 1
    assert [record["epoch"] for record in second.history] == [1, 2, 3]
    assert second.history[0] == first_payload["history"][0], (
        "resumed history must be preserved"
    )
    assert second.total_train_seconds >= first_payload["total_train_seconds"]
    assert second.total_train_seconds >= second.session_train_seconds


def test_resume_does_not_reset_the_best_metric(tmp_path):
    model = TinyModel()
    base = TrainConfig(output_dir=tmp_path, run_id="resume", epochs=3, use_amp=False)

    def interrupt_after_epoch_two(epoch, _record):
        if epoch == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train_model(
            model,
            make_loader(),
            make_loader(split_name="validation"),
            base,
            progress_callback=interrupt_after_epoch_two,
        )
    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    first_best = payload["best_metrics"]["f1_macro"]

    second = train_model(
        TinyModel(),
        make_loader(),
        make_loader(split_name="validation"),
        base,
        resume=True,
    )
    assert second.best_val_f1_macro >= first_best


def test_exact_resume_refuses_a_changed_epoch_horizon(tmp_path):
    train_model(
        TinyModel(),
        make_loader(),
        make_loader(split_name="validation"),
        TrainConfig(output_dir=tmp_path, run_id="resume", epochs=1, use_amp=False),
    )

    with pytest.raises(RuntimeError, match="'epochs' changed"):
        train_model(
            TinyModel(),
            make_loader(),
            make_loader(split_name="validation"),
            TrainConfig(output_dir=tmp_path, run_id="resume", epochs=2, use_amp=False),
            resume=True,
        )


def test_resume_does_not_train_past_a_saved_early_stop(tmp_path, monkeypatch):
    def constant_validation(*_args, **_kwargs):
        return {
            "loss": 1.0,
            "accuracy": 0.25,
            "f1_macro": 0.1,
            "confusion_matrix": [[0] * config.NUM_CLASSES] * config.NUM_CLASSES,
        }

    monkeypatch.setattr("src.train.evaluate_model", constant_validation)
    config_ = TrainConfig(
        output_dir=tmp_path,
        run_id="early-stop",
        epochs=3,
        patience=1,
        use_amp=False,
    )

    def interrupt_after_stopping_checkpoint(epoch, _record):
        if epoch == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train_model(
            TinyModel(),
            make_loader(),
            make_loader(split_name="validation"),
            config_,
            progress_callback=interrupt_after_stopping_checkpoint,
        )

    resumed = train_model(
        TinyModel(),
        make_loader(),
        make_loader(split_name="validation"),
        config_,
        resume=True,
    )
    assert resumed.resumed_from_epoch == 2
    assert [record["epoch"] for record in resumed.history] == [1, 2]


def test_resume_refuses_a_different_data_signature(tmp_path):
    model = TinyModel()
    train_model(
        model,
        make_loader(),
        make_loader(split_name="validation"),
        TrainConfig(
            output_dir=tmp_path,
            run_id="sig",
            epochs=1,
            use_amp=False,
            data_signature={"train_sha256": "aaa"},
        ),
    )
    with pytest.raises(RuntimeError, match="data signature"):
        train_model(
            TinyModel(),
            make_loader(),
            make_loader(split_name="validation"),
            TrainConfig(
                output_dir=tmp_path,
                run_id="sig",
                epochs=2,
                use_amp=False,
                data_signature={"train_sha256": "bbb"},
            ),
            resume=True,
        )


# ---------------------------------------------------------------------------
# Protocol guards
# ---------------------------------------------------------------------------
def test_training_rejects_a_test_loader(tmp_path):
    leaked = make_loader(split_name="test")
    with pytest.raises(ValueError, match="official test split must never"):
        train_model(
            TinyModel(),
            make_loader(),
            leaked,
            TrainConfig(output_dir=tmp_path, run_id="leak", epochs=1, use_amp=False),
        )


def test_training_flag_is_cleared_after_the_run(tmp_path):
    train_model(
        TinyModel(),
        make_loader(),
        make_loader(split_name="validation"),
        TrainConfig(output_dir=tmp_path, run_id="flag", epochs=1, use_amp=False),
    )
    assert guard.training_active() is False


def test_history_records_the_required_reporting_fields(tmp_path):
    result = train_model(
        TinyModel(),
        make_loader(),
        make_loader(split_name="validation"),
        TrainConfig(output_dir=tmp_path, run_id="fields", epochs=1, use_amp=False),
    )
    record = result.history[0]
    for key in (
        "epoch",
        "global_step",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_f1_macro",
        "epoch_time_sec",
        "peak_vram_gb",
    ):
        assert key in record


def test_evaluate_model_returns_loss_and_metrics():
    model = TinyModel()
    metrics = evaluate_model(
        model,
        make_loader(split_name="validation"),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    for key in ("loss", "accuracy", "f1_macro", "confusion_matrix", "per_class"):
        assert key in metrics


def test_evaluation_restores_training_mode():
    model = TinyModel()
    model.train()
    evaluate_model(
        model,
        make_loader(split_name="validation"),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    assert model.training is True


def test_gradient_audit_detects_unused_parameters():
    model = TinyModel()
    model.unused = nn.Linear(4, 4)  # never called in forward
    batch = next(iter(make_loader()))
    logits = model(batch["input_ids"], batch["attention_mask"])["logits"]
    nn.CrossEntropyLoss()(logits, batch["labels"]).backward()
    audit = gradient_audit(model)
    assert not audit["ok"]
    assert any("unused" in name for name in audit["missing_grad"])
