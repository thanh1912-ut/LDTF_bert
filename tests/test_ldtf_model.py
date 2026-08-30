"""End-to-end tests for LdtfBert, the baselines, and every ablation variant."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src import config
from src.models import LdtfBert
from src.registry import ABLATION_RUNS, BASELINE_RUNS, build_model
from src.train import (
    TrainConfig,
    build_optimizer,
    gradient_audit,
    optimizer_coverage_report,
)
from src.variants import LdtfVariant, ablation_variant

pytestmark = pytest.mark.slow

NEURAL_RUNS = list(ABLATION_RUNS) + list(BASELINE_RUNS)


def make_batch(batch: int = 2, tokens: int = 10) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    special = torch.zeros(batch, tokens, dtype=torch.long)
    special[:, 0] = 1
    special[:, -1] = 1
    return {
        "input_ids": torch.randint(999, 2000, (batch, tokens)),
        "attention_mask": torch.ones(batch, tokens, dtype=torch.long),
        "token_type_ids": torch.zeros(batch, tokens, dtype=torch.long),
        "special_tokens_mask": special,
        "labels": torch.randint(0, config.NUM_CLASSES, (batch,)),
    }


@pytest.fixture(scope="module")
def batch() -> dict[str, torch.Tensor]:
    return make_batch()


@pytest.fixture(scope="module")
def full_model() -> LdtfBert:
    return build_model("A0")


# -- reference model contract ----------------------------------------------
def test_default_forward_returns_only_logits(full_model, batch):
    outputs = full_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        special_tokens_mask=batch["special_tokens_mask"],
    )
    assert set(outputs) == {"logits"}
    assert outputs["logits"].shape == (2, config.NUM_CLASSES)


def test_return_routing_adds_attention_maps(full_model, batch):
    outputs = full_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
        return_routing=True,
    )
    assert set(outputs) == {"logits", "token_attention", "depth_attention"}
    tokens = batch["input_ids"].shape[1]
    assert outputs["token_attention"].shape == (2, config.NUM_CLASSES, 12, tokens)
    assert outputs["depth_attention"].shape == (2, config.NUM_CLASSES, 12)


def test_return_features_adds_intermediates(full_model, batch):
    outputs = full_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
        return_features=True,
    )
    for key in (
        "hidden_states",
        "label_queries",
        "token_features",
        "fused_features",
        "depth_attention",
    ):
        assert key in outputs
    assert outputs["label_queries"].shape == (config.NUM_CLASSES, 768)
    assert outputs["fused_features"].shape == (2, config.NUM_CLASSES, 768)


def test_model_does_not_apply_softmax_or_argmax(full_model, batch):
    logits = full_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
    )["logits"]
    assert logits.dtype.is_floating_point
    totals = logits.exp().sum(dim=-1)
    assert not torch.allclose(totals, torch.ones_like(totals), atol=1e-3)


def test_cross_entropy_loss_compatibility(full_model, batch):
    logits = full_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
    )["logits"]
    loss = nn.CrossEntropyLoss()(logits, batch["labels"])
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_shared_query_bank_feeds_both_routers(full_model, batch):
    """One bank, two consumers, gradient from both paths."""
    assert full_model.token_router is not None
    assert full_model.depth_router is not None
    full_model.zero_grad(set_to_none=True)
    outputs = full_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
    )
    nn.CrossEntropyLoss()(outputs["logits"], batch["labels"]).backward()
    assert float(full_model.label_queries.queries.grad.abs().sum()) > 0
    assert float(full_model.token_router.query_projection.weight.grad.abs().sum()) > 0
    assert float(full_model.depth_router.query_projection.weight.grad.abs().sum()) > 0


def test_depth_attention_differs_across_classes(full_model, batch):
    attention = full_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
        return_routing=True,
    )["depth_attention"]
    assert float((attention[:, 0, :] - attention[:, 1, :]).abs().max()) > 0.0


def test_batch_size_one_and_variable_lengths(full_model):
    for tokens in (4, 11, 32):
        single = make_batch(batch=1, tokens=tokens)
        logits = full_model(
            input_ids=single["input_ids"],
            attention_mask=single["attention_mask"],
            special_tokens_mask=single["special_tokens_mask"],
        )["logits"]
        assert logits.shape == (1, config.NUM_CLASSES)


# -- every variant ----------------------------------------------------------
@pytest.mark.parametrize("run_id", NEURAL_RUNS)
def test_every_variant_forward_backward_and_optimizer(run_id, batch, tmp_path):
    model = build_model(run_id)
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        special_tokens_mask=batch["special_tokens_mask"],
    )
    assert outputs["logits"].shape == (2, config.NUM_CLASSES)

    nn.CrossEntropyLoss()(outputs["logits"], batch["labels"]).backward()
    audit = gradient_audit(model)
    assert audit["ok"], f"{run_id} has parameters without a finite gradient: {audit}"

    train_config = TrainConfig(output_dir=tmp_path, run_id=run_id)
    coverage = optimizer_coverage_report(model, build_optimizer(model, train_config))
    assert coverage["covered"], f"{run_id} optimizer coverage failed: {coverage}"


@pytest.mark.parametrize(
    "run_id,absent",
    [
        ("A1", "depth_router"),
        ("A2", "depth_router"),
        ("B5_token_attention_only", "depth_router"),
    ],
)
def test_ablated_modules_are_not_registered(run_id, absent, batch, tmp_path):
    model = build_model(run_id)
    parameter_names = [name for name, _ in model.named_parameters()]
    assert not any(absent in name for name in parameter_names)
    assert not any(absent in key for key in model.state_dict())
    optimizer = build_optimizer(model, TrainConfig(output_dir=tmp_path))
    assert optimizer_coverage_report(model, optimizer)["covered"]


def test_a2_uses_only_the_final_layer(batch):
    model = build_model("A2")
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
        return_features=True,
    )
    assert outputs["hidden_states"].shape[1] == 1
    assert outputs["depth_attention"].shape == (2, config.NUM_CLASSES, 1)
    assert torch.allclose(
        outputs["depth_attention"], torch.ones_like(outputs["depth_attention"])
    )


def test_a5_frozen_queries_do_not_train(batch):
    model = build_model("A5")
    assert model.label_queries.queries.requires_grad is False
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
    )
    nn.CrossEntropyLoss()(outputs["logits"], batch["labels"]).backward()
    assert model.label_queries.queries.grad is None
    assert "label_queries.queries" in model.state_dict()


def test_a6_fixed_random_queries_are_buffers(batch):
    model = build_model("A6")
    assert not any("label_queries" in name for name, _ in model.named_parameters())
    assert "label_queries.queries" in model.state_dict()
    reference = build_model("A0")
    assert torch.equal(model.label_queries(), reference.label_queries())


def test_a13_and_a14_special_token_policies(batch):
    include = build_model("A13")(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
        return_routing=True,
    )["token_attention"]
    exclude = build_model("A14")(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
        return_routing=True,
    )["token_attention"]
    assert float(include[:, :, :, 0].abs().max()) > 0.0
    assert float(exclude[:, :, :, 0].abs().max()) == 0.0
    assert float(exclude[:, :, :, -1].abs().max()) == 0.0
    totals = exclude.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-4)


def test_a15_regime_pair_shares_one_architecture(batch, tmp_path):
    frozen = build_model("A0", frozen_backbone=True)
    finetuned = build_model("A0", frozen_backbone=False)
    assert frozen.variant.token_routing == finetuned.variant.token_routing
    assert frozen.variant.depth_routing == finetuned.variant.depth_routing
    assert set(frozen.state_dict()) == set(finetuned.state_dict())
    frozen_counts = frozen.count_parameters()["total"]
    assert frozen_counts["trainable"] == frozen_counts["non_backbone"]
    assert finetuned.count_parameters()["total"]["trainable"] > frozen_counts["trainable"]


def test_parameter_counts_match_the_design(batch):
    counts = build_model("A0").count_parameters()
    assert counts["label_queries"]["total"] == 4 * 768
    assert counts["token_router"]["total"] == 2 * 256 * 768
    assert counts["depth_router"]["total"] == 2 * 256 * 768
    assert counts["class_scorer"]["total"] == 769
    assert counts["total"]["non_backbone"] == 790_273


def test_shared_mlp_capacity_delta():
    reference = build_model("A0").count_parameters()["total"]["non_backbone"]
    mlp = build_model("A8").count_parameters()["total"]["non_backbone"]
    assert mlp - reference == 590_592


def test_no_token_router_without_direct_depth_is_rejected():
    """Degenerate configurations must be impossible to construct."""
    with pytest.raises(ValueError, match="degenerate"):
        LdtfVariant(name="bad", token_routing="none", depth_routing="uniform")
    with pytest.raises(ValueError, match="degenerate"):
        LdtfVariant(name="bad", token_routing="none", depth_routing="scalar_mix")
    # The one legal no-token variant keeps class identity via direct routing.
    LdtfVariant(name="ok", token_routing="none", depth_routing="direct")


def test_a15_is_not_a_separate_architecture():
    with pytest.raises(ValueError, match="regime pair"):
        ablation_variant("A15")


def test_frozen_ldtf_backbone_receives_no_gradient(batch):
    model = build_model("B6_full_ldtf_frozen")
    model.train()
    assert model.backbone.encoder.training is False
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
    )
    nn.CrossEntropyLoss()(outputs["logits"], batch["labels"]).backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert float(model.label_queries.queries.grad.abs().sum()) > 0


def test_finetuned_ldtf_backbone_receives_gradient(batch):
    model = build_model("B7_full_ldtf_finetuned")
    model.train()
    assert model.backbone.encoder.training is True
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        special_tokens_mask=batch["special_tokens_mask"],
    )
    nn.CrossEntropyLoss()(outputs["logits"], batch["labels"]).backward()
    grads = [
        p.grad for p in model.backbone.parameters() if p.grad is not None
    ]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_baselines_b1_to_b4_shapes_and_regimes(batch):
    for run_id, expect_frozen in [
        ("B1_bert_frozen_cls", True),
        ("B2_bert_finetuned_cls", False),
        ("B3_bert_frozen_mean_pool", True),
        ("B4_bert_scalar_mix", False),
    ]:
        model = build_model(run_id)
        model.train()
        assert model.backbone.frozen is expect_frozen
        if expect_frozen:
            assert model.backbone.encoder.training is False
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )["logits"]
        assert logits.shape == (2, config.NUM_CLASSES)


def test_b3_mean_pooling_ignores_padding():
    from src.models import masked_mean_pool

    hidden = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.long)
    mask[0, 4:] = 0
    pooled = masked_mean_pool(hidden, mask)
    assert torch.allclose(pooled[0], hidden[0, :4].mean(dim=0), atol=1e-6)
    assert torch.allclose(pooled[1], hidden[1].mean(dim=0), atol=1e-6)


def test_b4_scalar_mix_is_not_label_conditioned(batch):
    model = build_model("B4_bert_scalar_mix")
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_routing=True,
    )
    weights = outputs["layer_weights"]
    assert weights.shape == (12,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_save_and_load_round_trip(tmp_path, batch):
    model = build_model("A0")
    model.eval()
    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "special_tokens_mask": batch["special_tokens_mask"],
    }
    with torch.no_grad():
        before = model(**inputs)["logits"]

    path = tmp_path / "model.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "architecture": model.architecture_config()},
        path,
    )
    from src.evaluate import load_model_from_checkpoint

    restored, _ = load_model_from_checkpoint(path)
    restored.eval()
    with torch.no_grad():
        after = restored(**inputs)["logits"]
    assert torch.allclose(before, after, atol=1e-6)
