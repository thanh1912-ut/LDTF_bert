"""Unit tests for the depth-fusion modules."""

from __future__ import annotations

import pytest
import torch

from src.models import (
    DirectDepthRouter,
    GlobalScalarMix,
    IndirectDepthGating,
    LabelQueryBank,
    UniformLayerMean,
)


@pytest.fixture
def token_features(shapes):
    torch.manual_seed(1)
    return torch.randn(
        shapes["batch"], shapes["classes"], shapes["layers"], shapes["hidden"]
    )


@pytest.fixture
def queries(shapes):
    return LabelQueryBank(shapes["classes"], shapes["hidden"], seed=3)()


# -- direct router ----------------------------------------------------------
def test_direct_router_shapes(shapes, token_features, queries):
    router = DirectDepthRouter(shapes["hidden"], shapes["classes"], shapes["router_dim"])
    out = router(token_features, queries)
    assert out["depth_attention"].shape == (
        shapes["batch"],
        shapes["classes"],
        shapes["layers"],
    )
    assert out["class_features"].shape == (
        shapes["batch"],
        shapes["classes"],
        shapes["hidden"],
    )


def test_direct_router_normalizes_over_layers(shapes, token_features, queries):
    router = DirectDepthRouter(shapes["hidden"], shapes["classes"], shapes["router_dim"])
    attention = router(token_features, queries)["depth_attention"]
    totals = attention.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)
    assert torch.isfinite(attention).all()


def test_direct_router_manual_einsum_equivalence(shapes, token_features, queries):
    router = DirectDepthRouter(shapes["hidden"], shapes["classes"], shapes["router_dim"])
    out = router(token_features, queries)

    projected_q = router.query_projection(queries)  # [C, Rd]
    projected_k = router.key_projection(token_features)  # [B, C, L, Rd]
    scores = torch.einsum("cr,bclr->bcl", projected_q, projected_k)
    scores = scores * (shapes["router_dim"] ** -0.5)
    expected_attention = torch.softmax(scores, dim=-1)
    expected_features = torch.einsum("bcl,bcld->bcd", expected_attention, token_features)

    assert torch.allclose(out["depth_attention"], expected_attention, atol=1e-6)
    assert torch.allclose(out["class_features"], expected_features, atol=1e-6)


def test_direct_router_scores_each_layer_individually(shapes, token_features, queries):
    """Permuting the layer axis must permute the attention identically.

    This is the property the old implementation lacked: it averaged layers
    before scoring, which made its output invariant to layer permutation.
    """
    router = DirectDepthRouter(shapes["hidden"], shapes["classes"], shapes["router_dim"])
    baseline = router(token_features, queries)["depth_attention"]
    permutation = torch.randperm(shapes["layers"])
    permuted = router(token_features[:, :, permutation, :], queries)["depth_attention"]
    assert torch.allclose(permuted, baseline[:, :, permutation], atol=1e-6)
    assert not torch.allclose(permuted, baseline, atol=1e-4)


def test_direct_router_classes_can_differ(shapes, token_features, queries):
    router = DirectDepthRouter(shapes["hidden"], shapes["classes"], shapes["router_dim"])
    attention = router(token_features, queries)["depth_attention"]
    spread = (attention[:, 0, :] - attention[:, 1, :]).abs().max()
    assert float(spread) > 0.0


def test_direct_router_gradients_reach_queries(shapes, token_features):
    bank = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=3)
    router = DirectDepthRouter(shapes["hidden"], shapes["classes"], shapes["router_dim"])
    router(token_features, bank())["class_features"].sum().backward()
    assert bank.queries.grad is not None and float(bank.queries.grad.abs().sum()) > 0
    assert router.query_projection.weight.grad is not None
    assert router.key_projection.weight.grad is not None


def test_direct_router_does_not_hardcode_layer_count(shapes, queries):
    router = DirectDepthRouter(shapes["hidden"], shapes["classes"], shapes["router_dim"])
    for layers in (1, 3, 12, 13):
        features = torch.randn(2, shapes["classes"], layers, shapes["hidden"])
        assert router(features, queries)["depth_attention"].shape == (
            2,
            shapes["classes"],
            layers,
        )


# -- indirect gating ablation ----------------------------------------------
def test_indirect_gating_shapes_and_normalization(shapes, token_features):
    gate = IndirectDepthGating(shapes["hidden"], shapes["classes"], shapes["layers"])
    out = gate(token_features)
    assert out["depth_attention"].shape == (
        shapes["batch"],
        shapes["classes"],
        shapes["layers"],
    )
    assert out["class_features"].shape == (
        shapes["batch"],
        shapes["classes"],
        shapes["hidden"],
    )
    totals = out["depth_attention"].sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_indirect_gating_takes_no_label_queries(shapes, token_features, queries):
    gate = IndirectDepthGating(shapes["hidden"], shapes["classes"], shapes["layers"])
    with pytest.raises(TypeError):
        gate(token_features, queries)  # type: ignore[call-arg]


def test_indirect_gating_is_layer_permutation_invariant(shapes, token_features):
    """Documents precisely why this module is only an ablation."""
    gate = IndirectDepthGating(shapes["hidden"], shapes["classes"], shapes["layers"])
    baseline = gate(token_features)["depth_attention"]
    permutation = torch.randperm(shapes["layers"])
    permuted = gate(token_features[:, :, permutation, :])["depth_attention"]
    assert torch.allclose(permuted, baseline, atol=1e-6)


# -- global scalar mix ------------------------------------------------------
def test_scalar_mix_weights_are_shared_across_classes(shapes, token_features):
    mixer = GlobalScalarMix(num_layers=shapes["layers"])
    attention = mixer(token_features)["depth_attention"]
    assert torch.allclose(attention[:, 0, :], attention[:, 1, :], atol=1e-7)
    totals = attention.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_scalar_mix_starts_uniform(shapes, token_features):
    mixer = GlobalScalarMix(num_layers=shapes["layers"])
    attention = mixer(token_features)["depth_attention"]
    expected = torch.full_like(attention, 1.0 / shapes["layers"])
    assert torch.allclose(attention, expected, atol=1e-6)


def test_scalar_mix_parameter_count(shapes):
    mixer = GlobalScalarMix(num_layers=shapes["layers"])
    assert mixer.count_parameters()["total"] == shapes["layers"] + 1


# -- uniform mean -----------------------------------------------------------
def test_uniform_mean_is_parameter_free(shapes, token_features):
    mean = UniformLayerMean()
    out = mean(token_features)
    assert mean.count_parameters()["total"] == 0
    assert torch.allclose(out["class_features"], token_features.mean(dim=2), atol=1e-6)
    totals = out["depth_attention"].sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


@pytest.mark.parametrize("module_factory", [DirectDepthRouter, IndirectDepthGating])
def test_rejects_wrong_rank(shapes, module_factory, queries):
    module = module_factory(shapes["hidden"], shapes["classes"])
    bad = torch.randn(2, shapes["classes"], shapes["hidden"])
    with pytest.raises(ValueError, match=r"\[B,C,L,D\]"):
        if module_factory is DirectDepthRouter:
            module(bad, queries)
        else:
            module(bad)
