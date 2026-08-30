"""Unit tests for the three class scorers."""

from __future__ import annotations

import pytest
import torch

from src.models import (
    ClassSpecificLinearScorer,
    SharedLinearScorer,
    SharedMlpScorer,
    build_scorer,
)


@pytest.fixture
def features(shapes):
    torch.manual_seed(2)
    return torch.randn(shapes["batch"], shapes["classes"], shapes["hidden"])


def test_shared_linear_shapes_and_parameters(shapes, features):
    scorer = SharedLinearScorer(shapes["hidden"], dropout=0.0)
    logits = scorer(features)
    assert logits.shape == (shapes["batch"], shapes["classes"])
    assert scorer.projection.weight.shape == (1, shapes["hidden"])
    assert scorer.projection.bias.shape == (1,)
    assert scorer.count_parameters()["total"] == shapes["hidden"] + 1


def test_shared_linear_reference_count_at_768():
    assert SharedLinearScorer(768).count_parameters()["total"] == 769


def test_shared_linear_applies_identical_weights_to_every_class(shapes, features):
    scorer = SharedLinearScorer(shapes["hidden"], dropout=0.0)
    scorer.eval()
    logits = scorer(features)
    weight = scorer.projection.weight.squeeze(0)
    expected = torch.einsum("bcd,d->bc", features, weight) + scorer.projection.bias
    assert torch.allclose(logits, expected, atol=1e-5)


def test_shared_linear_is_class_permutation_equivariant(shapes, features):
    scorer = SharedLinearScorer(shapes["hidden"], dropout=0.0)
    scorer.eval()
    permutation = torch.randperm(shapes["classes"])
    assert torch.allclose(
        scorer(features[:, permutation, :]), scorer(features)[:, permutation], atol=1e-6
    )


def test_shared_mlp_shapes_and_parameter_count(shapes, features):
    scorer = SharedMlpScorer(shapes["hidden"], mlp_hidden_size=shapes["hidden"], dropout=0.0)
    assert scorer(features).shape == (shapes["batch"], shapes["classes"])
    hidden = shapes["hidden"]
    expected = (hidden * hidden + hidden) + (hidden + 1)
    assert scorer.count_parameters()["total"] == expected


def test_shared_mlp_adds_about_590k_parameters_at_768():
    """The documented capacity cost of A8 relative to the reference scorer."""
    mlp = SharedMlpScorer(768, mlp_hidden_size=768).count_parameters()["total"]
    linear = SharedLinearScorer(768).count_parameters()["total"]
    assert mlp == 591_361
    assert mlp - linear == 590_592


def test_class_specific_shapes_and_parameter_count(shapes, features):
    scorer = ClassSpecificLinearScorer(shapes["hidden"], shapes["classes"], dropout=0.0)
    assert scorer(features).shape == (shapes["batch"], shapes["classes"])
    assert scorer.class_weights.shape == (shapes["classes"], shapes["hidden"])
    assert scorer.class_bias.shape == (shapes["classes"],)
    expected = shapes["classes"] * shapes["hidden"] + shapes["classes"]
    assert scorer.count_parameters()["total"] == expected


def test_class_specific_uses_per_class_weight_rows(shapes, features):
    scorer = ClassSpecificLinearScorer(shapes["hidden"], shapes["classes"], dropout=0.0)
    scorer.eval()
    logits = scorer(features)
    expected = (
        torch.einsum("bcd,cd->bc", features, scorer.class_weights) + scorer.class_bias
    )
    assert torch.allclose(logits, expected, atol=1e-6)


def test_class_specific_is_not_linear_d_to_c_on_a_shared_vector(shapes):
    """A shared representation must still yield distinct per-class logits."""
    scorer = ClassSpecificLinearScorer(shapes["hidden"], shapes["classes"], dropout=0.0)
    scorer.eval()
    shared = torch.randn(2, 1, shapes["hidden"]).expand(-1, shapes["classes"], -1)
    logits = scorer(shared)
    assert float((logits[:, 0] - logits[:, 1]).abs().max()) > 0.0


def test_shared_scorers_collapse_on_identical_features(shapes):
    """Documents the degeneracy a shared scorer exhibits without class identity."""
    shared = torch.randn(2, 1, shapes["hidden"]).expand(-1, shapes["classes"], -1)
    for scorer in (
        SharedLinearScorer(shapes["hidden"], dropout=0.0),
        SharedMlpScorer(shapes["hidden"], mlp_hidden_size=8, dropout=0.0),
    ):
        scorer.eval()
        logits = scorer(shared)
        assert torch.allclose(logits[:, 0], logits[:, 1], atol=1e-6)


@pytest.mark.parametrize("name", ["shared_linear", "shared_mlp", "class_specific"])
def test_build_scorer_returns_only_the_requested_module(shapes, features, name):
    scorer = build_scorer(name, hidden_size=shapes["hidden"], num_classes=shapes["classes"])
    assert scorer(features).shape == (shapes["batch"], shapes["classes"])
    keys = set(scorer.state_dict())
    assert keys, "scorer must register parameters"


def test_build_scorer_rejects_unknown_name(shapes):
    with pytest.raises(ValueError, match="Unknown scorer"):
        build_scorer("mystery", hidden_size=shapes["hidden"], num_classes=4)


def test_scorers_reject_wrong_rank(shapes):
    bad = torch.randn(shapes["batch"], shapes["hidden"])
    for scorer in (
        SharedLinearScorer(shapes["hidden"]),
        SharedMlpScorer(shapes["hidden"], mlp_hidden_size=8),
        ClassSpecificLinearScorer(shapes["hidden"], shapes["classes"]),
    ):
        with pytest.raises(ValueError, match=r"\[B,C,D\]"):
            scorer(bad)
