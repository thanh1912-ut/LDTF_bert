"""Unit tests for the label-conditioned Token Router."""

from __future__ import annotations

import pytest
import torch

from src.models import LabelQueryBank, TokenRouter


def _router(shapes, *, include_special: bool = True) -> TokenRouter:
    torch.manual_seed(0)
    return TokenRouter(
        hidden_size=shapes["hidden"],
        num_classes=shapes["classes"],
        router_dim=shapes["router_dim"],
        include_special_tokens=include_special,
    )


def test_output_shapes(shapes, hidden_states, masks):
    attention_mask, special = masks
    router = _router(shapes)
    queries = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=0)()
    out = router(hidden_states, queries, attention_mask, special)
    assert out["token_attention"].shape == (
        shapes["batch"],
        shapes["classes"],
        shapes["layers"],
        shapes["tokens"],
    )
    assert out["token_features"].shape == (
        shapes["batch"],
        shapes["classes"],
        shapes["layers"],
        shapes["hidden"],
    )


def test_attention_normalizes_over_tokens(shapes, hidden_states, masks):
    attention_mask, special = masks
    router = _router(shapes)
    queries = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=0)()
    attention = router(hidden_states, queries, attention_mask, special)["token_attention"]
    totals = attention.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)
    assert torch.isfinite(attention).all()


def test_padding_never_receives_attention(shapes, hidden_states, masks):
    attention_mask, special = masks
    router = _router(shapes)
    queries = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=0)()
    attention = router(hidden_states, queries, attention_mask, special)["token_attention"]
    assert float(attention[1, :, :, 6:].abs().max()) == 0.0


def test_special_tokens_excluded_when_configured(shapes, hidden_states, masks):
    attention_mask, special = masks
    router = _router(shapes, include_special=False)
    queries = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=0)()
    attention = router(hidden_states, queries, attention_mask, special)["token_attention"]
    assert float(attention[:, :, :, 0].abs().max()) == 0.0  # [CLS]
    assert float(attention[0, :, :, -1].abs().max()) == 0.0  # [SEP]
    assert float(attention[1, :, :, 5].abs().max()) == 0.0
    content = attention.sum(dim=-1)
    assert torch.allclose(content, torch.ones_like(content), atol=1e-5)


def test_special_tokens_included_by_default(shapes, hidden_states, masks):
    attention_mask, special = masks
    router = _router(shapes, include_special=True)
    queries = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=0)()
    attention = router(hidden_states, queries, attention_mask, special)["token_attention"]
    assert float(attention[:, :, :, 0].abs().max()) > 0.0


def test_manual_equivalence(shapes, hidden_states, masks):
    attention_mask, special = masks
    router = _router(shapes, include_special=False)
    queries = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=0)()
    out = router(hidden_states, queries, attention_mask, special)

    projected_q = router.query_projection(queries)
    projected_k = router.key_projection(hidden_states)
    scores = torch.einsum("cr,bltr->bclt", projected_q, projected_k)
    scores = scores * (shapes["router_dim"] ** -0.5)
    valid = attention_mask.bool() & ~special.bool()
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    expected_attention = torch.softmax(scores, dim=-1)
    expected_features = torch.einsum("bclt,bltd->bcld", expected_attention, hidden_states)

    assert torch.allclose(out["token_attention"], expected_attention, atol=1e-6)
    assert torch.allclose(out["token_features"], expected_features, atol=1e-6)


def test_gradients_reach_queries_and_projections(shapes, hidden_states, masks):
    attention_mask, special = masks
    router = _router(shapes)
    bank = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=0)
    out = router(hidden_states, bank(), attention_mask, special)
    out["token_features"].sum().backward()
    assert bank.queries.grad is not None and float(bank.queries.grad.abs().sum()) > 0
    assert router.query_projection.weight.grad is not None
    assert router.key_projection.weight.grad is not None
    assert float(router.key_projection.weight.grad.abs().sum()) > 0


def test_all_padding_row_is_rejected(shapes, hidden_states, masks):
    _, special = masks
    router = _router(shapes)
    queries = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=0)()
    empty = torch.zeros(shapes["batch"], shapes["tokens"], dtype=torch.long)
    with pytest.raises(ValueError, match="valid routing token"):
        router(hidden_states, queries, empty, special)


def test_batch_size_one_and_variable_length(shapes):
    router = _router(shapes)
    queries = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=0)()
    for tokens in (3, 5, 17):
        hidden = torch.randn(1, shapes["layers"], tokens, shapes["hidden"])
        attention = torch.ones(1, tokens, dtype=torch.long)
        special = torch.zeros(1, tokens, dtype=torch.long)
        out = router(hidden, queries, attention, special)
        assert out["token_attention"].shape == (1, shapes["classes"], shapes["layers"], tokens)


def test_shape_mismatch_is_rejected(shapes, hidden_states, masks):
    attention_mask, special = masks
    router = _router(shapes)
    wrong = torch.randn(shapes["classes"] + 1, shapes["hidden"])
    with pytest.raises(ValueError):
        router(hidden_states, wrong, attention_mask, special)


def test_missing_special_mask_is_rejected_when_excluding(shapes, hidden_states, masks):
    attention_mask, _ = masks
    router = _router(shapes, include_special=False)
    queries = LabelQueryBank(shapes["classes"], shapes["hidden"], seed=0)()
    with pytest.raises(ValueError, match="special_tokens_mask is required"):
        router(hidden_states, queries, attention_mask, None)
