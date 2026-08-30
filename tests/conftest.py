"""Shared fixtures. Unit tests use a tiny random backbone to stay offline-fast."""

from __future__ import annotations

import pytest
import torch

TINY_HIDDEN = 32
TINY_LAYERS = 4
TINY_TOKENS = 9
NUM_CLASSES = 4
ROUTER_DIM = 8


@pytest.fixture(scope="session")
def shapes() -> dict[str, int]:
    return {
        "batch": 3,
        "hidden": TINY_HIDDEN,
        "layers": TINY_LAYERS,
        "tokens": TINY_TOKENS,
        "classes": NUM_CLASSES,
        "router_dim": ROUTER_DIM,
    }


@pytest.fixture
def hidden_states(shapes: dict[str, int]) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(
        shapes["batch"], shapes["layers"], shapes["tokens"], shapes["hidden"]
    )


@pytest.fixture
def masks(shapes: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (attention_mask, special_tokens_mask) with padding on row 1."""
    batch, tokens = shapes["batch"], shapes["tokens"]
    attention = torch.ones(batch, tokens, dtype=torch.long)
    attention[1, 6:] = 0
    special = torch.zeros(batch, tokens, dtype=torch.long)
    special[:, 0] = 1  # [CLS]
    special[0, tokens - 1] = 1  # [SEP]
    special[2, tokens - 1] = 1
    special[1, 5] = 1  # [SEP] before padding
    special[1, 6:] = 1  # padding counts as special
    return attention, special
