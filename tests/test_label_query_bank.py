"""Unit tests for the Label Query Bank."""

from __future__ import annotations

import pytest
import torch

from src.models import LabelQueryBank


def test_shape_and_dtype():
    bank = LabelQueryBank(num_classes=4, hidden_size=32, seed=7)
    queries = bank()
    assert queries.shape == (4, 32)
    assert queries.dtype == torch.float32


def test_initialization_is_reproducible():
    first = LabelQueryBank(num_classes=4, hidden_size=32, seed=123)()
    second = LabelQueryBank(num_classes=4, hidden_size=32, seed=123)()
    assert torch.equal(first, second)
    different = LabelQueryBank(num_classes=4, hidden_size=32, seed=124)()
    assert not torch.equal(first, different)


def test_all_modes_share_the_same_initial_values():
    trainable = LabelQueryBank(4, 32, mode="trainable", seed=5)()
    frozen = LabelQueryBank(4, 32, mode="frozen", seed=5)()
    fixed = LabelQueryBank(4, 32, mode="fixed_random", seed=5)()
    assert torch.equal(trainable, frozen)
    assert torch.equal(trainable, fixed)


def test_trainable_receives_gradient():
    bank = LabelQueryBank(4, 32, mode="trainable", seed=1)
    bank().sum().backward()
    assert bank.queries.grad is not None
    assert torch.isfinite(bank.queries.grad).all()
    assert bank.count_parameters() == {"total": 128, "trainable": 128, "frozen": 0}


def test_frozen_queries_have_no_gradient_but_are_saved():
    bank = LabelQueryBank(4, 32, mode="frozen", seed=1)
    assert bank.queries.requires_grad is False
    assert "queries" in bank.state_dict()
    assert bank.count_parameters() == {"total": 128, "trainable": 0, "frozen": 128}


def test_fixed_random_queries_are_a_buffer_not_a_parameter():
    bank = LabelQueryBank(4, 32, mode="fixed_random", seed=1)
    assert list(bank.parameters()) == []
    assert "queries" in bank.state_dict()  # persisted for reproducibility
    assert bank.count_parameters() == {"total": 0, "trainable": 0, "frozen": 0}


def test_freeze_and_unfreeze():
    bank = LabelQueryBank(4, 32, seed=1)
    bank.freeze()
    assert bank.queries.requires_grad is False and bank.mode == "frozen"
    bank.unfreeze()
    assert bank.queries.requires_grad is True and bank.mode == "trainable"


def test_save_and_load_round_trip():
    source = LabelQueryBank(4, 32, seed=11)
    target = LabelQueryBank(4, 32, seed=99)
    assert not torch.equal(source(), target())
    target.load_state_dict(source.state_dict())
    assert torch.equal(source(), target())


def test_device_placement():
    bank = LabelQueryBank(4, 32, seed=1)
    assert bank().device.type == "cpu"
    moved = LabelQueryBank(4, 32, mode="fixed_random", seed=1).to(torch.device("cpu"))
    assert moved().device.type == "cpu"


@pytest.mark.parametrize(
    "kwargs", [dict(num_classes=1), dict(hidden_size=0), dict(mode="bogus")]
)
def test_invalid_configuration_is_rejected(kwargs):
    base = dict(num_classes=4, hidden_size=32)
    base.update(kwargs)
    with pytest.raises(ValueError):
        LabelQueryBank(**base)
