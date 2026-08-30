"""Tests for the BERT backbone contract. These load the real pretrained model."""

from __future__ import annotations

import pytest
import torch

from src import config
from src.models import BertBackbone

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def backbone() -> BertBackbone:
    return BertBackbone(config.MODEL_NAME)


def test_no_pooler_parameters_exist(backbone):
    offenders = [name for name, _ in backbone.named_parameters() if ".pooler." in name]
    assert offenders == []


def test_geometry_is_read_from_the_pretrained_config(backbone):
    assert backbone.hidden_size == backbone.encoder.config.hidden_size
    assert backbone.num_transformer_layers == backbone.encoder.config.num_hidden_layers
    assert backbone.num_layers == 12  # bert-base, embedding output excluded


def test_hidden_state_output_shape(backbone):
    batch, tokens = 2, 7
    input_ids = torch.randint(999, 2000, (batch, tokens))
    attention_mask = torch.ones(batch, tokens, dtype=torch.long)
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert hidden.shape == (batch, 12, tokens, 768)


def test_embedding_layer_is_excluded_by_default(backbone):
    batch, tokens = 2, 6
    input_ids = torch.randint(999, 2000, (batch, tokens))
    attention_mask = torch.ones(batch, tokens, dtype=torch.long)
    outputs = backbone.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    assert len(outputs.hidden_states) == 13  # embeddings + 12 layers
    stacked = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert stacked.shape[1] == 12
    # Index 0 of our stack is layer 1, not the embedding output.
    assert torch.allclose(stacked[:, 0], outputs.hidden_states[1], atol=1e-5)
    assert not torch.allclose(stacked[:, 0], outputs.hidden_states[0], atol=1e-3)


def test_embedding_layer_can_be_included_explicitly():
    model = BertBackbone(config.MODEL_NAME, include_embedding_layer=True)
    assert model.num_layers == 13
    input_ids = torch.randint(999, 2000, (1, 5))
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    assert model(input_ids=input_ids, attention_mask=attention_mask).shape[1] == 13


def test_frozen_backbone_stays_in_eval_mode_after_train():
    model = BertBackbone(config.MODEL_NAME, freeze=True)
    model.train()
    assert model.encoder.training is False, "frozen backbone must not re-enable dropout"
    assert all(not p.requires_grad for p in model.encoder.parameters())
    counts = model.count_parameters()
    assert counts["trainable"] == 0 and counts["frozen"] == counts["total"]


def test_frozen_backbone_is_deterministic_under_train_mode():
    model = BertBackbone(config.MODEL_NAME, freeze=True)
    model.train()
    input_ids = torch.randint(999, 2000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    with torch.no_grad():
        first = model(input_ids=input_ids, attention_mask=attention_mask)
        second = model(input_ids=input_ids, attention_mask=attention_mask)
    assert torch.allclose(first, second, atol=1e-6)


def test_unfreeze_restores_training_mode():
    model = BertBackbone(config.MODEL_NAME, freeze=True)
    model.unfreeze()
    model.train()
    assert model.encoder.training is True
    assert all(p.requires_grad for p in model.encoder.parameters())
