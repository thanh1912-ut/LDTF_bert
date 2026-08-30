"""Neural baselines B1-B4, all sharing the LDTF output contract.

``B1`` frozen BERT, final-layer [CLS] -> Dropout -> Linear(D, C)
``B2`` fine-tuned BERT, final-layer [CLS] -> Dropout -> Linear(D, C)
``B3`` frozen BERT, masked mean pooling -> Dropout -> Linear(D, C)
``B4`` BERT scalar mix over layers -> masked mean pooling -> Linear(D, C)

Every model returns ``{"logits": [B, C]}`` with raw logits, computes no loss,
and applies no softmax over the class axis.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .. import config
from .bert_backbone import BertBackbone


def masked_mean_pool(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean over non-padding tokens. ``[B, T, D]`` and ``[B, T]`` -> ``[B, D]``."""
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class BertPooledClassifier(nn.Module):
    """BERT -> final-layer pooling -> Dropout -> Linear(D, C).

    ``pooling='cls'`` implements B1/B2, ``pooling='mean'`` implements B3.
    """

    POOLINGS = ("cls", "mean")

    def __init__(
        self,
        *,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        pooling: str = "cls",
        freeze_backbone: bool = False,
        dropout: float = config.SCORER_DROPOUT,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        if pooling not in self.POOLINGS:
            raise ValueError(f"pooling must be one of {self.POOLINGS}, got {pooling!r}.")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes!r}.")

        self.model_name = model_name
        self.pooling = pooling
        self.num_classes = num_classes
        self.backbone = BertBackbone(
            model_name=model_name, cache_dir=cache_dir, freeze=freeze_backbone
        )
        self.hidden_size = self.backbone.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze()

    def head_parameters(self):
        yield from self.dropout.parameters()
        yield from self.classifier.parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        hidden_state = self.backbone.last_hidden_state(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if self.pooling == "cls":
            pooled = hidden_state[:, 0, :]
        else:
            pooled = masked_mean_pool(hidden_state, attention_mask)
        return {"logits": self.classifier(self.dropout(pooled))}

    def count_parameters(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for name, module in self.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        counts["total"] = {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "non_backbone": total - backbone_total,
        }
        return counts

    def architecture_config(self) -> dict[str, object]:
        return {
            "class_name": "BertPooledClassifier",
            "model_name": self.model_name,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "backbone_frozen": self.backbone.frozen,
        }

    @staticmethod
    def build_tokenizer(
        model_name: str = config.MODEL_NAME, cache_dir: Optional[str] = None
    ):
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)


class BertScalarMixClassifier(nn.Module):
    """B4: global learned layer weights, no label conditioning.

    ``weights = softmax(w)`` over the ``L`` layer logits, one global vector
    shared by all classes and independent of the input, followed by a learned
    scale ``gamma`` and masked mean pooling. This mirrors the ELMo/AllenNLP
    ``ScalarMix`` parameterisation applied to BERT layer outputs; it is not
    ELMo itself, since the layers are Transformer layers rather than biLM
    states and the mixture feeds a classifier rather than a downstream encoder.
    """

    def __init__(
        self,
        *,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        freeze_backbone: bool = False,
        dropout: float = config.SCORER_DROPOUT,
        use_gamma: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = BertBackbone(
            model_name=model_name, cache_dir=cache_dir, freeze=freeze_backbone
        )
        self.hidden_size = self.backbone.hidden_size
        self.num_layers = self.backbone.num_layers
        self.layer_logits = nn.Parameter(torch.zeros(self.num_layers))
        self.gamma = nn.Parameter(torch.ones(1)) if use_gamma else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze()

    def head_parameters(self):
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        for parameter in self.parameters():
            if id(parameter) not in backbone_ids:
                yield parameter

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        *,
        return_routing: bool = False,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # [B, L, T, D]
        weights = torch.softmax(self.layer_logits.float(), dim=0).to(hidden_states.dtype)
        mixed = torch.einsum("l,bltd->btd", weights, hidden_states)
        if self.gamma is not None:
            mixed = mixed * self.gamma
        pooled = masked_mean_pool(mixed, attention_mask)
        outputs = {"logits": self.classifier(self.dropout(pooled))}
        if return_routing:
            outputs["layer_weights"] = weights
        return outputs

    def count_parameters(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for name, module in self.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        counts["scalar_mix"] = {
            "total": self.layer_logits.numel() + (self.gamma.numel() if self.gamma is not None else 0),
            "trainable": self.layer_logits.numel()
            + (self.gamma.numel() if self.gamma is not None else 0),
            "frozen": 0,
        }
        counts["total"] = {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "non_backbone": total - backbone_total,
        }
        return counts

    def architecture_config(self) -> dict[str, object]:
        return {
            "class_name": "BertScalarMixClassifier",
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "num_layers": self.num_layers,
            "backbone_frozen": self.backbone.frozen,
        }

    @staticmethod
    def build_tokenizer(
        model_name: str = config.MODEL_NAME, cache_dir: Optional[str] = None
    ):
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
