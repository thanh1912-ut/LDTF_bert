"""Full LDTF-BERT reference model and its ablation variants.

Reference architecture (A0 / B7)::

    BERT per-layer hidden states  [B, L, T, D]
      -> one shared Label Query Bank        [C, D]
      -> label-conditioned Token Router     [B, C, L, D]
      -> direct label-conditioned Depth Router  [B, C, D]
      -> shared Linear(D, 1) scorer         [B, C]

The model returns raw logits. It never computes the loss, never applies a
softmax over the class axis, and never takes an argmax.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .. import config
from ..variants import LdtfVariant
from .bert_backbone import BertBackbone
from .class_scorer import build_scorer
from .depth_router import (
    DirectDepthRouter,
    GlobalScalarMix,
    IndirectDepthGating,
    UniformLayerMean,
)
from .label_query_bank import LabelQueryBank
from .token_router import TokenRouter


class LdtfBert(nn.Module):
    """Label-Directed Token and Depth Fusion over BERT."""

    def __init__(
        self,
        variant: LdtfVariant | None = None,
        *,
        model_name: str = config.MODEL_NAME,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.variant = variant or LdtfVariant()
        self.model_name = model_name
        num_classes = self.variant.num_classes

        self.backbone = BertBackbone(
            model_name=model_name,
            cache_dir=cache_dir,
            freeze=self.variant.backbone_frozen,
            include_embedding_layer=self.variant.include_embedding_layer,
        )
        hidden_size = self.backbone.hidden_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        #: Number of layers actually routed over (1 when use_all_layers=False).
        self.num_routed_layers = (
            self.backbone.num_layers if self.variant.use_all_layers else 1
        )

        self.label_queries = LabelQueryBank(
            num_classes=num_classes,
            hidden_size=hidden_size,
            mode=self.variant.query_mode,
            seed=self.variant.query_seed,
        )

        if self.variant.token_routing == "label":
            self.token_router = TokenRouter(
                hidden_size=hidden_size,
                num_classes=num_classes,
                router_dim=self.variant.router_dim,
                include_special_tokens=self.variant.include_special_tokens,
            )
        else:
            self.token_router = None

        # Register exactly one depth module; ablated modules are absent from the
        # optimizer, the state dict and the parameter counts.
        depth = self.variant.depth_routing
        self.depth_router: nn.Module | None
        if depth == "direct":
            self.depth_router = DirectDepthRouter(
                hidden_size=hidden_size,
                num_classes=num_classes,
                router_dim=self.variant.depth_router_dim,
            )
        elif depth == "indirect":
            self.depth_router = IndirectDepthGating(
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_layers=self.num_routed_layers,
            )
        elif depth == "scalar_mix":
            self.depth_router = GlobalScalarMix(num_layers=self.num_routed_layers)
        elif depth == "uniform":
            self.depth_router = UniformLayerMean()
        else:
            self.depth_router = None

        self.class_scorer = build_scorer(
            self.variant.scorer,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=config.SCORER_DROPOUT,
            mlp_hidden_size=self.variant.scorer_hidden_size,
        )

    # -- regime control ----------------------------------------------------
    def freeze_backbone(self) -> None:
        """Freeze BERT and keep it in eval mode."""
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        """Unfreeze BERT."""
        self.backbone.unfreeze()

    def head_parameters(self):
        """Yield every non-backbone parameter."""
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        for parameter in self.parameters():
            if id(parameter) not in backbone_ids:
                yield parameter

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        *,
        return_routing: bool = False,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return ``{"logits": [B, C]}``, plus routing/features on request."""
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # [B, L, T, D]
        if not self.variant.use_all_layers:
            hidden_states = hidden_states[:, -1:, :, :]  # [B, 1, T, D]

        label_queries = self.label_queries()  # [C, D]

        if self.token_router is not None:
            token_out = self.token_router(
                hidden_states=hidden_states,
                label_queries=label_queries,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
            )
            token_attention = token_out["token_attention"]  # [B, C, L, T]
            token_features = token_out["token_features"]  # [B, C, L, D]
        else:
            # Class-agnostic masked mean pooling, broadcast over the class axis.
            # Class identity is then introduced solely by the direct depth router.
            mask = attention_mask[:, None, :, None].to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1e-6)
            token_features = pooled.unsqueeze(1).expand(
                -1, self.num_classes, -1, -1
            )  # [B, C, L, D]
            token_attention = None

        if self.depth_router is None:
            if token_features.shape[2] != 1:
                raise RuntimeError(
                    "No depth module is registered but multiple layers were routed."
                )
            class_features = token_features.squeeze(2)
            depth_attention = token_features.new_ones(
                token_features.shape[0], self.num_classes, 1
            )
        elif isinstance(self.depth_router, DirectDepthRouter):
            depth_out = self.depth_router(
                token_features=token_features, label_queries=label_queries
            )
            class_features = depth_out["class_features"]
            depth_attention = depth_out["depth_attention"]
        else:
            depth_out = self.depth_router(token_features)
            class_features = depth_out["class_features"]
            depth_attention = depth_out["depth_attention"]

        logits = self.class_scorer(class_features)  # [B, C]

        outputs: dict[str, torch.Tensor] = {"logits": logits}
        if return_routing or return_features:
            outputs["depth_attention"] = depth_attention
            if token_attention is not None:
                outputs["token_attention"] = token_attention
        if return_features:
            outputs["hidden_states"] = hidden_states
            outputs["label_queries"] = label_queries
            outputs["token_features"] = token_features
            outputs["fused_features"] = class_features
        return outputs

    # -- introspection -----------------------------------------------------
    def count_parameters(self) -> dict[str, dict[str, int]]:
        """Return parameter counts per registered module plus totals."""
        counts: dict[str, dict[str, int]] = {}
        for name, module in self.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {
                "total": total,
                "trainable": trainable,
                "frozen": total - trainable,
            }
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
        """Return a JSON-serializable description sufficient to rebuild self."""
        from dataclasses import asdict

        return {
            "class_name": "LdtfBert",
            "model_name": self.model_name,
            "variant": asdict(self.variant),
        }

    @staticmethod
    def build_tokenizer(
        model_name: str = config.MODEL_NAME, cache_dir: Optional[str] = None
    ):
        """Load the tokenizer matching the backbone."""
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
