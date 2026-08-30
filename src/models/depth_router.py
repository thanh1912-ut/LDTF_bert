"""Depth-fusion modules that reduce ``[B, C, L, D]`` to ``[B, C, D]``.

Four mechanisms, all sharing the output contract
``{"depth_attention": [B, C, L], "class_features": [B, C, D]}``:

``DirectDepthRouter``
    Label-conditioned attention over layers. The label queries are projected to
    queries and each layer's per-class feature vector is projected to a key, so
    every layer is scored **individually** and the layer distribution depends on
    both the class and the input. This is the mechanism used by Full LDTF.

``IndirectDepthGating``
    The original implementation of this project, corrected and renamed. It
    averages the layer axis first and then predicts one logit per layer index
    from that pooled vector. It is *not* label-conditioned (no label query
    enters it) and its layer scores are invariant to a permutation of the layer
    axis. Retained only as ablation A4; it must never be described as direct
    label-conditioned depth routing.

``GlobalScalarMix``
    One global learned weight per layer, shared by all classes and independent
    of the input, with an optional learned scale. Ablation A3 / baseline B4.

``UniformLayerMean``
    Parameter-free mean over layers. Ablation A1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import config


def _validate_token_features(token_features: torch.Tensor) -> None:
    if token_features.ndim != 4:
        raise ValueError(
            f"token_features must be [B,C,L,D], got {tuple(token_features.shape)}."
        )


class DirectDepthRouter(nn.Module):
    """Direct label-conditioned attention over the layer axis."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = config.NUM_CLASSES,
        router_dim: int = config.DEPTH_ROUTER_DIM,
        projection_bias: bool = config.PROJECTION_BIAS,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or router_dim <= 0 or num_classes < 2:
            raise ValueError("hidden_size, router_dim must be positive and num_classes >= 2.")

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.router_dim = router_dim
        self.scale = router_dim ** -0.5

        self.query_projection = nn.Linear(hidden_size, router_dim, bias=projection_bias)
        self.key_projection = nn.Linear(hidden_size, router_dim, bias=projection_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        if self.query_projection.bias is not None:
            nn.init.zeros_(self.query_projection.bias)
        if self.key_projection.bias is not None:
            nn.init.zeros_(self.key_projection.bias)

    def forward(
        self,
        token_features: torch.Tensor,
        label_queries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _validate_token_features(token_features)
        if label_queries.shape != (self.num_classes, self.hidden_size):
            raise ValueError(
                f"label_queries must be [{self.num_classes},{self.hidden_size}], "
                f"got {tuple(label_queries.shape)}."
            )
        if token_features.shape[1] != self.num_classes:
            raise ValueError(
                f"token_features has {token_features.shape[1]} classes, expected "
                f"{self.num_classes}."
            )

        projected_queries = self.query_projection(label_queries)  # [C,Rd]
        projected_keys = self.key_projection(token_features)  # [B,C,L,Rd]
        depth_scores = torch.einsum(
            "cr,bclr->bcl", projected_queries, projected_keys
        ) * self.scale
        depth_attention = torch.softmax(depth_scores.float(), dim=-1).to(token_features.dtype)
        class_features = torch.einsum("bcl,bcld->bcd", depth_attention, token_features)
        return {"depth_attention": depth_attention, "class_features": class_features}

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class IndirectDepthGating(nn.Module):
    """Layer gating from a layer-averaged summary vector (ablation A4).

    ``scores = Linear(mean_over_layers(token_features))`` produces one logit per
    layer *index*. No label query participates, and the scores do not depend on
    which layer contributed which vector.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = config.NUM_CLASSES,
        num_layers: int = 12,
        projection_bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_layers < 1 or num_classes < 2:
            raise ValueError("hidden_size, num_layers must be positive and num_classes >= 2.")
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.scale = hidden_size ** -0.5
        self.gate_projection = nn.Linear(hidden_size, num_layers, bias=projection_bias)
        nn.init.xavier_uniform_(self.gate_projection.weight)
        if self.gate_projection.bias is not None:
            nn.init.zeros_(self.gate_projection.bias)

    def forward(self, token_features: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_token_features(token_features)
        if token_features.shape[2] != self.num_layers:
            raise ValueError(
                f"token_features has {token_features.shape[2]} layers, expected "
                f"{self.num_layers}."
            )
        pooled = token_features.mean(dim=2)  # [B,C,D] — pools the LAYER axis
        layer_scores = self.gate_projection(pooled) * self.scale  # [B,C,L]
        depth_attention = torch.softmax(layer_scores.float(), dim=-1).to(token_features.dtype)
        class_features = torch.einsum("bcl,bcld->bcd", depth_attention, token_features)
        return {"depth_attention": depth_attention, "class_features": class_features}

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class GlobalScalarMix(nn.Module):
    """One global softmax-normalised weight per layer, shared by all classes.

    Follows the ELMo-style parameterisation ``gamma * sum_l softmax(w)_l * x_l``
    (AllenNLP ``ScalarMix``), with the layer weights initialised to zero so the
    mixture starts uniform. It is applied here to per-class token-routed
    features rather than to token-level biLM states; see
    docs/architecture_traceability.md for the difference.
    """

    def __init__(self, num_layers: int = 12, *, use_gamma: bool = True) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers!r}.")
        self.num_layers = num_layers
        self.use_gamma = use_gamma
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.ones(1)) if use_gamma else None

    def forward(self, token_features: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_token_features(token_features)
        if token_features.shape[2] != self.num_layers:
            raise ValueError(
                f"token_features has {token_features.shape[2]} layers, expected "
                f"{self.num_layers}."
            )
        weights = torch.softmax(self.layer_logits.float(), dim=0).to(token_features.dtype)
        class_features = torch.einsum("l,bcld->bcd", weights, token_features)
        if self.gamma is not None:
            class_features = class_features * self.gamma
        batch, classes = token_features.shape[0], token_features.shape[1]
        depth_attention = weights.view(1, 1, -1).expand(batch, classes, self.num_layers)
        return {"depth_attention": depth_attention, "class_features": class_features}

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class UniformLayerMean(nn.Module):
    """Parameter-free uniform mean across layers (ablation A1 / A2)."""

    def forward(self, token_features: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_token_features(token_features)
        num_layers = token_features.shape[2]
        class_features = token_features.mean(dim=2)
        depth_attention = token_features.new_full(
            (token_features.shape[0], token_features.shape[1], num_layers),
            1.0 / num_layers,
        )
        return {"depth_attention": depth_attention, "class_features": class_features}

    def count_parameters(self) -> dict[str, int]:
        return {"total": 0, "trainable": 0, "frozen": 0}
