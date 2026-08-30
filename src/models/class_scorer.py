"""Scorers mapping fused per-class features ``[B, C, D]`` to logits ``[B, C]``.

Three separate implementations. Exactly one is registered per model, so an
unused scorer never appears in the optimizer, the state dict, or the parameter
counts.

============================  ==========================  ==================
Scorer                        Parameters                  Count (D=768,C=4)
============================  ==========================  ==================
SharedLinearScorer            ``D + 1``                   769
SharedMlpScorer (H=768)       ``(D*H + H) + (H + 1)``     591,361
ClassSpecificLinearScorer     ``C*D + C``                 3,076
============================  ==========================  ==================

``SharedLinearScorer`` is the reference scorer for Full LDTF. The shared-MLP
variant adds 590,592 parameters over it when ``H = D = 768``, so it is a
capacity ablation and not a default.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import config


def _validate_class_features(class_features: torch.Tensor) -> None:
    if class_features.ndim != 3:
        raise ValueError(
            f"class_features must be [B,C,D], got {tuple(class_features.shape)}."
        )


class SharedLinearScorer(nn.Module):
    """``Linear(D, 1)`` applied identically to every class. Reference scorer."""

    def __init__(
        self,
        hidden_size: int = 768,
        *,
        dropout: float = config.SCORER_DROPOUT,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size!r}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}.")
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, class_features: torch.Tensor) -> torch.Tensor:
        _validate_class_features(class_features)
        return self.projection(self.dropout(class_features)).squeeze(-1)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class SharedMlpScorer(nn.Module):
    """``Dropout -> Linear(D,H) -> GELU -> Dropout -> Linear(H,1)``, shared."""

    def __init__(
        self,
        hidden_size: int = 768,
        *,
        mlp_hidden_size: int = config.SCORER_HIDDEN_SIZE,
        dropout: float = config.SCORER_DROPOUT,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or mlp_hidden_size <= 0:
            raise ValueError("hidden_size and mlp_hidden_size must be > 0.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}.")
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 1),
        )
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, class_features: torch.Tensor) -> torch.Tensor:
        _validate_class_features(class_features)
        return self.layers(class_features).squeeze(-1)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class ClassSpecificLinearScorer(nn.Module):
    """One weight row and bias per class: ``logit_c = <f_c, w_c> + b_c``.

    This is *not* ``Linear(D, C)`` applied to one shared representation; each
    class scores its own fused vector with its own weight row.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = config.NUM_CLASSES,
        *,
        dropout: float = config.SCORER_DROPOUT,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_classes < 2:
            raise ValueError("hidden_size must be > 0 and num_classes >= 2.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}.")
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.class_weights = nn.Parameter(torch.empty(num_classes, hidden_size))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.class_weights)

    def forward(self, class_features: torch.Tensor) -> torch.Tensor:
        _validate_class_features(class_features)
        if class_features.shape[1] != self.num_classes:
            raise ValueError(
                f"class_features has {class_features.shape[1]} classes, expected "
                f"{self.num_classes}."
            )
        features = self.dropout(class_features)
        return torch.einsum("bcd,cd->bc", features, self.class_weights) + self.class_bias

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


SCORERS = {
    "shared_linear": SharedLinearScorer,
    "shared_mlp": SharedMlpScorer,
    "class_specific": ClassSpecificLinearScorer,
}


def build_scorer(
    name: str,
    hidden_size: int,
    num_classes: int,
    *,
    dropout: float = config.SCORER_DROPOUT,
    mlp_hidden_size: int = config.SCORER_HIDDEN_SIZE,
) -> nn.Module:
    """Instantiate exactly one scorer by name."""
    if name not in SCORERS:
        raise ValueError(f"Unknown scorer {name!r}; expected one of {sorted(SCORERS)}.")
    if name == "shared_linear":
        return SharedLinearScorer(hidden_size, dropout=dropout)
    if name == "shared_mlp":
        return SharedMlpScorer(
            hidden_size, mlp_hidden_size=mlp_hidden_size, dropout=dropout
        )
    return ClassSpecificLinearScorer(hidden_size, num_classes, dropout=dropout)
