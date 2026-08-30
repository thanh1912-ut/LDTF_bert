"""Shared label query bank: one query vector per class.

The same ``[C, D]`` tensor is consumed by the Token Router and by the Direct
Depth Router, so both routing stages are conditioned on the same class
representation and both contribute gradient to it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import config


class LabelQueryBank(nn.Module):
    """Holds one query vector per class.

    Modes
    -----
    ``trainable``     learnable ``nn.Parameter`` (default, used by Full LDTF).
    ``frozen``        learnable-shaped parameter with ``requires_grad=False``,
                      initialised identically to ``trainable`` (ablation A5).
    ``fixed_random``  seeded random buffer, never trained and never in the
                      optimizer or in the parameter list (ablation A6).
    """

    MODES = ("trainable", "frozen", "fixed_random")

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden_size: int = 768,
        *,
        mode: str = "trainable",
        init_std: float = config.LABEL_QUERY_INIT_STD,
        seed: int = config.SEED,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, but received {num_classes!r}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, but received {hidden_size!r}.")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}.")

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.mode = mode
        self.init_std = init_std
        self.seed = seed

        # Draw from a dedicated generator so the initial values are identical
        # across modes and independent of surrounding module construction order.
        generator = torch.Generator().manual_seed(seed)
        initial = torch.empty(num_classes, hidden_size).normal_(
            mean=0.0, std=init_std, generator=generator
        )

        if mode == "fixed_random":
            self.register_buffer("queries", initial)
        else:
            self.queries = nn.Parameter(initial)
            if mode == "frozen":
                self.queries.requires_grad_(False)

    def forward(self) -> torch.Tensor:
        """Return the ``[C, D]`` query tensor."""
        return self.queries

    def freeze(self) -> None:
        """Disable gradients for the queries."""
        if isinstance(self.queries, nn.Parameter):
            self.queries.requires_grad_(False)
            self.mode = "frozen"

    def unfreeze(self) -> None:
        """Enable gradients for the queries."""
        if isinstance(self.queries, nn.Parameter):
            self.queries.requires_grad_(True)
            self.mode = "trainable"

    def count_parameters(self) -> dict[str, int]:
        """Return total / trainable / frozen parameter counts.

        ``fixed_random`` queries are a buffer and therefore contribute zero.
        """
        if not isinstance(self.queries, nn.Parameter):
            return {"total": 0, "trainable": 0, "frozen": 0}
        total = self.queries.numel()
        trainable = total if self.queries.requires_grad else 0
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, hidden_size={self.hidden_size}, "
            f"mode={self.mode}, init_std={self.init_std}, seed={self.seed}"
        )
