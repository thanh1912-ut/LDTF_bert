"""Direct label-conditioned token attention over every Transformer layer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .. import config


class TokenRouter(nn.Module):
    """Route valid tokens separately for every class and every layer.

    Inputs:
      hidden_states: ``[B, L, T, D]``
      label_queries: ``[C, D]``
      attention_mask: ``[B, T]`` (1 = non-padding)
      special_tokens_mask: optional ``[B, T]`` (1 = [CLS]/[SEP]/other special)

    Outputs:
      token_attention: ``[B, C, L, T]``
      token_features: ``[B, C, L, D]``
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = config.NUM_CLASSES,
        router_dim: int = config.TOKEN_ROUTER_DIM,
        projection_bias: bool = config.PROJECTION_BIAS,
        *,
        include_special_tokens: bool = config.INCLUDE_SPECIAL_TOKENS,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_classes < 2 or router_dim <= 0:
            raise ValueError("hidden_size, router_dim must be positive and num_classes >= 2.")
        if not isinstance(projection_bias, bool):
            raise ValueError("projection_bias must be bool.")

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.router_dim = router_dim
        self.projection_bias = projection_bias
        self.include_special_tokens = include_special_tokens
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

    def _valid_mask(
        self,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = attention_mask.bool()
        if not self.include_special_tokens and special_tokens_mask is not None:
            valid = valid & ~special_tokens_mask.bool()
        elif not self.include_special_tokens and special_tokens_mask is None:
            raise ValueError(
                "special_tokens_mask is required when include_special_tokens=False."
            )
        valid_counts = valid.sum(dim=-1)
        if (valid_counts == 0).any():
            bad = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(f"Each sample needs a valid routing token; bad rows: {bad}.")
        return valid

    def forward(
        self,
        hidden_states: torch.Tensor,
        label_queries: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden_states.ndim != 4:
            raise ValueError(f"hidden_states must be [B,L,T,D], got {tuple(hidden_states.shape)}.")
        if label_queries.shape != (self.num_classes, self.hidden_size):
            raise ValueError(
                f"label_queries must be [{self.num_classes},{self.hidden_size}], "
                f"got {tuple(label_queries.shape)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be [B,T], got {tuple(attention_mask.shape)}.")
        batch, _, tokens, hidden = hidden_states.shape
        if hidden != self.hidden_size or attention_mask.shape != (batch, tokens):
            raise ValueError(
                f"hidden_states [B,L,T,D] and attention_mask [B,T] disagree: "
                f"{tuple(hidden_states.shape)}, {tuple(attention_mask.shape)}."
            )
        if special_tokens_mask is not None and special_tokens_mask.shape != (batch, tokens):
            raise ValueError(
                f"special_tokens_mask must be [B,T]={batch,tokens}, "
                f"got {tuple(special_tokens_mask.shape)}."
            )

        projected_queries = self.query_projection(label_queries)  # [C,R]
        projected_keys = self.key_projection(hidden_states)  # [B,L,T,R]
        token_scores = torch.einsum(
            "cr,bltr->bclt", projected_queries, projected_keys
        ) * self.scale

        valid = self._valid_mask(attention_mask, special_tokens_mask)
        routing_mask = valid[:, None, None, :]
        # FP32 masked softmax is intentional for AMP stability.
        masked_scores = token_scores.float().masked_fill(~routing_mask, float("-inf"))
        token_attention = torch.softmax(masked_scores, dim=-1).to(hidden_states.dtype)
        if not torch.isfinite(token_attention).all():
            raise FloatingPointError("Token attention contains NaN or Inf values.")
        token_features = torch.einsum(
            "bclt,bltd->bcld", token_attention, hidden_states
        )
        return {
            "token_attention": token_attention,
            "token_features": token_features,
        }

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
