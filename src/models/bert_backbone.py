"""BERT encoder exposing per-layer hidden states as ``[B, L, T, D]``.

Output contract
---------------
``forward`` returns ``hidden_states`` of shape ``[B, L, T, D]``.

HuggingFace's ``BertModel(..., output_hidden_states=True)`` returns a tuple of
``num_hidden_layers + 1`` tensors of shape ``[B, T, D]``, where index 0 is the
*embedding output* and indices 1..L are the Transformer layer outputs
(https://huggingface.co/docs/transformers/main_classes/output).

This wrapper takes ``all_hidden_states[1:]`` by default, so ``L`` equals
``config.num_hidden_layers`` (12 for ``bert-base-uncased``) and the embedding
output is **excluded**. Set ``include_embedding_layer=True`` to keep it, in
which case ``L = num_hidden_layers + 1`` and the embedding output is explicitly
treated as a routable depth.

The pretrained pooler is not loaded (``add_pooling_layer=False``) because LDTF
consumes hidden states only and never ``pooler_output``; keeping it would
register 590,592 trainable parameters that receive no gradient.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import BertModel

from .. import config


class BertBackbone(nn.Module):
    """BERT encoder returning stacked per-layer hidden states."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        *,
        cache_dir: Optional[str] = None,
        freeze: bool = False,
        include_embedding_layer: bool = config.INCLUDE_EMBEDDING_LAYER,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.include_embedding_layer = bool(include_embedding_layer)

        self.encoder = BertModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            add_pooling_layer=False,
        )
        self.encoder.config.output_hidden_states = True

        # Read the geometry from the pretrained config; never hard-code it.
        self.hidden_size: int = self.encoder.config.hidden_size
        self.num_transformer_layers: int = self.encoder.config.num_hidden_layers
        self.num_layers: int = self.num_transformer_layers + (
            1 if self.include_embedding_layer else 0
        )

        self._frozen = False
        if freeze:
            self.freeze()

    # -- regime control ----------------------------------------------------
    def freeze(self) -> None:
        """Disable gradients and force eval mode (no dropout) permanently."""
        self._frozen = True
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def unfreeze(self) -> None:
        """Re-enable gradients and normal train/eval mode switching."""
        self._frozen = False
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    @property
    def frozen(self) -> bool:
        """Whether the backbone is frozen."""
        return self._frozen

    def train(self, mode: bool = True) -> "BertBackbone":
        """Keep a frozen encoder in eval mode so its dropout stays disabled."""
        super().train(mode)
        if self._frozen:
            self.encoder.eval()
        return self

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return hidden states ``[B, L, T, D]``."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        all_hidden_states = outputs.hidden_states
        if all_hidden_states is None:
            raise RuntimeError(
                "BertModel returned no hidden_states; output_hidden_states must be True."
            )
        selected = (
            all_hidden_states if self.include_embedding_layer else all_hidden_states[1:]
        )
        if len(selected) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} hidden-state tensors, got {len(selected)}."
            )
        return torch.stack(selected, dim=1)

    def last_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return only the final-layer hidden state ``[B, T, D]``."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def count_parameters(self) -> dict[str, int]:
        """Return total / trainable / frozen parameter counts."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def extra_repr(self) -> str:
        return (
            f"model_name={self.model_name}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, include_embedding_layer="
            f"{self.include_embedding_layer}, frozen={self._frozen}"
        )
