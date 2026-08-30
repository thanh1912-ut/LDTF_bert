"""Variant specifications for baselines and LDTF ablations."""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import config


@dataclass(frozen=True)
class LdtfVariant:
    """A complete, serializable LDTF architecture choice.

    ``token_routing`` is either ``"label"`` or ``"none"``. The latter is
    intentionally restricted: a no-token variant is only valid with direct
    depth routing because otherwise a shared scorer receives identical class
    representations and the four-way classifier is mathematically degenerate.
    """

    name: str = "A0_full_direct_ldtf"
    token_routing: str = "label"
    depth_routing: str = "direct"
    scorer: str = "shared_linear"
    router_dim: int = config.TOKEN_ROUTER_DIM
    depth_router_dim: int = config.DEPTH_ROUTER_DIM
    scorer_hidden_size: int = config.SCORER_HIDDEN_SIZE
    query_mode: str = "trainable"
    query_seed: int = config.SEED
    include_special_tokens: bool = config.INCLUDE_SPECIAL_TOKENS
    use_all_layers: bool = True
    backbone_frozen: bool = False
    include_embedding_layer: bool = config.INCLUDE_EMBEDDING_LAYER
    num_classes: int = config.NUM_CLASSES

    def __post_init__(self) -> None:
        if self.token_routing not in {"label", "none"}:
            raise ValueError(f"Unknown token_routing={self.token_routing!r}.")
        if self.depth_routing not in {"direct", "indirect", "scalar_mix", "uniform", "none"}:
            raise ValueError(f"Unknown depth_routing={self.depth_routing!r}.")
        if self.scorer not in {"shared_linear", "shared_mlp", "class_specific"}:
            raise ValueError(f"Unknown scorer={self.scorer!r}.")
        if self.query_mode not in {"trainable", "frozen", "fixed_random"}:
            raise ValueError(f"Unknown query_mode={self.query_mode!r}.")
        if self.token_routing == "none" and self.depth_routing != "direct":
            raise ValueError(
                "A no-token-router variant is degenerate with a shared scorer "
                "unless direct depth routing preserves class identity."
            )
        if not self.use_all_layers and self.depth_routing not in {"uniform", "none"}:
            raise ValueError("Final-layer-only variants cannot register a depth router.")


def baseline_variant(name: str) -> LdtfVariant:
    """Return the LDTF-side configuration for B5-B7."""
    mapping = {
        "B5_token_attention_only": dict(depth_routing="uniform"),
        "B6_full_ldtf_frozen": dict(backbone_frozen=True),
        "B7_full_ldtf_finetuned": dict(backbone_frozen=False),
    }
    if name not in mapping:
        raise ValueError(f"No LDTF variant exists for {name!r}.")
    return LdtfVariant(name=name, **mapping[name])


def ablation_variant(name: str) -> LdtfVariant:
    """Return one of the requested A0-A15 variants.

    A7 is an alias of A0 because shared linear is the A0 reference scorer;
    A12 is the reference router width; A13 is the reference special-token
    policy; A15 is a regime pair rather than a separate architecture.
    """
    variants = {
        "A0": LdtfVariant(name="A0_full_direct_ldtf"),
        "A1": LdtfVariant(name="A1_no_depth_router", depth_routing="uniform"),
        "A2": LdtfVariant(
            name="A2_final_layer_only", depth_routing="uniform", use_all_layers=False
        ),
        "A3": LdtfVariant(name="A3_global_scalar_mix", depth_routing="scalar_mix"),
        "A4": LdtfVariant(name="A4_indirect_depth_gating", depth_routing="indirect"),
        "A5": LdtfVariant(name="A5_frozen_label_queries", query_mode="frozen"),
        "A6": LdtfVariant(name="A6_fixed_random_label_queries", query_mode="fixed_random"),
        "A7": LdtfVariant(name="A7_shared_linear_scorer"),
        "A8": LdtfVariant(name="A8_shared_mlp_scorer", scorer="shared_mlp"),
        "A9": LdtfVariant(name="A9_class_specific_scorer", scorer="class_specific"),
        "A10": LdtfVariant(name="A10_router_dim_64", router_dim=64, depth_router_dim=64),
        "A11": LdtfVariant(name="A11_router_dim_128", router_dim=128, depth_router_dim=128),
        "A12": LdtfVariant(name="A12_router_dim_256", router_dim=256, depth_router_dim=256),
        "A13": LdtfVariant(name="A13_include_special_tokens", include_special_tokens=True),
        "A14": LdtfVariant(name="A14_exclude_special_tokens", include_special_tokens=False),
    }
    if name == "A15":
        raise ValueError("A15 is a frozen-vs-fine-tuned regime pair, not one architecture.")
    if name not in variants:
        raise ValueError(f"Unknown ablation {name!r}.")
    return variants[name]
