"""Model construction from a run identifier or a checkpoint's architecture block."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from . import config
from .models import BertPooledClassifier, BertScalarMixClassifier, LdtfBert
from .variants import LdtfVariant, ablation_variant, baseline_variant

#: Every runnable neural configuration. B0 is handled by experiments/run_tfidf.py.
BASELINE_RUNS = {
    "B1_bert_frozen_cls": dict(kind="pooled", pooling="cls", frozen=True),
    "B2_bert_finetuned_cls": dict(kind="pooled", pooling="cls", frozen=False),
    "B3_bert_frozen_mean_pool": dict(kind="pooled", pooling="mean", frozen=True),
    "B4_bert_scalar_mix": dict(kind="scalar_mix", frozen=False),
    "B5_token_attention_only": dict(kind="ldtf", variant="B5_token_attention_only"),
    "B6_full_ldtf_frozen": dict(kind="ldtf", variant="B6_full_ldtf_frozen"),
    "B7_full_ldtf_finetuned": dict(kind="ldtf", variant="B7_full_ldtf_finetuned"),
}

ABLATION_RUNS = tuple(f"A{index}" for index in range(15))  # A0..A14; A15 is a regime pair


def build_variant(run_id: str) -> LdtfVariant:
    """Return the LDTF variant for an ablation id or an LDTF baseline id."""
    if run_id in {"B5_token_attention_only", "B6_full_ldtf_frozen", "B7_full_ldtf_finetuned"}:
        return baseline_variant(run_id)
    return ablation_variant(run_id)


def build_model(
    run_id: str,
    *,
    model_name: str = config.MODEL_NAME,
    cache_dir: Optional[str] = None,
    frozen_backbone: bool | None = None,
) -> nn.Module:
    """Instantiate the model for *run_id*.

    ``run_id`` is either a baseline key from :data:`BASELINE_RUNS` or an
    ablation id ``A0``-``A14``. ``frozen_backbone`` overrides the variant's
    regime, which is how the A15 frozen-vs-fine-tuned pair is produced.
    """
    if run_id in BASELINE_RUNS:
        spec = BASELINE_RUNS[run_id]
        kind = spec["kind"]
        if kind == "pooled":
            frozen = spec["frozen"] if frozen_backbone is None else frozen_backbone
            return BertPooledClassifier(
                model_name=model_name,
                pooling=spec["pooling"],
                freeze_backbone=frozen,
                cache_dir=cache_dir,
            )
        if kind == "scalar_mix":
            frozen = spec["frozen"] if frozen_backbone is None else frozen_backbone
            return BertScalarMixClassifier(
                model_name=model_name, freeze_backbone=frozen, cache_dir=cache_dir
            )
        variant = baseline_variant(spec["variant"])
    else:
        variant = ablation_variant(run_id)

    if frozen_backbone is not None:
        variant = LdtfVariant(**{**variant.__dict__, "backbone_frozen": frozen_backbone})
    return LdtfBert(variant, model_name=model_name, cache_dir=cache_dir)


def build_model_from_architecture(
    architecture: dict, *, cache_dir: Optional[str] = None
) -> nn.Module:
    """Rebuild a model from a checkpoint's ``architecture`` block."""
    class_name = architecture.get("class_name")
    model_name = architecture.get("model_name", config.MODEL_NAME)
    if class_name == "LdtfBert":
        variant = LdtfVariant(**architecture["variant"])
        return LdtfBert(variant, model_name=model_name, cache_dir=cache_dir)
    if class_name == "BertPooledClassifier":
        return BertPooledClassifier(
            model_name=model_name,
            pooling=architecture["pooling"],
            num_classes=architecture.get("num_classes", config.NUM_CLASSES),
            freeze_backbone=architecture.get("backbone_frozen", False),
            cache_dir=cache_dir,
        )
    if class_name == "BertScalarMixClassifier":
        return BertScalarMixClassifier(
            model_name=model_name,
            num_classes=architecture.get("num_classes", config.NUM_CLASSES),
            freeze_backbone=architecture.get("backbone_frozen", False),
            cache_dir=cache_dir,
        )
    raise ValueError(f"Cannot rebuild unknown model class {class_name!r}.")


def is_frozen_run(run_id: str) -> bool:
    """Return whether *run_id* trains with a frozen backbone by default."""
    if run_id in BASELINE_RUNS:
        spec = BASELINE_RUNS[run_id]
        if spec["kind"] in {"pooled", "scalar_mix"}:
            return bool(spec["frozen"])
        return baseline_variant(spec["variant"]).backbone_frozen
    return ablation_variant(run_id).backbone_frozen
