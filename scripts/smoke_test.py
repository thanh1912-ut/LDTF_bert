"""Synthetic end-to-end smoke test.

Runs every baseline and ablation through a real training loop on a tiny
synthetic AG-News-shaped dataset: forward, backward, optimizer coverage,
gradient audit, checkpoint write, reload, resume, and evaluation. Uses the real
BERT tokenizer and a randomly initialised BERT with a reduced number of layers
so it is fast and requires no pretrained download beyond the tokenizer.

Run with::

    python -m scripts.smoke_test > smoke.log 2>&1
    status=$?; tail -n 40 smoke.log; echo "exit=$status"
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig, BertModel

from src import config, guard
from src.dataset import build_eval_dataloader, build_train_dataloader
from src.evaluate import evaluate_dataloader, load_model_from_checkpoint
from src.metrics import bootstrap_accuracy_difference, mcnemar_exact
from src.models import LdtfBert
from src.registry import ABLATION_RUNS, BASELINE_RUNS, build_model
from src.train import (
    TrainConfig,
    build_optimizer,
    forward_kwargs,
    gradient_audit,
    optimizer_coverage_report,
    train_model,
)
from src.utils import set_seed

TINY_BERT = BertConfig(
    vocab_size=30522,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=128,
    max_position_embeddings=128,
)

SAMPLE_TEXTS = [
    ("Peace talks resume between the two governments", 0),
    ("The striker scored twice in the second half", 1),
    ("Shares fell after the quarterly earnings report", 2),
    ("Researchers unveil a faster processor design", 3),
    ("Aid convoy reaches the border region", 0),
    ("The team advanced to the championship final", 1),
    ("Oil prices climbed on supply concerns", 2),
    ("A new satellite was launched this morning", 3),
]


class _Failure(Exception):
    """Raised when a smoke check fails."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise _Failure(message)
    print(f"  PASS  {message}", flush=True)


def patch_backbone_to_tiny() -> None:
    """Make BertModel.from_pretrained build a small randomly initialised model."""
    original = BertModel.from_pretrained

    def tiny(*args, **kwargs):  # noqa: ANN001
        add_pooling_layer = kwargs.get("add_pooling_layer", True)
        return BertModel(TINY_BERT, add_pooling_layer=add_pooling_layer)

    BertModel.from_pretrained = staticmethod(tiny)  # type: ignore[assignment]
    return original


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "text": text,
            "title_clean": text.split(" ")[0],
            "description_clean": " ".join(text.split(" ")[1:]),
            "label": label,
        }
        for text, label in SAMPLE_TEXTS
    ]
    frame = pd.DataFrame(rows * 4)
    return frame, pd.DataFrame(rows * 2)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic LDTF-BERT smoke test.")
    parser.add_argument("--quick", action="store_true", help="Only exercise A0 and B2.")
    args = parser.parse_args(argv)

    set_seed(config.SEED)
    patch_backbone_to_tiny()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
    train_frame, val_frame = build_frames()

    runs = ["A0", "B2_bert_finetuned_cls"] if args.quick else (
        list(ABLATION_RUNS) + list(BASELINE_RUNS)
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        print("\n[1] forward, backward, optimizer coverage, gradient audit", flush=True)
        for run_id in runs:
            model = build_model(run_id)
            loader = build_train_dataloader(train_frame, tokenizer, batch_size=4, seed=0)
            batch = next(iter(loader))
            outputs = model(**forward_kwargs(batch))
            check(
                outputs["logits"].shape == (4, config.NUM_CLASSES),
                f"{run_id}: logits shape {tuple(outputs['logits'].shape)}",
            )
            check(set(outputs) == {"logits"}, f"{run_id}: default output is logits only")
            nn.CrossEntropyLoss()(outputs["logits"], batch["labels"]).backward()
            audit = gradient_audit(model)
            check(audit["ok"], f"{run_id}: no unused or non-finite trainable parameters")
            coverage = optimizer_coverage_report(
                model, build_optimizer(model, TrainConfig(output_dir=root))
            )
            check(coverage["covered"], f"{run_id}: optimizer covers the trainable set")

        print("\n[2] routing diagnostics for the reference model", flush=True)
        reference = build_model("A0")
        loader = build_train_dataloader(train_frame, tokenizer, batch_size=4, seed=0)
        batch = next(iter(loader))
        outputs = reference(**forward_kwargs(batch), return_routing=True, return_features=True)
        token_attention = outputs["token_attention"]
        depth_attention = outputs["depth_attention"]
        check(
            torch.allclose(
                token_attention.sum(-1), torch.ones_like(token_attention.sum(-1)), atol=1e-4
            ),
            "token attention sums to 1 over valid tokens",
        )
        check(
            torch.allclose(
                depth_attention.sum(-1), torch.ones_like(depth_attention.sum(-1)), atol=1e-4
            ),
            "depth attention sums to 1 over layers",
        )
        check(torch.isfinite(token_attention).all(), "token attention has no NaN or Inf")
        check(torch.isfinite(depth_attention).all(), "depth attention has no NaN or Inf")
        padding = batch["attention_mask"] == 0
        if padding.any():
            check(
                float(token_attention.permute(0, 3, 1, 2)[padding].abs().max()) == 0.0,
                "padding tokens receive zero attention",
            )
        check(
            float((depth_attention[:, 0, :] - depth_attention[:, 1, :]).abs().max()) > 0,
            "different classes obtain different layer distributions",
        )

        print("\n[3] excluding special tokens", flush=True)
        excluded = build_model("A14")(**forward_kwargs(batch), return_routing=True)
        attention = excluded["token_attention"]
        special = batch["special_tokens_mask"].bool()
        check(
            float(attention.permute(0, 3, 1, 2)[special].abs().max()) == 0.0,
            "special and padding tokens receive zero attention under A14",
        )
        check(
            torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=1e-4),
            "content-token attention still sums to 1 under A14",
        )

        print("\n[4] batch size 1 and variable sequence lengths", flush=True)
        for size in (1, 2, 3):
            single = build_eval_dataloader(
                val_frame.head(size), tokenizer, batch_size=size, split_name="validation"
            )
            small_batch = next(iter(single))
            logits = reference(**forward_kwargs(small_batch))["logits"]
            check(
                logits.shape == (size, config.NUM_CLASSES),
                f"batch size {size} produces logits {tuple(logits.shape)}",
            )

        print("\n[5] training loop, checkpoints and resume", flush=True)
        train_loader = build_train_dataloader(train_frame, tokenizer, batch_size=4, seed=0)
        val_loader = build_eval_dataloader(val_frame, tokenizer, batch_size=4)
        output_dir = root / "A0_smoke"
        train_config = TrainConfig(
            output_dir=output_dir,
            run_id="A0_smoke",
            epochs=1,
            use_amp=False,
            grad_accum_steps=2,
            data_signature={"train_sha256": "smoke"},
            architecture=reference.architecture_config(),
        )
        first = train_model(reference, train_loader, val_loader, train_config)
        check(first.best_epoch == 1, "training completed and selected an epoch")
        check((output_dir / "best.pt").exists(), "best.pt was written")
        check((output_dir / "last.pt").exists(), "last.pt was written")

        best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
        check("optimizer_state_dict" not in best, "best.pt is slim (no optimizer state)")
        check(
            all(
                key in best
                for key in ("architecture", "data_signature", "protocol", "best_metrics", "seed")
            ),
            "best.pt carries architecture, data signature, protocol and metrics",
        )
        last = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
        check(
            all(
                key in last
                for key in (
                    "optimizer_state_dict",
                    "scheduler_state_dict",
                    "scaler_state_dict",
                    "global_step",
                    "rng_state",
                    "loader_generator_state",
                    "patience_counter",
                )
            ),
            "last.pt is fully resumable",
        )

        resumed = train_model(
            build_model("A0"),
            build_train_dataloader(train_frame, tokenizer, batch_size=4, seed=0),
            build_eval_dataloader(val_frame, tokenizer, batch_size=4),
            TrainConfig(
                output_dir=output_dir,
                run_id="A0_smoke",
                epochs=2,
                use_amp=False,
                grad_accum_steps=2,
                data_signature={"train_sha256": "smoke"},
                architecture=reference.architecture_config(),
            ),
            resume=True,
        )
        check(resumed.resumed_from_epoch == 1, "resume continued from the last epoch")
        check(
            [record["epoch"] for record in resumed.history] == [1, 2],
            "resumed run preserved and extended the history",
        )

        print("\n[6] checkpoint reload and evaluation", flush=True)
        restored, state = load_model_from_checkpoint(output_dir / "best.pt")
        metrics = evaluate_dataloader(restored, val_loader, device=torch.device("cpu"))
        check(
            {"accuracy", "f1_macro", "confusion_matrix", "per_class"} <= set(metrics),
            "evaluation returns accuracy, macro F1, confusion matrix and per-class scores",
        )
        check(0.0 <= metrics["accuracy"] <= 1.0, "accuracy lies in [0, 1]")

        print("\n[7] frozen regime", flush=True)
        frozen = build_model("B6_full_ldtf_frozen")
        frozen.train()
        check(
            frozen.backbone.encoder.training is False,
            "frozen backbone stays in eval mode after model.train()",
        )
        frozen_out = frozen(**forward_kwargs(batch))
        nn.CrossEntropyLoss()(frozen_out["logits"], batch["labels"]).backward()
        check(
            all(p.grad is None for p in frozen.backbone.parameters()),
            "frozen backbone receives no gradient",
        )
        check(
            float(frozen.label_queries.queries.grad.abs().sum()) > 0,
            "frozen-backbone run still trains the label queries",
        )

        print("\n[8] statistical comparison helpers", flush=True)
        import numpy as np

        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        first_predictions = np.array([0, 1, 2, 3, 0, 1, 2, 0])
        second_predictions = np.array([0, 1, 2, 0, 0, 1, 0, 0])
        test = mcnemar_exact(first_predictions, second_predictions, labels)
        check(0.0 <= test["p_value"] <= 1.0, "McNemar exact test returns a valid p-value")
        interval = bootstrap_accuracy_difference(
            first_predictions, second_predictions, labels, num_resamples=200
        )
        check(
            interval["ci_lower_95"] <= interval["mean_difference"] <= interval["ci_upper_95"],
            "bootstrap interval brackets the observed difference",
        )

        print("\n[9] official test split remains sealed", flush=True)
        check(not guard.training_active(), "training flag cleared after the run")
        try:
            guard.official_test_loader(tokenizer, reason="smoke", run_id="smoke")
            raise _Failure("official test loader must refuse without the unlock token")
        except guard.OfficialTestAccessError:
            check(True, "official test split refuses access without the unlock token")
        check(
            not config.TEST_ACCESS_LEDGER.exists()
            or config.TEST_ACCESS_LEDGER.stat().st_size >= 0,
            "official test access ledger untouched by the smoke test",
        )

    print("\nSMOKE TEST PASSED", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except _Failure as error:
        print(f"\nSMOKE TEST FAILED: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
