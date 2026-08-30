"""Central configuration for AG News topic classification with LDTF-BERT.

This module holds *paths*, *dataset facts*, and *default hyper-parameters*.
Per-run architecture choices live in :mod:`src.variants`; per-run training
choices live in :class:`src.train.TrainConfig`.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
DOCS_DIR = ROOT_DIR / "docs"
REFERENCES_DIR = ROOT_DIR / "references"

PROCESSED_TRAIN = PROCESSED_DIR / "research_train.parquet"
PROCESSED_VAL = PROCESSED_DIR / "research_validation.parquet"
PROCESSED_TEST = PROCESSED_DIR / "research_test.parquet"

#: Append-only ledger recording every access to the official test split.
TEST_ACCESS_LEDGER = REPORTS_DIR / "official_test_access.jsonl"

# ---------------------------------------------------------------------------
# Dataset facts
#
# Verified from data/processed/*.parquet on 2026-08-20:
#   research_train.parquet       107,735 rows  (26974/26965/26914/26882)
#   research_validation.parquet   11,971 rows  (2997/2996/2991/2987)
#   research_test.parquet          7,600 rows  (1900 per class, official split)
# Both CSV and Parquet are supported by :func:`src.dataset.load_split`.
# ---------------------------------------------------------------------------
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
TITLE_COLUMN = "title_clean"
DESCRIPTION_COLUMN = "description_clean"
NUM_CLASSES = 4
LABEL_NAMES = ("World", "Sports", "Business", "Sci/Tech")

#: How the two AG News text fields are fed to the tokenizer.
#: ``"single"``  -> the pre-joined ``text`` column, single-segment encoding.
#: ``"pair"``    -> (title_clean, description_clean) sentence-pair encoding,
#:                  which produces meaningful ``token_type_ids``.
#: All runs in a comparison MUST use the same value.
TEXT_ENCODING = "single"

# ---------------------------------------------------------------------------
# Tokenizer / backbone
# ---------------------------------------------------------------------------
MODEL_NAME = "bert-base-uncased"

#: Truncation policy. token_length_report.json (tokenizer=bert-base-uncased,
#: 127,306 samples) reports median=50, p95=73, p99=97, max=233 and
#: percentage_gt_128 = 0.253%. MAX_LENGTH=128 therefore truncates ~0.25% of
#: examples from the right (``truncation=True`` = longest_first).
MAX_LENGTH = 128

# ---------------------------------------------------------------------------
# LDTF module defaults
# ---------------------------------------------------------------------------
TOKEN_ROUTER_DIM = 256
DEPTH_ROUTER_DIM = 256
PROJECTION_BIAS = False
SCORER_DROPOUT = 0.1
SCORER_HIDDEN_SIZE = 768  # only used by the shared-MLP scorer ablation
LABEL_QUERY_INIT_STD = 0.02  # matches BERT's initializer_range

#: Whether [CLS]/[SEP] may receive token-routing attention mass.
#: A13 = True (include), A14 = False (exclude). Reference A0 = True.
INCLUDE_SPECIAL_TOKENS = True

#: Whether the BERT embedding output (hidden_states[0]) counts as a routable
#: "layer". HuggingFace returns num_hidden_layers + 1 tensors, index 0 being
#: the embedding output. We route over the 12 Transformer layers only, i.e.
#: ``all_hidden_states[1:]``. See docs/architecture_traceability.md.
INCLUDE_EMBEDDING_LAYER = False

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
SEED = 42
SEEDS = (42, 1337, 2024)
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 3
FROZEN_EPOCHS = 10  # frozen probes need more passes at a fixed backbone
BACKBONE_LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
GRAD_ACCUM_STEPS = 1
LABEL_SMOOTHING = 0.0
EARLY_STOPPING_PATIENCE = 0  # 0 disables early stopping
NUM_WORKERS = 0

#: Checkpoint selection. Primary = validation macro F1 (maximise).
#: Tie-break order: higher macro F1 -> lower validation loss -> earlier epoch.
SELECTION_METRIC = "val_f1_macro"

#: TF-IDF baseline (B0). Tuned on validation only; never on the official test.
TFIDF_MAX_FEATURES = 200_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_C_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)


def experiment_output_dir(run_id: str) -> Path:
    """Return (and do not create) the output directory for *run_id*."""
    return OUTPUTS_DIR / run_id


def ensure_output_dirs() -> None:
    """Create every on-disk directory used during training and reporting."""
    for directory in (OUTPUTS_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR, DOCS_DIR):
        os.makedirs(directory, exist_ok=True)
