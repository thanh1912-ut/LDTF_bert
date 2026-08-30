"""B0: TF-IDF + Logistic Regression.

Classical baseline and data-pipeline sanity check. The regularisation strength
``C`` is selected on the *validation* split only; the official test split is
never read here.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import config
from src.dataset import data_signature, load_split
from src.metrics import compute_classification_metrics
from src.utils import save_json, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TF-IDF + LR baseline (B0).")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-features", type=int, default=config.TFIDF_MAX_FEATURES)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_output_dirs()
    set_seed(args.seed)

    run_id = f"B0_tfidf_logreg_seed{args.seed}"
    output_dir = config.experiment_output_dir(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = load_split(config.PROCESSED_TRAIN)
    val_frame = load_split(config.PROCESSED_VAL)
    train_text = train_frame[config.TEXT_COLUMN].astype(str).tolist()
    val_text = val_frame[config.TEXT_COLUMN].astype(str).tolist()
    train_labels = train_frame[config.LABEL_COLUMN].astype(int).to_numpy()
    val_labels = val_frame[config.LABEL_COLUMN].astype(int).to_numpy()

    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_features = vectorizer.fit_transform(train_text)
    val_features = vectorizer.transform(val_text)

    search: list[dict[str, float]] = []
    best_score = -1.0
    best_c = config.TFIDF_C_GRID[0]
    best_model: LogisticRegression | None = None
    for penalty_c in config.TFIDF_C_GRID:
        classifier = LogisticRegression(
            C=penalty_c, max_iter=args.max_iter, n_jobs=-1, random_state=args.seed
        )
        classifier.fit(train_features, train_labels)
        scores = classifier.decision_function(val_features)
        metrics = compute_classification_metrics(scores, val_labels)
        search.append(
            {
                "C": penalty_c,
                "val_f1_macro": metrics["f1_macro"],
                "val_accuracy": metrics["accuracy"],
            }
        )
        print(
            f"[B0] C={penalty_c}: val macro F1={metrics['f1_macro']:.4f} "
            f"accuracy={metrics['accuracy']:.4f}",
            flush=True,
        )
        if metrics["f1_macro"] > best_score:
            best_score = float(metrics["f1_macro"])
            best_c = penalty_c
            best_model = classifier

    assert best_model is not None
    val_metrics = compute_classification_metrics(
        best_model.decision_function(val_features), val_labels
    )
    elapsed = time.perf_counter() - started

    import joblib

    joblib.dump(
        {"vectorizer": vectorizer, "classifier": best_model}, output_dir / "best.joblib"
    )
    num_params = int(
        np.prod(best_model.coef_.shape) + best_model.intercept_.size
    )
    save_json(
        {
            "run_id": run_id,
            "model": "TF-IDF + Logistic Regression",
            "seed": args.seed,
            "selected_C": best_c,
            "selection_metric": "val_f1_macro",
            "search": search,
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "total_parameters": num_params,
            "trainable_parameters": num_params,
            "best_val_f1_macro": val_metrics["f1_macro"],
            "best_val_accuracy": val_metrics["accuracy"],
            "total_train_seconds": round(elapsed, 2),
            "data_signature": data_signature(),
        },
        output_dir / "run_summary.json",
    )
    print(
        f"[B0] selected C={best_c}; val macro F1={val_metrics['f1_macro']:.4f} "
        f"accuracy={val_metrics['accuracy']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
