# Baseline Results (seed 42)

> Rows marked PENDING have not been executed. No metric in these tables is
> estimated, fabricated, or copied from a fixture. Official-test metrics are
> produced solely by `python -m experiments.final_eval`.

| ID | Model | Backbone regime | Token mechanism | Depth mechanism | Scorer | Total params | Trainable params | Accuracy | Macro F1 | Peak VRAM | Time/epoch |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| B0 | TF-IDF + Logistic Regression | none | bag-of-ngrams | none | class-specific LR | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B1 | BERT Frozen CLS | frozen | final-layer [CLS] | final layer | Linear(D,C) | 108,894,724 | 3,076 | PENDING | PENDING | PENDING | PENDING |
| B2 | BERT Fine-Tuned CLS | fine-tuned | final-layer [CLS] | final layer | Linear(D,C) | 108,894,724 | 108,894,724 | PENDING | PENDING | PENDING | PENDING |
| B3 | BERT Frozen Mean Pool | frozen | masked mean pool | final layer | Linear(D,C) | 108,894,724 | 3,076 | PENDING | PENDING | PENDING | PENDING |
| B4 | BERT Scalar Mix | fine-tuned | masked mean pool | global scalar mix | Linear(D,C) | 108,894,737 | 108,894,737 | PENDING | PENDING | PENDING | PENDING |
| B5 | Label Token Attention Only | fine-tuned | label-conditioned | uniform layer mean | shared linear | 109,288,705 | 109,288,705 | PENDING | PENDING | PENDING | PENDING |
| B6 | Full LDTF Frozen | frozen | label-conditioned | direct label-conditioned | shared linear | 109,681,921 | 790,273 | PENDING | PENDING | PENDING | PENDING |
| B7 | Full LDTF Fine-Tuned | fine-tuned | label-conditioned | direct label-conditioned | shared linear | 109,681,921 | 109,681,921 | PENDING | PENDING | PENDING | PENDING |
