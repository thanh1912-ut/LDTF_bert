# Ablation Results (seed 42)

> Rows marked PENDING have not been executed. No metric in these tables is
> estimated, fabricated, or copied from a fixture. Official-test metrics are
> produced solely by `python -m experiments.final_eval`.

> A0 is the reference. Delta F1 is computed only when both A0 and the row
> have real official-test metrics.

| ID | Variant | Token routing | Depth routing | Layer policy | Query policy | Scorer | Params | Macro F1 | Delta F1 vs A0 |
|---|---|---|---|---|---|---|---:|---:|---:|
| A0 | Full Direct LDTF | label-conditioned | direct label-conditioned | all layers | trainable | shared linear | 109,681,921 | PENDING | PENDING |
| A1 | No Depth Router | label-conditioned | uniform layer mean | all layers | trainable | shared linear | 109,288,705 | PENDING | PENDING |
| A2 | Final Layer Only | label-conditioned | none | final layer | trainable | shared linear | 109,288,705 | PENDING | PENDING |
| A3 | Global Scalar Mix | label-conditioned | global scalar mix | all layers | trainable | shared linear | 109,288,718 | PENDING | PENDING |
| A4 | Indirect Depth Gating | label-conditioned | indirect gating | all layers | trainable | shared linear | 109,297,921 | PENDING | PENDING |
| A5 | Frozen Label Queries | label-conditioned | direct label-conditioned | all layers | frozen | shared linear | 109,678,849 | PENDING | PENDING |
| A6 | Fixed Random Label Queries | label-conditioned | direct label-conditioned | all layers | fixed random | shared linear | 109,678,849 | PENDING | PENDING |
| A7 | Shared Linear Scorer | label-conditioned | direct label-conditioned | all layers | trainable | shared linear | 109,681,921 | PENDING | PENDING |
| A8 | Shared MLP Scorer | label-conditioned | direct label-conditioned | all layers | trainable | shared MLP | 110,272,513 | PENDING | PENDING |
| A9 | Class-Specific Linear Scorer | label-conditioned | direct label-conditioned | all layers | trainable | class-specific | 109,684,228 | PENDING | PENDING |
| A10 | Router Dimension 64 | label-conditioned (R=64) | direct (R=64) | all layers | trainable | shared linear | 109,092,097 | PENDING | PENDING |
| A11 | Router Dimension 128 | label-conditioned (R=128) | direct (R=128) | all layers | trainable | shared linear | 109,288,705 | PENDING | PENDING |
| A12 | Router Dimension 256 | label-conditioned (R=256) | direct (R=256) | all layers | trainable | shared linear | 109,681,921 | PENDING | PENDING |
| A13 | Include Special Tokens | label-conditioned (+special) | direct label-conditioned | all layers | trainable | shared linear | 109,681,921 | PENDING | PENDING |
| A14 | Exclude Special Tokens | label-conditioned (-special) | direct label-conditioned | all layers | trainable | shared linear | 109,681,921 | PENDING | PENDING |
| A15 | Frozen vs Fine-Tuned | label-conditioned | direct label-conditioned | all layers | trainable | shared linear | 109,681,921 | PENDING | PENDING |
