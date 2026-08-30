# Run Plan — what to run, in what order, and why

Generated from the code, not from memory: the run ids below come from
`src/registry.py` and `experiments/run_suite.py`.

**Total distinct configurations: 19** (not 24 — five listed ids are duplicates of
others and must not be run twice; see §4).

Every run trains and validates only. The official test split stays sealed until
§6.

---

## 1. Baselines (8 rows in the baseline table)

| ID | Run id | Backbone | Epochs | Purpose |
|---|---|---|---|---|
| B0 | `B0_tfidf_logreg` | none | n/a | classical floor, data sanity check |
| B1 | `B1_bert_frozen_cls` | frozen | 10 | frozen `[CLS]` probe |
| B2 | `B2_bert_finetuned_cls` | fine-tuned | 3 | **the baseline everything must beat** |
| B3 | `B3_bert_frozen_mean_pool` | frozen | 10 | is B1 artificially weak? |
| B4 | `B4_bert_scalar_mix` | fine-tuned | 3 | multi-layer, **no** label conditioning |
| B5 | `B5_token_attention_only` | fine-tuned | 3 | = **A1**, do not run twice |
| B6 | `B6_full_ldtf_frozen` | frozen | 10 | full LDTF, frozen |
| B7 | `B7_full_ldtf_finetuned` | fine-tuned | 3 | = **A0**, do not run twice |

```bash
python -m experiments.run_tfidf_baseline
python -m experiments.run_suite --preset baselines --seed 42
```

## 2. Ablations (16 rows in the ablation table)

| ID | Run id | What changes vs A0 | Non-backbone params |
|---|---|---|---|
| A0 | `A0` | reference | 790,273 |
| A1 | `A1` | uniform layer mean, no depth router | 397,057 |
| A2 | `A2` | final layer only | 397,057 |
| A3 | `A3` | global scalar mix (label-agnostic depth) | 397,070 |
| A4 | `A4` | indirect depth gating | 406,273 |
| A5 | `A5` | frozen label queries | 790,273 |
| A6 | `A6` | fixed random label queries | 787,201 |
| A7 | `A7` | **identical to A0** — do not run | 790,273 |
| A8 | `A8` | shared MLP scorer (+590,592) | 1,380,865 |
| A9 | `A9` | class-specific scorer | 792,580 |
| A10 | `A10` | router dim 64 | 200,449 |
| A11 | `A11` | router dim 128 | 397,057 |
| A12 | `A12` | **identical to A0** — do not run | 790,273 |
| A13 | `A13` | **identical to A0** — do not run | 790,273 |
| A14 | `A14` | exclude `[CLS]`/`[SEP]` | 790,273 |
| A15 | `A0 --frozen-backbone` | regime pair, not a new architecture | 790,273 |

```bash
python -m experiments.run_suite --preset ablations --seed 42
python -m experiments.run_experiment --run A0 --frozen-backbone --seed 42   # A15
```

## 3. Minimum set that answers the research question

If GPU time is limited, run these five first. They decide whether the
contribution is real:

```bash
python -m experiments.run_suite --preset core --seed 42     # A0, A1, A3, A4, A11
python -m experiments.run_experiment --run B2_bert_finetuned_cls --seed 42
```

| Comparison | Question | Parameter-matched? |
|---|---|---|
| **A0 vs A3** | Does *label conditioning* of depth matter, or is a global learned mixture enough? | no |
| **A11 vs A1** | Does depth routing help at an identical budget (both 397,057)? | **yes, exactly** |
| A0 vs A1 | Does depth routing help at all? | no (A0 is 2x the head) |
| A0 vs A4 | Direct vs indirect conditioning | no (A4 is 43x smaller) |
| A0 vs B2 | Does any of this beat plain fine-tuned BERT? | no |

**A0 vs A3 is the decisive one.** A3 keeps a learned per-layer mixture and
removes only the label conditioning. If A3 matches A0, the contribution reduces
to multi-layer fusion, which is ELMo (2018), and the novelty claim fails.

## 4. Do not run these

| Skip | Reason |
|---|---|
| A7 | identical config to A0 (shared linear is the A0 reference scorer) |
| A12 | identical config to A0 (router dim 256 is the reference width) |
| A13 | identical config to A0 (include-special is the reference policy) |
| B5 | identical config to A1 |
| B7 | identical config to A0 |

Report A0's number in those rows and note the aliasing. Running them again would
produce a second sample of the same configuration and inflate the apparent
number of independent results.

## 5. Seeds

One seed is not enough to support any claim. At n = 7,600 and accuracy near
0.94, a single run's standard error is about 0.27 pp, so a gap below roughly
0.75 pp is indistinguishable from noise.

```bash
python -m experiments.run_suite --preset core --seed 42 --seed 1337 --seed 2024
```

Report mean +/- std across seeds, and use the paired tests in `src/metrics.py`
(`mcnemar_exact`, `bootstrap_accuracy_difference`) rather than comparing two
single numbers.

## 6. Final evaluation — once, at the very end

Only after every architecture, hyper-parameter, epoch and seed decision is
final:

```bash
export LDTF_ALLOW_OFFICIAL_TEST=I_AM_REPORTING_FINAL_RESULTS
python -m experiments.final_eval --run A0_seed42 --run A1_seed42 \
    --reason "final reported numbers"
```

Each access is logged to `reports/official_test_access.jsonl`. The
`access_index` there is how many times the test set has ever been touched and
belongs in the write-up.

## 7. Export the tables

```bash
python -m experiments.export_tables --with-params
```

Writes `reports/tables/baseline_table.md` and `ablation_table.md`. Rows not yet
run show `PENDING`; nothing is estimated or back-filled.

## 8. Cost

Rough guidance only — measure the first epoch and extrapolate rather than
trusting these numbers.

- Fine-tuned runs: 3 epochs over 107,735 rows, batch 32 = 3,367 steps/epoch.
- Frozen runs: 10 epochs, but no backward pass through BERT.
- LDTF variants materialise a `[B, C, L, T]` attention tensor, so they cost
  meaningfully more per step than B1-B4. Reduce `--batch-size` if you hit OOM.
- 19 configurations x 3 seeds = 57 runs. On a single Colab T4 that is a
  multi-day job; start with §3 at one seed, then widen.

Disk: each run writes `best.pt` (~440 MB) plus `last.pt` (~1.3 GB). 19 runs is
roughly 33 GB. Colab's disk will not hold the full suite, so copy results to
Drive and delete `last.pt` for finished runs, or run in batches.
