# LDTF-BERT — Label-Directed Token and Depth Fusion for AG News

Label-conditioned routing over BERT's per-layer hidden states for single-label
4-way topic classification.

```
BERT per-layer hidden states  [B, L, T, D]
  -> one shared Label Query Bank          [C, D]
  -> label-conditioned Token Router       [B, C, L, D]
  -> direct label-conditioned Depth Router [B, C, D]
  -> shared Linear(D, 1) scorer            [B, C]
```

The distinguishing component is the **Direct Depth Router**: the layer
distribution is produced by attention between the label queries and each layer's
per-class feature vector, so layer weights are `[B, C, L]` — per example and per
class — rather than a single global `[L]` vector as in a scalar mix.

**Status: implementation complete, experiments not run.** Every metric in
`reports/tables/` is `PENDING`. No claim is made that LDTF outperforms any
baseline. See `docs/architecture_traceability.md` for what each comparison would
have to show.

## Layout

```
src/
  config.py            paths, dataset facts, defaults
  variants.py          LdtfVariant — one serializable architecture choice
  registry.py          run id -> model; checkpoint -> model
  dataset.py           batched tokenization, dynamic padding, DDP samplers
  distributed.py       torchrun/DDP runtime and rank-safe collectives
  guard.py             the seal around the official test split
  metrics.py           accuracy, macro F1, McNemar, paired bootstrap
  train.py             optimizer groups, AMP, accumulation, checkpoints, resume
  evaluate.py          inference and checkpoint evaluation
  models/
    bert_backbone.py   BERT -> [B, L, T, D], no pooler
    label_query_bank.py  [C, D], trainable / frozen / fixed-random
    token_router.py    label-conditioned token attention
    depth_router.py    DirectDepthRouter, IndirectDepthGating,
                       GlobalScalarMix, UniformLayerMean
    class_scorer.py    shared-linear, shared-MLP, class-specific
    ldtf_bert.py       the full model
    baselines.py       B1-B4
experiments/           run_experiment, run_tfidf_baseline, final_eval, export_tables
scripts/               smoke_test, build_data_report
tests/                 unit tests
docs/                  prior art, traceability, source audit, changelog
```

## Setup

```bash
pip install -r requirements.txt
```

Expected data at `data/processed/`: `research_train.parquet` (107,735 rows),
`research_validation.parquet` (11,971), `research_test.parquet` (7,600, sealed).
On Kaggle, set `LDTF_DATA_DIR=/kaggle/input/<dataset-name>` when that directory
contains `processed/`; no copy into the repository is required.

## Running

```bash
# classical baseline
python -m experiments.run_tfidf_baseline

# any baseline or ablation
python -m experiments.run_experiment --run B2_bert_finetuned_cls
python -m experiments.run_experiment --run A0 --seed 42

# the A15 frozen-vs-fine-tuned pair: same architecture, different regime
python -m experiments.run_experiment --run A0 --frozen-backbone
python -m experiments.run_experiment --run A0 --finetune-backbone

# resume an interrupted run from last.pt
python -m experiments.run_experiment --run A0 --resume
```

If a run directory already contains artifacts, the single-run entrypoint asks
for either `--resume` or `--force`; force-started artifacts are moved under that
run's `previous_runs/` directory instead of being deleted. Suite runs resume an
incomplete `last.pt` automatically unless `--restart-incomplete` is supplied.

### Kaggle dual T4

The entrypoints auto-detect `torchrun`; no separate DDP flag or shell script is
needed. `--batch-size` is the **global micro-batch**, so 32 is split into 16
examples per GPU and preserves the single-GPU protocol. Gradient accumulation
then gives `effective batch = global batch × accumulation steps`.

In a Kaggle notebook cell:

```python
!python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m experiments.run_experiment --run A0 --batch-size 32 \
    --eval-batch-size 64 --num-workers 1 --pad-to-multiple-of 8
```

For a suite, use the same launcher and remove `--continue-on-error`:

```python
!python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m experiments.run_suite --preset core --seed 42 \
    --batch-size 32 --eval-batch-size 64 --num-workers 1 \
    --pad-to-multiple-of 8
```

The T4 training path automatically uses FP16 autocast with gradient scaling and
fused AdamW; validation/model selection stays FP32 to preserve the original
protocol. Validation is sharded without padding, so every one of the 11,971 rows is
counted exactly once. Training uses PyTorch's equal-length distributed sampler;
because 107,735 is odd, one training index is repeated per epoch and that count
is recorded as `runtime.train_sampler_padding`.

Start with one loader worker per rank on Kaggle's small CPU allocation; benchmark
0/1/2 for the actual notebook. `--pad-to-multiple-of 8` is an explicit throughput
choice and is recorded in every summary/checkpoint. Omit it if strict continuity
with an older run matters. `--fast-nondeterministic` enables cuDNN autotuning but
must be applied consistently across every compared run.

`scripts.benchmark_input_pipeline` measures one rank's tokenizer path (use batch
16 for global batch 32 on two GPUs); it does not emulate two ranks contending for
the same CPUs. Use `run_summary.json` seconds/epoch for the final 0/1/2-worker
comparison on the real Kaggle runtime.

Protocol-safe `--resume` requires the same planned epoch count, world size,
global/per-device batch, accumulation, determinism and padding settings as the
interrupted run. Once official-test artifacts exist, that run directory is
immutable for training; use a new output directory/run id for any new fit. A
slim `best.pt` remains portable to normal single-process inference/evaluation.

Do **not** launch `experiments.final_eval` with `torchrun`; it intentionally
refuses multi-process execution so the sealed test ledger is touched once.

Run ids: `A0`–`A14` for ablations (`A15` is a regime pair, not an architecture),
and `B1_bert_frozen_cls`, `B2_bert_finetuned_cls`, `B3_bert_frozen_mean_pool`,
`B4_bert_scalar_mix`, `B5_token_attention_only`, `B6_full_ldtf_frozen`,
`B7_full_ldtf_finetuned`.

Each run writes to `outputs/<run_id>_seed<seed>/`: `best.pt` (slim),
`last.pt` (resumable), `train_log.jsonl`, `val_metrics.json`, `run_summary.json`.

## The official test split is sealed

Training never constructs a test loader. `src.dataset.load_split` refuses the
official test path outright. The only way through is:

```bash
export LDTF_ALLOW_OFFICIAL_TEST=I_AM_REPORTING_FINAL_RESULTS
python -m experiments.final_eval --run A0_seed42 --reason "final reported numbers"
```

which additionally refuses while a training loop is active and appends a
hash-stamped record to `reports/official_test_access.jsonl`. The `access_index`
in that ledger is the number of times the test set has ever been touched, and it
belongs in any write-up.

Do not use test results to choose an architecture, a training regime, a scorer,
a router dimension, an epoch, or a seed.

## Protocol

- **Selection metric:** validation macro F1. Ties broken by lower validation
  loss, then earlier epoch.
- **Learning rates:** 2e-5 backbone, 1e-3 head, four optimizer groups split by
  decay / no-decay.
- **Frozen runs:** backbone `requires_grad=False`, kept in `eval()` so its
  dropout stays off, and absent from the optimizer.
- **Held constant across runs:** split, cleaner, tokenizer, max length,
  validation protocol, selection metric, seed, training engine, metrics, world
  size, effective global batch and deterministic/throughput settings.

## Verification

```bash
python -m compileall src experiments scripts tests

python -m pytest tests -q > tests.log 2>&1
rc=$?; tail -n 20 tests.log; echo "exit=$rc"

python -m scripts.smoke_test > smoke.log 2>&1
rc=$?; tail -n 40 smoke.log; echo "exit=$rc"

# input-only per-rank benchmark for global batch 32 on T4 x2
python -m scripts.benchmark_input_pipeline --batch-size 16 --num-workers 1

python -m experiments.export_tables --with-params
python -m scripts.build_data_report
```

The smoke test builds a small randomly initialised BERT and drives all 22 neural
configurations through forward, backward, optimizer coverage, gradient audit,
checkpointing, resume and evaluation.

## Reading the results

`reports/tables/baseline_table.md` and `ablation_table.md` are generated from run
artifacts. Rows that have not been executed show `PENDING`; nothing is estimated
or back-filled.

Two cautions before interpreting any delta:

1. At n = 7,600 and accuracy near 0.94, a single run's standard error is about
   0.27 pp, so differences below roughly 0.75 pp are not distinguishable from
   noise. Use `src.metrics.mcnemar_exact` and
   `src.metrics.bootstrap_accuracy_difference` on paired predictions, across the
   three seeds in `config.SEEDS`.
2. **A0 vs A3** is the decisive test: A3 keeps a learned layer mixture and
   removes only the label conditioning. If A3 matches A0, the contribution
   reduces to multi-layer fusion, which is ELMo (2018). **A11 vs A1** is the
   parameter-matched control — both have exactly 397,057 non-backbone
   parameters.

## Documentation

- `docs/prior_art_source_audit.md` — papers and official repositories actually
  retrieved and read, with confidence levels, licenses, and an explicit account
  of what the search did **not** cover.
- `docs/architecture_traceability.md` — each component's equation, source
  location, closest prior art and key difference; class-identity paths;
  parameter accounting.
- `docs/source_audit.md` — the pre-refactor defects and their resolutions.
- `docs/CHANGELOG.md` — refactor history.

## Known limitations

- No training run has been completed; all metrics are `PENDING`.
- The upstream pipeline identified 281 of 7,600 official test rows (3.70%) as
  near-duplicates of training data and wrote a decontaminated id list, but no
  code consumes it. Test metrics computed today are on the contaminated set.
- With only 4 classes, AG News gives a label-conditioned router limited room to
  specialise; a dataset with more classes would test the mechanism harder.
- The prior-art search covered titles and abstracts only, and the citation-graph
  search failed. The novelty statement is scoped accordingly.
