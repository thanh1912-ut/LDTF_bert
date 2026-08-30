# Source Audit — pre-refactor state and resolutions

Audit of the repository as it stood before this refactor, with the resolution
applied to each finding. Line references point at the **original** files.

## Blockers (the repository did not run)

| File | Function/Class | Problem | Severity | Evidence | Resolution |
|---|---|---|---|---|---|
| `src/train.py` | `train_model` | `IndentationError`: resume block over-indented, so the module did not compile and every entry point failed | BLOCKER | line 177 `        if resume and best_checkpoint_path.exists():` after a 4-space body; `compileall` reported `IndentationError: unexpected indent (train.py, line 177)` | `src/train.py` rewritten; `compileall` exits 0 |
| `src/models/ldtf_bert.py` | `PeftLdtfBert` | `config.LORA_RANK/ALPHA/DROPOUT` used as **default argument values**, evaluated at class-definition time, so `import src.models` raised `AttributeError` | BLOCKER | lines 189–191, 216; `config.py` defined no `LORA_*` | LoRA removed entirely |
| `src/train.py` | `TrainConfig` | Same missing constants as frozen-dataclass field defaults | BLOCKER | lines 43–45 | LoRA removed entirely |
| `src/models/ldtf_bert.py` | `PeftLdtfBert.__init__` | Assigned `self.backbone.transformer`, but `BertBackbone` defines `self.encoder`; the adapter would have been a no-op even if it resolved | BLOCKER | lines 219–221 vs `bert_backbone.py:32,51` | LoRA removed entirely |
| `experiments/run_*.py` (4 files) | imports | `from src.utils import ensure_output_dirs`, but the function is defined in `src/config.py` | BLOCKER | `run_finetune.py:18`, `run_frozen.py:17`, `run_bert_base.py:17`, `run_bert_frozen.py:19` | Experiment layer rewritten; `ensure_output_dirs` imported from `src.config` |
| `scripts/build_notebook.py` | `_build_train_cell` | Emitted `config.BERT_BASE_OUTPUT` from `exp_key.upper()`, but config defined `BERT_OUTPUT` | BLOCKER (generated notebook) | line 398 | Notebook generator and stale notebooks deleted |

## Scientific correctness

| File | Function/Class | Problem | Severity | Evidence | Resolution |
|---|---|---|---|---|---|
| `src/models/depth_router.py` | `DepthRouter.forward` | **The headline mechanism was inert.** Input is `[B,C,L,D]`, so `.mean(dim=-2)` averaged the **layer** axis (the comment claimed it pooled tokens, which had already been consumed upstream). Layer scores were computed from a single layer-averaged vector, making the output provably invariant to permutation of the layer axis. Shapes were valid, so it trained and reported plausible accuracy while measuring a constant | BLOCKER (silent) | lines 53–54, 57; permutation test returned `True` | Replaced by `DirectDepthRouter` (per-layer query–key scoring). The old behaviour is preserved, corrected and renamed as `IndirectDepthGating` for ablation A4, and `tests/test_depth_routers.py` now *asserts* its permutation invariance to document why it is only an ablation |
| `src/models/depth_router.py` | `DepthRouter` | Label queries were never passed in; `forward(self, token_features)` had no query argument, so depth routing was not label-conditioned at all | BLOCKER | line 35; `ldtf_bert.py:132` | `DirectDepthRouter.forward(token_features, label_queries)`; a test asserts gradient flows from the depth path into the query bank |
| `src/models/depth_router.py` | `DepthRouter.__init__` | Scale was `num_layers ** -0.5` applied to a dot product in `D`-dimensional space | HIGH | line 30 | Direct router scales by `router_dim ** -0.5`; the indirect ablation scales by `hidden_size ** -0.5` |
| `src/models/token_router.py` | `TokenRouter.forward` | No `special_tokens_mask` support, so `[CLS]`/`[SEP]` always received attention mass | HIGH | lines 95–97 | Added, with `INCLUDE_SPECIAL_TOKENS` policy and ablations A13/A14 |
| `src/models/token_router.py` | `TokenRouter.forward` | No all-padding guard; an all-`-inf` row would produce silent NaN across the batch | HIGH | line 97 | Explicit `ValueError`, plus a finiteness assertion |
| `src/models/bert_backbone.py` | `BertBackbone.__init__` | Loaded `BertModel` with a pooler; `pooler.dense.*` = **590,592** parameters were trainable, optimizer-tracked, weight-decayed and checkpointed, but never used by `forward` | HIGH | line 32; `forward` reads only `outputs.hidden_states` | `add_pooling_layer=False`; a test asserts no parameter name contains `.pooler.` |
| `src/models/class_scorer.py` | `ClassScorer` | The only scorer was a 2-layer MLP (591,361 parameters), so the "reference" configuration silently included the capacity ablation | HIGH | lines 33–39 | Three separate scorers; `shared_linear` (769 parameters) is the reference and only one is ever registered |
| `src/models/ldtf_bert.py` | `_init_cross_module_weights` | Re-applied Xavier over all submodules after each had run its own initialisation | LOW | lines 73–93 | Removed; each module owns its initialisation |

## Training protocol

| File | Function/Class | Problem | Severity | Evidence | Resolution |
|---|---|---|---|---|---|
| `src/train.py` | `train_model` | **Frozen encoder ran with dropout enabled.** `model.train()` recursively set the encoder to train mode; the remediation loop only cleared `requires_grad`. Frozen features were therefore stochastic during training but deterministic at validation, so train and validation measured different functions | CRITICAL | lines 208–211; `bert_backbone.py:42` called `.eval()` only at construction | `BertBackbone.train()` is overridden to keep a frozen encoder in `eval()`; the loop also calls `model.backbone.eval()`; tests assert both mode and determinism |
| `src/train.py` | `build_optimizer` | Single learning rate 2e-5 for BERT **and** ~1M randomly initialised head parameters. Frozen runs, where there is no co-adaptation, were badly under-trained | HIGH | line 72; `config.py:49` | Four groups: `{backbone, head} × {decay, no-decay}` with 2e-5 / 1e-3 |
| `src/train.py` | `train_model` | Optimizer was built before the freeze was applied, so frozen parameters could enter it | HIGH | line 165 vs 209 | Freeze happens before `build_optimizer`; `optimizer_coverage_report` asserts the invariant |
| `src/train.py` | `train_model` | Final partial accumulation group was never stepped; its gradients were discarded at the next epoch. Loss was also divided by `G` rather than the true group size | MEDIUM | line 239 | Boundary is `(i+1) % G == 0 or (i+1) == N`; normalisation uses the actual group size; a parametrised test asserts `ceil(N/G)` optimizer steps |
| `src/train.py` | `train_model` | `total_steps` used `floor(N/G)`, consistent only because the tail was dropped | MEDIUM | lines 166–168 | `ceil(N/G) * epochs` |
| `src/train.py` | — | No AMP anywhere | MEDIUM | no `autocast`/`GradScaler` in the tree | BF16 or FP16 autocast on CUDA with `GradScaler` only for FP16; CPU/MPS stay FP32 |
| `src/train.py` | `train_model` | No non-finite gradient handling; a NaN norm would propagate into every parameter | MEDIUM | lines 240–244 | `unscale_ → clip → isfinite check → step`; non-finite gradients skip both optimizer and scheduler and are logged |
| `src/train.py` | `train_model` | Checkpoint selection on **accuracy**, while the reports used macro F1; macro F1 was never computed at validation time | MEDIUM | lines 199, 284; `evaluate_loss` returned only loss and accuracy | Selection on validation macro F1, tie-broken by lower loss then earlier epoch; both metrics recorded per epoch |
| `src/train.py` | `train_model` | Resume read **`best_model.pt`**, treated its epoch as "last completed", replayed scheduler steps by hand, and reset `best_val_accuracy` to `-1.0` — so the first post-resume epoch always "improved" and **overwrote the genuinely best checkpoint** | CRITICAL | lines 172–194 | Separate `best.pt` (slim) and `last.pt` (fully resumable); scheduler/scaler/RNG/generator/best-metrics/patience all restored; resume refuses a mismatched data signature |
| `src/train.py` | `train_model` | `resume=True` by default, so a stale checkpoint silently short-circuited a rerun | MEDIUM | line 134 | Default `resume=False`, explicit `--resume` flag |
| `src/train.py` | `train_model` | `train_log.json` overwritten with a single epoch record each epoch | LOW | line 298 | Appended `train_log.jsonl`, plus `val_metrics.json` flushed every epoch |
| `src/train.py` | — | No peak-VRAM tracking, so the results table could not be filled | MEDIUM | no `max_memory_allocated` in the tree | `reset_peak_memory_stats` / `max_memory_allocated` per epoch |
| `src/utils.py` | `set_seed` | No cuDNN determinism, no `CUBLAS_WORKSPACE_CONFIG`, and `PYTHONHASHSEED` set after interpreter start | LOW | lines 17–24 | Determinism flags added; DataLoaders take a seeded `generator` and `worker_init_fn` |

## Data pipeline and test-set protocol

| File | Function/Class | Problem | Severity | Evidence | Resolution |
|---|---|---|---|---|---|
| `src/dataset.py` | `build_all_dataloaders` | **Always** built the official test loader, in every training entry point | HIGH | lines 159, 163; called by all four run scripts | `build_dataloaders` returns train and validation only; `load_split` refuses the official test path; access requires `src.guard.official_test_loader` with an environment unlock, no active training loop, and an append-only ledger entry |
| `experiments/run_*.py` | `main` | Trained and printed test accuracy **in the same process**, inviting test-set-driven iteration | HIGH | e.g. `run_finetune.py:58-68` | Training and final evaluation split into `run_experiment.py` and `final_eval.py` |
| `src/dataset.py` | `AgNewsDataset.__getitem__` | Static `padding="max_length"` to 128 while the median length is 50, so roughly 60% of every batch was padding | MEDIUM | line 68; `token_length_report.json` median 50, p95 73 | Dynamic per-batch padding via `DynamicPaddingCollator` |
| `src/dataset.py` | `AgNewsDataset` | No `return_special_tokens_mask`, so the router could not implement a special-token policy | MEDIUM | lines 65–71 | Requested and propagated; padding is marked special so it is never routable |
| `src/dataset.py` | `build_dataloader` | No `generator`, no `worker_init_fn`, no `persistent_workers` | LOW | lines 111–119 | All added |
| `src/evaluate.py` | `build_model_from_checkpoint` | Hard-coded `LdtfBert`, so baseline checkpoints could not be rebuilt | MEDIUM | line 113 | Checkpoints carry an `architecture` block; `src/registry.py` rebuilds any model |
| `src/evaluate.py` | `compute_metrics` | Dumped full prediction and label arrays into the metrics JSON | LOW | lines 81–82 | Predictions written to a separate `.npz` |

## Dead code and repository hygiene

| Item | Problem | Resolution |
|---|---|---|
| `fix3.py`, `fix4.py`, `fix_script.py`, `fix_script2.py` | One-off byte patchers hard-coding a Windows path from another machine, mutually contradictory | Deleted |
| `phase2_colab_walkthrough.md` | Zero-byte file | Deleted |
| `notebooks/*.ipynb` | Stale artifacts importing `PeftLdtfBert` (never exported) and `config.PEFT_OUTPUT` (never defined) | Deleted |
| `scripts/build_notebook.py` | Generated a notebook referencing a non-existent config attribute | Deleted |
| `experiments/compare_results.py` | Aggregated only 2 of the 4 experiments; required `tabulate`, which was not a dependency | Replaced by `experiments/export_tables.py` |
| `src/train.py::aggregate_history`, `src/utils.py::timer`, `config.PAD_TOKEN_ID`, `config.REFERENCES_DIR` | Defined and never used | Removed or given a real use |
| `src/models/token_router.py` lines 87–90 | Silent `reshape` + truncate that could drop classes instead of raising | Replaced by a strict shape check |

## Data facts confirmed during the audit

Read directly from `data/processed/*.parquet`:

| Split | Rows | Class distribution (World/Sports/Business/Sci-Tech) |
|---|---|---|
| train | 107,735 | 26,974 / 26,965 / 26,914 / 26,882 |
| validation | 11,971 | 2,997 / 2,996 / 2,991 / 2,987 |
| test (official, sealed) | 7,600 | 1,900 each |

The upstream pipeline (`src/data/scripts/prepare_agnews.py`) performs checksummed
ingestion, versioned cleaning with raw preservation, exact and near-duplicate
deduplication, and group-stratified splitting on `near_duplicate_group_id`.

Two upstream issues are recorded but **not** fixed here, as they belong to the
data pipeline rather than the model code:

1. `src/data/reports/split_overlap_report.json` reports **281 of 7,600** official
   test rows (3.70%) as near-duplicates of training data, and
   `manifests/decontaminated_test_ids.json` lists the 7,319 clean ids — but no
   code consumes that manifest. Any reported test metric is therefore computed on
   the contaminated set unless this is addressed.
2. `verify_pass_conditions` hard-codes 11 of its 26 gates to `True`, so
   `final_data_report.json`'s `"overall_status": "PASS"` is weaker evidence than
   it appears.
