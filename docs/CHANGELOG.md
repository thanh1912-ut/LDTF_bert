# Changelog

## Refactor — LDTF-BERT clean implementation

### Removed: LoRA / PEFT (historical note)

**This is historical information only. The project does not support LoRA and no
LoRA functionality is planned or retained.**

The previous revision contained a partial LoRA integration that was removed in
full. For the record, what existed and why it went:

- `PeftLdtfBert` in `src/models/ldtf_bert.py` — it referenced
  `config.LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT` and `LORA_TARGET_MODULES`,
  none of which were ever defined in `src/config.py`. Because these were used as
  default argument values, evaluated when the class body executes, `import
  src.models` raised `AttributeError` and the entire package was unimportable.
- The same four missing constants were also used as `TrainConfig` field
  defaults, so `import src.train` failed for the same reason.
- The adapter was attached to `self.backbone.transformer`, an attribute that
  does not exist (`BertBackbone` defines `self.encoder`), so the adapter would
  have been a no-op even had the constants existed.
- `TrainConfig.use_peft`, `lora_rank`, `lora_alpha` and `lora_dropout` were never
  read by the training loop.
- `PeftLdtfBert` was not exported from `src/models/__init__.py`, so its only
  caller — a stale notebook — could not have imported it.

Removed in this revision: the `PeftLdtfBert` class, every `peft` import, all
`LORA_*` configuration, the `use_peft` switch and its checkpoint fields, the
LoRA branches in the notebooks, and the `peft` dependency. No dead LoRA classes,
conditional imports or "future LoRA support" comments remain in any production
path.

Verification:

```
grep -RniE "lora|peft|PeftLdtfBert|LoraConfig|get_peft_model|LORA_" \
  src experiments scripts tests requirements.txt README.md
```

returns no matches in production source. (Note: a case-insensitive `lora` search
over `src/data/**` matches the word "Colorado" in the raw AG News CSVs; scope the
search to source files.)

### Architecture

- **Direct label-conditioned depth routing implemented.** The previous
  `DepthRouter` averaged the layer axis before predicting layer weights, which
  made its output invariant to layer permutation and meant the advertised
  mechanism did nothing. It also never received the label queries. Replaced by
  `DirectDepthRouter`, which projects the label queries to queries and each
  layer's per-class features to keys, scoring every layer individually.
- **Old behaviour retained as an ablation.** Corrected and renamed to
  `IndirectDepthGating` (ablation A4). Its permutation invariance is now asserted
  by a test that documents why it is only an ablation.
- **BERT pooler removed** via `add_pooling_layer=False`, eliminating 590,592
  trainable parameters that received no gradient.
- **Embedding output excluded** from the routed layers by default
  (`all_hidden_states[1:]`), with an explicit `include_embedding_layer` opt-in.
- **Three separate scorers** — shared linear (reference, 769 parameters),
  shared MLP (capacity ablation, +590,592) and class-specific linear. Exactly one
  is registered per model.
- **Special-token policy** added to the Token Router, with ablations A13/A14.
- **Label Query Bank** gained reproducible seeded initialisation plus frozen and
  fixed-random modes for ablations A5 and A6.
- **Degenerate configurations rejected at construction**: a variant with no token
  router and no direct depth router would give every class an identical
  representation under a shared scorer, and now raises.

### Training protocol

- Frozen backbones stay in `eval()` mode, so their dropout is disabled and frozen
  features are deterministic. Previously `model.train()` silently re-enabled it.
- Differential learning rates: 2e-5 backbone, 1e-3 head, across four optimizer
  groups split by decay/no-decay.
- Gradient accumulation flushes the final partial group and normalises by the
  true group size; the scheduler steps only when the optimizer steps.
- AMP with BF16 or FP16 on CUDA, `GradScaler` only where FP16 requires it, and
  `unscale_ → clip → step` ordering. Non-finite gradients skip the step.
- Checkpoint selection on validation macro F1, tie-broken by lower validation
  loss then earlier epoch.
- `best.pt` is slim; `last.pt` is fully resumable including RNG and DataLoader
  generator state. Resume refuses a mismatched data signature.

### Data and test-set protocol

- Dynamic per-batch padding replaces static padding to 128.
- `special_tokens_mask` and `token_type_ids` are produced and propagated; CSV and
  Parquet are both supported; sentence-pair encoding is available.
- The official test split is sealed by `src/guard.py`: it requires an environment
  unlock token, refuses while a training loop is active, and appends every access
  to `reports/official_test_access.jsonl`.

### Repository hygiene

Deleted: `fix3.py`, `fix4.py`, `fix_script.py`, `fix_script2.py`, the empty
`phase2_colab_walkthrough.md`, both stale notebooks, `scripts/build_notebook.py`,
and the four superseded experiment scripts.

### Status

Code is complete and tested. **No full training run has been performed**, so all
baseline and ablation metrics are `PENDING` and no accuracy claim is made.
