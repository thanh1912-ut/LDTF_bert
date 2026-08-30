# Architecture Traceability

Every LDTF component, its mathematical definition, where it lives in the source,
and the closest verified prior art. Prior-art claims are supported by
`docs/prior_art_source_audit.md`; entries there record whether a source was
actually retrieved.

Notation: `B` batch, `L` layers, `T` tokens, `D` hidden size (768),
`C` classes (4), `Rt`/`Rd` token/depth router widths.

## Component table

| LDTF component | Mathematical definition | Source file | Function/Class | Closest prior art | Key difference |
|---|---|---|---|---|---|
| Backbone hidden states | `H = stack(all_hidden_states[1:]) ∈ R^{B×L×T×D}`, `L = num_hidden_layers = 12` | `src/models/bert_backbone.py` | `BertBackbone.forward` | HuggingFace `BertModel` output contract (VERIFIED docs) | Embedding output is explicitly excluded and the pooler is not loaded (`add_pooling_layer=False`) |
| Label Query Bank | `Q ∈ R^{C×D}`, `Q ~ N(0, 0.02²)` seeded | `src/models/label_query_bank.py` | `LabelQueryBank` | LEAM class embeddings; LSAN `label_embed`; CAML `U` | One bank shared by **both** routers; LSAN detaches its label embeddings, CAML keeps two separate label matrices |
| Token Router scores | `s_{bclt} = ⟨Wq_t Q_c, Wk_t H_{blt}⟩ / √Rt` | `src/models/token_router.py` | `TokenRouter.forward` | CAML `alpha = softmax(U·xᵀ)`; LEAM compatibility `G` | Scores are computed **per layer** (`bltr`), giving a `[B,C,L,T]` map instead of `[B,C,T]` |
| Token masking | mask = `attention_mask ∧ ¬special_tokens_mask` (policy-dependent), `-inf` before softmax | `src/models/token_router.py` | `TokenRouter._valid_mask` | CAML, LSAN, LAAT all lack explicit attention masking | Padding and (optionally) `[CLS]`/`[SEP]` are hard-masked; an all-masked row raises instead of producing NaN |
| Token attention | `A^tok = softmax_T(s)`, computed in FP32 | `src/models/token_router.py` | `TokenRouter.forward` | LEAM `partial_softmax` | Max-stable masked softmax; LEAM's variant lacks max-subtraction |
| Token features | `U_{bcl} = Σ_t A^tok_{bclt} H_{blt} ∈ R^{B×C×L×D}` | `src/models/token_router.py` | `TokenRouter.forward` | CAML `m = alpha @ x` | Retains a layer axis, which CAML does not have |
| **Direct Depth Router** | `z_{bcl} = ⟨Wq_d Q_c, Wk_d U_{bcl}⟩ / √Rd`; `A^dep = softmax_L(z)` | `src/models/depth_router.py` | `DirectDepthRouter.forward` | **No verified prior art conditions layer weights on labels** — see gap analysis | Layer weights are `[B,C,L]` (per-example, per-class); ScalarMix weights are `[L]` |
| Depth fusion | `f_{bc} = Σ_l A^dep_{bcl} U_{bcl} ∈ R^{B×C×D}` | `src/models/depth_router.py` | `DirectDepthRouter.forward` | ELMo ScalarMix convex combination | The combination is label- and input-conditioned rather than global |
| Indirect Depth Gating (A4) | `z_{bc} = W_g · mean_l(U_{bcl}) / √D`; `A^dep = softmax_L(z)` | `src/models/depth_router.py` | `IndirectDepthGating` | none claimed | Not label-conditioned; provably invariant to layer permutation. Retained only as an ablation |
| Global Scalar Mix (A3/B4) | `w = softmax(θ) ∈ R^L`; `f = γ Σ_l w_l U_{·l}` | `src/models/depth_router.py`, `src/models/baselines.py` | `GlobalScalarMix`, `BertScalarMixClassifier` | ELMo / AllenNLP `ScalarMix` (VERIFIED source) | Reimplemented from the published equation; applied to BERT Transformer layers and to per-class routed features, not to biLM states |
| Uniform Layer Mean (A1/A2) | `f_{bc} = (1/L) Σ_l U_{bcl}` | `src/models/depth_router.py` | `UniformLayerMean` | standard layer averaging | Parameter-free; registered as a module only so the output contract is uniform |
| Shared Linear Scorer (A0/A7) | `logit_{bc} = wᵀ f_{bc} + b`, `w ∈ R^D` | `src/models/class_scorer.py` | `SharedLinearScorer` | — | 769 parameters; identical function applied to every class, so class identity must come from the representation |
| Shared MLP Scorer (A8) | `logit_{bc} = w₂ᵀ GELU(W₁ f_{bc} + b₁) + b₂` | `src/models/class_scorer.py` | `SharedMlpScorer` | — | Capacity ablation: +590,592 parameters at `H = D = 768` |
| Class-Specific Scorer (A9) | `logit_{bc} = ⟨f_{bc}, w_c⟩ + b_c` | `src/models/class_scorer.py` | `ClassSpecificLinearScorer` | CAML `final.weight.mul(m).sum(2)`; LAAT `third_linear` | Same idiom as CAML; here it is an ablation rather than the default, because `C=4` does not need per-class scoring capacity |

## Class-identity path

Class identity must survive to the scorer, otherwise a shared scorer assigns
identical logits to every class and the model cannot learn. In the reference
model it is injected **twice**:

1. `token_router.py` — `einsum("cr,bltr->bclt")` makes `A^tok` depend on `c`.
2. `depth_router.py` — `einsum("cr,bclr->bcl")` makes `A^dep` depend on `c`.

Per variant:

| Variant | Class identity mechanism | Degenerate? |
|---|---|---|
| A0, A5–A14, B6, B7 | token routing **and** direct depth routing | No |
| A1, A2, A3, B5 | token routing only; the layer reduction is applied independently per class and therefore preserves the class axis | No |
| A4 | token routing only; the gate reads already-class-specific features | No |
| A9 | token + depth routing **and** per-class scorer weights | No (two paths; interpret with care) |
| B1, B2, B3, B4 | per-class rows of `Linear(D, C)` | No |
| hypothetical: no token router + uniform/scalar-mix/indirect depth | none — every class receives the same vector | **Yes — rejected at construction** |

`src/variants.py::LdtfVariant.__post_init__` raises when
`token_routing="none"` is combined with anything other than `depth_routing="direct"`,
so degenerate configurations cannot be built. The single legal no-token variant
keeps class identity through direct depth routing alone.

## Parameter accounting (D=768, C=4, L=12, Rt=Rd=256)

| Module | Formula | Count |
|---|---|---|
| Label Query Bank | `C·D` | 3,072 |
| Token Router (`Wq`+`Wk`, no bias) | `2·Rt·D` | 393,216 |
| Direct Depth Router (`Wq`+`Wk`, no bias) | `2·Rd·D` | 393,216 |
| Indirect Depth Gating | `D·L` | 9,216 |
| Global Scalar Mix (+γ) | `L + 1` | 13 |
| Shared Linear Scorer | `D + 1` | 769 |
| Shared MLP Scorer (H=768) | `(D·H + H) + (H + 1)` | 591,361 |
| Class-Specific Scorer | `C·D + C` | 3,076 |
| Baseline head `Linear(D,C)` | `D·C + C` | 3,076 |
| **A0 non-backbone total** | | **790,273** |

Verified programmatically by `tests/test_ldtf_model.py::test_parameter_counts_match_the_design`
and `::test_shared_mlp_capacity_delta`.

Note the exact match `A1 = A2 = A11 = 397,057` (`2·256·768 = 4·128·768`), which
makes **A11 vs A1** a parameter-matched comparison.

## Which comparison isolates what

| Question | Comparison | Parameter-matched? |
|---|---|---|
| Does direct label-conditioned depth routing help? | **A0 vs A1** | No (A0 is 2× the head) |
| Does it still help at a fixed budget? | **A11 vs A1** (both 397,057) | **Yes, exactly** |
| Does *label conditioning* of depth matter, versus a learned but global mixture? | **A0 vs A3** | No (+393,204) |
| Direct versus indirect depth conditioning? | **A0 vs A4** | No (A4 is 43× smaller; cannot be matched by tuning `Rd`) |
| Do intermediate layers carry class-relevant signal? | **A1 vs A2** (both 397,057) | **Yes, exactly** |
| Does the shared-scorer constraint cost accuracy? | A7 vs A9 | Effectively (+2,307) |
| Is the gain merely scorer capacity? | A7 vs A8 | No, by construction (+590,592) |
| Backbone regime effect | B6 vs B7 (A15 pair) | No, by construction |

**A0 vs A3 is the decisive test of the novelty claim**, because A3 keeps a
learned layer mixture and removes only the label conditioning. If A3 matches A0,
the contribution reduces to multi-layer fusion, which is ELMo (2018).

## Statistical caveat

At `n = 7,600` and accuracy near 0.94, the standard error of a single run is
about 0.27 pp, so a difference between two independent runs below roughly
0.75 pp is not distinguishable from noise. Expected effect sizes on AG News are
plausibly within that band. `src/metrics.py` therefore provides
`mcnemar_exact` (paired, exact) and `bootstrap_accuracy_difference` (paired
bootstrap CI), and `config.SEEDS` defines three seeds. Any reported improvement
should carry a paired test and a confidence interval, not a single accuracy delta.

## Novelty statement

To the best of our knowledge, and based on the search whose scope and failures
are documented in `docs/prior_art_source_audit.md`, we are not aware of prior
work that conditions transformer layer-mixing weights on label representations.
The token-routing stage is **not** claimed as novel: it follows the
LEAM/CAML/LSAN/LAAT line of label-wise attention. No claim about accuracy,
superiority over baselines, or state of the art is made anywhere in this
repository, because no full training run has been completed.
