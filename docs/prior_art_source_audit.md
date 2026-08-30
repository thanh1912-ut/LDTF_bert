# Prior-Art Source Audit

Scope: research works and official implementations closest to LDTF-BERT
(label-conditioned token attention over BERT layers, plus label-conditioned
depth fusion), for AG News single-label 4-way topic classification.

Method and honesty statement
- Entries marked **VERIFIED** were retrieved during the audit session via HTTP
  fetches of ACL Anthology / arXiv / IJCAI pages, or via `raw.githubusercontent.com`
  and the GitHub REST API. Commit SHAs were resolved through
  `api.github.com/repos/{repo}/commits`; every file path listed under
  "Files inspected" was actually retrieved and read.
- Where only the landing page or abstract was retrieved, this is stated and no
  claim is made about the paper body.
- Statements recalled without retrieval are explicitly labelled
  "from memory, NOT verified in this session".
- Where no author repository could be located, the entry says
  **official source not found** rather than attributing a third-party
  reimplementation to the authors.
- A Semantic Scholar Graph API search was attempted and **failed** (empty
  responses); this gap is recorded in the gap analysis.

---

## 1. LEAM — Joint Embedding of Words and Labels for Text Classification

- **Paper:** Joint Embedding of Words and Labels for Text Classification
- **Authors:** Guoyin Wang, Chunyuan Li, Wenlin Wang, Yizhe Zhang, Dinghan Shen,
  Xinyuan Zhang, Ricardo Henao, Lawrence Carin
- **Year / Venue:** 2018, ACL 2018 (Long Papers), pp. 2321–2331. DOI 10.18653/v1/P18-1216
- **Official paper:** https://aclanthology.org/P18-1216/ (also arXiv:1805.04174)
- **Official repository:** https://github.com/guoyinwang/LEAM
- **Commit/revision:** `bfd341ccd92009ab6b9e6b0578216f57bb2b921e` (HEAD of `master`)
- **Files inspected:** `model.py`, `main.py`
- **Functions/classes inspected:** `embedding`, `embedding_class`,
  `att_emb_ngram_encoder_maxout`, `partial_softmax`, `discriminator_2layer`,
  `emb_classifier`
- **Relevant equations (read from the code, not the PDF body):**
  L2-normalise token and class embeddings; compatibility `G = x_emb_norm · W_class_norm`
  `[b,s,c]`; 1-D convolution over `G` with window `ngram`; `reduce_max` over the
  class axis to `[b,s,1]`; masked softmax over tokens; weighted sum to `[b,e]`.
- **Tensor flow:** `x [b,s] -> [b,s,e] -> G [b,s,c] -> max over c -> [b,s,1] ->
  softmax over s -> H_enc [b,e] -> 2-layer MLP -> logits [b,c]`
- **Masking:** `partial_softmax` multiplies `exp(logits)` by the mask and
  renormalises; no max-subtraction, so it is numerically fragile.
- **Pooling:** one attention-weighted document vector; the class axis is collapsed.
- **Similarities to LDTF:** the direct ancestor of our Token Router. A learned
  class-embedding matrix lives in the token space and drives attention over
  tokens by inner product. `main.py` contains an AG News configuration with
  `num_class = 4` and class names `['World','Sports','Business','Science']`.
- **Differences from LDTF:** static word embeddings, a single representation
  layer, therefore **no depth axis at all**; LEAM max-pools over classes and
  produces one document vector, whereas LDTF keeps `[B,C,D]` through to scoring.
- **What informed our implementation:** class-name-derived label semantics as a
  natural initialisation story for the Label Query Bank; the confirmation that
  label-token compatibility by inner product is the standard formulation.
- **What was not reused:** max-over-classes pooling (it would destroy our
  `[B,C,D]` contract), the TensorFlow 1.x code, and the unstable `partial_softmax`.
- **Not applicable to AG News:** nothing about the loss — LEAM is genuinely
  single-label softmax cross-entropy, matching our setting.
- **License:** no license file present; GitHub reports `license: null`. Treated as
  all-rights-reserved; **no code was copied**.
- **Confidence:** High (VERIFIED).

## 2. LSAN — Label-Specific Document Representation for Multi-Label Text Classification

- **Authors:** Lin Xiao, Xin Huang, Boli Chen, Liping Jing
- **Year / Venue:** 2019, EMNLP-IJCNLP 2019, pp. 466–475. DOI 10.18653/v1/D19-1044
- **Official paper:** https://aclanthology.org/D19-1044/
- **Repository:** https://github.com/EMNLP2019LSAN/LSAN
- **Commit/revision:** `1d130c525861280167c99d8e11473222258ef709`
- **Files inspected:** `attention/model.py`, `attention/train.py`, `classification.py`
- **Functions/classes inspected:** `StructuredSelfAttention.__init__`/`forward`,
  `load_labelembedd`, `multilabel_classification`
- **Relevant equations:** self-attention branch
  `softmax(W2 tanh(W1 H))` giving `[B,C,T]` then `[B,C,2H]`; label-attention branch
  `label_embed @ hᵀ` giving `[B,C,T]` then `[B,C,2H]`; sigmoid-gated fusion
  `w1 = σ(Linear(label_att))`, renormalised, `doc = w1·label_att + w2·self_att`.
- **Tensor flow:** `[B,T] -> BiLSTM [B,T,2H] -> two branches [B,C,2H] -> gated
  fusion [B,C,2H] -> mean over C -> [B,2H] -> Linear -> sigmoid`
- **Masking:** **none.** `forward(self, x)` receives no mask; the softmax over the
  token axis includes padding. We deliberately did not replicate this.
- **Similarities to LDTF:** an explicit label-embedding bank used as attention
  queries against hidden states.
- **Differences from LDTF:** BiLSTM, one representation layer, **no depth
  routing**; the released code uses `self.label_embed.weight.data`, which
  **detaches** the label embeddings so they are effectively frozen; the final
  code averages over the class axis before a `Linear(2H, C)`, which does not
  match the paper's "label-specific" framing. We flag this discrepancy rather
  than resolve it.
- **What informed our implementation:** three anti-patterns to avoid —
  detached label queries, missing masking, and collapsing the class axis.
- **Not applicable to AG News:** multi-label throughout — `BCELoss` on sigmoid
  outputs, precision@k and nDCG@k metrics, 54 labels.
- **License:** no license file; GitHub reports `license: null`.
- **Confidence:** Medium-high for the code (VERIFIED). The account
  `EMNLP2019LSAN` is a submission-time pseudonymous org, so "official" is
  probable rather than certain.

## 3. CAML — Explainable Prediction of Medical Codes from Clinical Text

- **Authors:** James Mullenbach, Sarah Wiegreffe, Jon Duke, Jimeng Sun, Jacob Eisenstein
- **Year / Venue:** 2018, NAACL-HLT 2018, pp. 1101–1111. DOI 10.18653/v1/N18-1100
- **Official paper:** https://aclanthology.org/N18-1100/
- **Official repository:** https://github.com/jamesmullenbach/caml-mimic
- **Commit/revision:** `44a47455070d3d5c6ee69fb5305e32caec104960`
- **Files inspected:** `learn/models.py`
- **Functions/classes inspected:** `BaseModel._get_loss`, `ConvAttnPool.__init__`,
  `ConvAttnPool._code_emb_init`, `ConvAttnPool.forward`
- **Relevant equations:** `x = tanh(Conv1d(embed(x)))`;
  `alpha = softmax(U.weight @ xᵀ, dim=2)` giving `[B,Y,T]`; `m = alpha @ x` giving
  `[B,Y,d]`; `y = final.weight.mul(m).sum(dim=2).add(final.bias)`.
- **Tensor flow:** `[B,T] -> [B,T,e] -> conv [B,T,d] -> alpha [B,Y,T] ->
  m [B,Y,d] -> per-label dot product -> [B,Y]`
- **Masking:** no explicit attention mask; `padding_idx=0` zeroes pad embeddings
  but the softmax still assigns mass to padded positions.
- **Pooling:** genuine per-label attention pooling, `[B,Y,d]` preserved to the scorer.
- **Similarities to LDTF:** the canonical label-wise attention output contract,
  matching our Token Router: one pooled vector per class, class axis never
  collapsed before scoring.
- **Differences from LDTF:** CNN encoder, one representation layer, no depth
  axis; CAML uses **two** label-indexed matrices (`U` for attention, `final` for
  scoring), whereas LDTF shares **one** Label Query Bank across both routers and
  uses a shared `Linear(D,1)` scorer.
- **What informed our implementation:** the `[B,C,D] -> [B,C]` reduction idiom,
  which is exactly what our class-specific scorer (A9) implements as
  `einsum("bcd,cd->bc")`.
- **Not applicable to AG News:** multi-label BCE over thousands of sigmoid
  outputs, micro-F1 and precision@8 metrics; CAML's per-label scorer is motivated
  by extreme label counts, which does not transfer to `C=4`.
- **License:** MIT (GitHub SPDX `MIT`; a `LICENSE` blob is present in the tree,
  though its text was not fetched). No code was copied.
- **Confidence:** High (VERIFIED).

## 4. LAAT — A Label Attention Model for ICD Coding from Clinical Text

- **Authors:** Thanh Vu, Dat Quoc Nguyen, Anthony Nguyen
- **Year / Venue:** 2020, IJCAI-PRICAI 2020, pp. 3335–3341. DOI 10.24963/ijcai.2020/461
- **Official paper:** https://www.ijcai.org/proceedings/2020/461
- **Official repository:** https://github.com/aehrc/LAAT
- **Commit/revision:** `cd5c0ec0b0b8098289042be6d68363a760d7bbca`
- **Files inspected:** `src/models/attentions/attention_layer.py`,
  `src/models/attentions/util.py`, `src/models/rnn.py`
- **Functions/classes inspected:** `AttentionLayer.forward`, `_init_weights`,
  `l2_matrix_norm`, `init_attention_layer`, `perform_attention`, `RNN.forward`
- **Relevant equations:** `weights = tanh(W1 H)`; `att = softmax(W2 weights, dim=1)`
  transposed to `[B,L,T]`; `out = att @ H` giving `[B,L,size]`; per-label scoring
  `third_linear.weight.mul(out).sum(dim=2).add(bias)`.
- **Masking:** no explicit `-inf` masking. `pack_padded_sequence` zeroes padded
  timesteps, but `tanh(0)=0` and a bias-free projection yields logit 0, which
  still receives `exp(0)=1` weight in the softmax — padding leaks attention mass.
- **Similarities to LDTF:** a configurable `attention_mode` switch
  (`hard`/`self`/`label`/`caml`) that implements several label-attention
  variants inside one module. This modular-ablation design directly influenced
  our variant registry.
- **Differences from LDTF:** the "levels" in LAAT are **label-hierarchy levels**
  (ICD chapter vs. code) operating on the same single BiLSTM output, **not**
  encoder depths. Despite the word "hierarchical", LAAT does not route over
  network depth.
- **What informed our implementation:** the one-flag-per-ablation module design;
  the published initialisation scale `normal_(0, 0.03)` for label-attention
  projections as a reference point.
- **Not applicable to AG News:** multi-label ICD coding, label hierarchy,
  extreme class imbalance, micro/macro-F1 and P@k.
- **License:** GitHub reports `NOASSERTION`; a `LICENSE` blob exists but its text
  was not fetched, so its terms are unknown. No code was copied.
- **Confidence:** High for the code (VERIFIED); the org `aehrc` is the CSIRO
  Australian e-Health Research Centre and file headers name the first author.

## 5. ELMo / ScalarMix — Deep Contextualized Word Representations

- **Authors:** Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner,
  Christopher Clark, Kenton Lee, Luke Zettlemoyer
- **Year / Venue:** 2018, NAACL-HLT 2018, pp. 2227–2237. DOI 10.18653/v1/N18-1202
- **Official paper:** https://aclanthology.org/N18-1202/
- **Reference implementation:** https://github.com/allenai/allennlp (archived)
- **Commit/revision:** `80fb6061e568cb9d6ab5d45b661e86eb61b92c82` (HEAD of `main`)
- **Files inspected:** `allennlp/modules/scalar_mix.py`
- **Functions/classes inspected:** `ScalarMix.__init__`, `ScalarMix.forward`,
  inner `_do_layer_norm`
- **Relevant equation (verbatim from the retrieved module docstring):**
  "Computes a parameterised scalar mixture of N tensors,
  `mixture = gamma * sum(s_k * tensor_k)` where `s = softmax(w)`, with `w` and
  `gamma` scalar parameters."
- **Tensor flow:** `N tensors [..,D] -> softmax(w) over N (a length-N vector) ->
  gamma * sum_k s_k · tensor_k -> [..,D]`. The mixture weights have shape `[N]`
  only: **no batch, no token, and no class dimension**.
- **Masking:** the mask is used only when `do_layer_norm=True`, to compute masked
  normalisation statistics; it is not an attention mask.
- **Similarities to LDTF:** the canonical prior art for our depth-fusion stage
  and the primary baseline against which the Depth Router must be measured.
- **Differences from LDTF:** decisive. ScalarMix weights are (i) a single global
  set shared by every class, (ii) static after training and independent of the
  input, and (iii) free parameters rather than a query-key interaction. LDTF's
  depth weights are `[B,C,L]` — per-example and per-class.
- **What was reused:** the parameterisation `gamma * sum_l softmax(w)_l x_l` and
  the zero-initialisation of the layer logits (uniform mixture at start) are
  reimplemented in `src/models/depth_router.py::GlobalScalarMix` and
  `src/models/baselines.py::BertScalarMixClassifier`. Written from the published
  equation; **no AllenNLP code was copied**.
- **What was not reused:** the optional masked layer normalisation.
- **License:** Apache-2.0.
- **Confidence:** High (VERIFIED).

## 6. Linguistic Knowledge and Transferability of Contextual Representations

- **Authors:** Nelson F. Liu, Matt Gardner, Yonatan Belinkov, Matthew E. Peters, Noah A. Smith
- **Year / Venue:** 2019, NAACL-HLT 2019, pp. 1073–1094. DOI 10.18653/v1/N19-1112
- **Official paper:** https://aclanthology.org/N19-1112/
- **Official repository:** **official source not found** (not located in this session;
  no exhaustive repository search was performed, so absence is weak evidence).
- **Files inspected:** none — landing page and abstract only.
- **Relevant claims (from the retrieved abstract only):** the authors probe ELMo,
  a transformer LM and BERT with sixteen probing tasks and "quantify differences
  in the transferability of individual layers within contextualizers", finding
  that "higher layers of RNNs are more task-specific, while transformer layers do
  not exhibit the same monotonic trend."
- **Relation to LDTF:** motivation, not mechanism. The non-monotonic layer-utility
  finding is the published justification for not simply using the final layer.
- **Not applicable to AG News:** the probing suite is token/span-level linguistic
  structure, not topic classification; its per-layer conclusions cannot be
  assumed to transfer without our own measurement.
- **Confidence:** Medium. Metadata and abstract VERIFIED; nothing beyond the
  abstract is claimed.

## 7. What Does BERT Learn about the Structure of Language?

- **Authors:** Ganesh Jawahar, Benoît Sagot, Djamé Seddah
- **Year / Venue:** 2019, ACL 2019, pp. 3651–3657. DOI 10.18653/v1/P19-1356
- **Official paper:** https://aclanthology.org/P19-1356/
- **Official repository:** **official source not found** in this session.
- **Files inspected:** none — landing page and abstract only.
- **Relevant claims (retrieved abstract only):** "BERT's phrasal representation
  captures phrase-level information in the lower layers. The intermediate layers
  of BERT compose a rich hierarchy of linguistic information, starting with
  surface features at the bottom, syntactic features in the middle followed by
  semantic features at the top."
- **Relation to LDTF:** motivational only; supports the premise that different
  layers encode different information. No mechanism is proposed.
- **Confidence:** Medium (metadata and abstract VERIFIED).

## 8. BERT Rediscovers the Classical NLP Pipeline

- **Authors:** Ian Tenney, Dipanjan Das, Ellie Pavlick
- **Year / Venue:** 2019, ACL 2019, pp. 4593–4601. DOI 10.18653/v1/P19-1452;
  arXiv:1905.05950
- **Official paper:** https://aclanthology.org/P19-1452/
- **Official repository:** **official source not found** in this session.
- **Files inspected:** none — Anthology landing page and arXiv v2 abstract only.
- **Relation to LDTF:** the nearest prior art to "learn a distribution over BERT
  layers and interpret it". Crucially, the layer-mixing weights are learned
  **per probing task**, with a separate probe per task. That is *task*
  conditioning; it is not conditioned on the class label within a task, is not
  computed from label queries, and is not input-dependent.
- **Caveat:** a recollection that this paper introduces a "centre of gravity"
  layer statistic is **from memory, NOT verified in this session**; the PDF body
  was not retrieved. Do not cite that term without checking the PDF.
- **Confidence:** Medium (metadata VERIFIED; content limited to the abstract).

## 9. Revealing the Dark Secrets of BERT

- **Authors:** Olga Kovaleva, Alexey Romanov, Anna Rogers, Anna Rumshisky
- **Year / Venue:** 2019, EMNLP-IJCNLP 2019, pp. 4365–4374. DOI 10.18653/v1/D19-1445
- **Official paper:** https://aclanthology.org/D19-1445/
- **Official repository:** **official source not found** in this session. The
  Anthology page lists a supplementary attachment which was observed but **not
  downloaded or inspected**.
- **Relation to LDTF:** thematic only — it concerns attention-*head* redundancy
  inside the encoder, whereas LDTF routes *across* layers. Its overparameterisation
  finding motivates a risk we explicitly test for: our per-class attention
  distributions may collapse to near-identical patterns, which would reduce LDTF
  to a scalar mix. This is why inter-class divergence of the depth distributions
  is a required diagnostic.
- **Confidence:** Medium (metadata and abstract VERIFIED).

## 10. HuggingFace Transformers — hidden-state output contract

- **Source:** official documentation,
  https://huggingface.co/docs/transformers/en/main_classes/output
- **Version:** page corresponds to Transformers v5.15.1 (retrieved during audit).
  The installed version in this project is recorded in `requirements.txt`.
- **Classes inspected:** `ModelOutput`, `BaseModelOutput`,
  `BaseModelOutputWithPooling`, `SequenceClassifierOutput`
- **Documented contract (quoted from the retrieved page):**
  - `hidden_states` — "returned when `output_hidden_states=True` ... Tuple of
    `torch.FloatTensor` (one for the output of the embeddings, if the model has
    an embedding layer, + one for the output of each layer) of shape
    `(batch_size, sequence_length, hidden_size)`."
  - `logits` — "Classification (or regression if config.num_labels==1) scores
    (before SoftMax)."
- **How this determined our implementation:**
  1. `L = num_hidden_layers + 1` and index 0 is the **embedding output, not a
     Transformer layer**. `src/models/bert_backbone.py` therefore takes
     `all_hidden_states[1:]` and documents that decision; including the embedding
     output is an explicit opt-in (`include_embedding_layer=True`).
  2. Raw logits before softmax are the ecosystem convention, so `LdtfBert`
     returns raw logits and never applies softmax or argmax internally.
  3. `pooler_output` is produced by a layer trained for next-sentence
     prediction and is unused by LDTF, so the backbone is loaded with
     `add_pooling_layer=False`.
- **License:** Transformers is Apache-2.0 (from memory, NOT verified in this
  session — the repository license file was not fetched during the audit).
- **Confidence:** High for the documented API contract (VERIFIED).

---

## Gap analysis — is label-conditioned depth routing novel?

### What was searched
- arXiv API, roughly twelve query formulations, including
  `abs:"label-specific" AND abs:"layer"`, `abs:"label attention" AND abs:"layer-wise"`,
  `abs:"label queries" AND abs:"transformer layers"`,
  `abs:"each label" AND abs:"different layers" AND abs:"BERT"`,
  `abs:"layer fusion" AND abs:"label"`, `abs:"layer weights" AND abs:"class-specific"`,
  `abs:"depth" AND abs:"routing" AND abs:"label"`, `abs:"label-wise attention"`,
  `abs:"scalar mix"`.
- ACL Anthology and IJCAI pages for the works above.
- GitHub REST API and raw file contents for the four architecture repositories.
- **Semantic Scholar Graph API: attempted and failed** (three queries returned
  empty responses). No citation-graph search was performed.

### Nearest candidates and why each falls short
- **ScalarMix / ELMo** (source read): learns a softmax distribution over layers,
  but as a single global `[N]` vector shared by all classes and all inputs.
- **Tenney et al. 2019** (abstract only): learns layer-mixing weights *per
  probing task*. Task conditioning, not per-class conditioning within a task,
  and not input-dependent.
- **LAAT** (source read): its "hierarchical" levels are label-hierarchy levels
  over a single BiLSTM output; there is no encoder-depth axis.
- **HiLAT** (arXiv:2204.10716, abstract retrieved): hierarchical label-wise
  attention over **tokens then chunks** of a document — token/chunk hierarchy,
  not layer hierarchy.
- **MHLAT** (arXiv:2309.08868), **Pseudo Label-wise Attention** (arXiv:2106.06822),
  **HLAN** (arXiv:2010.15728): surfaced by search but **abstracts were not
  retrieved**; flagged as unexamined nearest neighbours, not dismissed.

### Honest conclusion
To the best of our knowledge, and within the explicit limits of this search, no
prior work was found that conditions BERT layer weights on label or class
queries. The literature separates into two families that LDTF joins:

1. **Label-conditioned attention over tokens** — LEAM, CAML, LSAN, LAAT. All
   produce `[B,C,D]` per-class representations, and all operate on a *single*
   representation layer. **This half of LDTF is well precedented and is
   presented as such, not as novel.**
2. **Learned weighting over layers** — ScalarMix and the probing literature that
   uses it as an instrument. All such weights are global, or at most
   task-specific. None are per-class.

The contribution being tested is the composition: making the depth distribution
a function of the label query, so layer weights become `[B,C,L]` rather than `[L]`.

### Limits that weaken this conclusion, to be stated in any write-up
- The Semantic Scholar API failed; **no forward-citation sweep of ScalarMix or
  LEAM was performed**. This is the highest-value missing step.
- Google Scholar, DBLP, OpenReview and Papers-with-Code were not queried.
- Only titles and abstracts were searched on arXiv; **full text was never
  searched**. A method described only in a paper's Section 3 would be invisible
  to this search. This is the most serious gap.
- Only English-language sources were searched.
- Workshop papers, theses and non-archival venues were not covered.
- Code was read for four systems; the **full text of none of the ten papers**
  was read — every paper entry rests on metadata plus abstract.

Accordingly the defensible claim strength is:

> To the best of our knowledge, and based on a title/abstract-level search of
> arXiv together with a source-level review of the principal label-attention
> implementations, we are not aware of prior work that conditions transformer
> layer-mixing weights on label representations.

A bare "we are the first" is **not** supported by the evidence gathered here.

### Consequence for the experimental design
Because the token-routing half is thoroughly precedented and the depth-weighting
half is precedented in its unconditioned form, the decisive experiment is the
one that removes label conditioning from the depth stage while holding
everything else fixed. That is **A0 vs A3** (label-conditioned vs global scalar
mix) and **A0 vs A1** (depth routing vs uniform mean), with **A11 vs A1** as the
parameter-matched control. See `docs/architecture_traceability.md`.
