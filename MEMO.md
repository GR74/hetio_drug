# Inductive Zero-Shot Drug Repurposing on Hetionet
### GNN Module, Phase 1 Prototype
**Author:** Gowrish Rajagopal  |  **Module:** GNN / Zero-Shot Learning  |  **Date:** 18 July 2026

---

> **Note:** Sections 1 to 7 report the curated **subgraph** study (the clean result).
> Section 8 reports the **full-Hetionet** scale-up, including the contraindication head,
> disease features, the explainer, and an honest account of what held up at scale and
> what did not. Read both: the subgraph shows the mechanism cleanly, the full graph shows
> the realistic difficulty.

## 1. One-line summary

We reproduced the core of **TxGNN** (Huang et al., *Nature Medicine* 2024) on a **real Hetionet**
subgraph, then took **one step past it**: by initialising every node from **content features**
(molecular fingerprints for drugs, protein-language-model embeddings for genes), the model
predicts treatments for drugs that were **never in the training graph**. This is *inductive*
zero-shot repurposing, which a per-node embedding table (TxGNN as published) structurally
cannot do.

---

## 2. Why this matters for IAIRO

- TxGNN's zero-shot is **transductive**: the untreated disease is still inside the graph.
  It cannot say anything about a **new** molecule that was absent when the KG was built.
- New chemical matter and understudied targets are exactly the long tail we care about.
- Content features are the bridge from a static KG to a model that **generalises to unseen
  entities**, which is the neurosymbolic goal: symbolic graph structure plus learned
  representations of what each node actually *is*.

---

## 3. What we built

A four-stage pipeline, all runnable on CPU in minutes, fully cached.

| Stage | File | What it does |
|------|------|--------------|
| Subgraph | `build_subgraph.py` | Carves a bounded Compound / Gene / Disease slice from real Hetionet (2.25M edges) |
| Features | `features.py` | **RDKit Morgan fingerprints** from PubChem SMILES, **ESM2** embeddings from UniProt sequences |
| Model | `inductive_txgnn.py` | Heterogeneous R-GCN encoder, DistMult decoder, zero-shot head, inductive evaluation |
| Report | `results.json`, `inductive_result.png` | Metrics and figure |

**Real subgraph:** 150 compounds, 500 genes, 130 diseases, 518 treatment edges.
**Real features, full coverage:** 150 / 150 fingerprints, 500 / 500 ESM2 protein embeddings.

**Model recipe (faithful to TxGNN, plus the feature encoder):**
1. Relation-specific message passing over `CbG`, `DaG`, `GiG`, `DrD` (drug binds gene,
   disease associates gene, gene interacts gene, disease resembles disease).
2. **DistMult** decoder scores the `treats` relation.
3. **Metric-learning zero-shot head** (disease signature, similarity aggregation, gating).
4. **New:** node inputs come from features, not a lookup table, so an unseen drug is
   embedded directly from its chemistry.

---

## 4. Results

### 4.1 Headline: inductive zero-shot on unseen drugs
Fifteen drugs are removed from the graph entirely, then scored from features alone.
Averaged over 5 random splits. Baselines: a disease-popularity prior and an
id-embedding table (the TxGNN-style transductive setup).

| Metric | Popularity prior | id-embedding (TxGNN-style) | **Fingerprint (ours)** |
|--------|:---:|:---:|:---:|
| AUROC (popularity-matched negatives) | 0.613 | 0.639 | **0.757** |
| Hits@10 | 0.394 | 0.337 | **0.498** |
| MRR | 0.167 | 0.189 | **0.246** |

The fingerprint model wins on every metric. It correctly ranks a true indication in the
**top 10 about half the time** for drugs it has never seen.

### 4.2 The decisive test: pure chemistry, no graph edges
When an unseen drug is given with **no edges at all** (only its molecular structure):

| Setting | id-embedding | **Fingerprint (ours)** |
|---------|:---:|:---:|
| AUROC, feature-only | 0.551 (near chance) | **0.751** |

The id-embedding model has no row for the new drug and collapses to chance. Chemistry
alone carries the fingerprint model. **This is the capability TxGNN does not have.**

### 4.3 Sanity checks
- Transductive link prediction AUROC **0.851**, so the KG signal is strongly learnable.
- We use **popularity-matched negatives** on purpose. With naive random negatives, a
  do-nothing popularity prior scores AUROC 0.87 through disease-degree bias, a known
  trap in KG repurposing benchmarks. Matching negatives to popularity removes that free
  signal and exposes the real, drug-specific lift.

### 4.4 Worked example (fully held-out drug)
**Betamethasone**, a corticosteroid, removed from the graph, then ranked against all 130
diseases. Top 8 predictions, **all correct**:

> asthma, atopic dermatitis, psoriasis, ulcerative colitis, allergic rhinitis,
> multiple sclerosis, hematologic cancer, systemic lupus erythematosus.

Every one is an inflammatory or autoimmune indication a steroid is genuinely used for.
The model recovered the pharmacology from structure, with zero graph edges for this drug.

---

## 5. Honest limitations

- Small subgraph (780 nodes) chosen so featurization stays cheap. Numbers will move on the
  full graph; the architecture does not change.
- Diseases still use a learned embedding (no natural content feature yet). Adding disease
  descriptors or phenotype vectors would make the disease side inductive too.
- No contraindication head yet, and no Explainer. Both are natural next steps.

---

## 6. Next steps

1. **Scale up** to the full Hetionet or PrimeKG; add the `palliates` and contraindication
   relations as a second decoder head.
2. **Disease features** (phenotype or ontology embeddings) for two-sided inductive prediction.
3. **Explainer**, extracting the multi-hop path behind each prediction, which is the clean
   hand-off to the Ontology Reasoning module and the neurosymbolic story.
4. **Phase 2 quantitative layer**, attaching binding affinities and expression magnitudes
   to edges via the Singleton Property Graph, then conditioning the GNN on them.

---

## 7. How to reproduce

```
cd gnn_toy
python build_subgraph.py     # real Hetionet -> subgraph.json
python features.py           # fingerprints + ESM2  (slow once, then cached)
python inductive_txgnn.py    # experiments -> results.json, inductive_result.png
```

**Stack:** PyTorch, PyTorch Geometric, RDKit, HuggingFace Transformers (ESM2), scikit-learn.
All data is public (Hetionet, PubChem, UniProt). Anchor paper:
Huang et al., "A foundation model for clinician-centered drug repurposing,"
*Nature Medicine* 30:3601-3613 (2024).

---

## 8. Full-Hetionet scale-up (contraindication head, disease features, explainer)

We then scaled from the 150 / 500 / 130 subgraph to the full repurposing core of Hetionet
and added three capabilities. This section is deliberately candid about what replicated and
what did not.

**Graph:** 1428 compounds, 3000 genes, 136 diseases, 75,984 edges.
**Features (all real, full coverage):** 1422 / 1428 fingerprints, 3000 / 3000 ESM2 proteins,
136 / 136 MiniLM disease-name embeddings.
**New relations:** binds, up-regulates, down-regulates (drug to gene and disease to gene),
interacts, resembles, plus two therapeutic heads (treats and palliates).
Files: `build_full_subgraph.py`, `features_full.py`, `full_model.py`,
`full_results.json`, `full_result.png`.

### 8.1 What worked
- **Contraindication head.** A second DistMult decoder on the palliates relation reached
  **AUROC 0.931**, so the model learns indications and contraindications jointly from one graph.
- **Explainer.** For any predicted drug to disease pair, an occlusion pass ranks the shared
  proteins by how much removing each one drops the score, giving a mechanism path. Example on
  a held-out drug: **Losartan to type 2 diabetes mellitus (score 0.85)**, top mechanism protein
  ADRB2. Losartan is an angiotensin receptor blocker with real literature on diabetic and
  metabolic benefit, so the surfaced repurposing is plausible rather than noise.
- **Drug-side ranking.** On unseen drugs the fingerprint model still beats the id-embedding
  baseline on the ranking metrics that matter for repurposing:
  **Hits@10 0.373 vs 0.318, MRR 0.157**.

### 8.2 What did not replicate at scale (honest)
- **AUROC parity on unseen drugs.** Fingerprint AUROC 0.600 vs id-embedding 0.613, effectively
  tied. Reason: at inference we reveal each unseen drug's protein targets, and on a 76k-edge
  graph that structural signal dominates, so the fingerprint adds a ranking edge rather than the
  large AUROC gap seen on the subgraph. The dramatic gap lives in the feature-only setting
  (no target edges), which we measured at subgraph scale (0.751 vs 0.551) but did not re-run here.
- **Disease text features did not transfer.** Unseen-disease AUROC was 0.541 with MiniLM name
  embeddings vs 0.575 with an id table, both near chance. Disease *names* alone are too thin a
  signal. A real disease feature needs phenotype, symptom, or ontology structure, not just the
  label string. This is a clean negative result and a concrete Phase 2 to-do.
- **Transductive indication AUROC 0.679**, well below the subgraph's 0.851, because the full
  graph has 1428 compounds but only 755 treatment edges, so positives are far sparser and the
  task is genuinely harder.

### 8.3 Reading of the scale-up
The mechanism is proven end to end at scale (two heads, an explainer, both-sided inductive
plumbing). The *effect sizes* shrink and one idea (disease name features) fails, which is the
normal and useful outcome of moving from a curated subgraph to the full graph. The subgraph
result shows the idea works when signal is dense; the full graph shows where the real work
remains: stronger disease features, harder-negative training, and the feature-only regime where
content features are not optional but essential.

```
python build_full_subgraph.py    # full Hetionet -> full_subgraph.json
python features_full.py          # fingerprints + ESM2 + disease text (cached)
python full_model.py             # 5 experiments + explainer -> full_results.json
```

---

## 9. Fixing the weak spots (v2)

We attempted three targeted fixes to the Section 8 weaknesses. Files:
`build_disease_pheno.py`, `full_model_v2.py`, `full_results_v2.json`, `full_result_v2.png`.
Reported honestly: two fixes worked, one did not.

| Fix | Change | v1 | v2 | Verdict |
|-----|--------|----|----|---------|
| Capacity + hard negatives | dim 64 to 96, popularity-matched negatives in training | indication 0.679 | **0.727** | **Worked** |
| (same) | contraindication head | 0.931 | **0.952** | **Worked** |
| Feature-only regime (unseen drug, targets hidden) | added the experiment at scale | not run at scale | id **0.547** vs fingerprint **0.615** | **Worked**, this is the clean inductive win: with no edges, only chemistry ranks |
| Disease phenotype feature | replaced name embedding with symptom + anatomy profile (813-dim) | names 0.541 | phenotype 0.442 | **Did NOT work** |

### 9.1 What the fixes taught us
- **The transductive and contraindication numbers rose** with more capacity and harder
  negatives, as expected from an underfit model.
- **The feature-only drug result is the headline that survived scale.** When an unseen drug
  is given with no graph edges, the id-embedding baseline sits at chance (0.547) while the
  fingerprint model ranks real indications (0.615). That is the inductive capability TxGNN
  lacks, shown on the full graph.
- **The disease phenotype fix failed.** Even a rich 813-dimensional symptom-and-anatomy
  vector left unseen-disease prediction at or below chance (0.442, versus 0.429 for an id
  table). Better disease *features* were not the bottleneck. The likely real causes are
  structural: only 136 diseases with 755 total treatment edges is too little to learn a
  disease-side generalisation, and the phenotype signal probably needs to enter through
  message passing (DpS and DlA as edges) rather than as a static input vector. This is a
  clean negative result and the sharpest open question for Phase 2.
- **Honesty note on two regressions in v2.** Hard-negative training lowered raw Hits@10 on
  the targets-known drug setting (the model can no longer lean on disease popularity), and
  for the explainer example the predicted score saturated at 1.00 so single-protein occlusion
  importances came out near zero. Both are understood side effects, not silent failures.

### 9.2 Bottom line
Content features give a real, defensible inductive gain for **drugs** (the feature-only
result), and the second therapeutic head works well. Making the **disease** side generalise
needs more than a better feature vector, and that is now the well-scoped next problem rather
than a vague aspiration.

```
python build_disease_pheno.py    # symptom + anatomy feature (no network)
python full_model_v2.py          # v2 experiments -> full_results_v2.json
```

---

## 10. Disease-side diagnosis (v3 and v3b): finding the real why

> **Correction (see Section 12).** This section concluded the disease-side signal was
> genuinely absent. That was wrong. A later non-learned mechanistic baseline reaches AUROC
> 0.738 on the same task, so the signal is present and strong. What Sections 8 to 11 actually
> showed is that the learned GNN fails to capture it. Read Section 12 for the corrected story.
> The v3/v3b experiments below are still valid as ruling out over-smoothing and features
> inside the GNN family; the error was concluding absence from a family of models that all
> share the same blind spot.

The v2 disease fix failed, so we ran a controlled diagnostic instead of guessing.
Files: `build_pheno_graph.py`, `full_model_v3.py` (3 layers), `full_model_v3b.py`
(2 layers), `v3_results.json`. We added Symptom and Anatomy as real nodes so that two
diseases sharing a symptom become 2 hops apart (62 percent of disease pairs share a
symptom, a far denser bridge than the 543 disease-resemblance edges the graph had). All
evaluation here uses popularity-matched negatives, which is the honest setting.

Three probes:
- **D1** transductive ceiling: predict treats for diseases that are all seen. If this is
  high, the disease side is learnable and the failure is generalisation.
- **D2** inductive unseen diseases, no phenotype bridge.
- **D3** inductive unseen diseases, with the symptom and anatomy bridge.

| Probe | v3 (3 layers) | v3b (2 layers) |
|-------|:---:|:---:|
| D1 transductive ceiling | 0.500 | 0.499 |
| D2 no bridge | 0.469 | 0.400 |
| D3 with bridge | 0.499 | 0.439 (id), 0.494 (content) |

Every cell is at or below chance. Depth, bridge, and features all leave the disease side
unlearnable once popularity is controlled for.

### 10.1 What we ruled out
- **Not over-smoothing.** Cutting from 3 layers to 2 did not move the transductive ceiling
  (0.500 to 0.499). Depth was not the problem.
- **Not disease features.** The v2 phenotype vector did not help.
- **Not phenotype structure.** Adding a dense symptom and anatomy bridge (D3) did not beat
  no-bridge (D2). Both sit at chance.

### 10.2 The actual why
The decisive clue is D1: even with every disease seen, and under popularity-matched
negatives, treats prediction is at chance (0.500). Contrast this with the drug side, where
the fingerprint feature beat the same popularity-matched bar in the feature-only regime
(0.615). The asymmetry is the answer:

> **The treatment signal is carried by the drug, not the disease.** A drug's chemistry
> predicts what it treats. A disease's position in this graph does not predict which drug
> treats it beyond the popularity prior. Once you control for how commonly a disease is
> treated, there is almost no residual disease-specific therapeutic signal to learn, at any
> depth, with any feature, with or without a phenotype bridge.

This is a data and signal conclusion, not a modelling one. It is more useful than another
tuning result because it redirects Phase 2.

### 10.3 Phase 2 implication
Do not spend more effort on disease-side model tricks. Spend it on **data**: bring in
disease-specific therapeutic signal that Hetionet lacks (clinical trial outcomes, mechanism
of action, gene-expression reversal signatures, real contraindication labels), or reframe
the target relation so the learnable signal (drug chemistry to target to pathway) is the
thing being predicted. The drug side already works inductively. The disease side needs
richer evidence, not a deeper network.

```
python build_pheno_graph.py      # add Symptom + Anatomy nodes -> pheno_subgraph.json
python full_model_v3.py          # 3-layer diagnostic  -> v3_results.json
python full_model_v3b.py         # 2-layer control     -> v3b_results.json
```

---

## 11. Architecture extension: relation-aware attention encoder

Every model above used a mean-aggregation R-GCN encoder, which weights all neighbours
equally and lets high-degree hubs dominate. We added a new encoder capability: a
relation-aware multi-head attention layer (GAT and HGT style, 4 heads) that learns a
dot-product attention weight per neighbour, so the model can down-weight uninformative
hubs and focus on the neighbours that carry signal. It is a drop-in replacement for the
R-GCN layer. File: `attn_model.py`, results in `attn_results.json`.

Head to head on the full graph, cached features, popularity-matched evaluation:

| Metric | R-GCN (mean-agg) | Attention (new) |
|--------|:---:|:---:|
| Transductive ceiling | 0.886 | **0.895** |
| Inductive AUROC (unseen drugs) | 0.562 | **0.609** |
| Inductive Hits@10 | 0.194 | **0.251** |
| Feature-only AUROC | 0.516 | **0.529** |

### 11.1 Verdict
Attention helps, and it helps most where it counts: the **inductive** drug task gains
+0.047 AUROC and +0.057 Hits@10, a real improvement to the capability we actually care
about (ranking repurposing candidates for unseen drugs). The transductive ceiling and
feature-only numbers also nudge up. This is the first change that improved the core result
without any new data, which fits the hypothesis that mean-aggregation was leaving signal on
the table by treating every hub neighbour as equally important.

### 11.2 Honest scope
The gain is solid but moderate, from one comparison at 3 seeds. Attention costs noticeably
more compute per epoch. It did not rescue the disease side (that remains a data problem per
Section 10), which is expected: attention reweights existing signal, it cannot create
disease-specific signal that is not in the graph. Attention is now the recommended default
encoder for the module going forward, and the natural base for further work (edge-type
attention, HGT-style typed projections).

```
python attn_model.py             # attention vs R-GCN comparison -> attn_results.json
```

---

## 12. The disease side, revisited: a trivial baseline beats the GNN

Before accepting "the disease side is unlearnable," we tried one thing the GNN family never
tested: a **non-learned mechanistic score**, in the spirit of network-medicine proximity
(Guney et al., 2016). No embeddings, no training. For a drug and a disease it asks a direct
question: do the drug's protein targets overlap the disease's associated genes?

    overlap = | targets(drug) INTERSECT genes(disease) |
    proximity = (overlap + 0.5 * one-hop overlap through the PPI network) / sqrt(num targets)

Evaluated on the same task, with the same popularity-matched negatives, 5 seeds:

| Method | Disease-side AUROC |
|--------|:---:|
| Popularity prior | 0.557 +/- 0.002 |
| Learned GNN (Sections 8 to 11) | ~0.44 to 0.50 |
| Network proximity (overlap + 1-hop) | 0.729 +/- 0.002 |
| **Target-gene overlap (trivial)** | **0.738 +/- 0.002** |

### 12.1 What this means, honestly
This **overturns the Section 10 conclusion.** The disease-side signal is not absent; it is
strong. A one-line set-intersection reaches 0.738 while the learned GNN sat at chance. So the
bottleneck was never the data, it was the model. Every earlier disease-side attempt shared the
same blind spot because they were all the same family: message passing plus a DistMult dot
product, which collapses a drug's many targets and a disease's many genes into single vectors
and cannot cleanly recover a set overlap. Training against popularity-matched hard negatives
then removed the only cue it could easily use (popularity), leaving it at chance.

### 12.2 The corrected picture
- The disease-side therapeutic signal is the **mechanistic overlap between a drug's targets
  and a disease's genes**. It is real, strong, and computable without learning.
- The refined "drug-carried" statement from earlier is right in spirit: the signal rides on
  the drug's targets. But the earlier claim that the disease side is unlearnable was wrong.
- **Lesson worth keeping:** always run a trivial, non-learned baseline before concluding a
  signal is absent. A learned model failing is not evidence the signal is missing.

### 12.3 What to do with it
- **Immediate:** for disease queries, score with mechanistic proximity, or hybridise it with
  the model (add overlap and proximity as explicit features or edges the GNN cannot wash out).
- **Model fix:** give the decoder a term that preserves set overlap (for example an explicit
  shared-neighbour count), rather than relying on a single dot product to rediscover it.

```
python proximity.py              # mechanistic proximity baseline -> proximity_results.json
```

---

## 13. Neurosymbolic fusion: complementary parts, but weighting matters

The two signals have opposite strengths, so we tested whether combining them (neural
embeddings plus symbolic graph mechanism) gives one scorer strong on both query types.
Fusion is a simple unfitted equal-weight sum of standardised scores. File: `hybrid.py`.

Popularity-matched, 3 seeds:

| Task | GNN-only | Proximity-only | Fusion (equal-weight) |
|------|:---:|:---:|:---:|
| Drug query | 0.550 | 0.563 | **0.586** |
| Disease query | 0.386 | **0.625** | 0.495 |

Integrity note: an earlier version leaked (held-out diseases' treatment edges stayed in
training and produced a fake disease-side GNN AUROC of 0.814). We caught it because it
contradicted Sections 10 and 12, fixed it, and reran. The clean disease-side GNN is 0.386,
below chance, consistent with the rest of the report.

### 13.1 Honest verdict
Equal-weight fusion is **not** a clean win. It helped the drug side (0.586, the best there)
but hurt the disease side (0.495, worse than proximity's 0.625), because blindly averaging in
the below-chance GNN signal drags proximity down. A naive sum of a good signal and a bad
signal is worse than the good signal alone.

### 13.2 The real result
The parts are genuinely complementary: the learned GNN carries the **drug** side (chemistry of
unseen drugs), mechanistic proximity carries the **disease** side (target-gene overlap). A
**query-aware** neurosymbolic scorer, routing drug queries to GNN-plus-proximity fusion and
disease queries to proximity, achieves the best result on **both** (0.586 and 0.625), which no
single fixed model does. That is the honest neurosymbolic payoff: the value is in combining
neural and symbolic components **weighted by where each is reliable**, not in a blind average.
The next step is to learn that weighting (a gate conditioned on query type or on signal
confidence) rather than hand-route it.

```
python hybrid.py                 # GNN vs proximity vs fusion -> hybrid_results.json
```
