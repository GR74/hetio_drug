# helio_drug

Inductive zero-shot drug repurposing on Hetionet, built for the IAIRO GNN module.

This project reproduces the core of **TxGNN** (Huang et al., *Nature Medicine* 2024) on the
real Hetionet knowledge graph, then extends it one step: by initialising every node from
**content features** (molecular fingerprints for drugs, ESM2 protein embeddings for genes),
the model can predict treatments for drugs that were **never in the training graph**. That is
inductive zero-shot repurposing, which a per-node embedding table cannot do.

The full technical write-up, including the honest account of what worked and what did not, is
in [`MEMO.md`](MEMO.md).

## What is here

| Stage | Files |
|-------|-------|
| Subgraph (demo) | `build_subgraph.py`, `features.py`, `inductive_txgnn.py` |
| Full Hetionet | `build_full_subgraph.py`, `features_full.py`, `full_model.py` |
| Fixes (v2) | `build_disease_pheno.py`, `full_model_v2.py` |
| Disease diagnostic (v3) | `build_pheno_graph.py`, `full_model_v3.py`, `full_model_v3b.py` |
| Results | `results.json`, `full_results*.json`, `v3*_results.json`, `*.png` |
| Write-up | `MEMO.md` |

## Headline results

- **Subgraph, inductive unseen drugs:** fingerprint model beats id-embedding and popularity
  baselines (AUROC 0.757, Hits@10 0.498). Feature-only (no edges): 0.751 vs 0.551.
- **Full Hetionet:** contraindication head AUROC 0.931; feature-only unseen-drug regime
  0.615 vs 0.547, the clean inductive win at scale.
- **Disease side:** unseen-disease prediction stays near chance even with rich phenotype
  features. The v3 diagnostic investigates why (reach, phenotype bridge, sparsity ceiling).

## Reproduce

```bash
pip install -r requirements.txt

# demo subgraph
python build_subgraph.py
python features.py            # slow once (PubChem + ESM2), then cached
python inductive_txgnn.py

# full Hetionet
python build_full_subgraph.py
python features_full.py
python full_model.py
```

The build scripts download Hetionet (public), and the feature scripts pull SMILES from PubChem
and protein sequences from UniProt, then embed them. All fetched data is cached locally, so the
slow steps run only once.

## Data and credits

- Hetionet v1.0 (het.io), public domain graph of 47k nodes and 2.25M edges.
- PubChem (SMILES), UniProt (protein sequences), RDKit (fingerprints), ESM2 and MiniLM
  (embeddings via HuggingFace Transformers).
- Anchor paper: Huang, Chandak, ..., Zitnik. "A foundation model for clinician-centered drug
  repurposing." *Nature Medicine* 30:3601-3613 (2024).
