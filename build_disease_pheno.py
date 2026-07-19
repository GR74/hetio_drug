"""
build_disease_pheno.py  --  a REAL phenotype feature for diseases.

The earlier disease feature was just a MiniLM embedding of the disease NAME, which
did not transfer (unseen-disease AUROC 0.541, near chance). Names are too thin.

Here we build a genuine biomedical content vector for each disease from Hetionet:
  symptom profile   (Disease -presents-  Symptom, DpS)
  anatomy profile   (Disease -localizes- Anatomy, DlA)
concatenated into one binary vector. This is phenotype content, independent of the
treats label, so it is a legitimate inductive feature (a new disease can be
described by its symptoms and affected anatomy even with zero treatment edges).

Output: disease_pheno.npz  (aligned to full_subgraph.json disease order)
No network calls; parses the already-downloaded Hetionet edge file.
"""
import gzip, json, pathlib
import numpy as np

HERE = pathlib.Path(__file__).parent
SUB = json.loads((HERE / "full_subgraph.json").read_text())
dis_nodes = [d["node"] for d in SUB["diseases"]]
dis_idx = {n: i for i, n in enumerate(dis_nodes)}
ND = len(dis_nodes)

sym_edges, ana_edges = [], []
with gzip.open(HERE / "hetionet-v1.0-edges.sif.gz", "rt") as f:
    next(f)
    for line in f:
        s, m, t = line.rstrip("\n").split("\t")
        if s in dis_idx and m == "DpS": sym_edges.append((s, t))
        if s in dis_idx and m == "DlA": ana_edges.append((s, t))

def profile(edges):
    vocab = sorted({t for _, t in edges})
    col = {v: j for j, v in enumerate(vocab)}
    M = np.zeros((ND, len(vocab)), dtype=np.float32)
    for s, t in edges:
        M[dis_idx[s], col[t]] = 1.0
    return M

sym = profile(sym_edges)
ana = profile(ana_edges)
pheno = np.concatenate([sym, ana], axis=1)
np.savez(HERE / "disease_pheno.npz", pheno=pheno)

covered = int((pheno.sum(1) > 0).sum())
print(f"disease phenotype feature: {pheno.shape}  "
      f"(symptoms {sym.shape[1]} + anatomy {ana.shape[1]})")
print(f"  {covered}/{ND} diseases have at least one symptom/anatomy term")
print("  saved -> disease_pheno.npz")
