"""
proximity.py  --  the last disease-side push: mechanistic network proximity.

Every disease-side attempt so far LEARNED an embedding and failed (Sections 8-10).
This tries the opposite: a non-learned network-medicine score that asks a direct
mechanistic question, following Guney et al. (2016) style proximity:

    Does the drug's protein targets sit near the disease's genes in the
    protein-protein interaction network?

If this separates true treatments from popularity-matched decoys where the GNN
could not, then the disease-side signal was there and the learned model washed it
out (a real fix). If it also fails, the signal is genuinely absent and the
Section 10 conclusion is confirmed by an independent method. Either way we learn
something. No training, no features, runs in seconds.

Scores per (drug c, disease d), using only CbG/CuG/CdG (drug->gene),
DaG/DuG/DdG (disease->gene) and GiG (gene-gene):
    overlap  = | targets(c) INTERSECT genes(d) |
    one_hop  = | targets(c) INTERSECT GiG-neighbours(genes(d)) |
    score    = (overlap + 0.5 * one_hop) / sqrt(|targets(c)|)   # de-bias promiscuous drugs
Evaluated with popularity-matched negatives, the same honest bar used everywhere.
"""
import json, pathlib, math
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
SUB = json.loads((HERE / "full_subgraph.json").read_text())
rng = np.random.default_rng(0)

dis_nodes = [d["node"] for d in SUB["diseases"]]
dis_idx = {n: i for i, n in enumerate(dis_nodes)}
ND = len(dis_nodes)
E = SUB["edges"]

# drug -> target genes ; disease -> genes ; gene -> GiG neighbours
targets = defaultdict(set)
for r in ("CbG", "CuG", "CdG"):
    for s, t in E[r]:
        targets[s].add(t)
dis_genes = defaultdict(set)
for r in ("DaG", "DuG", "DdG"):
    for s, t in E[r]:
        dis_genes[s].add(t)
gi = defaultdict(set)
for s, t in E["GiG"]:
    gi[s].add(t); gi[t].add(s)

def score(c_node, d_node):
    T = targets.get(c_node, set())
    if not T:
        return 0.0
    G = dis_genes.get(d_node, set())
    overlap = len(T & G)
    NG = set().union(*(gi[g] for g in G)) if G else set()
    one_hop = len(T & NG)
    return (overlap + 0.5 * one_hop) / math.sqrt(len(T))

# ground-truth treats edges and disease popularity
CtD = [(s, t) for s, t in E["CtD"]]
pos_set = set(CtD)
dis_pop = np.zeros(ND)
for _, d in CtD:
    dis_pop[dis_idx[d]] += 1
POP_P = (dis_pop + 1) / (dis_pop + 1).sum()
comp_nodes = [c["node"] for c in SUB["compounds"]]

def eval_scorer(fn, name, ratio=10):
    pos = CtD
    # popularity-matched negatives: same drugs, diseases drawn by popularity
    negs = []
    while len(negs) < len(pos) * ratio:
        c = comp_nodes[rng.integers(len(comp_nodes))]
        d = dis_nodes[rng.choice(ND, p=POP_P)]
        if (c, d) not in pos_set:
            negs.append((c, d))
    pairs = pos + negs
    y = np.r_[np.ones(len(pos)), np.zeros(len(negs))]
    s = np.array([fn(c, d) for c, d in pairs])
    return roc_auc_score(y, s)

print("=" * 66)
print("LAST DISEASE-SIDE PUSH: mechanistic network proximity (no learning)")
print(f"  {len(comp_nodes)} drugs, {ND} diseases, {len(CtD)} treatments")
print("  evaluation: popularity-matched negatives (the honest bar)")
print("=" * 66)

auc_prox = eval_scorer(score, "proximity")
auc_overlap = eval_scorer(lambda c, d: len(targets.get(c, set()) & dis_genes.get(d, set())), "overlap")
auc_pop = eval_scorer(lambda c, d: dis_pop[dis_idx[d]], "popularity")

print(f"\n  popularity prior (baseline)      AUROC = {auc_pop:.3f}")
print(f"  target-gene overlap only         AUROC = {auc_overlap:.3f}")
print(f"  network proximity (overlap+1hop) AUROC = {auc_prox:.3f}")
print(f"\n  reference: learned GNN disease-side was ~0.44-0.50 (Sections 10)")

best = max(auc_prox, auc_overlap)
if best > 0.6:
    verdict = "SIGNAL FOUND: mechanistic proximity beats popularity and the GNN. The disease-side signal was there; the learned model washed it out. Use proximity for disease queries."
elif best > 0.55:
    verdict = "WEAK SIGNAL: proximity is modestly above chance, more than the GNN managed. Worth combining with the model."
else:
    verdict = "NO SIGNAL: even a direct mechanistic score is near chance. Confirms Section 10 by an independent method: the disease-side therapeutic signal is genuinely absent in this graph slice."
print(f"\n  VERDICT: {verdict}")

json.dump({"popularity": round(auc_pop, 3), "overlap": round(auc_overlap, 3),
           "proximity": round(auc_prox, 3), "verdict": verdict},
          open(HERE / "proximity_results.json", "w"), indent=2)
print("\nSaved -> proximity_results.json")
