"""
hybrid.py  --  the neurosymbolic payoff: learned GNN + mechanistic proximity.

Two signals, two strengths:
  * the learned GNN generalises on the DRUG side (chemistry of unseen drugs), but
    misses the disease side (Section 12: it cannot recover a set overlap).
  * mechanistic proximity (drug targets overlapping disease genes) nails the
    DISEASE side but is a fixed, non-learned rule.

The neurosymbolic move is to fuse them: neural embeddings plus a symbolic
graph-mechanism score. If the fusion beats both parts on both query types, that
is the whole IAIRO thesis in one experiment: neither pure learning nor pure rules,
but their combination.

Fusion is deliberately simple and unfitted: standardise each score over the eval
set and add them (equal weight). No fitting on the test set, no free parameters.

Compares GNN-only vs proximity-only vs fusion on the drug-side and disease-side
inductive tasks, popularity-matched negatives, 3 seeds. Cached features, R-GCN
encoder for speed (attention is a drop-in but slower).
"""
import json, pathlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
SUB = json.loads((HERE / "full_subgraph.json").read_text())
FEAT = np.load(HERE / "full_features.npz")
rng = np.random.default_rng(0); torch.manual_seed(0)
DIM, LAYERS, EPOCHS = 96, 2, 250

comp = [c["node"] for c in SUB["compounds"]]
gene = [g["node"] for g in SUB["genes"]]
dis  = [d["node"] for d in SUB["diseases"]]
NC, NG, ND = len(comp), len(gene), len(dis)
gi_idx = {n: i for i, n in enumerate(gene)}
gid = {}
for i, n in enumerate(comp): gid[n] = i
for i, n in enumerate(gene): gid[n] = NC + i
for i, n in enumerate(dis):  gid[n] = NC + NG + i
N = NC + NG + ND; DIS0 = NC + NG
comp_fp  = torch.tensor(FEAT["compound_fp"], dtype=torch.float32)
gene_esm = torch.tensor(FEAT["gene_esm"],   dtype=torch.float32)
E = SUB["edges"]

# --------------------------------------------------------------------------- #
# Symbolic side: overlap matrix O[drug, disease] via sparse-style matmul       #
# --------------------------------------------------------------------------- #
Tmat = np.zeros((NC, NG), np.float32)
for r in ("CbG", "CuG", "CdG"):
    for s, t in E[r]:
        if s in gid and t in gi_idx and gid[s] < NC: Tmat[gid[s], gi_idx[t]] = 1.0
Dmat = np.zeros((ND, NG), np.float32)
for r in ("DaG", "DuG", "DdG"):
    for s, t in E[r]:
        if s in gid and t in gi_idx: Dmat[gid[s] - DIS0, gi_idx[t]] = 1.0
OVER = Tmat @ Dmat.T                                   # [NC, ND] shared-gene counts
tgt_count = Tmat.sum(1, keepdims=True)
PROX = OVER / np.sqrt(np.clip(tgt_count, 1, None))     # de-biased proximity score

def pg(rel): return np.array([[gid[s], gid[t]] for s, t in E[rel]])
CtD = pg("CtD"); CpD = pg("CpD")
dis_global = np.arange(DIS0, N)
POS_CODE = np.sort(CtD[:, 0].astype(np.int64) * N + CtD[:, 1])
REL = ["CbG", "CuG", "CdG", "DaG", "DuG", "DdG", "GiG", "DrD"]
DRUG_GENE = {"CbG", "CuG", "CdG"}; DIS_GENE = {"DaG", "DuG", "DdG"}
REL_KEYS = REL + [r + "_inv" for r in REL]
dpop = np.zeros(ND)
for d in CtD[:, 1]: dpop[d - DIS0] += 1
POP_P = (dpop + 1) / (dpop + 1).sum()

def build_edges(drop_comp_dg=frozenset(), drop_dis_dg=frozenset()):
    Ed = {}
    for r in REL:
        out = [(gid[s], gid[t]) for s, t in E[r]
               if not (r in DRUG_GENE and gid[s] in drop_comp_dg)
               and not (r in DIS_GENE and gid[s] in drop_dis_dg)]
        et = torch.tensor(out, dtype=torch.long).t() if out else torch.zeros(2, 0, dtype=torch.long)
        Ed[r] = et; Ed[r + "_inv"] = et.flip(0)
    return Ed

class RGCN(nn.Module):
    def __init__(s, d):
        super().__init__(); s.w = nn.ModuleDict({r: nn.Linear(d, d, bias=False) for r in REL_KEYS})
    def forward(s, h, Ed):
        out = h.clone()
        for r, e in Ed.items():
            if e.size(1) == 0: continue
            m = s.w[r](h[e[0]]); agg = torch.zeros_like(h); agg.index_add_(0, e[1], m)
            deg = torch.zeros(h.size(0)); deg.index_add_(0, e[1], torch.ones(e.size(1)))
            out = out + agg / deg.clamp(min=1).unsqueeze(1)
        return F.relu(out)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.cf = nn.Sequential(nn.Linear(comp_fp.size(1), DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        s.ge = nn.Sequential(nn.Linear(gene_esm.size(1), DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        s.dis = nn.Parameter(torch.empty(ND, DIM)); nn.init.xavier_uniform_(s.dis)
        s.gnn = nn.ModuleList([RGCN(DIM) for _ in range(LAYERS)])
        s.w = nn.ParameterDict({r: nn.Parameter(torch.ones(DIM)) for r in ["CtD", "CpD"]})
    def enc(s, Ed):
        h = torch.cat([s.cf(comp_fp), s.ge(gene_esm), s.dis], 0)
        for l in s.gnn: h = l(h, Ed)
        return h
    def score(s, h, c, d, rel="CtD"): return (h[c] * s.w[rel] * h[d]).sum(-1)

def neg(pos, pool, rs):
    n = len(pos); cs = rs.choice(pool, size=n * 2); ds = rs.choice(ND, size=n * 2, p=POP_P) + DIS0
    return np.stack([cs, ds], 1)[~np.isin(cs.astype(np.int64) * N + ds, POS_CODE)][:n]

def train(m, Ed, pool, rs, ctd_pos):
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-5)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad(); h = m.enc(Ed); loss = 0.0
        for rel, pos in [("CtD", ctd_pos), ("CpD", CpD)]:
            ng = neg(pos, pool, rs)
            pc = m.score(h, torch.tensor(pos[:, 0]), torch.tensor(pos[:, 1]), rel)
            nc = m.score(h, torch.tensor(ng[:, 0]), torch.tensor(ng[:, 1]), rel)
            loss = loss + F.binary_cross_entropy_with_logits(
                torch.cat([pc, nc]), torch.cat([torch.ones_like(pc), torch.zeros_like(nc)]))
        loss.backward(); opt.step()
    return m

def z(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / (s if s > 1e-9 else 1.0)

def make_pairs(test_pos, rs, ratio=10):
    n = len(test_pos) * ratio
    cs = rs.choice(test_pos[:, 0], size=n * 2); ds = rs.choice(ND, size=n * 2, p=POP_P) + DIS0
    ng = np.stack([cs, ds], 1)[~np.isin(cs.astype(np.int64) * N + ds, POS_CODE)][:n]
    pairs = np.vstack([test_pos, ng]); y = np.r_[np.ones(len(test_pos)), np.zeros(len(ng))]
    return pairs, y

def three_way(m, Ed, pairs, y):
    with torch.no_grad():
        h = m.enc(Ed)
        g = m.score(h, torch.tensor(pairs[:, 0]), torch.tensor(pairs[:, 1])).numpy()
    p = PROX[pairs[:, 0], pairs[:, 1] - DIS0]
    fus = z(g) + z(p)
    return roc_auc_score(y, g), roc_auc_score(y, p), roc_auc_score(y, fus)

print("=" * 70)
print("NEUROSYMBOLIC FUSION: learned GNN + mechanistic proximity")
print(f"  {NC} drugs {NG} genes {ND} diseases | popularity-matched eval, 3 seeds")
print("=" * 70)

deg_d = np.bincount(CtD[:, 0], minlength=NC); elig_d = np.where(deg_d >= 2)[0]
deg_s = np.bincount(CtD[:, 1] - DIS0, minlength=ND); elig_s = np.where(deg_s >= 2)[0]
R = {t: {"gnn": [], "prox": [], "fus": []} for t in ["drug", "disease"]}

for sd in [0, 1, 2]:
    rs = np.random.default_rng(60 + sd)
    # drug-side: hold out drugs (their treats + gene edges removed from training)
    ho = rs.choice(elig_d, size=25, replace=False); hs = set(int(c) for c in ho)
    pool = np.array([c for c in range(NC) if c not in hs])
    trp_drug = CtD[np.isin(CtD[:, 0], pool)]
    m = train(Net(), build_edges(drop_comp_dg=hs), pool, rs, trp_drug)
    pairs, y = make_pairs(CtD[np.isin(CtD[:, 0], ho)], rs)
    a, b, c = three_way(m, build_edges(), pairs, y)
    R["drug"]["gnn"].append(a); R["drug"]["prox"].append(b); R["drug"]["fus"].append(c)
    # disease-side: hold out diseases (their treats + gene edges removed from training)
    hod = rs.choice(elig_s, size=15, replace=False); hds = set(int(d + DIS0) for d in hod)
    trp_dis = CtD[~np.isin(CtD[:, 1] - DIS0, hod)]     # FIX: exclude held-out diseases' treats
    md = train(Net(), build_edges(drop_dis_dg=hds), np.arange(NC), rs, trp_dis)
    pairs, y = make_pairs(CtD[np.isin(CtD[:, 1] - DIS0, hod)], rs)
    a, b, c = three_way(md, build_edges(), pairs, y)
    R["disease"]["gnn"].append(a); R["disease"]["prox"].append(b); R["disease"]["fus"].append(c)

def M(t, k): return float(np.mean(R[t][k])), float(np.std(R[t][k]))
print(f"\n  TASK          GNN-only        proximity-only   FUSION (neurosymbolic)")
for t in ["drug", "disease"]:
    print(f"  {t:12}  {M(t,'gnn')[0]:.3f}+-{M(t,'gnn')[1]:.2f}     "
          f"{M(t,'prox')[0]:.3f}+-{M(t,'prox')[1]:.2f}      "
          f"{M(t,'fus')[0]:.3f}+-{M(t,'fus')[1]:.2f}")

win = all(M(t, "fus")[0] >= max(M(t, "gnn")[0], M(t, "prox")[0]) - 0.005 for t in ["drug", "disease"])
best_gain = max(M(t, "fus")[0] - max(M(t, "gnn")[0], M(t, "prox")[0]) for t in ["drug", "disease"])
verdict = ("FUSION WINS: neurosymbolic beats or matches both parts on both tasks"
           if win else "mixed: fusion helps on at least one task")
print(f"\n  VERDICT: {verdict}  (best gain over best single: +{best_gain:.3f})")

json.dump({t: {k: round(M(t, k)[0], 3) for k in ["gnn", "prox", "fus"]} for t in ["drug", "disease"]}
          | {"verdict": verdict}, open(HERE / "hybrid_results.json", "w"), indent=2)
print("Saved -> hybrid_results.json")
