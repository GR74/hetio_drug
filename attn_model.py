"""
attn_model.py  --  ARCHITECTURE EXTENSION: relation-aware attention encoder.

So far the encoder was a mean-aggregation R-GCN: every neighbour of a node is
averaged with equal weight. That is exactly what let high-degree hub nodes
(popular genes, common symptoms) wash out the signal and collapse the disease
side (see MEMO Section 10).

This adds a new capability: a relation-aware multi-head ATTENTION layer
(GAT / HGT style). For each relation and each target node, neighbours are
weighted by a learned dot-product attention score instead of averaged, so the
model can down-weight uninformative hubs and focus on the neighbours that matter.

We run it head to head against the mean-agg R-GCN on the drug-side inductive task
(the one that works) and the disease-side ceiling (the one that collapsed), using
the already-cached full-Hetionet features, so this is training-only and fast.

Run after build_full_subgraph.py + features_full.py.
"""
import json, pathlib, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
SUB = json.loads((HERE / "full_subgraph.json").read_text())
FEAT = np.load(HERE / "full_features.npz")
rng = np.random.default_rng(0); torch.manual_seed(0)
DIM, LAYERS, EPOCHS, HEADS = 96, 2, 250, 4

comp = [c["node"] for c in SUB["compounds"]]
gene = [g["node"] for g in SUB["genes"]]
dis  = [d["node"] for d in SUB["diseases"]]
NC, NG, ND = len(comp), len(gene), len(dis)
gid = {}
for i, n in enumerate(comp): gid[n] = i
for i, n in enumerate(gene): gid[n] = NC + i
for i, n in enumerate(dis):  gid[n] = NC + NG + i
N = NC + NG + ND; DIS0 = NC + NG

comp_fp  = torch.tensor(FEAT["compound_fp"], dtype=torch.float32)
gene_esm = torch.tensor(FEAT["gene_esm"],   dtype=torch.float32)

MP_RELS = ["CbG", "CuG", "CdG", "DaG", "DuG", "DdG", "GiG", "DrD"]
DRUG_GENE = {"CbG", "CuG", "CdG"}
def pg(rel): return np.array([[gid[s], gid[t]] for s, t in SUB["edges"][rel]])
CtD = pg("CtD"); CpD = pg("CpD")
dis_global = np.arange(DIS0, N)
POS_CODE = np.sort(CtD[:, 0].astype(np.int64) * N + CtD[:, 1])
REL_KEYS = MP_RELS + [r + "_inv" for r in MP_RELS]

def build_edges(drop_comp_dg=frozenset()):
    E = {}
    for r in MP_RELS:
        out = [(gid[s], gid[t]) for s, t in SUB["edges"][r]
               if not (r in DRUG_GENE and gid[s] in drop_comp_dg)]
        et = torch.tensor(out, dtype=torch.long).t() if out else torch.zeros(2, 0, dtype=torch.long)
        E[r] = et; E[r + "_inv"] = et.flip(0)
    return E

_dpop = np.zeros(ND)
for d in CtD[:, 1]: _dpop[d - DIS0] += 1
_POPP = (_dpop + 1) / (_dpop + 1).sum()


# --------------------------------------------------------------------------- #
# Two encoders: mean-agg R-GCN (baseline) and relation-aware attention (new)   #
# --------------------------------------------------------------------------- #
class RGCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.ModuleDict({r: nn.Linear(dim, dim, bias=False) for r in REL_KEYS})
    def forward(self, h, E):
        out = h.clone()
        for r, e in E.items():
            if e.size(1) == 0: continue
            msg = self.w[r](h[e[0]])
            agg = torch.zeros_like(h); agg.index_add_(0, e[1], msg)
            deg = torch.zeros(h.size(0)); deg.index_add_(0, e[1], torch.ones(e.size(1)))
            out = out + agg / deg.clamp(min=1).unsqueeze(1)
        return F.relu(out)

class GATLayer(nn.Module):
    """Relation-aware multi-head dot-product attention (down-weights hubs)."""
    def __init__(self, dim, heads=HEADS):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.w = nn.ModuleDict({r: nn.Linear(dim, dim, bias=False) for r in REL_KEYS})
    def forward(self, h, E):
        out = h.clone()
        for r, e in E.items():
            if e.size(1) == 0: continue
            src, dst = e[0], e[1]
            hw = self.w[r](h).view(-1, self.h, self.dh)          # (N, H, dh)
            score = (hw[src] * hw[dst]).sum(-1) / math.sqrt(self.dh)   # (E, H)
            score = score.clamp(-10, 10).exp()                   # segment softmax
            denom = torch.zeros(h.size(0), self.h); denom.index_add_(0, dst, score)
            alpha = score / denom[dst].clamp(min=1e-9)           # (E, H)
            msg = hw[src] * alpha.unsqueeze(-1)                  # (E, H, dh)
            agg = torch.zeros(h.size(0), self.h, self.dh); agg.index_add_(0, dst, msg)
            out = out + agg.reshape(h.size(0), -1)
        return F.relu(out)


class Model(nn.Module):
    def __init__(self, encoder="rgcn", comp_mode="fp"):
        super().__init__()
        self.comp_mode = comp_mode
        self.cf = nn.Sequential(nn.Linear(comp_fp.size(1), DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        self.ge = nn.Sequential(nn.Linear(gene_esm.size(1), DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        self.comp_id = nn.Parameter(torch.empty(NC, DIM)); nn.init.xavier_uniform_(self.comp_id)
        self.dis = nn.Parameter(torch.empty(ND, DIM)); nn.init.xavier_uniform_(self.dis)
        Layer = RGCNLayer if encoder == "rgcn" else GATLayer
        self.gnn = nn.ModuleList([Layer(DIM) for _ in range(LAYERS)])
        self.w = nn.ParameterDict({r: nn.Parameter(torch.ones(DIM)) for r in ["CtD", "CpD"]})
    def inputs(self):
        c = self.cf(comp_fp) if self.comp_mode == "fp" else self.comp_id
        return torch.cat([c, self.ge(gene_esm), self.dis], 0)
    def encode(self, E):
        h = self.inputs()
        for l in self.gnn: h = l(h, E)
        return h
    def score(self, h, c, d, rel="CtD"): return (h[c] * self.w[rel] * h[d]).sum(-1)


def neg(pos, pool, rng_s):
    n = len(pos)
    cs = rng_s.choice(pool, size=n * 2); ds = rng_s.choice(ND, size=n * 2, p=_POPP) + DIS0
    code = cs.astype(np.int64) * N + ds
    return np.stack([cs, ds], 1)[~np.isin(code, POS_CODE)][:n]

def train(model, E, pool, rng_s):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad(); h = model.encode(E)
        loss = 0.0
        for rel, pos in [("CtD", CtD[np.isin(CtD[:, 0], pool)]), ("CpD", CpD)]:
            ng = neg(pos, pool, rng_s)
            pc = model.score(h, torch.tensor(pos[:, 0]), torch.tensor(pos[:, 1]), rel)
            nc = model.score(h, torch.tensor(ng[:, 0]), torch.tensor(ng[:, 1]), rel)
            loss = loss + F.binary_cross_entropy_with_logits(
                torch.cat([pc, nc]), torch.cat([torch.ones_like(pc), torch.zeros_like(nc)]))
        loss.backward(); opt.step()
    return model

def evalA(model, E, test_pos, rng_s, ratio=10):
    n = len(test_pos) * ratio
    cs = rng_s.choice(test_pos[:, 0], size=n * 2); ds = rng_s.choice(ND, size=n * 2, p=_POPP) + DIS0
    code = cs.astype(np.int64) * N + ds
    ng = np.stack([cs, ds], 1)[~np.isin(code, POS_CODE)][:n]
    pairs = np.vstack([test_pos, ng]); y = np.r_[np.ones(len(test_pos)), np.zeros(len(ng))]
    with torch.no_grad():
        h = model.encode(E)
        s = torch.sigmoid(model.score(h, torch.tensor(pairs[:, 0]), torch.tensor(pairs[:, 1]))).numpy()
    return roc_auc_score(y, s)

def truth(c): return {gid[t] - DIS0 for s, t in SUB["edges"]["CtD"] if gid[s] == c}
def hits10(model, E, hold):
    with torch.no_grad(): h = model.encode(E)
    hit = tot = 0
    for c in hold:
        s = model.score(h, torch.full((ND,), int(c)), torch.tensor(dis_global)).detach().numpy()
        top = set(np.argsort(-s)[:10])
        for d in truth(int(c)): hit += int(d in top); tot += 1
    return hit / max(tot, 1)


print("=" * 68)
print(f"ARCHITECTURE COMPARISON: mean-agg R-GCN  vs  relation-aware ATTENTION")
print(f"  {NC} drugs {NG} genes {ND} diseases | {LAYERS} layers, {HEADS} heads")
print("=" * 68)

deg = np.bincount(CtD[:, 0], minlength=NC); elig = np.where(deg >= 2)[0]
out = {enc: {"auc": [], "hits": [], "only": []} for enc in ["rgcn", "gat"]}
trans = {}
# transductive ceiling per encoder (does attention beat mean-agg with all seen?)
for enc in ["rgcn", "gat"]:
    Ef = build_edges(); perm = rng.permutation(len(CtD)); nt = int(0.2 * len(CtD))
    m = train(Model(enc), Ef, np.arange(NC), rng)
    trans[enc] = evalA(m, Ef, CtD[perm[:nt]], np.random.default_rng(7))
# inductive unseen drugs, 3 seeds
for sd in [0, 1, 2]:
    rs = np.random.default_rng(40 + sd)
    ho = rs.choice(elig, size=25, replace=False); hs = set(int(c) for c in ho)
    tep = CtD[np.isin(CtD[:, 0], ho)]; pool = np.array([c for c in range(NC) if c not in hs])
    Etr = build_edges(drop_comp_dg=hs); Ekn = build_edges()
    for enc in ["rgcn", "gat"]:
        torch.manual_seed(sd)
        m = train(Model(enc), Etr, pool, rs)
        out[enc]["auc"].append(evalA(m, Ekn, tep, rs))
        out[enc]["only"].append(evalA(m, Etr, tep, rs))     # feature-only (targets hidden)
        out[enc]["hits"].append(hits10(m, Ekn, ho))

def mm(enc, k): return float(np.mean(out[enc][k]))
print(f"\n                       R-GCN (mean-agg)   ATTENTION (new)")
print(f"  transductive ceiling   {trans['rgcn']:.3f}             {trans['gat']:.3f}")
print(f"  inductive AUROC        {mm('rgcn','auc'):.3f}             {mm('gat','auc'):.3f}")
print(f"  inductive Hits@10      {mm('rgcn','hits'):.3f}             {mm('gat','hits'):.3f}")
print(f"  feature-only AUROC     {mm('rgcn','only'):.3f}             {mm('gat','only'):.3f}")

better = "attention helps" if (mm('gat','auc') + mm('gat','hits')) > (mm('rgcn','auc') + mm('rgcn','hits')) + 0.02 else "attention comparable / no clear gain"
print(f"\n  VERDICT: {better}")

json.dump({"encoders": ["rgcn", "gat"],
           "transductive": {"rgcn": round(trans['rgcn'], 3), "gat": round(trans['gat'], 3)},
           "inductive_auroc": {"rgcn": round(mm('rgcn','auc'), 3), "gat": round(mm('gat','auc'), 3)},
           "inductive_hits10": {"rgcn": round(mm('rgcn','hits'), 3), "gat": round(mm('gat','hits'), 3)},
           "feature_only": {"rgcn": round(mm('rgcn','only'), 3), "gat": round(mm('gat','only'), 3)},
           "verdict": better},
          open(HERE / "attn_results.json", "w"), indent=2)
print("Saved -> attn_results.json")
