"""
full_model_v3.py  --  diagnosing WHY unseen-disease prediction fails.

v2 finding: giving diseases a rich phenotype FEATURE did not lift unseen-disease
AUROC above chance, even with their gene edges revealed. v3 tests three concrete
explanations, disease-side only, so it runs fast:

  [D1] Transductive disease ceiling.  Random 80/20 split of treats edges, all
       diseases seen. If this is high, the model CAN do the disease side, so the
       unseen failure is a generalisation or reach problem, not a modelling one.

  [D2] Inductive unseen diseases, NO phenotype bridge  (replicates v2, 3 layers).
  [D3] Inductive unseen diseases, WITH Symptom+Anatomy nodes as a bridge.
       Two diseases sharing a symptom become 2 hops apart, a shortcut that does
       not exist through genes. If D3 >> D2, structure (not features) was the fix.
       D3-id uses a disease id table; D3-content adds the phenotype feature too.

Reach: 3 GNN layers (v2 used 2), so drug info can travel drug->gene->disease->symptom.
Run after build_pheno_graph.py (needs pheno_subgraph.json + full_features.npz).
"""
import json, pathlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
SUB = json.loads((HERE / "pheno_subgraph.json").read_text())
FEAT = np.load(HERE / "full_features.npz")
PHENO = np.load(HERE / "disease_pheno.npz")["pheno"]
rng = np.random.default_rng(0); torch.manual_seed(0)
DIM, LAYERS, EPOCHS = 96, 2, 250          # v3b: 2 layers (test over-smoothing hypothesis)

comp = [c["node"] for c in SUB["compounds"]]
gene = [g["node"] for g in SUB["genes"]]
dis  = [d["node"] for d in SUB["diseases"]]
sym  = [s["node"] for s in SUB["symptoms"]]
ana  = [a["node"] for a in SUB["anatomy"]]
NC, NG, ND, NS, NA = len(comp), len(gene), len(dis), len(sym), len(ana)
off = {}
gid = {}
for i, n in enumerate(comp): gid[n] = i
for i, n in enumerate(gene): gid[n] = NC + i
for i, n in enumerate(dis):  gid[n] = NC + NG + i
for i, n in enumerate(sym):  gid[n] = NC + NG + ND + i
for i, n in enumerate(ana):  gid[n] = NC + NG + ND + NS + i
N = NC + NG + ND + NS + NA
DIS0 = NC + NG

comp_fp  = torch.tensor(FEAT["compound_fp"], dtype=torch.float32)
gene_esm = torch.tensor(FEAT["gene_esm"],   dtype=torch.float32)
dis_content = torch.tensor(np.concatenate([PHENO, FEAT["disease_txt"]], 1), dtype=torch.float32)

MP_RELS = ["CbG", "CuG", "CdG", "DaG", "DuG", "DdG", "GiG", "DrD", "DpS", "DlA"]
DIS_GENE = {"DaG", "DuG", "DdG"}
PHENO_REL = {"DpS", "DlA"}

def pairs_global(rel): return np.array([[gid[s], gid[t]] for s, t in SUB["edges"][rel]])
CtD = pairs_global("CtD")
dis_global = np.arange(DIS0, DIS0 + ND)
POS_CODE = np.sort(CtD[:, 0].astype(np.int64) * N + CtD[:, 1])

def build_edges(include_pheno=True, drop_dis_dg=frozenset(), drop_nodes=frozenset()):
    E = {}
    for r in MP_RELS:
        if r in PHENO_REL and not include_pheno: continue
        out = []
        for s, t in SUB["edges"][r]:
            gs, gt = gid[s], gid[t]
            if gs in drop_nodes or gt in drop_nodes: continue
            if r in DIS_GENE and gs in drop_dis_dg: continue
            out.append((gs, gt))
        et = (torch.tensor(out, dtype=torch.long).t() if out
              else torch.zeros(2, 0, dtype=torch.long))
        E[r] = et; E[r + "_inv"] = et.flip(0)
    return E

_dpop = np.zeros(ND)
for d in CtD[:, 1]: _dpop[d - DIS0] += 1
_POPP = (_dpop + 1) / (_dpop + 1).sum()

class RGCN(nn.Module):
    def __init__(self, dim, rels):
        super().__init__()
        self.w = nn.ModuleDict({r: nn.Linear(dim, dim, bias=False) for r in rels})
    def forward(self, h, E):
        out = h.clone()
        for r, e in E.items():
            if e.size(1) == 0: continue
            msg = self.w[r](h[e[0]])
            agg = torch.zeros_like(h); agg.index_add_(0, e[1], msg)
            deg = torch.zeros(h.size(0)); deg.index_add_(0, e[1], torch.ones(e.size(1)))
            out = out + agg / deg.clamp(min=1).unsqueeze(1)
        return F.relu(out)

class Model(nn.Module):
    def __init__(self, dis_mode="id"):
        super().__init__()
        self.dis_mode = dis_mode
        self.cf = nn.Sequential(nn.Linear(comp_fp.size(1), DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        self.ge = nn.Sequential(nn.Linear(gene_esm.size(1), DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        self.dc = nn.Sequential(nn.Linear(dis_content.size(1), DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        self.dis_id = nn.Parameter(torch.empty(ND, DIM)); nn.init.xavier_uniform_(self.dis_id)
        self.sym_id = nn.Parameter(torch.empty(NS, DIM)); nn.init.xavier_uniform_(self.sym_id)
        self.ana_id = nn.Parameter(torch.empty(NA, DIM)); nn.init.xavier_uniform_(self.ana_id)
        rels = [r for r in MP_RELS] + [r + "_inv" for r in MP_RELS]
        self.gnn = nn.ModuleList([RGCN(DIM, rels) for _ in range(LAYERS)])
        self.w = nn.Parameter(torch.ones(DIM))
    def inputs(self):
        d = self.dc(dis_content) if self.dis_mode == "content" else self.dis_id
        return torch.cat([self.cf(comp_fp), self.ge(gene_esm), d, self.sym_id, self.ana_id], 0)
    def encode(self, E):
        h = self.inputs()
        for l in self.gnn: h = l(h, E)
        return h
    def score(self, h, c, d): return (h[c] * self.w * h[d]).sum(-1)

def neg(pos, comp_pool, rng_s):
    n = len(pos)
    cs = rng_s.choice(comp_pool, size=n * 2)
    ds = rng_s.choice(ND, size=n * 2, p=_POPP) + DIS0
    code = cs.astype(np.int64) * N + ds
    return np.stack([cs, ds], 1)[~np.isin(code, POS_CODE)][:n]

def train(model, E, pos, comp_pool, rng_s):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        h = model.encode(E)
        ng = neg(pos, comp_pool, rng_s)
        pc = model.score(h, torch.tensor(pos[:, 0]), torch.tensor(pos[:, 1]))
        nc = model.score(h, torch.tensor(ng[:, 0]), torch.tensor(ng[:, 1]))
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pc, nc]), torch.cat([torch.ones_like(pc), torch.zeros_like(nc)]))
        loss.backward(); opt.step()
    return model

def evalA(model, E, test_pos, rng_s, ratio=10):
    n = len(test_pos) * ratio
    cs = rng_s.choice(test_pos[:, 0], size=n * 2)
    ds = rng_s.choice(ND, size=n * 2, p=_POPP) + DIS0
    code = cs.astype(np.int64) * N + ds
    ng = np.stack([cs, ds], 1)[~np.isin(code, POS_CODE)][:n]
    pairs = np.vstack([test_pos, ng]); y = np.r_[np.ones(len(test_pos)), np.zeros(len(ng))]
    with torch.no_grad():
        h = model.encode(E)
        s = torch.sigmoid(model.score(h, torch.tensor(pairs[:, 0]), torch.tensor(pairs[:, 1]))).numpy()
    return roc_auc_score(y, s)

print("=" * 70)
print(f"v3 diagnostic  |  nodes: {NC} drug {NG} gene {ND} disease {NS} symptom {NA} anatomy")
print(f"               |  {LAYERS} layers, phenotype nodes as a disease-disease bridge")
print("=" * 70)

# [D1] transductive disease ceiling
Ef = build_edges(include_pheno=True)
perm = rng.permutation(len(CtD)); nt = int(0.2 * len(CtD))
mdl = train(Model("content"), Ef, CtD[perm[nt:]], np.arange(NC), rng)
d1 = evalA(mdl, Ef, CtD[perm[:nt]], np.random.default_rng(9))
print(f"\n[D1] Transductive disease ceiling (seen diseases)   AUROC={d1:.3f}")

# [D2] no bridge  vs  [D3] bridge (id and content), 3 seeds
ddeg = np.bincount(CtD[:, 1] - DIS0, minlength=ND); delig = np.where(ddeg >= 2)[0]
res = {k: [] for k in ["d2", "d3_id", "d3_content"]}
for sd in [0, 1, 2]:
    rs = np.random.default_rng(30 + sd)
    hod = rs.choice(delig, size=15, replace=False)
    hglob = set(int(d + DIS0) for d in hod)
    mask = np.isin(CtD[:, 1] - DIS0, hod)
    trp, tep = CtD[~mask], CtD[mask]
    # D2: NO phenotype bridge; D3: WITH bridge. Held-out disease keeps DaG+pheno at inference.
    E2tr = build_edges(include_pheno=False, drop_dis_dg=hglob); E2kn = build_edges(include_pheno=False)
    E3tr = build_edges(include_pheno=True,  drop_dis_dg=hglob); E3kn = build_edges(include_pheno=True)
    torch.manual_seed(sd)
    m2 = train(Model("id"), E2tr, trp, np.arange(NC), rs)
    res["d2"].append(evalA(m2, E2kn, tep, rs))
    m3 = train(Model("id"), E3tr, trp, np.arange(NC), rs)
    res["d3_id"].append(evalA(m3, E3kn, tep, rs))
    m3c = train(Model("content"), E3tr, trp, np.arange(NC), rs)
    res["d3_content"].append(evalA(m3c, E3kn, tep, rs))
def MS(k): return float(np.mean(res[k])), float(np.std(res[k]))
print(f"\nINDUCTIVE unseen diseases (mean+-std, 3 splits, 15 diseases):")
print(f"  [D2] no bridge, id            AUROC={MS('d2')[0]:.3f}+-{MS('d2')[1]:.2f}   (v2 replication)")
print(f"  [D3] + symptom/anatomy bridge AUROC={MS('d3_id')[0]:.3f}+-{MS('d3_id')[1]:.2f}   (id embedding)")
print(f"  [D3] + bridge + phenotype ft  AUROC={MS('d3_content')[0]:.3f}+-{MS('d3_content')[1]:.2f}")

verdict = ("bridge helps" if MS('d3_id')[0] - MS('d2')[0] > 0.03 else
           "bridge does not help; likely sparsity ceiling" if d1 > 0.7 else
           "disease side weak even transductively (modelling ceiling)")
print(f"\nDIAGNOSIS: ceiling(D1)={d1:.3f}, D2={MS('d2')[0]:.3f}, D3={MS('d3_id')[0]:.3f}  ->  {verdict}")

json.dump({"ceiling_d1": round(d1, 3),
           "d2_no_bridge": round(MS('d2')[0], 3),
           "d3_bridge_id": round(MS('d3_id')[0], 3),
           "d3_bridge_content": round(MS('d3_content')[0], 3),
           "verdict": verdict},
          open(HERE / "v3b_results.json", "w"), indent=2)
print("Saved -> v3b_results.json")
