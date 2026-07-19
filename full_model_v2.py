"""
full_model.py  --  Full-scale inductive TxGNN on Hetionet, with:
  * three INDUCTIVE feature encoders   (drug fingerprint / gene ESM2 / disease text)
  * TWO decoder heads                  (indication = treats, contraindication = palliates)
  * a metric-learning zero-shot head
  * an occlusion-based EXPLAINER        (which protein mechanism carries a prediction)

Experiments:
  [1] Transductive indication link prediction        (KG signal is learnable)
  [2] Contraindication head                          (second relation works)
  [3] INDUCTIVE zero-shot, UNSEEN DRUGS              (fingerprint vs id-embedding)
  [4] INDUCTIVE zero-shot, UNSEEN DISEASES           (text feature vs id-embedding)
  [5] Explainer demo on a held-out drug's top prediction

Run after build_full_subgraph.py + features_full.py.
"""
import json, pathlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = pathlib.Path(__file__).parent
SUB = json.loads((HERE / "full_subgraph.json").read_text())
FEAT = np.load(HERE / "full_features.npz")
PHENO = np.load(HERE / "disease_pheno.npz")["pheno"]      # v2: real disease phenotype
rng = np.random.default_rng(0); torch.manual_seed(0)
DIM = 96                                                   # v2: more capacity

comp_nodes = [c["node"] for c in SUB["compounds"]]
gene_nodes = [g["node"] for g in SUB["genes"]]
dis_nodes  = [d["node"] for d in SUB["diseases"]]
NC, NG, ND = len(comp_nodes), len(gene_nodes), len(dis_nodes)
gid = {n: i for i, n in enumerate(comp_nodes)}
gid.update({n: NC + i for i, n in enumerate(gene_nodes)})
gid.update({n: NC + NG + i for i, n in enumerate(dis_nodes)})
N = NC + NG + ND
DIS0 = NC + NG

comp_fp  = torch.tensor(FEAT["compound_fp"], dtype=torch.float32)
gene_esm = torch.tensor(FEAT["gene_esm"],   dtype=torch.float32)
# v2: disease CONTENT = phenotype (symptom + anatomy) concatenated with name embedding
dis_txt  = torch.tensor(np.concatenate([PHENO, FEAT["disease_txt"]], axis=1),
                        dtype=torch.float32)

TARGETS = ["CtD", "CpD"]                                     # treats, palliates
MP_RELS = ["CbG", "CuG", "CdG", "DaG", "DuG", "DdG", "GiG", "DrD"]
DRUG_GENE = {"CbG", "CuG", "CdG"}
DIS_GENE  = {"DaG", "DuG", "DdG"}

def pairs_global(rel): return np.array([[gid[s], gid[t]] for s, t in SUB["edges"][rel]])
CtD = pairs_global("CtD"); CpD = pairs_global("CpD")

def build_edges(drop_comp_dg=frozenset(), drop_dis_dg=frozenset(), drop_nodes=frozenset()):
    """Message-passing edges (+ inverses). Can hide an unseen node's typed edges."""
    E = {}
    for r in MP_RELS:
        ps = SUB["edges"][r]
        out = []
        for s, t in ps:
            gs, gt = gid[s], gid[t]
            if gs in drop_nodes or gt in drop_nodes: continue
            if r in DRUG_GENE and gs in drop_comp_dg: continue
            if r in DIS_GENE and gs in drop_dis_dg: continue
            out.append((gs, gt))
        if not out:
            E[r] = torch.zeros(2, 0, dtype=torch.long)
            E[r + "_inv"] = torch.zeros(2, 0, dtype=torch.long); continue
        et = torch.tensor(out, dtype=torch.long).t()
        E[r] = et; E[r + "_inv"] = et.flip(0)
    return E
REL_KEYS = [r for r in MP_RELS] + [r + "_inv" for r in MP_RELS]

# disease signature (gene-association profile) for the zero-shot metric head
prof = torch.zeros(ND, NG)
for s, t in SUB["edges"]["DaG"]:
    prof[gid[s] - DIS0, gid[t] - NC] = 1.0


class RGCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.ModuleDict({r: nn.Linear(dim, dim, bias=False) for r in REL_KEYS})

    def forward(self, h, E):
        out = h.clone()
        for r, e in E.items():
            if e.size(1) == 0: continue
            src, dst = e[0], e[1]
            msg = self.w[r](h[src])
            agg = torch.zeros_like(h); agg.index_add_(0, dst, msg)
            deg = torch.zeros(h.size(0)); deg.index_add_(0, dst, torch.ones(dst.numel()))
            out = out + agg / deg.clamp(min=1).unsqueeze(1)
        return F.relu(out)


class FullTxGNN(nn.Module):
    def __init__(self, comp_mode="fp", dis_mode="txt", layers=2, zero_shot=True):
        super().__init__()
        self.comp_mode, self.dis_mode = comp_mode, dis_mode
        self.comp_fp = nn.Sequential(nn.Linear(comp_fp.size(1), DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        self.gene_e  = nn.Sequential(nn.Linear(gene_esm.size(1), DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        self.dis_t   = nn.Sequential(nn.Linear(dis_txt.size(1), DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        self.comp_id = nn.Parameter(torch.empty(NC, DIM)); nn.init.xavier_uniform_(self.comp_id)
        self.dis_id  = nn.Parameter(torch.empty(ND, DIM)); nn.init.xavier_uniform_(self.dis_id)
        self.gnn = nn.ModuleList([RGCNLayer(DIM) for _ in range(layers)])
        self.w = nn.ParameterDict({r: nn.Parameter(torch.ones(DIM)) for r in TARGETS})
        self.zs = zero_shot
        if zero_shot:
            self.gate = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(), nn.Linear(DIM, 1))

    def inputs(self):
        comp = self.comp_fp(comp_fp) if self.comp_mode == "fp" else self.comp_id
        dis  = self.dis_t(dis_txt) if self.dis_mode == "txt" else self.dis_id
        return torch.cat([comp, self.gene_e(gene_esm), dis], 0)

    def encode(self, E, seen_dis=None):
        h = self.inputs()
        for layer in self.gnn: h = layer(h, E)
        if self.zs and seen_dis is not None and len(seen_dis):
            p = F.normalize(prof, dim=1)
            sim = p @ p[seen_dis].t()
            aux = F.softmax(sim, 1) @ h[DIS0:][seen_dis]
            g = torch.sigmoid(self.gate(h[DIS0:]))
            h = h.clone(); h[DIS0:] = g * h[DIS0:] + (1 - g) * aux
        return h

    def score(self, h, c_ids, d_ids, rel="CtD"):
        return (h[c_ids] * self.w[rel] * h[d_ids]).sum(-1)


# ----------------------------- training utils ------------------------------ #
dis_global = np.arange(DIS0, N)
POS = {r: set(map(tuple, pairs_global(r).tolist())) for r in TARGETS}
POS_CODE = {r: np.sort(pairs_global(r)[:, 0].astype(np.int64) * N + pairs_global(r)[:, 1])
            for r in TARGETS}

_dpop = np.zeros(ND)
for _d in CtD[:, 1]: _dpop[_d - DIS0] += 1
_POPP = (_dpop + 1) / (_dpop + 1).sum()                    # v2: popularity-weighted

def neg_sample(pos, comp_pool, dis_pool, rng_s, rel):
    """v2: HARD negatives -- diseases drawn in proportion to how treated they are,
    so the model must learn drug-specific signal rather than 'popular disease'."""
    n = len(pos)
    cs = rng_s.choice(comp_pool, size=n * 2)
    ds = rng_s.choice(ND, size=n * 2, p=_POPP) + DIS0
    code = cs.astype(np.int64) * N + ds
    keep = ~np.isin(code, POS_CODE[rel])
    out = np.stack([cs[keep], ds[keep]], 1)[:n]
    return out

def train(model, E, pos_by_rel, comp_pool, seen_dis=None, epochs=250, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        model.train(); opt.zero_grad(); loss = 0.0
        h = model.encode(E, seen_dis)
        for rel, pos in pos_by_rel.items():
            if len(pos) == 0: continue
            neg = neg_sample(pos, comp_pool, dis_global, rng, rel)
            pc = model.score(h, torch.tensor(pos[:, 0]), torch.tensor(pos[:, 1]), rel)
            nc = model.score(h, torch.tensor(neg[:, 0]), torch.tensor(neg[:, 1]), rel)
            loss = loss + F.binary_cross_entropy_with_logits(
                torch.cat([pc, nc]), torch.cat([torch.ones_like(pc), torch.zeros_like(nc)]))
        loss.backward(); opt.step()
    return model

dis_pop = np.zeros(ND)
for d in CtD[:, 1]: dis_pop[d - DIS0] += 1
POP_P = (dis_pop + 1) / (dis_pop + 1).sum()

def eval_auroc(model, E, test_pos, rng_s, rel="CtD", ratio=10, pop_matched=True):
    n = len(test_pos) * ratio
    cs = rng_s.choice(test_pos[:, 0], size=n * 2)
    if pop_matched:
        ds = rng_s.choice(ND, size=n * 2, p=POP_P) + DIS0
    else:
        ds = rng_s.choice(ND, size=n * 2) + DIS0
    code = cs.astype(np.int64) * N + ds
    neg = np.stack([cs, ds], 1)[~np.isin(code, POS_CODE[rel])][:n]
    pairs = np.vstack([test_pos, neg])
    y = np.r_[np.ones(len(test_pos)), np.zeros(len(neg))]
    with torch.no_grad():
        h = model.encode(E, None)
        s = torch.sigmoid(model.score(h, torch.tensor(pairs[:, 0]),
                                      torch.tensor(pairs[:, 1]), rel)).numpy()
    return roc_auc_score(y, s), average_precision_score(y, s)

def truth_treats(c): return {gid[t] - DIS0 for s, t in SUB["edges"]["CtD"] if gid[s] == c}

def rank_metrics(ranker, holdout, truth_fn, k=10):
    hits, tot, rr = 0, 0, []
    for c in holdout:
        s = np.asarray(ranker(int(c)))
        pos = np.empty(ND, int); pos[np.argsort(-s)] = np.arange(ND)
        for d in truth_fn(int(c)):
            r = pos[d] + 1; rr.append(1 / r); tot += 1; hits += int(r <= k)
    return hits / max(tot, 1), float(np.mean(rr)) if rr else 0.0

def drug_ranker(model, E):
    with torch.no_grad(): h = model.encode(E, None)
    return lambda c: model.score(h, torch.full((ND,), c), torch.tensor(dis_global), "CtD").detach().numpy()


# =========================== EXPERIMENTS =================================== #
print("=" * 72)
print(f"FULL Hetionet:  {NC} compounds  {NG} genes  {ND} diseases  "
      f"{sum(len(v) for v in SUB['edges'].values())} edges")
print(f"Features: fp{tuple(comp_fp.shape)} esm{tuple(gene_esm.shape)} txt{tuple(dis_txt.shape)}")
print("=" * 72)

# ---- [1] transductive indication + [2] contraindication ---- #
E_full = build_edges()
perm = rng.permutation(len(CtD)); nt = int(0.2 * len(CtD))
m = train(FullTxGNN("fp", "txt", zero_shot=False), E_full,
          {"CtD": CtD[perm[nt:]], "CpD": CpD}, np.arange(NC))
a_tr, p_tr = eval_auroc(m, E_full, CtD[perm[:nt]], np.random.default_rng(1), "CtD", pop_matched=False)
permp = rng.permutation(len(CpD)); ntp = int(0.2 * len(CpD))
a_cp, p_cp = eval_auroc(m, E_full, CpD[permp[:ntp]], np.random.default_rng(2), "CpD", pop_matched=False)
print(f"\n[1] Transductive INDICATION (treats)       AUROC={a_tr:.3f}  AUPRC={p_tr:.3f}")
print(f"[2] CONTRAINDICATION head (palliates)      AUROC={a_cp:.3f}  AUPRC={p_cp:.3f}")

# ---- [3] inductive unseen DRUGS: fingerprint vs id ---- #
deg = np.bincount(CtD[:, 0], minlength=NC); elig = np.where(deg >= 2)[0]
SEEDS = [0, 1, 2]
D = {k: [] for k in ["auc_id", "auc_fp", "hits_id", "hits_fp", "mrr_fp",
                     "only_id", "only_fp"]}       # v2: + feature-only regime
last = None
for sd in SEEDS:
    rs = np.random.default_rng(50 + sd)
    ho = rs.choice(elig, size=25, replace=False); hset = set(ho.tolist())
    hglobal = set(int(c) for c in ho)
    trp = CtD[~np.isin(CtD[:, 0], ho)]; tep = CtD[np.isin(CtD[:, 0], ho)]
    pool = np.array([c for c in range(NC) if c not in hset])
    Etr = build_edges(drop_comp_dg=hglobal); Ekn = build_edges()
    torch.manual_seed(sd)
    mfp = train(FullTxGNN("fp", "txt", zero_shot=False), Etr, {"CtD": trp, "CpD": CpD}, pool)
    mid = train(FullTxGNN("id", "txt", zero_shot=False), Etr, {"CtD": trp, "CpD": CpD}, pool)
    D["auc_fp"].append(eval_auroc(mfp, Ekn, tep, rs, "CtD")[0])
    D["auc_id"].append(eval_auroc(mid, Ekn, tep, rs, "CtD")[0])
    # v2: feature-only -- keep the unseen drugs' target edges HIDDEN at inference
    D["only_fp"].append(eval_auroc(mfp, Etr, tep, rs, "CtD")[0])
    D["only_id"].append(eval_auroc(mid, Etr, tep, rs, "CtD")[0])
    hf, mf = rank_metrics(drug_ranker(mfp, Ekn), ho, truth_treats)
    hi, _ = rank_metrics(drug_ranker(mid, Ekn), ho, truth_treats)
    D["hits_fp"].append(hf); D["mrr_fp"].append(mf); D["hits_id"].append(hi)
    last = (mfp, Ekn, ho)
def M(k): return float(np.mean(D[k])), float(np.std(D[k]))
print(f"\n[3] INDUCTIVE zero-shot, UNSEEN DRUGS  (mean+-std, {len(SEEDS)} splits, 25 drugs each)")
print(f"      id-embedding   AUROC={M('auc_id')[0]:.3f}+-{M('auc_id')[1]:.2f}   Hits@10={M('hits_id')[0]:.3f}")
print(f"      fingerprint    AUROC={M('auc_fp')[0]:.3f}+-{M('auc_fp')[1]:.2f}   Hits@10={M('hits_fp')[0]:.3f}   MRR={M('mrr_fp')[0]:.3f}")
print(f"    feature-only (targets hidden):  id={M('only_id')[0]:.3f}   fingerprint={M('only_fp')[0]:.3f}")

# ---- [4] inductive unseen DISEASES: text feature vs id ---- #
ddeg = np.bincount(CtD[:, 1] - DIS0, minlength=ND); delig = np.where(ddeg >= 2)[0]
DD = {k: [] for k in ["auc_id", "auc_txt"]}
for sd in SEEDS:
    rs = np.random.default_rng(70 + sd)
    hod = rs.choice(delig, size=15, replace=False)
    hod_glob = set(int(d + DIS0) for d in hod)
    mask = np.isin(CtD[:, 1] - DIS0, hod)
    trp, tep = CtD[~mask], CtD[mask]
    Etr = build_edges(drop_dis_dg=hod_glob); Ekn = build_edges()
    seen = np.array([d for d in range(ND) if d not in set(hod.tolist())])
    torch.manual_seed(sd)
    mtx = train(FullTxGNN("fp", "txt", zero_shot=False), Etr, {"CtD": trp, "CpD": CpD}, np.arange(NC))
    mdi = train(FullTxGNN("fp", "id", zero_shot=False), Etr, {"CtD": trp, "CpD": CpD}, np.arange(NC))
    DD["auc_txt"].append(eval_auroc(mtx, Ekn, tep, rs, "CtD")[0])
    DD["auc_id"].append(eval_auroc(mdi, Ekn, tep, rs, "CtD")[0])
def MD(k): return float(np.mean(DD[k])), float(np.std(DD[k]))
print(f"\n[4] INDUCTIVE zero-shot, UNSEEN DISEASES  (mean+-std, {len(SEEDS)} splits, 15 diseases each)")
print(f"      id-embedding       AUROC={MD('auc_id')[0]:.3f}+-{MD('auc_id')[1]:.2f}   (no id row -> chance)")
print(f"      phenotype feature  AUROC={MD('auc_txt')[0]:.3f}+-{MD('auc_txt')[1]:.2f}   (symptom+anatomy+name)")

# ---- [5] Explainer: occlusion over shared proteins ---- #
def explain(model, E, c, d, topk=5):
    with torch.no_grad():
        base = torch.sigmoid(model.score(model.encode(E, None),
               torch.tensor([c]), torch.tensor([d]), "CtD")).item()
    c_genes = {gid[t] for r in DRUG_GENE for s, t in SUB["edges"][r] if gid[s] == c}
    d_genes = {gid[s] for r in DIS_GENE for s, t in SUB["edges"][r] if gid[t] == d}
    cand = list(c_genes & d_genes) or list(c_genes)          # shared mechanism, else targets
    imp = []
    for g in cand:
        Eg = build_edges(drop_nodes={g})
        with torch.no_grad():
            s = torch.sigmoid(model.score(model.encode(Eg, None),
                torch.tensor([c]), torch.tensor([d]), "CtD")).item()
        imp.append((base - s, g))
    imp.sort(reverse=True)
    return base, imp[:topk]

mfp, Ekn, ho = last
rk = drug_ranker(mfp, Ekn)
best = max(ho, key=lambda c: len(set(np.argsort(-rk(int(c)))[:10]) & truth_treats(int(c))))
sc = rk(int(best)); top_d = int(np.argsort(-sc)[0])
cname = SUB["compounds"][int(best)]["name"]; dname = SUB["diseases"][top_d]["name"]
gene_name = {gid[g["node"]]: g["name"] for g in SUB["genes"]}
base, top = explain(mfp, Ekn, int(best), top_d + DIS0)
print(f"\n[5] EXPLAINER  --  unseen drug '{cname}'  ->  '{dname}'  (score {base:.2f})")
print(f"    mechanism proteins (occlusion importance, drop in score when removed):")
for drop, g in top:
    print(f"      {gene_name.get(g,'?'):<12} importance {drop:+.3f}   "
          f"path: {cname} -> [{gene_name.get(g,'?')}] -> {dname}")

# ---- figure + json ---- #
try:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax[0].bar([0, 1], [M("auc_id")[0], M("auc_fp")[0]], yerr=[M("auc_id")[1], M("auc_fp")[1]],
              color=["#c9603f", "#3f9a6b"], capsize=3)
    ax[0].axhline(0.5, ls="--", lw=.8, color="k"); ax[0].set_ylim(0, 1)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["id-embed", "fingerprint"])
    ax[0].set_title("Unseen DRUGS (AUROC)"); ax[0].set_ylabel("AUROC")
    ax[1].bar([0, 1], [MD("auc_id")[0], MD("auc_txt")[0]], yerr=[MD("auc_id")[1], MD("auc_txt")[1]],
              color=["#c9603f", "#3f6f9a"], capsize=3)
    ax[1].axhline(0.5, ls="--", lw=.8, color="k"); ax[1].set_ylim(0, 1)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["id-embed", "phenotype"])
    ax[1].set_title("Unseen DISEASES (AUROC)")
    fig.suptitle("v2: phenotype disease features + hard negatives (full Hetionet)",
                 fontsize=10, weight="bold")
    fig.tight_layout(); fig.savefig(HERE / "full_result_v2.png", dpi=140)
    print("\nSaved figure -> full_result_v2.png")
except Exception as e:
    print("plot skipped:", e)

json.dump({
    "graph": {"compounds": NC, "genes": NG, "diseases": ND,
              "edges": sum(len(v) for v in SUB["edges"].values())},
    "transductive": {"indication_auroc": round(a_tr, 3), "contraindication_auroc": round(a_cp, 3)},
    "inductive_drugs": {"id_auroc": round(M("auc_id")[0], 3), "fp_auroc": round(M("auc_fp")[0], 3),
                        "id_hits10": round(M("hits_id")[0], 3), "fp_hits10": round(M("hits_fp")[0], 3),
                        "fp_mrr": round(M("mrr_fp")[0], 3),
                        "feature_only_id": round(M("only_id")[0], 3),
                        "feature_only_fp": round(M("only_fp")[0], 3)},
    "inductive_diseases": {"id_auroc": round(MD("auc_id")[0], 3),
                           "phenotype_auroc": round(MD("auc_txt")[0], 3)},
    "explainer_example": {"drug": cname, "disease": dname, "score": round(base, 3),
                          "top_proteins": [gene_name.get(g, "?") for _, g in top]},
}, open(HERE / "full_results_v2.json", "w"), indent=2)
print("Saved -> full_results_v2.json")
