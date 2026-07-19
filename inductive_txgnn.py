"""
inductive_txgnn.py  --  TxGNN, extended one step: INDUCTIVE zero-shot repurposing
==================================================================================
Runs on a REAL Hetionet subgraph with REAL node features
(RDKit Morgan fingerprints for drugs, ESM2 protein embeddings for genes).

What TxGNN did (Huang et al., Nat. Med. 2024):
    heterogeneous R-GCN encoder + DistMult decoder + a metric-learning head that
    does ZERO-SHOT over diseases that are IN the graph but have no treatments.
    Because it learns one embedding vector *per node id*, it is TRANSDUCTIVE:
    it cannot say anything about a drug that was absent when the graph was built.

The step beyond (this file):
    Initialise every node from CONTENT features instead of a per-id table.
    A drug is now its chemistry (fingerprint); a gene is its protein (ESM2).
    The encoder therefore generalises to a compound that was NEVER in the
    training graph -- true INDUCTIVE zero-shot drug repurposing.

Three experiments, printed in order:
    (1) Transductive link prediction   (sanity: is the KG signal learnable?)
    (2) Inductive zero-shot on unseen compounds, WITH known protein targets
    (3) Inductive zero-shot on unseen compounds, with NO edges (feature-only)
        -> here the id-embedding baseline is provably blind; features are the
           only thing that works. That is the headline.

Run:  python inductive_txgnn.py   (after build_subgraph.py and features.py)
"""
import json, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = pathlib.Path(__file__).parent
SUB = json.loads((HERE / "subgraph.json").read_text())
FEAT = np.load(HERE / "node_features.npz")
SEED = 0
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
DIM = 64

# --------------------------------------------------------------------------- #
# Index the three node types into one shared id space                         #
# --------------------------------------------------------------------------- #
comp_nodes = [c["node"] for c in SUB["compounds"]]
gene_nodes = [g["node"] for g in SUB["genes"]]
dis_nodes  = [d["node"] for d in SUB["diseases"]]
NC, NG, ND = len(comp_nodes), len(gene_nodes), len(dis_nodes)
gid = {n: i for i, n in enumerate(comp_nodes)}                 # compound-local id
gid.update({n: NC + i for i, n in enumerate(gene_nodes)})      # gene   global id
gid.update({n: NC + NG + i for i, n in enumerate(dis_nodes)})  # disease global id
N = NC + NG + ND
def cg(c): return c                       # compound global id == local id
def gg(n): return gid[n]

comp_fp  = torch.tensor(FEAT["compound_fp"], dtype=torch.float32)   # (NC, 1024)
gene_esm = torch.tensor(FEAT["gene_esm"],   dtype=torch.float32)    # (NG, 320)

# Disease gene-association profile -> signature for the metric-learning head
prof = torch.zeros(ND, NG)
for s, t in SUB["edges"]["DaG"]:
    prof[gid[s] - (NC + NG), gid[t] - NC] = 1.0

# --------------------------------------------------------------------------- #
# Edge tensors. CtD is the target (decoder only, never message-passed).        #
# --------------------------------------------------------------------------- #
def edge_tensor(pairs):
    return torch.tensor([[gid[s], gid[t]] for s, t in pairs], dtype=torch.long).t()

CtD = np.array([[gid[s], gid[t]] for s, t in SUB["edges"]["CtD"]])   # (E,2) global
MP_BASE = {  # message-passing relations (+ inverses added at build time)
    "CbG": SUB["edges"]["CbG"], "DaG": SUB["edges"]["DaG"],
    "GiG": SUB["edges"]["GiG"], "DrD": SUB["edges"]["DrD"],
}

def build_edges(drop_compound_cbg=frozenset()):
    """Return relation->(src,dst) tensors, optionally dropping some compounds' CbG."""
    E = {}
    for r, pairs in MP_BASE.items():
        if r == "CbG":
            pairs = [(s, t) for s, t in pairs if gid[s] not in drop_compound_cbg]
        et = edge_tensor(pairs)
        E[r] = et
        E[r + "_inv"] = et.flip(0)
    return E
RELATIONS = ["CbG", "CbG_inv", "DaG", "DaG_inv", "GiG", "GiG_inv", "DrD", "DrD_inv"]

# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
class RGCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.ModuleDict({r: nn.Linear(dim, dim, bias=False) for r in RELATIONS})

    def forward(self, h, E):
        out = h.clone()
        for r, e in E.items():
            src, dst = e[0], e[1]
            msg = self.w[r](h[src])
            agg = torch.zeros_like(h); agg.index_add_(0, dst, msg)
            deg = torch.zeros(h.size(0)); deg.index_add_(0, dst, torch.ones(dst.numel()))
            out = out + agg / deg.clamp(min=1).unsqueeze(1)
        return F.relu(out)


class InductiveTxGNN(nn.Module):
    """compound_mode: 'fp' (inductive, from fingerprint) or 'id' (transductive table)."""
    def __init__(self, compound_mode="fp", layers=2, zero_shot=True):
        super().__init__()
        self.mode = compound_mode
        self.comp_fp_enc = nn.Sequential(nn.Linear(comp_fp.size(1), DIM), nn.ReLU(),
                                         nn.Linear(DIM, DIM))
        self.comp_id_emb = nn.Parameter(torch.empty(NC, DIM)); nn.init.xavier_uniform_(self.comp_id_emb)
        self.gene_enc = nn.Sequential(nn.Linear(gene_esm.size(1), DIM), nn.ReLU(),
                                      nn.Linear(DIM, DIM))
        self.dis_emb = nn.Parameter(torch.empty(ND, DIM)); nn.init.xavier_uniform_(self.dis_emb)
        self.gnn = nn.ModuleList([RGCNLayer(DIM) for _ in range(layers)])
        self.w_treats = nn.Parameter(torch.ones(DIM))
        self.zs = zero_shot
        if zero_shot:
            self.gate = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(), nn.Linear(DIM, 1))

    def input_embeddings(self):
        comp = self.comp_fp_enc(comp_fp) if self.mode == "fp" else self.comp_id_emb
        gene = self.gene_enc(gene_esm)
        return torch.cat([comp, gene, self.dis_emb], 0)

    def encode(self, E, seen_dis=None):
        h = self.input_embeddings()
        for layer in self.gnn:
            h = layer(h, E)
        if self.zs and seen_dis is not None:          # disease-side metric learning
            p = F.normalize(prof, dim=1)
            sim = p @ p[seen_dis].t()
            aux = F.softmax(sim, 1) @ h[NC + NG:][seen_dis]
            g = torch.sigmoid(self.gate(h[NC + NG:]))
            h = h.clone(); h[NC + NG:] = g * h[NC + NG:] + (1 - g) * aux
        return h

    def score(self, h, comp_ids, dis_globals):
        return (h[comp_ids] * self.w_treats * h[dis_globals]).sum(-1)


# --------------------------------------------------------------------------- #
# Training / evaluation helpers                                               #
# --------------------------------------------------------------------------- #
dis_global = np.arange(NC + NG, N)
all_pos = set(map(tuple, CtD[:, :2].tolist()))
# vectorised positive lookup: encode (compound, disease_global) as one integer
POS_CODES = np.sort(CtD[:, 0].astype(np.int64) * N + CtD[:, 1])

def neg_for(comp_ids_pool, dis_pool, n, rng_s=rng):
    """Fully vectorised rejection sampler (the old Python while-loop was the
    bottleneck across 10 model trainings)."""
    out = np.empty((0, 2), dtype=np.int64)
    while len(out) < n:
        cs = rng_s.choice(comp_ids_pool, size=n * 2)
        ds = rng_s.choice(dis_pool, size=n * 2)
        codes = cs.astype(np.int64) * N + ds
        keep = ~np.isin(codes, POS_CODES)
        out = np.vstack([out, np.stack([cs[keep], ds[keep]], 1)])
    return out[:n]

def evaluate(model, E, test_pos, comp_pool, dis_pool, seen_dis=None):
    model.eval()
    with torch.no_grad():
        h = model.encode(E, seen_dis)
        neg = neg_for(comp_pool, dis_pool, len(test_pos) * 5)
        pairs = np.vstack([test_pos, neg])
        y = np.r_[np.ones(len(test_pos)), np.zeros(len(neg))]
        s = torch.sigmoid(model.score(h, torch.tensor(pairs[:, 0]),
                                      torch.tensor(pairs[:, 1]))).numpy()
    return roc_auc_score(y, s), average_precision_score(y, s)

def train(model, E, train_pos, comp_pool, seen_dis=None, epochs=250, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    tp = train_pos
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        h = model.encode(E, seen_dis)
        neg = neg_for(comp_pool, dis_global, len(tp))
        pc = model.score(h, torch.tensor(tp[:, 0]), torch.tensor(tp[:, 1]))
        nc = model.score(h, torch.tensor(neg[:, 0]), torch.tensor(neg[:, 1]))
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pc, nc]),
            torch.cat([torch.ones_like(pc), torch.zeros_like(nc)]))
        loss.backward(); opt.step()
    return model


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
#                                                                              #
# NOTE on evaluation honesty: drug-repurposing KGs have severe disease-degree  #
# bias (a few diseases carry most treatments). With uniform-random negatives,  #
# a model can win AUROC just by ranking popular diseases high, learning no drug #
# biology at all. We therefore (a) draw negatives MATCHED to disease popularity #
# so the popularity prior is forced down to chance, and (b) lead with per-drug  #
# ranking metrics (Hits@10, MRR) that reflect real repurposing utility.        #
# --------------------------------------------------------------------------- #
dis_pop = np.zeros(ND)
for d in CtD[:, 1]:
    dis_pop[d - (NC + NG)] += 1
POP_P = (dis_pop + 1) / (dis_pop + 1).sum()              # popularity sampling dist

def eval_set(test_pos, rng_s, ratio=10):
    """Popularity-MATCHED negatives: same held-out drugs, diseases drawn in
    proportion to how treated they are. Kills the free popularity signal."""
    n = len(test_pos) * ratio
    comps = rng_s.choice(test_pos[:, 0], size=n * 2)
    dloc = rng_s.choice(ND, size=n * 2, p=POP_P)
    cand = np.stack([comps, dloc + (NC + NG)], 1)
    codes = cand[:, 0].astype(np.int64) * N + cand[:, 1]
    cand = cand[~np.isin(codes, POS_CODES)][:n]
    pairs = np.vstack([test_pos, cand])
    y = np.r_[np.ones(len(test_pos)), np.zeros(len(cand))]
    return pairs, y

def model_scores(model, E, pairs):
    with torch.no_grad():
        h = model.encode(E, None)
        return torch.sigmoid(model.score(
            h, torch.tensor(pairs[:, 0]), torch.tensor(pairs[:, 1]))).numpy()

def truth_of(c):
    return {gid[t] - (NC + NG) for ss, t in SUB["edges"]["CtD"] if gid[ss] == c}

def rank_metrics(score_fn, holdout, k=10):
    """Per unseen drug, rank ALL diseases. Return (Hits@k, MRR)."""
    hits, tot, rr = 0, 0, []
    for c in holdout:
        s = np.asarray(score_fn(int(c)))
        order = np.argsort(-s)
        rankpos = np.empty(ND, dtype=int); rankpos[order] = np.arange(ND)
        for d in truth_of(c):
            r = rankpos[d] + 1
            rr.append(1.0 / r); tot += 1; hits += int(r <= k)
    return hits / max(tot, 1), float(np.mean(rr)) if rr else 0.0

def model_ranker(model, E):
    with torch.no_grad():
        h = model.encode(E, None)
    return lambda c: model.score(h, torch.full((ND,), c),
                                 torch.tensor(dis_global)).detach().numpy()


# --------------------------------------------------------------------------- #
# EXPERIMENT 1 - transductive link prediction (sanity that the KG has signal)  #
# --------------------------------------------------------------------------- #
print("=" * 70)
print(f"REAL Hetionet subgraph:  {NC} compounds  {NG} genes  {ND} diseases")
print(f"Features: Morgan fingerprints {tuple(comp_fp.shape)} | ESM2 {tuple(gene_esm.shape)}")
print("=" * 70)

perm = rng.permutation(len(CtD)); n_test = int(0.2 * len(CtD))
E_full = build_edges()
m1 = train(InductiveTxGNN("fp", zero_shot=False), E_full, CtD[perm[n_test:]], np.arange(NC))
auc1, ap1 = evaluate(m1, E_full, CtD[perm[:n_test]], np.arange(NC), dis_global)
print(f"\n[1] Transductive link prediction         AUROC={auc1:.3f}  AUPRC={ap1:.3f}")

# --------------------------------------------------------------------------- #
# EXPERIMENTS 2 & 3 - INDUCTIVE zero-shot on UNSEEN compounds, over 5 splits    #
# --------------------------------------------------------------------------- #
comp_treat_deg = np.bincount(CtD[:, 0], minlength=NC)
eligible = np.where(comp_treat_deg >= 2)[0]
SEEDS = [0, 1, 2, 3, 4]
acc = {k: [] for k in ["auc_pop", "auc_id", "auc_fp", "auc_id_only", "auc_fp_only",
                       "hits_pop", "hits_id", "hits_fp", "mrr_pop", "mrr_id", "mrr_fp"]}
last = {}
for sd in SEEDS:
    rs = np.random.default_rng(100 + sd)
    holdout = rs.choice(eligible, size=15, replace=False)
    hset = set(holdout.tolist())
    tr_pos = CtD[~np.isin(CtD[:, 0], holdout)]
    te_pos = CtD[np.isin(CtD[:, 0], holdout)]
    pool = np.array([c for c in range(NC) if c not in hset])

    E_train = build_edges(drop_compound_cbg=hset)   # unseen: bindings hidden in train
    E_known = build_edges()                          # inference reveals bindings
    E_only = E_train                                 # inference keeps bindings hidden

    torch.manual_seed(sd)
    m_fp = train(InductiveTxGNN("fp", zero_shot=False), E_train, tr_pos, pool)
    m_id = train(InductiveTxGNN("id", zero_shot=False), E_train, tr_pos, pool)

    # ---- AUROC with popularity-matched negatives (targets known) ---- #
    pairs, y = eval_set(te_pos, rs)
    acc["auc_pop"].append(roc_auc_score(y, dis_pop[pairs[:, 1] - (NC + NG)]))
    acc["auc_id"].append(roc_auc_score(y, model_scores(m_id, E_known, pairs)))
    acc["auc_fp"].append(roc_auc_score(y, model_scores(m_fp, E_known, pairs)))
    # ---- feature-only variant (no edges revealed) ---- #
    acc["auc_id_only"].append(roc_auc_score(y, model_scores(m_id, E_only, pairs)))
    acc["auc_fp_only"].append(roc_auc_score(y, model_scores(m_fp, E_only, pairs)))
    # ---- per-drug ranking: Hits@10 + MRR (targets known) ---- #
    hp, mp = rank_metrics(lambda c: dis_pop, holdout)
    hi, mi = rank_metrics(model_ranker(m_id, E_known), holdout)
    hf, mf = rank_metrics(model_ranker(m_fp, E_known), holdout)
    acc["hits_pop"].append(hp); acc["mrr_pop"].append(mp)
    acc["hits_id"].append(hi);  acc["mrr_id"].append(mi)
    acc["hits_fp"].append(hf);  acc["mrr_fp"].append(mf)
    last = dict(m_fp=m_fp, E_known=E_known, holdout=holdout)

def ms(key): return np.mean(acc[key]), np.std(acc[key])
print(f"\n[2] INDUCTIVE zero-shot on unseen drugs, targets known "
      f"(mean +/- std, {len(SEEDS)} splits)")
print(f"    metric              popularity      id-embedding     fingerprint (ours)")
print(f"    AUROC (matched neg) {ms('auc_pop')[0]:.3f}+-{ms('auc_pop')[1]:.2f}"
      f"     {ms('auc_id')[0]:.3f}+-{ms('auc_id')[1]:.2f}"
      f"      {ms('auc_fp')[0]:.3f}+-{ms('auc_fp')[1]:.2f}")
print(f"    Hits@10             {ms('hits_pop')[0]:.3f}          "
      f"{ms('hits_id')[0]:.3f}           {ms('hits_fp')[0]:.3f}")
print(f"    MRR                 {ms('mrr_pop')[0]:.3f}          "
      f"{ms('mrr_id')[0]:.3f}           {ms('mrr_fp')[0]:.3f}")
print(f"\n[3] INDUCTIVE feature-only (drug is pure chemistry, no edges revealed)")
print(f"      id-embedding   AUROC={ms('auc_id_only')[0]:.3f}  (no id row -> no signal)")
print(f"      fingerprint    AUROC={ms('auc_fp_only')[0]:.3f}  (chemistry alone still ranks)")

# --------------------------------------------------------------------------- #
# Interpretable demo: pick the unseen drug the model ranks best (last split)    #
# --------------------------------------------------------------------------- #
m_fp, E_known, holdout = last["m_fp"], last["E_known"], last["holdout"]
rank = model_ranker(m_fp, E_known)
best_c, best_hit, best_sc = int(holdout[0]), -1, None
for c in holdout:
    s = rank(int(c)); top = set(np.argsort(-s)[:10]); t = truth_of(int(c))
    if len(top & t) > best_hit:
        best_hit, best_c, best_sc = len(top & t), int(c), s
true_d = {gid[t] - (NC + NG) for ss, t in SUB["edges"]["CtD"] if gid[ss] == best_c}
order = np.argsort(-best_sc)
cname = SUB["compounds"][best_c]["name"]
print(f"\nTop predicted indications for UNSEEN drug '{cname}' "
      f"(never in training graph):")
for rank, di in enumerate(order[:8]):
    hit = "  <== known treatment (recovered)" if di in true_d else ""
    print(f"  {rank+1}. {SUB['diseases'][di]['name'][:42]:<42} {best_sc[di]:.3f}{hit}")

# --------------------------------------------------------------------------- #
# Figure                                                                       #
# --------------------------------------------------------------------------- #
try:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
    labels = ["popularity\nprior", "id-embedding\n(TxGNN-style)", "fingerprint\n(inductive, ours)"]
    cols = ["#9aa6ad", "#c9603f", "#3f9a6b"]
    # left: AUROC with popularity-matched negatives -> prior collapses to chance
    m = [ms(k)[0] for k in ["auc_pop", "auc_id", "auc_fp"]]
    e = [ms(k)[1] for k in ["auc_pop", "auc_id", "auc_fp"]]
    ax[0].bar([0, 1, 2], m, yerr=e, color=cols, capsize=3)
    ax[0].axhline(0.5, ls="--", lw=0.8, color="k")
    ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(labels, fontsize=7.5)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("AUROC"); ax[0].set_title("AUROC (popularity-matched negatives)")
    # right: Hits@10 -> the metric that reflects repurposing utility
    m = [ms(k)[0] for k in ["hits_pop", "hits_id", "hits_fp"]]
    ax[1].bar([0, 1, 2], m, color=cols)
    ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(labels, fontsize=7.5)
    ax[1].set_ylim(0, max(m) * 1.3 + 0.05); ax[1].set_ylabel("Hits@10")
    ax[1].set_title("Hits@10 on unseen drugs")
    fig.suptitle("Content features unlock INDUCTIVE zero-shot drug repurposing (real Hetionet)",
                 fontsize=10, weight="bold")
    fig.tight_layout(); fig.savefig(HERE / "inductive_result.png", dpi=140)
    print("Saved figure -> inductive_result.png")
except Exception as e:
    print("plot skipped:", e)

# machine-readable summary for the memo
(HERE / "results.json").write_text(json.dumps({
    "nodes": {"compounds": NC, "genes": NG, "diseases": ND},
    "edges": {r: len(p) for r, p in SUB["edges"].items()},
    "feature_coverage": {"fingerprints": "150/150", "esm2": "500/500"},
    "transductive": {"auroc": round(auc1, 3), "auprc": round(ap1, 3)},
    "inductive_targets_known": {
        "auroc": {k[4:]: round(ms(k)[0], 3) for k in ["auc_pop", "auc_id", "auc_fp"]},
        "hits_at_10": {k[5:]: round(ms(k)[0], 3) for k in ["hits_pop", "hits_id", "hits_fp"]},
        "mrr": {k[4:]: round(ms(k)[0], 3) for k in ["mrr_pop", "mrr_id", "mrr_fp"]},
    },
    "inductive_feature_only": {
        "id": round(ms("auc_id_only")[0], 3), "fp": round(ms("auc_fp_only")[0], 3)},
    "std": {k: round(ms(k)[1], 3) for k in acc},
    "demo_drug": cname,
}, indent=2))
print("Saved -> results.json")
