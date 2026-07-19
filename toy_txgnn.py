"""
toy_txgnn.py  --  A minimal, readable re-implementation of the TxGNN idea
=========================================================================
Paper: Huang, Chandak, ... Zitnik. "A foundation model for clinician-centered
       drug repurposing." Nature Medicine 30, 3601-3613 (2024).

TxGNN in one breath:
  (1) a heterogeneous R-GCN encoder does relation-specific message passing over
      a medical knowledge graph,
  (2) a DistMult decoder scores drug<->disease pairs for a relation
      (indication / "treats"),
  (3) a *zero-shot* metric-learning head enriches the embedding of a disease
      that has NO known treatments by borrowing from *similar* diseases
      (disease-signature -> aggregate -> gate).

This toy reproduces that structure at tiny scale on a Hetionet-style metagraph,
so the IAIRO GNN module can see the moving parts end-to-end and extend it.

The synthetic KG has a *real* latent mechanism, so the signal is learnable:
  - genes are grouped into latent "mechanism modules",
  - a disease is associated with genes from 1-2 modules  (Disease-associates-Gene),
  - a compound binds genes from ONE module              (Compound-binds-Gene),
  - a compound TREATS a disease iff they share a module  (Compound-treats-Disease),
  - diseases that share modules "resemble" each other    (Disease-resembles-Disease).
So `treats(c,d)` is recoverable from the path  c -binds- g -associates- d,
which is exactly the biological premise TxGNN exploits.

We hold out a set of diseases as ZERO-SHOT: all their `treats` edges are removed
from training. They keep their DaG/DrD/GiG edges (so they still get an embedding),
and we ask the model to recover their treatments at test time -- the TxGNN setting.

Run:  python toy_txgnn.py
Deps: torch, numpy, scikit-learn, matplotlib  (all already in your env)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# 1. Build a tiny Hetionet-style heterogeneous knowledge graph                #
# --------------------------------------------------------------------------- #
N_DISEASE, N_COMPOUND, N_GENE = 40, 60, 90
N_MODULE = 6                      # latent mechanism modules (the hidden signal)
GENES_PER_MODULE = N_GENE // N_MODULE
N_ZEROSHOT = 10                   # diseases whose treatments are fully hidden

# Assign each gene to a latent module.
gene_module = np.repeat(np.arange(N_MODULE), GENES_PER_MODULE)
gene_module = np.concatenate([gene_module,
                              rng.integers(0, N_MODULE, N_GENE - len(gene_module))])

# Each disease belongs to 1-2 modules; each compound to exactly 1.
disease_modules = [rng.choice(N_MODULE, size=rng.integers(1, 3), replace=False)
                   for _ in range(N_DISEASE)]
compound_module = rng.integers(0, N_MODULE, N_COMPOUND)

# ---- Global node indexing (one shared id space for the R-GCN) -------------- #
# disease ids: 0..ND-1 ; compound ids: ND..ND+NC-1 ; gene ids: rest
ND, NC, NG = N_DISEASE, N_COMPOUND, N_GENE
N_NODES = ND + NC + NG
def d_id(i): return i
def c_id(i): return ND + i
def g_id(i): return ND + NC + i

node_type = np.array(["disease"] * ND + ["compound"] * NC + ["gene"] * NG)

# ---- Edge builders --------------------------------------------------------- #
def genes_in_modules(mods, k):
    """Sample k genes drawn from the given modules."""
    pool = np.where(np.isin(gene_module, mods))[0]
    return rng.choice(pool, size=min(k, len(pool)), replace=False)

DaG = []   # Disease -associates- Gene
for d, mods in enumerate(disease_modules):
    for g in genes_in_modules(mods, k=6):
        DaG.append((d_id(d), g_id(g)))

CbG = []   # Compound -binds- Gene
for c in range(NC):
    for g in genes_in_modules([compound_module[c]], k=4):
        CbG.append((c_id(c), g_id(g)))

GiG = []   # Gene -interacts- Gene   (within-module wiring)
for m in range(N_MODULE):
    gs = np.where(gene_module == m)[0]
    for _ in range(len(gs)):
        a, b = rng.choice(gs, size=2, replace=False)
        GiG.append((g_id(a), g_id(b)))

# Compound -treats- Disease  (ground truth = shared module, with a little noise)
CtD_all = []
for c in range(NC):
    for d in range(ND):
        shares = compound_module[c] in disease_modules[d]
        if shares and rng.random() > 0.15:          # 15% false negatives
            CtD_all.append((c, d))
        elif (not shares) and rng.random() < 0.02:  # 2% false positives
            CtD_all.append((c, d))
CtD_all = np.array(CtD_all)

# Disease -resembles- Disease  (share >=1 module) -- fuels zero-shot transfer
DrD = []
for i in range(ND):
    for j in range(i + 1, ND):
        if set(disease_modules[i]) & set(disease_modules[j]):
            DrD.append((d_id(i), d_id(j)))

# --------------------------------------------------------------------------- #
# 2. Train / zero-shot split                                                  #
# --------------------------------------------------------------------------- #
zeroshot_diseases = set(rng.choice(ND, size=N_ZEROSHOT, replace=False).tolist())
train_mask = np.array([d not in zeroshot_diseases for (_, d) in CtD_all])
CtD_train = CtD_all[train_mask]      # supervision: treats-edges of "seen" diseases
CtD_test  = CtD_all[~train_mask]     # held-out: treats-edges of zero-shot diseases
seen_diseases = sorted(set(range(ND)) - zeroshot_diseases)

print(f"KG: {N_NODES} nodes  ({ND} disease / {NC} compound / {NG} gene)")
print(f"    edges  DaG={len(DaG)}  CbG={len(CbG)}  GiG={len(GiG)}  DrD={len(DrD)}")
print(f"    treats: {len(CtD_all)} total  ->  {len(CtD_train)} train / "
      f"{len(CtD_test)} zero-shot (across {N_ZEROSHOT} untreated diseases)\n")


def to_edge(pairs):
    e = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return e.to(DEVICE)

# Message-passing relations (each with an explicit inverse). NOTE: `treats` is
# deliberately NOT a message-passing relation -- the encoder learns purely from
# binding / association / interaction structure, and the decoder predicts treats.
# That keeps zero-shot honest: untreated diseases have no treats edges anyway.
EDGES = {
    "DaG": to_edge(DaG),            "DaG_inv": to_edge([(b, a) for a, b in DaG]),
    "CbG": to_edge(CbG),            "CbG_inv": to_edge([(b, a) for a, b in CbG]),
    "GiG": to_edge(GiG),            "GiG_inv": to_edge([(b, a) for a, b in GiG]),
    "DrD": to_edge(DrD),            "DrD_inv": to_edge([(b, a) for a, b in DrD]),
}
RELATIONS = list(EDGES.keys())

# Disease "signature" = its binary gene-association profile. Available even for
# zero-shot diseases (they keep DaG edges). Drives metric-learning similarity.
gene_profile = torch.zeros(ND, NG)
for d, g in DaG:
    gene_profile[d, g - (ND + NC)] = 1.0
gene_profile = gene_profile.to(DEVICE)


# --------------------------------------------------------------------------- #
# 3. Model:  R-GCN encoder  ->  DistMult decoder  ->  zero-shot head          #
# --------------------------------------------------------------------------- #
class RGCNLayer(nn.Module):
    """One relation-specific message-passing layer (TxGNN Methods, steps 2-4)."""
    def __init__(self, dim):
        super().__init__()
        self.rel_w = nn.ModuleDict(
            {r: nn.Linear(dim, dim, bias=False) for r in RELATIONS})

    def forward(self, h):
        out = h.clone()                                    # residual  h_i^{l-1}
        for r, (src, dst) in EDGES.items():
            msg = self.rel_w[r](h[src])                    # step 2: W_r h_j
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)                    # step 3: sum over N_r(i)
            deg = torch.zeros(h.size(0), device=h.device)
            deg.index_add_(0, dst, torch.ones(dst.size(0), device=h.device))
            agg = agg / deg.clamp(min=1).unsqueeze(1)      #         mean-aggregate
            out = out + agg                                # step 4: h_i += sum_r
        return F.relu(out)


class ZeroShotHead(nn.Module):
    """Signature -> aggregate similar diseases -> gate  (TxGNN metric-learning)."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                  nn.Linear(dim, 1))

    def forward(self, h, seen_ids):
        # cosine similarity between every disease signature and the SEEN diseases
        prof = F.normalize(gene_profile, dim=1)
        sim = prof @ prof[seen_ids].t()                    # (ND, |seen|)
        w = F.softmax(sim, dim=1)
        aux = w @ h[seen_ids]                              # borrowed embedding
        g = torch.sigmoid(self.gate(h[:ND]))               # per-disease gate in [0,1]
        h_dis = g * h[:ND] + (1 - g) * aux                 # modulate original<->aux
        h = h.clone()
        h[:ND] = h_dis
        return h


class ToyTxGNN(nn.Module):
    def __init__(self, dim=64, layers=2, zero_shot=True):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(N_NODES, dim))
        nn.init.xavier_uniform_(self.emb)                  # step 1: init X_i
        self.gnn = nn.ModuleList([RGCNLayer(dim) for _ in range(layers)])
        self.w_treats = nn.Parameter(torch.ones(dim))      # DistMult relation vec
        self.zs = ZeroShotHead(dim) if zero_shot else None

    def encode(self, seen_ids):
        h = self.emb
        for layer in self.gnn:
            h = layer(h)
        if self.zs is not None:
            h = self.zs(h, seen_ids)
        return h

    def score(self, h, compounds, diseases):
        """DistMult:  sigma( <h_c , w , h_d> )  for the treats relation."""
        hc, hd = h[[c_id(c) for c in compounds]], h[diseases]
        return (hc * self.w_treats * hd).sum(-1)           # logits


# --------------------------------------------------------------------------- #
# 4. Train + evaluate one model                                               #
# --------------------------------------------------------------------------- #
def sample_negatives(pos_pairs, n):
    pos = set(map(tuple, pos_pairs.tolist()))
    negs = []
    while len(negs) < n:
        c, d = rng.integers(0, NC), int(rng.choice(seen_diseases))
        if (c, d) not in pos:
            negs.append((c, d))
    return np.array(negs)

seen_ids_t = torch.tensor(seen_diseases, dtype=torch.long, device=DEVICE)

def build_eval_set():
    """Rank test positives (zero-shot diseases) against sampled negatives."""
    pos = CtD_test
    pos_set = set(map(tuple, CtD_all.tolist()))
    negs = []
    for _ in range(len(pos) * 5):
        c, d = rng.integers(0, NC), int(rng.choice(list(zeroshot_diseases)))
        if (c, d) not in pos_set:
            negs.append((c, d))
    negs = np.array(negs)
    pairs = np.vstack([pos, negs])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(negs))])
    return pairs, labels

eval_pairs, eval_labels = build_eval_set()

def train(zero_shot: bool, epochs=300):
    model = ToyTxGNN(zero_shot=zero_shot).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    pos = CtD_train
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        h = model.encode(seen_ids_t)
        neg = sample_negatives(pos, len(pos))
        pc = model.score(h, pos[:, 0], torch.tensor(pos[:, 1], device=DEVICE))
        nc = model.score(h, neg[:, 0], torch.tensor(neg[:, 1], device=DEVICE))
        logits = torch.cat([pc, nc])
        target = torch.cat([torch.ones_like(pc), torch.zeros_like(nc)])
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward(); opt.step()

    # ---- zero-shot evaluation ---- #
    model.eval()
    with torch.no_grad():
        h = model.encode(seen_ids_t)
        s = model.score(h, eval_pairs[:, 0],
                        torch.tensor(eval_pairs[:, 1], device=DEVICE))
        s = torch.sigmoid(s).cpu().numpy()
    return model, roc_auc_score(eval_labels, s), average_precision_score(eval_labels, s)


print("Training baseline R-GCN (no zero-shot head) ...")
_, auc_base, ap_base = train(zero_shot=False)
print(f"   zero-shot  AUROC={auc_base:.3f}  AUPRC={ap_base:.3f}\n")

print("Training TxGNN-style model (+ metric-learning zero-shot head) ...")
model_zs, auc_zs, ap_zs = train(zero_shot=True)
print(f"   zero-shot  AUROC={auc_zs:.3f}  AUPRC={ap_zs:.3f}\n")

print("=" * 62)
print(f"  Zero-shot lift from the metric-learning head:")
print(f"     AUROC  {auc_base:.3f} -> {auc_zs:.3f}   ({auc_zs-auc_base:+.3f})")
print(f"     AUPRC  {ap_base:.3f} -> {ap_zs:.3f}   ({ap_zs-ap_base:+.3f})")
print("=" * 62)

# --------------------------------------------------------------------------- #
# 5. Interpretable demo: rank repurposing candidates for ONE zero-shot disease #
# --------------------------------------------------------------------------- #
demo_d = sorted(zeroshot_diseases)[0]
with torch.no_grad():
    h = model_zs.encode(seen_ids_t)
    scores = torch.sigmoid(model_zs.score(
        h, list(range(NC)), torch.full((NC,), demo_d, device=DEVICE))).cpu().numpy()
true_treats = {c for (c, d) in CtD_all if d == demo_d}
order = np.argsort(-scores)
print(f"\nRepurposing ranking for zero-shot disease D{demo_d} "
      f"(latent modules {disease_modules[demo_d].tolist()}):")
print("  rank  compound  score   binds-module  actually-treats?")
for rank, c in enumerate(order[:8]):
    hit = "<-- TRUE" if c in true_treats else ""
    print(f"  {rank+1:>4}   C{c:<7} {scores[c]:.3f}      module {compound_module[c]}"
          f"        {'yes' if c in true_treats else 'no ':>3}  {hit}")

# --------------------------------------------------------------------------- #
# 6. Save a small comparison plot                                             #
# --------------------------------------------------------------------------- #
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.arange(2)
    ax.bar(x - 0.2, [auc_base, ap_base], 0.4, label="R-GCN baseline")
    ax.bar(x + 0.2, [auc_zs, ap_zs], 0.4, label="+ zero-shot head (TxGNN-style)")
    ax.set_xticks(x); ax.set_xticklabels(["AUROC", "AUPRC"])
    ax.set_ylim(0, 1); ax.set_title("Zero-shot drug repurposing (toy)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("zero_shot_lift.png", dpi=130)
    print("\nSaved plot -> gnn_toy/zero_shot_lift.png")
except Exception as e:
    print("plot skipped:", e)
