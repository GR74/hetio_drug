"""
build_subgraph.py  --  Extract a bounded, connected subgraph from REAL Hetionet.

Hetionet (het.io) is a public biomedical knowledge graph: 47k nodes, 2.25M edges.
We carve out a Compound / Gene / Disease neighbourhood small enough to featurize
with real chemistry + protein models, yet rich enough for drug-repurposing.

Relations kept (Hetionet metaedge abbreviations):
    CtD  Compound -treats-      Disease   <-- prediction target (indication)
    CbG  Compound -binds-       Gene      <-- drug -> protein target
    DaG  Disease  -associates-  Gene      <-- disease -> protein
    GiG  Gene     -interacts-   Gene      <-- protein-protein interaction
    DrD  Disease  -resembles-   Disease   <-- fuels zero-shot transfer

Output: subgraph.json  (node lists with DrugBank/Entrez IDs + typed edge lists)
"""
import gzip, json, collections, pathlib

HERE = pathlib.Path(__file__).parent
EDGES = HERE / "hetionet-v1.0-edges.sif.gz"
NODES = HERE / "hetionet-v1.0-nodes.tsv"

KEEP = {"CtD", "CbG", "DaG", "GiG", "DrD"}
# Budget for the subgraph (keeps featurization + CPU training minutes-scale).
MAX_COMPOUNDS = 150     # compounds with the most treatments (repurposable)
MAX_GENES     = 500     # highest-degree genes touching our compounds/diseases

# --------------------------------------------------------------------------- #
# 1. Load node kinds + human-readable names                                   #
# --------------------------------------------------------------------------- #
name, kind = {}, {}
with open(NODES, encoding="utf-8") as f:
    next(f)
    for line in f:
        nid, nm, kd = line.rstrip("\n").split("\t")
        name[nid] = nm
        kind[nid] = kd

# --------------------------------------------------------------------------- #
# 2. Stream edges once, bucket the relations we care about                    #
# --------------------------------------------------------------------------- #
buckets = collections.defaultdict(list)
with gzip.open(EDGES, "rt") as f:
    next(f)
    for line in f:
        s, m, t = line.rstrip("\n").split("\t")
        if m in KEEP:
            buckets[m].append((s, t))

# --------------------------------------------------------------------------- #
# 3. Choose compounds (most treatments) -> diseases they treat -> genes        #
# --------------------------------------------------------------------------- #
treat_deg = collections.Counter(s for s, _ in buckets["CtD"])
compounds = {c for c, _ in treat_deg.most_common(MAX_COMPOUNDS)}

diseases = {t for s, t in buckets["CtD"] if s in compounds}
# add resemblance neighbours so DrD has something to transfer along
for a, b in buckets["DrD"]:
    if a in diseases or b in diseases:
        diseases.add(a); diseases.add(b)

# candidate genes: bound by our compounds OR associated with our diseases
gene_deg = collections.Counter()
for s, t in buckets["CbG"]:
    if s in compounds: gene_deg[t] += 1
for s, t in buckets["DaG"]:
    if s in diseases: gene_deg[t] += 1
genes = {g for g, _ in gene_deg.most_common(MAX_GENES)}

def keep_edge(s, t, ss, ts):
    return s in ss and t in ts

E = {
    "CtD": [(s, t) for s, t in buckets["CtD"] if keep_edge(s, t, compounds, diseases)],
    "CbG": [(s, t) for s, t in buckets["CbG"] if keep_edge(s, t, compounds, genes)],
    "DaG": [(s, t) for s, t in buckets["DaG"] if keep_edge(s, t, diseases, genes)],
    "GiG": [(s, t) for s, t in buckets["GiG"] if keep_edge(s, t, genes, genes)],
    "DrD": [(s, t) for s, t in buckets["DrD"] if keep_edge(s, t, diseases, diseases)],
}

# Keep only nodes that actually survived in an edge, so the graph is connected.
used = set()
for pairs in E.values():
    for s, t in pairs:
        used.add(s); used.add(t)
compounds &= used; diseases &= used; genes &= used
E = {r: [(s, t) for s, t in ps if s in used and t in used] for r, ps in E.items()}

# --------------------------------------------------------------------------- #
# 4. Serialise (strip the "Type::" prefix to raw DrugBank / Entrez / DOID ids) #
# --------------------------------------------------------------------------- #
def raw(nid): return nid.split("::", 1)[1]

out = {
    "compounds": [{"node": c, "drugbank": raw(c), "name": name[c]} for c in sorted(compounds)],
    "genes":     [{"node": g, "entrez":   raw(g), "name": name[g]} for g in sorted(genes)],
    "diseases":  [{"node": d, "doid":     raw(d), "name": name[d]} for d in sorted(diseases)],
    "edges": E,
}
(HERE / "subgraph.json").write_text(json.dumps(out), encoding="utf-8")

print("Real Hetionet subgraph extracted:")
print(f"  compounds {len(compounds):>4}   genes {len(genes):>4}   diseases {len(diseases):>4}")
for r, ps in E.items():
    print(f"  {r}: {len(ps)} edges")
print(f"  treats (CtD) positives available for training/eval: {len(E['CtD'])}")
print("  saved -> subgraph.json")
