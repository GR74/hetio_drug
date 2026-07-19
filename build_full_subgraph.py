"""
build_full_subgraph.py  --  FULL Hetionet drug-repurposing graph.

Scales up from the 150/500/130 demo to the whole repurposing core of Hetionet:
ALL compounds, ALL diseases, and the genes in their neighbourhood. Adds richer
drug-gene and disease-gene relations, and BOTH therapeutic relations so the
model can learn indications and contraindications jointly.

Node types : Compound, Gene, Disease
Target relations (decoder heads):
    CtD  Compound -treats-      Disease   (indication)
    CpD  Compound -palliates-   Disease   (contraindication analogue in Hetionet)
Message-passing relations (+ inverses):
    CbG  binds        CuG upregulates   CdG downregulates   (drug  -> gene)
    DaG  associates   DuG upregulates   DdG downregulates   (disease -> gene)
    GiG  interacts                                          (gene  -> gene)
    DrD  resembles                                          (disease -> disease)

Genes are degree-capped so protein featurisation stays tractable on CPU.
Output: full_subgraph.json
"""
import gzip, json, collections, pathlib

HERE = pathlib.Path(__file__).parent
EDGES = HERE / "hetionet-v1.0-edges.sif.gz"
NODES = HERE / "hetionet-v1.0-nodes.tsv"

TARGETS = {"CtD", "CpD"}
MP = {"CbG", "CuG", "CdG", "DaG", "DuG", "DdG", "GiG", "DrD"}
KEEP = TARGETS | MP
MAX_GENES = 3000          # degree-ranked cap (protein ESM featurisation budget)

name, kind = {}, {}
with open(NODES, encoding="utf-8") as f:
    next(f)
    for line in f:
        nid, nm, kd = line.rstrip("\n").split("\t")
        name[nid], kind[nid] = nm, kd

buckets = collections.defaultdict(list)
with gzip.open(EDGES, "rt") as f:
    next(f)
    for line in f:
        s, m, t = line.rstrip("\n").split("\t")
        if m in KEEP:
            buckets[m].append((s, t))

# All compounds / diseases that participate in ANY kept relation.
compounds, diseases = set(), set()
for m, pairs in buckets.items():
    for s, t in pairs:
        if kind.get(s) == "Compound": compounds.add(s)
        if kind.get(t) == "Compound": compounds.add(t)
        if kind.get(s) == "Disease":  diseases.add(s)
        if kind.get(t) == "Disease":  diseases.add(t)

# Candidate genes = genes touching our compounds or diseases; keep highest-degree.
gene_deg = collections.Counter()
for m, pairs in buckets.items():
    for s, t in pairs:
        if kind.get(t) == "Gene" and (s in compounds or s in diseases):
            gene_deg[t] += 1
        if kind.get(s) == "Gene" and (t in compounds or t in diseases):
            gene_deg[s] += 1
genes = {g for g, _ in gene_deg.most_common(MAX_GENES)}

def node_ok(n):
    return n in compounds or n in diseases or n in genes

E = {}
for m, pairs in buckets.items():
    E[m] = [(s, t) for s, t in pairs if node_ok(s) and node_ok(t)]

# prune to nodes that survive in at least one edge
used = set()
for pairs in E.values():
    for s, t in pairs:
        used.add(s); used.add(t)
compounds &= used; diseases &= used; genes &= used
E = {m: [(s, t) for s, t in ps if s in used and t in used] for m, ps in E.items()}

def raw(nid): return nid.split("::", 1)[1]
out = {
    "compounds": [{"node": c, "drugbank": raw(c), "name": name[c]} for c in sorted(compounds)],
    "genes":     [{"node": g, "entrez":   raw(g), "name": name[g]} for g in sorted(genes)],
    "diseases":  [{"node": d, "doid":     raw(d), "name": name[d]} for d in sorted(diseases)],
    "edges": E,
}
(HERE / "full_subgraph.json").write_text(json.dumps(out), encoding="utf-8")

print("FULL Hetionet repurposing graph:")
print(f"  compounds {len(compounds)}   genes {len(genes)}   diseases {len(diseases)}")
tot = 0
for m in sorted(E):
    print(f"  {m}: {len(E[m])}"); tot += len(E[m])
print(f"  total edges {tot}")
print(f"  indications  (CtD): {len(E['CtD'])}   contraindications (CpD): {len(E['CpD'])}")
print("  saved -> full_subgraph.json")
