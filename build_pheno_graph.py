"""
build_pheno_graph.py  --  extend the full graph with Symptom + Anatomy NODES.

v2 gave diseases a phenotype FEATURE and it did not help. The v3 hypothesis is
that phenotype should be STRUCTURE, not a feature: adding Symptom and Anatomy as
real nodes lets message passing form disease-to-disease shortcuts (two diseases
that share a symptom become 2 hops apart). 62 percent of our disease pairs share
at least one symptom, versus only 543 disease-resemblance edges, so this is a far
denser bridge than the graph currently has.

Adds node types Symptom, Anatomy and relations:
    DpS  Disease -presents-  Symptom
    DlA  Disease -localizes- Anatomy
Output: pheno_subgraph.json  (superset of full_subgraph.json)
"""
import gzip, json, pathlib

HERE = pathlib.Path(__file__).parent
SUB = json.loads((HERE / "full_subgraph.json").read_text())
dis = set(d["node"] for d in SUB["diseases"])

name = {}
with open(HERE / "hetionet-v1.0-nodes.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        nid, nm, kd = line.rstrip("\n").split("\t")
        name[nid] = nm

sym, ana, dps, dla = set(), set(), [], []
with gzip.open(HERE / "hetionet-v1.0-edges.sif.gz", "rt") as f:
    next(f)
    for line in f:
        s, m, t = line.rstrip("\n").split("\t")
        if s in dis and m == "DpS": sym.add(t); dps.append((s, t))
        if s in dis and m == "DlA": ana.add(t); dla.append((s, t))

def raw(n): return n.split("::", 1)[1]
SUB["symptoms"] = [{"node": s, "name": name[s]} for s in sorted(sym)]
SUB["anatomy"]  = [{"node": a, "name": name[a]} for a in sorted(ana)]
SUB["edges"]["DpS"] = dps
SUB["edges"]["DlA"] = dla
(HERE / "pheno_subgraph.json").write_text(json.dumps(SUB), encoding="utf-8")

print("pheno_subgraph.json written")
print(f"  + Symptom nodes {len(sym)}   + Anatomy nodes {len(ana)}")
print(f"  + DpS edges {len(dps)}   + DlA edges {len(dla)}")
