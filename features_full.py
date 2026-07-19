"""
features_full.py  --  REAL content features for the FULL Hetionet graph.

Compounds -> RDKit Morgan fingerprints (1024-bit)         from PubChem SMILES
Genes     -> ESM2 protein embeddings (320-d)              from UniProt sequences
Diseases  -> MiniLM sentence embeddings (384-d)  [NEW]    from disease names

The disease featuriser is the new piece: it gives diseases a CONTENT vector, so
the disease side of the model also becomes inductive (a new disease can be
scored from its description alone, not just its graph position).

All three are cached PER ENTITY, so re-runs and supersets are cheap.
Run:  python features_full.py
"""
import json, time, pathlib, hashlib
import numpy as np
import requests

HERE = pathlib.Path(__file__).parent
SUB = json.loads((HERE / "full_subgraph.json").read_text())
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
FP_BITS = 1024
UA = {"User-Agent": "IAIRO-GNN/1.0 (research)"}

def _load(p): return json.loads(p.read_text()) if p.exists() else {}
def _save(p, d): p.write_text(json.dumps(d))


# ----------------------------- compounds ----------------------------------- #
def fetch_smiles(dbs, names):
    cache = _load(CACHE / "smiles.json")
    todo = [(db, nm) for db, nm in zip(dbs, names) if db not in cache]
    for i, (db, nm) in enumerate(todo):
        smi = None
        for url in (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/{db}/property/CanonicalSMILES/JSON",
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(nm)}/property/CanonicalSMILES/JSON"):
            try:
                r = requests.get(url, headers=UA, timeout=25)
                if r.ok:
                    props = r.json()["PropertyTable"]["Properties"][0]
                    smi = next((v for k, v in props.items() if k.endswith("SMILES")), None)
                    if smi: break
            except Exception:
                pass
        cache[db] = smi
        if (i + 1) % 100 == 0:
            _save(CACHE / "smiles.json", cache); print(f"    SMILES {i+1}/{len(todo)}")
        time.sleep(0.18)
    _save(CACHE / "smiles.json", cache)
    return cache

def morgan(smi):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None: return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_BITS)
    a = np.zeros(FP_BITS, np.float32); ConvertToNumpyArray(bv, a); return a

def compound_features():
    dbs = [c["drugbank"] for c in SUB["compounds"]]
    nms = [c["name"] for c in SUB["compounds"]]
    smiles = fetch_smiles(dbs, nms)
    feats, hit = [], 0
    for db in dbs:
        fp = morgan(smiles.get(db))
        if fp is None:
            r = np.random.default_rng(int(hashlib.md5(db.encode()).hexdigest(), 16) % 2**32)
            fp = (r.random(FP_BITS) < 0.02).astype(np.float32)
        else: hit += 1
        feats.append(fp)
    print(f"  compounds: real fingerprint {hit}/{len(dbs)} ({100*hit/len(dbs):.0f}%)")
    return np.vstack(feats)


# ------------------------------- genes ------------------------------------- #
def fetch_sequences(entrez):
    cache = _load(CACHE / "seqs.json")
    todo = [e for e in entrez if e not in cache]
    for i in range(0, len(todo), 200):
        chunk = todo[i:i + 200]
        try:
            r = requests.post("https://mygene.info/v3/gene",
                              data={"ids": ",".join(chunk), "fields": "uniprot"},
                              headers=UA, timeout=60)
            info = {str(d["query"]): d for d in r.json()} if r.ok else {}
        except Exception:
            info = {}
        for e in chunk:
            up = None
            u = info.get(e, {}).get("uniprot")
            sp = u.get("Swiss-Prot") if isinstance(u, dict) else None
            up = sp if isinstance(sp, str) else (sp[0] if sp else None)
            seq = None
            if up:
                try:
                    fa = requests.get(f"https://rest.uniprot.org/uniprotkb/{up}.fasta",
                                      headers=UA, timeout=25)
                    if fa.ok: seq = "".join(fa.text.split("\n")[1:])
                except Exception: pass
            cache[e] = seq
            time.sleep(0.03)
        _save(CACHE / "seqs.json", cache)
        print(f"    sequences {min(i+200, len(todo))}/{len(todo)}")
    return cache

def gene_features():
    ids = [g["entrez"] for g in SUB["genes"]]
    seqs = fetch_sequences(ids)
    # per-gene ESM cache so supersets never recompute
    store = CACHE / "esm_by_gene.npz"
    cache = {}
    if store.exists():
        d = np.load(store, allow_pickle=True)
        cache = {k: v for k, v in zip(d["ids"], d["emb"])}
    missing = [e for e in ids if e not in cache]
    if missing:
        import torch
        from transformers import AutoTokenizer, AutoModel
        print(f"  genes: embedding {len(missing)} new proteins with ESM2 ...")
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        esm = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval()
        dim = esm.config.hidden_size
        with torch.no_grad():
            for k, e in enumerate(missing):
                seq = seqs.get(e)
                if not seq:
                    cache[e] = np.random.default_rng(
                        int(hashlib.md5(e.encode()).hexdigest(), 16) % 2**32).normal(0, .1, dim).astype(np.float32)
                else:
                    t = tok(seq[:1022], return_tensors="pt", truncation=True, max_length=1024)
                    cache[e] = esm(**t).last_hidden_state[0].mean(0).numpy().astype(np.float32)
                if (k + 1) % 200 == 0: print(f"    ESM {k+1}/{len(missing)}")
        np.savez(store, ids=np.array(list(cache.keys())),
                 emb=np.vstack(list(cache.values())))
    hit = sum(1 for e in ids if seqs.get(e))
    print(f"  genes: real ESM2 for {hit}/{len(ids)} proteins ({100*hit/len(ids):.0f}%)")
    return np.vstack([cache[e] for e in ids])


# ------------------------------ diseases (NEW) ----------------------------- #
def disease_features():
    names = [d["name"] for d in SUB["diseases"]]
    import torch
    from transformers import AutoTokenizer, AutoModel
    print("  diseases: embedding names with MiniLM (sentence-transformers/all-MiniLM-L6-v2) ...")
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(names), 32):
            batch = names[i:i + 32]
            t = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=32)
            out = mdl(**t).last_hidden_state                       # (B, L, 384)
            mask = t["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pool
            embs.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
    emb = np.vstack(embs).astype(np.float32)
    print(f"  diseases: text embedding for {len(names)}/{len(names)} (dim {emb.shape[1]})")
    return emb


if __name__ == "__main__":
    print("Featurizing FULL Hetionet graph (cached per entity):")
    cf = compound_features()
    gf = gene_features()
    df = disease_features()
    np.savez(HERE / "full_features.npz", compound_fp=cf, gene_esm=gf, disease_txt=df)
    print(f"  saved -> full_features.npz  "
          f"(compounds {cf.shape}, genes {gf.shape}, diseases {df.shape})")
