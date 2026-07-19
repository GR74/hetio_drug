"""
features.py  --  REAL content features for the Hetionet subgraph nodes.

Compounds -> RDKit Morgan fingerprints (1024-bit) from PubChem SMILES.
Genes     -> ESM2 protein-language-model embeddings (320-d) from UniProt sequences.

Both are *content* features: they describe WHAT a node is, independent of the
graph. That is precisely what lets the GNN embed a node it never saw during
training (the inductive step beyond TxGNN). Everything is cached to disk, so
this slow step runs once.

Run:  python features.py
"""
import json, time, pathlib, hashlib
import numpy as np
import requests

HERE = pathlib.Path(__file__).parent
SUB = json.loads((HERE / "subgraph.json").read_text())
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)

FP_BITS = 1024
UA = {"User-Agent": "IAIRO-GNN-toy/1.0 (research)"}


def _load(p):  return json.loads(p.read_text()) if p.exists() else {}
def _save(p, d): p.write_text(json.dumps(d))


# --------------------------------------------------------------------------- #
# Compounds: DrugBank id -> SMILES (PubChem) -> Morgan fingerprint (RDKit)     #
# --------------------------------------------------------------------------- #
def fetch_smiles(drugbank_ids, names):
    cache = _load(CACHE / "smiles.json")
    for db, nm in zip(drugbank_ids, names):
        if db in cache:
            continue
        smi = None
        # (a) DrugBank registry cross-reference
        try:
            u = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/"
                 f"RegistryID/{db}/property/CanonicalSMILES/JSON")
            r = requests.get(u, headers=UA, timeout=25)
            if r.ok:
                props = r.json()["PropertyTable"]["Properties"][0]
                smi = next((v for k, v in props.items() if k.endswith("SMILES")), None)
        except Exception:
            pass
        # (b) fall back to name search
        if not smi:
            try:
                u = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                     f"{requests.utils.quote(nm)}/property/CanonicalSMILES/JSON")
                r = requests.get(u, headers=UA, timeout=25)
                if r.ok:
                    props = r.json()["PropertyTable"]["Properties"][0]
                    smi = next((v for k, v in props.items() if k.endswith("SMILES")), None)
            except Exception:
                pass
        cache[db] = smi
        time.sleep(0.2)                      # be polite to PubChem
    _save(CACHE / "smiles.json", cache)
    return cache


def morgan(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
    arr = np.zeros(FP_BITS, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(bv, arr)
    return arr


def compound_features():
    dbs   = [c["drugbank"] for c in SUB["compounds"]]
    names = [c["name"]     for c in SUB["compounds"]]
    smiles = fetch_smiles(dbs, names)
    feats, hit = [], 0
    for db in dbs:
        fp = morgan(smiles.get(db))
        if fp is None:
            # deterministic hashed fallback so the pipeline is complete
            h = int(hashlib.md5(db.encode()).hexdigest(), 16)
            r = np.random.default_rng(h % (2**32))
            fp = (r.random(FP_BITS) < 0.02).astype(np.float32)
        else:
            hit += 1
        feats.append(fp)
    print(f"  compounds: real fingerprint for {hit}/{len(dbs)} "
          f"({100*hit/len(dbs):.0f}% real chemistry)")
    return np.vstack(feats)


# --------------------------------------------------------------------------- #
# Genes: Entrez id -> UniProt sequence (MyGene) -> ESM2 embedding (320-d)      #
# --------------------------------------------------------------------------- #
def fetch_sequences(entrez_ids):
    cache = _load(CACHE / "seqs.json")
    todo = [e for e in entrez_ids if e not in cache]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        try:
            r = requests.post("https://mygene.info/v3/gene",
                              data={"ids": ",".join(chunk), "fields": "uniprot"},
                              headers=UA, timeout=40)
            info = {str(d["query"]): d for d in r.json()} if r.ok else {}
        except Exception:
            info = {}
        for e in chunk:
            up = None
            d = info.get(e, {})
            sp = d.get("uniprot", {}).get("Swiss-Prot") if isinstance(d.get("uniprot"), dict) else None
            up = sp if isinstance(sp, str) else (sp[0] if sp else None)
            seq = None
            if up:
                try:
                    fa = requests.get(f"https://rest.uniprot.org/uniprotkb/{up}.fasta",
                                      headers=UA, timeout=25)
                    if fa.ok:
                        seq = "".join(fa.text.split("\n")[1:])
                except Exception:
                    pass
            cache[e] = seq
            time.sleep(0.05)
        _save(CACHE / "seqs.json", cache)
    return cache


def gene_features():
    npz = CACHE / "esm.npz"
    ids = [g["entrez"] for g in SUB["genes"]]
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        if list(d["ids"]) == ids:
            print(f"  genes: loaded cached ESM2 embeddings ({d['emb'].shape})")
            return d["emb"]
    seqs = fetch_sequences(ids)

    import torch
    from transformers import AutoTokenizer, AutoModel
    print("  genes: loading ESM2 (facebook/esm2_t6_8M_UR50D) ...")
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval()
    dim = esm.config.hidden_size

    embs, hit = [], 0
    with torch.no_grad():
        for k, e in enumerate(ids):
            seq = seqs.get(e)
            if not seq:
                h = int(hashlib.md5(e.encode()).hexdigest(), 16)
                embs.append(np.random.default_rng(h % 2**32).normal(0, 0.1, dim).astype(np.float32))
                continue
            t = tok(seq[:1022], return_tensors="pt", truncation=True, max_length=1024)
            out = esm(**t).last_hidden_state[0]         # (L, dim)
            embs.append(out.mean(0).numpy().astype(np.float32))
            hit += 1
            if (k + 1) % 50 == 0:
                print(f"    embedded {k+1}/{len(ids)} proteins")
    emb = np.vstack(embs)
    np.savez(npz, ids=np.array(ids), emb=emb)
    print(f"  genes: real ESM2 embedding for {hit}/{len(ids)} "
          f"({100*hit/len(ids):.0f}% real proteins), dim={emb.shape[1]}")
    return emb


if __name__ == "__main__":
    print("Featurizing REAL Hetionet nodes (cached after first run):")
    cf = compound_features()
    gf = gene_features()
    np.savez(HERE / "node_features.npz", compound_fp=cf, gene_esm=gf)
    print(f"  saved -> node_features.npz  (compounds {cf.shape}, genes {gf.shape})")
