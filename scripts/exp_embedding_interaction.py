import os, json, time, hashlib
import numpy as np, pandas as pd, faiss

N      = int(os.environ.get("N", "120000"))
K      = int(os.environ.get("K", "10"))
SEED   = int(os.environ.get("SEED", "0"))
NQ     = int(os.environ.get("NQ", "5000"))
ROOT   = os.path.join(os.path.dirname(__file__), "..")
DDIR   = os.path.join(ROOT, "data", "wdc_full")
CACHE  = os.path.join(ROOT, "outputs", "cache"); os.makedirs(CACHE, exist_ok=True)
OUT_DIR= os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR= os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG_DIR, exist_ok=True)

MODELS = [
    ("minilm_l6", "sentence-transformers/all-MiniLM-L6-v2", ""),
    ("bge_small", "BAAI/bge-small-en-v1.5",                 ""),
    ("bge_base",  "BAAI/bge-base-en-v1.5",                  ""),
    ("e5_base",   "intfloat/e5-base-v2",                    "passage: "),
    ("gte_base",  "thenlper/gte-base",                      ""),
]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def rec(I, eI): return float(np.mean([len(set(a) & set(b)) for a, b in zip(I, eI)]) / K)
def hnsw_max_indeg(h, n):
    hn = h.hnsw
    offs = faiss.vector_to_array(hn.offsets).astype(np.int64)
    nbr  = faiss.vector_to_array(hn.neighbors).astype(np.int64)
    nb0 = hn.nb_neighbors(0); deg = np.zeros(n, np.int64)
    idx = (offs[:n, None] + np.arange(nb0)[None, :]).ravel(); seg = nbr[idx]
    seg = seg[(seg >= 0) & (seg < n)]; np.add.at(deg, seg, 1)
    return int(deg.max())

# ---- load a fixed WDC text subset (same rows for every model) ----
log("loading WDC text subset")
df = pd.read_pickle(os.path.join(DDIR, "wdc_full_cleaned.pkl")).reset_index(drop=True)
combo = (df["title_clean"].fillna("") + " " + df["brand_clean"].fillna("") + " "
         + df["description_clean"].fillna("")).str.strip()
rng = np.random.default_rng(SEED)
# keep rows with non-empty text, then sample N deterministically
nonempty = np.flatnonzero((combo != "").to_numpy())
sub = np.sort(rng.choice(nonempty, min(N, len(nonempty)), replace=False))
texts = combo.iloc[sub].tolist()
n = len(texts)
text_hashes = np.array([int(hashlib.md5(t.encode("utf-8", "ignore")).hexdigest()[:15], 16) for t in texts], dtype=np.int64)
exact_text_dup = n - len(np.unique(text_hashes))
log(f"n={n} exact-TEXT-dup rows={exact_text_dup} ({100*exact_text_dup/n:.1f}%)")

def vec_dup_frac(X):
    """fraction of rows that are byte-identical (fp32) to an earlier row."""
    # hash each row's bytes
    hh = {}; dup = 0
    for i in range(len(X)):
        key = X[i].tobytes()
        if key in hh: dup += 1
        else: hh[key] = 1
    return dup, len(hh)

def build_measure(X):
    d = X.shape[1]; m = len(X)
    qidx = np.random.default_rng(SEED).choice(m, min(NQ, m), replace=False)
    q = np.ascontiguousarray(X[qidx])
    _, exact = faiss.knn(q, X, K, metric=faiss.METRIC_INNER_PRODUCT)
    h = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200; h.add(X); h.hnsw.efSearch = 200
    _, Ih = h.search(q, K); hr = rec(Ih, exact); mi = hnsw_max_indeg(h, m)
    return round(hr, 4), mi

rows = []
emb_cache_key = f"embint_wdc_n{n}_s{SEED}"
for name, hf, prefix in MODELS:
    cf = os.path.join(CACHE, f"{emb_cache_key}_{name}.npy")
    if os.path.exists(cf):
        log(f"[{name}] cached"); X = np.load(cf)
    else:
        from sentence_transformers import SentenceTransformer
        import torch
        log(f"[{name}] embedding {n} texts on GPU")
        mdl = SentenceTransformer(hf, device="cuda" if torch.cuda.is_available() else "cpu")
        inp = [prefix + t for t in texts] if prefix else texts
        X = mdl.encode(inp, batch_size=1024, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype("float32")
        np.save(cf, X); del mdl; torch.cuda.empty_cache()
    X = np.ascontiguousarray(X.astype("float32"))
    vdup, vuniq = vec_dup_frac(X)
    # raw
    hr_raw, mi_raw = build_measure(X)
    # np.unique on VECTORS
    _, keep = np.unique(X, axis=0, return_index=True)
    Xu = np.ascontiguousarray(X[np.sort(keep)])
    hr_u, mi_u = build_measure(Xu)
    # the key cross-tab: do text-dups map to vector-dups? compare counts
    row = {"model": name, "dim": int(X.shape[1]),
           "exact_text_dup": int(exact_text_dup), "exact_text_dup_frac": round(exact_text_dup/n, 4),
           "exact_vec_dup": int(vdup), "exact_vec_dup_frac": round(vdup/n, 4),
           "text_dup_preserved_as_vec_dup": bool(vdup >= exact_text_dup * 0.99),
           "raw_max_indeg": mi_raw, "raw_hnsw_recall": hr_raw,
           "npunique_max_indeg": mi_u, "npunique_hnsw_recall": hr_u,
           "npunique_removed": int(n - len(Xu))}
    rows.append(row)
    log(f"  [{name}] d={X.shape[1]} vecdup={vdup}({100*vdup/n:.1f}%) raw indeg={mi_raw} rec={hr_raw:.3f}"
        f" -> npunique indeg={mi_u} rec={hr_u:.3f} (removed {n-len(Xu)})")

out = {"config": {"corpus": "WDC subset", "n": int(n), "K": K, "NQ": NQ, "SEED": SEED,
                  "models": [m[0] for m in MODELS]},
       "per_model": rows}
# verdict
text_always_preserved = all(r["text_dup_preserved_as_vec_dup"] for r in rows)
all_collapse_raw = all(r["raw_hnsw_recall"] < 0.9 for r in rows)
all_fixed = all(r["npunique_hnsw_recall"] > 0.95 for r in rows)
out["verdict"] = {
    "exact_text_dup_frac": round(exact_text_dup/n, 4),
    "vec_dup_frac_range": [min(r["exact_vec_dup_frac"] for r in rows),
                           max(r["exact_vec_dup_frac"] for r in rows)],
    "text_dups_always_become_vec_dups": bool(text_always_preserved),
    "all_models_collapse_raw": bool(all_collapse_raw),
    "raw_recall_range": [min(r["raw_hnsw_recall"] for r in rows), max(r["raw_hnsw_recall"] for r in rows)],
    "npunique_fixes_all_models": bool(all_fixed),
    "npunique_recall_range": [min(r["npunique_hnsw_recall"] for r in rows), max(r["npunique_hnsw_recall"] for r in rows)],
    "interpretation": (
        "Tests whether the exact-degeneracy trigger and the np.unique fix are embedding-model "
        "invariant. Deterministic embedders map identical TEXT to identical VECTORS, so the "
        "byte-identical-text duplicates that drive the mega-hub are preserved as byte-identical "
        "VECTORS in every model -- the trigger and the np.unique fix transfer across models, even "
        "though absolute recall and near-dup geometry vary by model. If text_dups_always_become_vec_dups "
        "is true and np.unique fixes every model, the phenomenon is not a MiniLM artifact.")
}
json.dump(out, open(os.path.join(OUT_DIR, "embedding_interaction.json"), "w"), indent=2)
log("saved outputs/validation/embedding_interaction.json")
print(json.dumps(out["verdict"], indent=2))

