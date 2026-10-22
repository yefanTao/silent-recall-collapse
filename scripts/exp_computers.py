import os, json, time, hashlib
import numpy as np, pandas as pd, pickle, faiss

MODEL = os.environ.get("MODEL", "all-MiniLM-L6-v2")
K     = int(os.environ.get("K", "10"))
SEED  = int(os.environ.get("SEED", "0"))
NQ    = int(os.environ.get("NQ", "5000"))
ROOT  = os.path.join(os.path.dirname(__file__), "..")
DATA  = os.path.join(ROOT, "data", "wdc_products", "computers_cleaned.pkl")
EMB   = os.path.join(ROOT, "data", "wdc_products", "computers_minilm.npy")
OUT_DIR = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG_DIR, exist_ok=True)

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

# ---- load text + labels ----
log("loading WDC-Computers")
df = pickle.load(open(DATA, "rb")).reset_index(drop=True)
text = (df.title_clean.fillna("") + " " + df.description_clean.fillna("")).str.strip().tolist()
cluster = df.cluster_id.to_numpy()
N = len(text)
ndup = N - len(set(text))
log(f"N={N} entities={df.cluster_id.nunique()} exact-dup-rows={ndup} ({100*ndup/N:.1f}%)")

# ---- embed (cached) ----
if os.path.exists(EMB):
    log(f"loading cached embeddings {EMB}")
    X = np.load(EMB)
else:
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"embedding {N} texts with {MODEL} on {dev}")
    m = SentenceTransformer(MODEL, device=dev)
    X = m.encode(text, batch_size=2048, show_progress_bar=True,
                 convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    np.save(EMB, X)
    log(f"saved {EMB} shape {X.shape}")
X = np.ascontiguousarray(X.astype("float32"))
D = X.shape[1]

def build_measure(x):
    """pure-ANN recall@10 (vs exact kNN on the SAME gallery) for HNSW and IVF, + HNSW indeg."""
    n = len(x)
    rng = np.random.default_rng(SEED)
    qidx = rng.choice(n, min(NQ, n), replace=False)
    q = np.ascontiguousarray(x[qidx])
    _, exact = faiss.knn(q, x, K, metric=faiss.METRIC_INNER_PRODUCT)
    h = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200; h.add(x); h.hnsw.efSearch = 200
    _, Ih = h.search(q, K); hr = rec(Ih, exact); mi = hnsw_max_indeg(h, n)
    nlist = max(64, int(np.sqrt(n)))
    iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, nlist, faiss.METRIC_INNER_PRODUCT)
    iv.train(x); iv.add(x); iv.nprobe = 64
    _, Iiv = iv.search(q, K); ir = rec(Iiv, exact)
    return round(hr, 4), round(ir, 4), mi

out = {"config": {"model": MODEL, "N": N, "D": D, "K": K, "NQ": NQ, "SEED": SEED,
                  "entities": int(df.cluster_id.nunique()),
                  "exact_dup_rows": int(ndup), "exact_dup_frac": round(ndup / N, 4)}}

# ---- (1) redundancy sweep: keep up to R offers per entity (Table-1 analogue) ----
log("=== redundancy sweep (keep R per entity) ===")
order = np.arange(N)
by_ent = {}
for i in order:
    by_ent.setdefault(cluster[i], []).append(i)
sweep = []
for R in [1, 2, 3, 5, 8, 12, 0]:   # 0 == all
    if R == 0:
        keep = order
    else:
        keep = np.array([i for ids in by_ent.values() for i in ids[:R]])
    xR = np.ascontiguousarray(X[keep])
    hr, ir, mi = build_measure(xR)
    sweep.append({"R": ("all" if R == 0 else R), "n": int(len(keep)),
                  "max_indeg": mi, "hnsw_recall": hr, "ivf_recall": ir})
    log(f"  R={('all' if R==0 else R):<4} n={len(keep):<7d} indeg={mi:<6d} hnsw={hr:.3f} ivf={ir:.3f}")
out["redundancy_sweep"] = sweep

# ---- (2) exact-dedup triad on a fixed subset (Table-4 analogue) ----
log("=== exact-dedup triad ===")
rng = np.random.default_rng(SEED)
sub = rng.choice(N, min(195000, N), replace=False)
Xs = X[sub]; txt_s = [text[i] for i in sub]
# raw
hr_raw, ir_raw, mi_raw = build_measure(np.ascontiguousarray(Xs))
# exact np.unique on text hash
seen = {}; keep_idx = []
for j, t in enumerate(txt_s):
    hh = hashlib.md5(t.encode("utf-8", "ignore")).hexdigest()
    if hh not in seen:
        seen[hh] = 1; keep_idx.append(j)
Xu = np.ascontiguousarray(Xs[np.array(keep_idx)])
hr_u, ir_u, mi_u = build_measure(Xu)
# + "near-dup" arm: collapse vectors identical after rounding to 4 decimals.
# CAVEAT: this is a much stricter proxy than the cos>=0.99 union-find clustering
# used on WDC in exp_real_dedup.py (it merges only per-coordinate diffs < 1e-4),
# so "+near-dup adds ~0" here is weaker evidence than the WDC arm.
rounded = np.round(Xu, 4)
_, uniq_idx = np.unique(rounded, axis=0, return_index=True)
Xn = np.ascontiguousarray(Xu[np.sort(uniq_idx)])
hr_n, ir_n, mi_n = build_measure(Xn)
triad = [
    {"treatment": "raw", "n": len(Xs), "max_indeg": mi_raw, "hnsw_recall": hr_raw},
    {"treatment": "exact np.unique", "n": len(Xu), "max_indeg": mi_u, "hnsw_recall": hr_u},
    {"treatment": "+ near-dup", "n": len(Xn), "max_indeg": mi_n, "hnsw_recall": hr_n},
]
out["dedup_triad"] = triad
for t in triad:
    log(f"  {t['treatment']:<16} n={t['n']:<7d} indeg={t['max_indeg']:<6d} hnsw={t['hnsw_recall']:.3f}")

# ---- verdict ----
from scipy.stats import spearmanr
xs = [(s["max_indeg"], s["hnsw_recall"]) for s in sweep]
rho, pv = spearmanr([np.log10(max(a, 1)) for a, _ in xs], [b for _, b in xs])
out["verdict"] = {
    "spearman_logindeg_hnsw_recall": round(float(rho), 3), "p": float(f"{pv:.2e}"),
    "hnsw_recall_full_vs_R1": [sweep[-1]["hnsw_recall"], sweep[0]["hnsw_recall"]],
    "ivf_min_recall": min(s["ivf_recall"] for s in sweep),
    "dedup_recovers": [triad[0]["hnsw_recall"], triad[1]["hnsw_recall"]],
    "neardup_adds_over_exact": round(triad[2]["hnsw_recall"] - triad[1]["hnsw_recall"], 4),
    "interpretation": "Third independent real corpus (WDC-Computers, 58% exact-dup). If HNSW recall "
                      "falls with R while IVF stays flat (rho<<0), and exact np.unique recovers recall "
                      "while near-dup adds ~0, the in-degree law and the exact-vs-near dichotomy "
                      "replicate outside the two main-paper corpora.",
}
json.dump(out, open(os.path.join(OUT_DIR, "computers_corpus.json"), "w"), indent=2)
log("saved outputs/validation/computers_corpus.json")
print(json.dumps(out["verdict"], indent=2))

