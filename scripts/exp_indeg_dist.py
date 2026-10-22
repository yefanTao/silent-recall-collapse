import os, json, time
import numpy as np, pandas as pd, faiss

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wdc_full")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
SEED = 0; K = 10
NBASE = int(os.environ.get("NBASE", "60000"))
SINK_FRAC = float(os.environ.get("SINK_FRAC", "0.3"))

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def indeg_vector(h, n):
    hn = h.hnsw
    offs = faiss.vector_to_array(hn.offsets).astype(np.int64)
    nbr  = faiss.vector_to_array(hn.neighbors).astype(np.int64)
    nb0 = hn.nb_neighbors(0); deg = np.zeros(n, np.int64)
    idx = (offs[:n, None] + np.arange(nb0)[None, :]).ravel(); seg = nbr[idx]
    seg = seg[(seg >= 0) & (seg < n)]; np.add.at(deg, seg, 1)
    return deg

log("Loading WDC-clean embeddings for a real base ...")
df = pd.read_pickle(os.path.join(DATA_DIR, "wdc_full_cleaned.pkl")).reset_index(drop=True)
emb = np.load(os.path.join(DATA_DIR, "embeddings.npy")).astype(np.float32)
faiss.normalize_L2(emb)
combo = (df["title_clean"].fillna("") + df["brand_clean"].fillna("")
         + df["description_clean"].fillna("")).str.strip()
keep = (combo != "").to_numpy()
emb = emb[keep]; d = emb.shape[1]
rng = np.random.default_rng(SEED)
# de-dup the base so "clean" truly has no mega-hub, then sample NBASE
uq = np.unique(emb, axis=0)
sel = rng.choice(len(uq), min(NBASE, len(uq)), replace=False)
base = np.ascontiguousarray(uq[sel]); n_base = len(base)
log(f"  clean base = {n_base:,} unique vectors (d={d})")

def build_indeg(x):
    h = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200; h.add(np.ascontiguousarray(x))
    return indeg_vector(h, len(x))

# clean
deg_clean = build_indeg(base)
# collapsed: + an exact-duplicate sink (a real verbatim base vector repeated)
S = int(SINK_FRAC * n_base)
sink = base[rng.integers(0, n_base)].copy()
xhub = np.concatenate([base, np.repeat(sink[None, :], S, 0)])
deg_hub = build_indeg(xhub)
log(f"  clean max-indeg={deg_clean.max()}  collapsed max-indeg={deg_hub.max()} (sink S={S:,})")

def ccdf(deg):
    vals = np.sort(deg)
    # P(X >= x)
    x = np.unique(vals)
    ccdf = np.array([(deg >= xi).mean() for xi in x])
    return x, ccdf
def gini(deg):
    d = np.sort(deg.astype(np.float64)); n = len(d); cum = np.cumsum(d)
    if cum[-1] == 0: return 0.0
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
def top_share(deg, frac):
    d = np.sort(deg.astype(np.float64))[::-1]; k = max(1, int(frac * len(d)))
    return float(d[:k].sum() / d.sum())

out = {"config": {"corpus": "WDC-clean unique base", "n_base": int(n_base), "d": int(d),
                  "sink_frac": SINK_FRAC, "sink_size": int(S)},
       "clean":     {"max_indeg": int(deg_clean.max()), "mean": round(float(deg_clean.mean()), 2),
                     "p99": int(np.percentile(deg_clean, 99)), "gini": round(gini(deg_clean), 4),
                     "top0.01pct_edge_share": round(top_share(deg_clean, 1e-4), 4),
                     "top1pct_edge_share": round(top_share(deg_clean, 0.01), 4)},
       "collapsed": {"max_indeg": int(deg_hub.max()), "mean": round(float(deg_hub.mean()), 2),
                     "p99": int(np.percentile(deg_hub, 99)), "gini": round(gini(deg_hub), 4),
                     "top0.01pct_edge_share": round(top_share(deg_hub, 1e-4), 4),
                     "top1pct_edge_share": round(top_share(deg_hub, 0.01), 4)}}
out["interpretation"] = (
    "The mechanism is a degenerate clique, not a uniform shift. Clean HNSW in-degree has a bounded "
    "tail (max ~O(M), low Gini); adding one exact-duplicate sink produces a single node whose "
    "in-degree is orders of magnitude larger, and a tiny fraction of nodes captures a large share of "
    "all incoming edges (Gini jumps; top-0.01% edge share spikes). This is what the scalar max-in-degree "
    "summarizes and what collapses greedy navigation.")
json.dump(out, open(os.path.join(OUT_DIR, "indeg_dist.json"), "w"), indent=2)
log("saved outputs/validation/indeg_dist.json")
print(json.dumps(out, indent=2))

