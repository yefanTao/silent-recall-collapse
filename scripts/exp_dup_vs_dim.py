import os, json, time
import numpy as np, faiss
from scipy.stats import skew, spearmanr

DFIX    = int(os.environ.get("DFIX", "128"))
N_BASE  = int(os.environ.get("N_BASE", "20000"))
K       = int(os.environ.get("K", "10"))
EF      = int(os.environ.get("EF", "200"))
NPROBE  = int(os.environ.get("NPROBE", "64"))
SEED    = int(os.environ.get("SEED", "0"))
GPU     = os.environ.get("GPU", "0")

ROOT    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "validation")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

SINK_FRAC = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]      # ARM 1: duplication
D_INTRINSIC = [2, 4, 8, 16, 32, 64, 128]                # ARM 2: effective rank in ambient DFIX

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype("float32")

def indeg_dist(index, n):
    h = index.hnsw
    offs = faiss.vector_to_array(h.offsets); nbr = faiss.vector_to_array(h.neighbors)
    nb0 = h.nb_neighbors(0); deg = np.zeros(n, np.int64)
    for i in range(n):
        seg = nbr[offs[i]:offs[i]+nb0]; seg = seg[(seg >= 0) & (seg < n)]; deg[seg] += 1
    return deg

def rec(I, exact_I): return float(np.mean([len(set(a) & set(b)) for a, b in zip(I, exact_I)]) / K)

def build_eval(x, q, exact_I, n_for_indeg):
    """Return (hnsw_recall, ivf_recall, max_indeg, indeg_skew_over_clean)."""
    x = np.ascontiguousarray(_unit(x))
    h = faiss.IndexHNSWFlat(x.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200; h.add(x); h.hnsw.efSearch = EF
    _, Ih = h.search(q, K); hnr = rec(Ih, exact_I)
    nlist = max(16, int(np.sqrt(len(x))))
    iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(x.shape[1]), x.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    iv.train(x); iv.add(x); iv.nprobe = NPROBE
    _, Iiv = iv.search(q, K); ivr = rec(Iiv, exact_I)
    deg = indeg_dist(h, len(x))
    # hubness skewness measured over the CLEAN rows only (first n_for_indeg), to compare arms fairly
    sk = float(skew(deg[:n_for_indeg])) if n_for_indeg > 2 else 0.0
    return hnr, ivr, int(deg.max()), sk

log(f"=== DUP vs DIM separation (DFIX={DFIX}, N_BASE={N_BASE}, EF={EF}, NPROBE={NPROBE}) ===")
out = {"config": {"DFIX": DFIX, "N_BASE": N_BASE, "K": K, "EF": EF, "NPROBE": NPROBE, "SEED": SEED,
                  "sink_frac": SINK_FRAC, "d_intrinsic": D_INTRINSIC},
       "arm1_duplication": [], "arm2_dimensionality": []}

# ---------------- ARM 1: duplication axis (dimension fixed) ----------------
log("ARM 1 -- duplication: fixed D=%d, growing exact-dup sink" % DFIX)
rng = np.random.default_rng(SEED)
base = _unit(rng.standard_normal((N_BASE, DFIX)))
q = np.ascontiguousarray(_unit(base + rng.normal(0, 1e-3, base.shape)))
_, exact_I = faiss.knn(q, np.ascontiguousarray(base), K, metric=faiss.METRIC_INNER_PRODUCT)
sink_vec = _unit(rng.standard_normal((1, DFIX)))[0]
print("  sinkfrac   n      HNSW    IVF   max_indeg  hub_skew")
for sf in SINK_FRAC:
    S = int(sf * N_BASE)
    x = base if S == 0 else np.concatenate([base, np.repeat(sink_vec[None, :], S, axis=0)])
    hnr, ivr, im, sk = build_eval(x, q, exact_I, N_BASE)
    out["arm1_duplication"].append({"sink_frac": sf, "n": int(len(x)), "hnsw": round(hnr,4),
                                     "ivf": round(ivr,4), "max_indeg": im, "hub_skew": round(sk,3)})
    print(f"  {sf:<8g} {len(x):<7d} {hnr:.3f}  {ivr:.3f}  {im:<9d} {sk:.3f}")

# ---------------- ARM 2: dimensionality axis (NO duplicates) ----------------
log("ARM 2 -- dimensionality: NO dups, vary intrinsic rank in ambient D=%d" % DFIX)
print("  d_intr    n      HNSW    IVF   max_indeg  hub_skew")
for di in D_INTRINSIC:
    rng = np.random.default_rng(SEED + 100 + di)
    # low-rank Gaussian: N_BASE points living on a di-dim subspace of R^DFIX -> intrinsic dim = di
    latent = rng.standard_normal((N_BASE, di)).astype("float32")
    basis  = rng.standard_normal((di, DFIX)).astype("float32")
    x = _unit(latent @ basis)
    q = np.ascontiguousarray(_unit(x + rng.normal(0, 1e-3, x.shape)))   # queries are perturbed copies of points
    _, exact_I = faiss.knn(q, np.ascontiguousarray(x), K, metric=faiss.METRIC_INNER_PRODUCT)
    hnr, ivr, im, sk = build_eval(x, q, exact_I, N_BASE)
    out["arm2_dimensionality"].append({"d_intrinsic": di, "n": int(len(x)), "hnsw": round(hnr,4),
                                       "ivf": round(ivr,4), "max_indeg": im, "hub_skew": round(sk,3)})
    print(f"  {di:<8d} {len(x):<7d} {hnr:.3f}  {ivr:.3f}  {im:<9d} {sk:.3f}")

# ---------------- shared-plane analysis ----------------
a1 = out["arm1_duplication"]; a2 = out["arm2_dimensionality"]
# does the duplication arm follow a monotone indeg->recall law?
if len({p["max_indeg"] for p in a1}) > 2:
    rho1, p1 = spearmanr([np.log10(max(p["max_indeg"],1)) for p in a1], [p["hnsw"] for p in a1])
else:
    rho1, p1 = float("nan"), float("nan")
arm2_max_indeg = max(p["max_indeg"] for p in a2)
arm2_min_recall = min(p["hnsw"] for p in a2)
arm1_max_indeg = max(p["max_indeg"] for p in a1)
arm1_min_recall = min(p["hnsw"] for p in a1)

verdict = {
    "arm1_duplication_indeg_recall_spearman": None if np.isnan(rho1) else round(float(rho1),3),
    "arm1_duplication_p": None if np.isnan(p1) else float(f"{p1:.2e}"),
    "arm1_pushes_indeg_high_and_recall_low": bool(arm1_max_indeg > 2000 and arm1_min_recall < 0.85),
    "arm2_dim_keeps_indeg_low": bool(arm2_max_indeg < 500),
    "arm2_dim_keeps_recall_high": bool(arm2_min_recall > 0.90),
    "arm1_max_indeg": arm1_max_indeg, "arm1_min_recall": round(arm1_min_recall,4),
    "arm2_max_indeg": arm2_max_indeg, "arm2_min_recall": round(arm2_min_recall,4),
    "separation_holds": bool(arm1_max_indeg > 2000 and arm1_min_recall < 0.85
                             and arm2_max_indeg < 500 and arm2_min_recall > 0.90),
    "interpretation": "separation_holds=True means: DUPLICATION drives max-in-degree high and HNSW "
                      "recall low (our mechanism), while pure INTRINSIC DIMENSIONALITY (no dups) keeps "
                      "natural max-in-degree LOW and recall HIGH over the reachable range. The collapse "
                      "is governed by ABSOLUTE max-in-degree (inflated by duplication), NOT by "
                      "dimensionality -- the clean separation from PACMMOD 2026 hubness.",
}
out["verdict"] = verdict
fp = os.path.join(OUT_DIR, "dup_vs_dim.json")
json.dump(out, open(fp, "w"), indent=2)
log(f"saved {fp}")
print("\n=== DUP vs DIM VERDICT ===")
print(json.dumps(verdict, indent=2))

# ---------------- figure: both arms on the shared (indeg, recall) plane ----------------
