import os, json, time
import numpy as np, faiss
from scipy.stats import spearmanr

D       = int(os.environ.get("D", "128"))
N_BASE  = int(os.environ.get("N_BASE", "20000"))
K       = int(os.environ.get("K", "10"))
SEED    = int(os.environ.get("SEED", "0"))
ROOT    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG_DIR, exist_ok=True)

# fine sweep to resolve a knee
SINK_FRAC = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype("float32")
def rec(I, exact_I): return float(np.mean([len(set(a) & set(b)) for a, b in zip(I, exact_I)]) / K)
def max_indeg(h, n):
    hn = h.hnsw; offs = faiss.vector_to_array(hn.offsets); nbr = faiss.vector_to_array(hn.neighbors)
    nb0 = hn.nb_neighbors(0); deg = np.zeros(n, np.int64)
    for i in range(n):
        seg = nbr[offs[i]:offs[i]+nb0]; seg = seg[(seg >= 0) & (seg < n)]; deg[seg] += 1
    return int(deg.max())

log(f"=== predictive model + phase transition (D={D}, N_BASE={N_BASE}) ===")
rng = np.random.default_rng(SEED)
base = _unit(rng.standard_normal((N_BASE, D)))
q = np.ascontiguousarray(_unit(base + rng.normal(0, 1e-3, base.shape)))
_, exact_I = faiss.knn(q, np.ascontiguousarray(base), K, metric=faiss.METRIC_INNER_PRODUCT)
sink_vec = _unit(rng.standard_normal((1, D)))[0]

pts = []
for sf in SINK_FRAC:
    S = int(sf * N_BASE)
    x = base if S == 0 else np.ascontiguousarray(np.concatenate([base, np.repeat(sink_vec[None, :], S, 0)]))
    h = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200; h.add(x); h.hnsw.efSearch = 200
    _, I = h.search(q, K); r = rec(I, exact_I); mi = max_indeg(h, len(x))
    pts.append({"sink_frac": sf, "n": int(len(x)), "recall": round(r, 4), "max_indeg": mi})
    log(f"  sf={sf:<5g} n={len(x):<7d} recall={r:.4f} max_indeg={mi}")

xs = np.array([np.log10(max(p["max_indeg"], 1)) for p in pts])
ys = np.array([p["recall"] for p in pts])

# ---- #4: fit recall_hat = a + b*log10(d_max), clipped to [0,1]; LOO MAE ----
def fit_predict_loo(xs, ys):
    errs = []
    for i in range(len(xs)):
        m = np.ones(len(xs), bool); m[i] = False
        b, a = np.polyfit(xs[m], ys[m], 1)
        yhat = np.clip(a + b*xs[i], 0, 1)
        errs.append(abs(yhat - ys[i]))
    return float(np.mean(errs)), float(np.max(errs))
loo_mae, loo_max = fit_predict_loo(xs, ys)
b, a = np.polyfit(xs, ys, 1)
rho, pv = spearmanr(xs, ys)

# ---- #7: knee = steepest local drop in recall per unit log10(d_max) ----
order = np.argsort(xs)
xo, yo = xs[order], ys[order]
slopes = np.diff(yo) / np.maximum(np.diff(xo), 1e-6)
knee_idx = int(np.argmin(slopes))           # most negative slope segment
knee = {"between_indeg": [int(10**xo[knee_idx]), int(10**xo[knee_idx+1])],
        "recall_drop": round(float(yo[knee_idx] - yo[knee_idx+1]), 4),
        "slope_recall_per_decade": round(float(slopes[knee_idx]), 4)}
# safe threshold: largest d_max with recall >= 0.95
safe = [p["max_indeg"] for p in pts if p["recall"] >= 0.95]
safe_indeg = int(max(safe)) if safe else None

verdict = {
    "spearman_logindeg_recall": round(float(rho), 3), "p": float(f"{pv:.2e}"),
    "linear_fit": {"recall_hat": f"{a:.3f} + ({b:.3f})*log10(d_max)"},
    "leave_one_out_MAE": round(loo_mae, 4), "leave_one_out_maxerr": round(loo_max, 4),
    "predictive_health_check_viable": bool(loo_mae < 0.05),
    "knee": knee,
    "safe_max_indeg_for_recall>=0.95": safe_indeg,
    "interpretation": "LOO MAE is the error of predicting HNSW recall from log10(max in-degree) alone, "
                      "with no ground-truth queries. Low MAE => a single integer read off the index is a "
                      "usable recall health check. The knee localizes the phase transition; "
                      "safe_max_indeg is the operating margin below which recall stays >= 0.95.",
}
out = {"config": {"D": D, "N_BASE": N_BASE, "K": K, "SEED": SEED, "sink_frac": SINK_FRAC},
       "points": pts, "verdict": verdict}
json.dump(out, open(os.path.join(OUT_DIR, "predict_threshold.json"), "w"), indent=2)
log("saved outputs/validation/predict_threshold.json")
print(json.dumps(verdict, indent=2))

