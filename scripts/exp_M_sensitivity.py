import os, json, time
import numpy as np, faiss

D        = int(os.environ.get("D", "128"))
N_BASE   = int(os.environ.get("N_BASE", "20000"))
SINK_FRAC= float(os.environ.get("SINK_FRAC", "0.3"))
K        = int(os.environ.get("K", "10"))
SEED     = int(os.environ.get("SEED", "0"))
ROOT     = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR  = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG_DIR, exist_ok=True)

M_LIST   = [8, 16, 32, 64]
EFC_LIST = [40, 200, 500]

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

log(f"=== M / efConstruction sensitivity (D={D}, sink={SINK_FRAC}) ===")
rng = np.random.default_rng(SEED)
base = _unit(rng.standard_normal((N_BASE, D)))
S = int(SINK_FRAC * N_BASE); sink_vec = _unit(rng.standard_normal((1, D)))[0]
x = np.ascontiguousarray(np.concatenate([base, np.repeat(sink_vec[None, :], S, 0)])); n = len(x)
q = np.ascontiguousarray(_unit(base + rng.normal(0, 1e-3, base.shape)))
_, exact_I = faiss.knn(q, np.ascontiguousarray(x), K, metric=faiss.METRIC_INNER_PRODUCT)

grid = []
print("   M  efC   recall   max_indeg")
for M in M_LIST:
    for efc in EFC_LIST:
        h = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
        h.hnsw.efConstruction = efc; h.add(x); h.hnsw.efSearch = 200
        _, I = h.search(q, K); r = rec(I, exact_I); mi = max_indeg(h, n)
        grid.append({"M": M, "efC": efc, "recall": round(r, 4), "max_indeg": mi})
        print(f"  {M:<3d} {efc:<4d}  {r:.4f}   {mi}")

best = max(grid, key=lambda g: g["recall"]); worst = min(grid, key=lambda g: g["recall"])
# does raising M (at fixed efC=200) help?
m200 = sorted([g for g in grid if g["efC"] == 200], key=lambda g: g["M"])
m_effect = round(m200[-1]["recall"] - m200[0]["recall"], 4) if len(m200) > 1 else None
verdict = {
    "best": best, "worst": worst,
    "raising_M_effect_at_efC200 (M64 - M8)": m_effect,
    "M_helps": bool(m_effect is not None and m_effect > 0.05),
    "M_hurts": bool(m_effect is not None and m_effect < -0.05),
    "any_setting_recovers (recall>=0.95)": bool(any(g["recall"] >= 0.95 for g in grid)),
    "interpretation": "If no (M, efConstruction) setting reaches recall>=0.95, the mega-hub collapse "
                      "cannot be tuned away with build-time graph knobs -- it requires dedup/reduction. "
                      "The sign of raising_M_effect says whether more neighbors spread or amplify the hub.",
}
out = {"config": {"D": D, "N_BASE": N_BASE, "SINK_FRAC": SINK_FRAC, "K": K, "n": n,
                  "M_list": M_LIST, "efC_list": EFC_LIST}, "grid": grid, "verdict": verdict}
json.dump(out, open(os.path.join(OUT_DIR, "M_sensitivity.json"), "w"), indent=2)
log("saved outputs/validation/M_sensitivity.json")
print(json.dumps(verdict, indent=2))

