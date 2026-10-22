import os, json, time
import numpy as np, faiss

D        = int(os.environ.get("D", "128"))
RANK     = int(os.environ.get("RANK", "16"))
NCLUST   = int(os.environ.get("NCLUST", "2000"))
K        = int(os.environ.get("K", "10"))
SINK_FRAC= float(os.environ.get("SINK_FRAC", "0.3"))
SEED     = int(os.environ.get("SEED", "0"))
N_LIST   = [int(s) for s in os.environ.get("N_LIST", "100000,500000,2000000,5000000").split(",")]
NQ       = int(os.environ.get("NQ", "5000"))
IVF_PROBE_FRAC = float(os.environ.get("IVF_PROBE_FRAC", "0.05"))
ROOT     = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR  = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG_DIR, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    return (x / nrm).astype("float32")
def rec(I, eI): return float(np.mean([len(set(a) & set(b)) for a, b in zip(I, eI)]) / K)
def max_indeg(h, n):
    hn = h.hnsw
    offs = faiss.vector_to_array(hn.offsets).astype(np.int64)
    nbr  = faiss.vector_to_array(hn.neighbors).astype(np.int64)
    nb0 = hn.nb_neighbors(0); deg = np.zeros(n, np.int64)
    idx = (offs[:n, None] + np.arange(nb0)[None, :]).ravel(); seg = nbr[idx]
    seg = seg[(seg >= 0) & (seg < n)]; np.add.at(deg, seg, 1)
    return int(deg.max())

def clustered(rng, N):
    """N points: each from one of NCLUST centers living in a RANK-dim subspace of R^D, + small noise."""
    basis = rng.standard_normal((RANK, D)).astype("float32")
    centers = (rng.standard_normal((NCLUST, RANK)).astype("float32")) @ basis
    assign = rng.integers(0, NCLUST, size=N)
    pts = centers[assign] + rng.normal(0, 0.15, (N, D)).astype("float32")
    return _unit(pts)

def build_eval_pair(x_clean, q, exact_clean):
    h = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200; h.add(np.ascontiguousarray(x_clean)); h.hnsw.efSearch = 200
    _, Ih = h.search(q, K); hr = rec(Ih, exact_clean); mi = max_indeg(h, len(x_clean))
    nlist = max(64, int(np.sqrt(len(x_clean))))
    iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, nlist, faiss.METRIC_INNER_PRODUCT)
    iv.train(np.ascontiguousarray(x_clean)); iv.add(np.ascontiguousarray(x_clean))
    iv.nprobe = max(1, int(IVF_PROBE_FRAC * nlist)); _, Iiv = iv.search(q, K); ir = rec(Iiv, exact_clean)
    return hr, ir, mi

log(f"=== scale (clustered, rank={RANK}, nclust={NCLUST}, D={D}, sink={SINK_FRAC}) ===")
out = {"config": {"D": D, "RANK": RANK, "NCLUST": NCLUST, "K": K, "SINK_FRAC": SINK_FRAC, "SEED": SEED,
                  "N_list": N_LIST, "NQ": NQ, "IVF_PROBE_FRAC": IVF_PROBE_FRAC,
                  "design": "clustered low-rank background keeps clean HNSW recall high; "
                            "marginal hub damage at matched N"},
       "points": []}
for N in N_LIST:
    rng = np.random.default_rng(SEED + N)
    base = clustered(rng, N)
    qidx = rng.choice(N, NQ, replace=False)
    q = np.ascontiguousarray(_unit(base[qidx] + rng.normal(0, 0.02, (NQ, D))))
    t0 = time.time()
    _, exact_clean = faiss.knn(q, np.ascontiguousarray(base), K, metric=faiss.METRIC_INNER_PRODUCT)
    hr0, ir0, mi0 = build_eval_pair(base, q, exact_clean)
    S = int(SINK_FRAC * N); sink_vec = _unit(rng.standard_normal((1, D)))[0]
    xhub = np.concatenate([base, np.repeat(sink_vec[None, :], S, 0)])
    hr1, ir1, mi1 = build_eval_pair(xhub, q, exact_clean)
    dt = time.time() - t0
    rowd = {"N": N, "n_with_hub": int(N + S),
            "hnsw_clean": round(hr0, 4), "hnsw_hub": round(hr1, 4), "hnsw_drop": round(hr0 - hr1, 4),
            "ivf_clean": round(ir0, 4), "ivf_hub": round(ir1, 4), "ivf_drop": round(ir0 - ir1, 4),
            "max_indeg_clean": mi0, "max_indeg_hub": mi1, "build_search_s": round(dt, 1)}
    out["points"].append(rowd)
    log(f"  N={N:<9,} HNSW {hr0:.3f}->{hr1:.3f} (drop {hr0-hr1:+.3f}) | "
        f"IVF {ir0:.3f}->{ir1:.3f} (drop {ir0-ir1:+.3f}) | indeg {mi0}->{mi1} ({dt:.0f}s)")

hdrops = [p["hnsw_drop"] for p in out["points"]]; idrops = [p["ivf_drop"] for p in out["points"]]
hcleans = [p["hnsw_clean"] for p in out["points"]]
verdict = {
    "clean_hnsw_recall_range": [round(min(hcleans), 4), round(max(hcleans), 4)],
    "clean_baseline_stable": bool(min(hcleans) > 0.85),
    "hnsw_hub_drop_range": [round(min(hdrops), 4), round(max(hdrops), 4)],
    "ivf_hub_drop_range": [round(min(idrops), 4), round(max(idrops), 4)],
    "hnsw_marginal_damage_persists": bool(min(hdrops) > 0.10),
    "ivf_marginally_immune": bool(max(idrops) < 0.05),
    "interpretation": "With a stable clean baseline (clustered low-rank data, like real embeddings), "
                      "the mega-hub's marginal damage to HNSW is isolated from the 'task gets harder' "
                      "confound. If hnsw_drop stays large and ivf_drop ~0 up to millions of points, the "
                      "pathology and IVF immunity are scale-robust on realistic data.",
}
out["verdict"] = verdict
json.dump(out, open(os.path.join(OUT_DIR, "scale_clustered.json"), "w"), indent=2)
log("saved outputs/validation/scale_clustered.json")
print(json.dumps(verdict, indent=2))

