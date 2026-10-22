import os, json, time
import numpy as np, faiss

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
D = 128; N_BASE = 20000; SINK_FRAC = 0.30; SEED = 0; EF = 200
K_LIST = [1, 10, 50, 100]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype("float32")
def rec(I, exact_I, k):
    return float(np.mean([len(set(a[:k].tolist()) & set(b[:k].tolist())) for a, b in zip(I, exact_I)]) / k)
def indeg_max(h, n):
    hn = h.hnsw
    offs = faiss.vector_to_array(hn.offsets).astype(np.int64)
    nbr = faiss.vector_to_array(hn.neighbors).astype(np.int64)
    nb0 = hn.nb_neighbors(0); deg = np.zeros(n, np.int64)
    idx = (offs[:n, None] + np.arange(nb0)[None, :]).ravel(); seg = nbr[idx]
    seg = seg[(seg >= 0) & (seg < n)]; np.add.at(deg, seg, 1)
    return int(deg.max())

rng = np.random.default_rng(SEED)
base = _unit(rng.standard_normal((N_BASE, D)))
q = np.ascontiguousarray(_unit(base + rng.normal(0, 1e-3, base.shape)))
sink_vec = _unit(rng.standard_normal((1, D)))[0]
S = int(SINK_FRAC * N_BASE)
kmax = max(K_LIST)
_, exact_I = faiss.knn(q, np.ascontiguousarray(base), kmax, metric=faiss.METRIC_INNER_PRODUCT)

def build(x_rows, ids):
    """Build HNSW by adding rows in the given order; return index searching in ORIGINAL id space."""
    h = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 200
    idm = faiss.IndexIDMap(h); idm.add_with_ids(np.ascontiguousarray(x_rows), ids.astype(np.int64))
    return h, idm

out = {"config": {"D": D, "N_BASE": N_BASE, "sink_frac": SINK_FRAC, "sink_size": S, "SEED": SEED,
                  "M": 16, "efConstruction": 200, "efSearch": EF, "k_list": K_LIST}}

# ---------------- (1) insertion order ----------------
log("=== (1) insertion order of the duplicate sink ===")
x_all = np.concatenate([base, np.repeat(sink_vec[None, :], S, axis=0)])  # ids 0..N_BASE-1 = clean
ids_all = np.arange(len(x_all))
orders = {
    "duplicates_last (paper default)": ids_all,
    "duplicates_first": np.concatenate([ids_all[N_BASE:], ids_all[:N_BASE]]),
    "interleaved_random (streaming)": np.random.default_rng(1).permutation(ids_all),
}
res1 = []
for name, perm in orders.items():
    h, idm = build(x_all[perm], perm)
    h.hnsw.efSearch = EF; _, I = idm.search(q, 10)
    r = rec(I, exact_I, 10); im = indeg_max(h, len(x_all))
    res1.append({"order": name, "recall@10": round(r, 4), "indeg_max": im})
    log(f"  {name:<34} recall@10={r:.4f} indeg_max={im}")
h0, idm0 = build(x_all, ids_all); h0.hnsw.efSearch = EF
_, I = idm0.search(q, 10)
hc, idmc = build(base, np.arange(N_BASE)); hc.hnsw.efSearch = EF
_, Ic = idmc.search(q, 10)
res1.append({"order": "clean (no sink)", "recall@10": round(rec(Ic, exact_I, 10), 4), "indeg_max": indeg_max(hc, N_BASE)})
out["insertion_order"] = res1

# ---------------- (2) recall@k ----------------
log("=== (2) recall@k on the collapsed vs clean graph ===")
res2 = []
for k in K_LIST:
    ef = max(EF, k)
    h0.hnsw.efSearch = ef; _, I = idm0.search(q, k)
    hc.hnsw.efSearch = ef; _, Ic = idmc.search(q, k)
    rk, rck = rec(I, exact_I, k), rec(Ic, exact_I, k)
    hub_in_topk = float(np.mean([(a >= N_BASE).any() for a in I]))
    res2.append({"k": k, "efSearch": ef, "collapsed_recall": round(rk, 4), "clean_recall": round(rck, 4),
                 "drop": round(rck - rk, 4), "frac_queries_with_hub_in_topk": round(hub_in_topk, 4)})
    log(f"  k={k:<4} ef={ef:<4} collapsed={rk:.4f} clean={rck:.4f} drop={rck-rk:.4f} hub_in_topk={hub_in_topk:.3f}")
out["recall_at_k"] = res2

json.dump(out, open(os.path.join(OUT_DIR, "deploy_questions.json"), "w"), indent=2)
log("saved deploy_questions.json")
