import os, json, time
import numpy as np, faiss

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
N_GROUPS = 4000; R = 8; K = 10; EF = 200; SEED = 0; EPS = 1e-5
D_LIST = [32, 64, 128, 384]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype("float32")
def indeg_max(h, n):
    hn = h.hnsw
    offs = faiss.vector_to_array(hn.offsets).astype(np.int64)
    nbr = faiss.vector_to_array(hn.neighbors).astype(np.int64)
    nb0 = hn.nb_neighbors(0); deg = np.zeros(n, np.int64)
    idx = (offs[:n, None] + np.arange(nb0)[None, :]).ravel(); seg = nbr[idx]
    seg = seg[(seg >= 0) & (seg < n)]; np.add.at(deg, seg, 1)
    return int(deg.max())
def measure(x, q):
    x = np.ascontiguousarray(x)
    Dgt, Igt = faiss.knn(q, x, K, metric=faiss.METRIC_INNER_PRODUCT)
    h = faiss.IndexHNSWFlat(x.shape[1], 16, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 200
    h.add(x); h.hnsw.efSearch = EF; Dh, Ih = h.search(q, K)
    idr = float(np.mean([len(set(a) & set(b)) for a, b in zip(Ih, Igt)]) / K)
    tie = float(np.mean((Dh >= (Dgt[:, K - 1] - EPS)[:, None]).sum(1) / K))
    return round(idr, 4), round(tie, 4), indeg_max(h, len(x))

out = {"config": {"n_groups": N_GROUPS, "R": R, "K": K, "ef": EF, "SEED": SEED, "d_list": D_LIST}, "rows": []}
print("  d     exact groups: id   tie-aware  indeg | jitter 1e-3: id  tie  indeg | np.unique on exact: id  indeg")
for d in D_LIST:
    rng = np.random.default_rng(SEED + d)
    proto = _unit(rng.standard_normal((N_GROUPS, d)))
    q = np.ascontiguousarray(_unit(proto + rng.normal(0, 1e-3, proto.shape)))
    x_exact = _unit(np.repeat(proto, R, axis=0))
    x_jit = _unit(np.repeat(proto, R, axis=0) + rng.normal(0, 1e-3, (N_GROUPS * R, d)))
    e_id, e_tie, e_in = measure(x_exact, q)
    j_id, j_tie, j_in = measure(x_jit, q)
    uniq, uidx = np.unique(x_exact, axis=0, return_index=True); xu = x_exact[np.sort(uidx)]
    u_id, u_tie, u_in = measure(xu, q)
    out["rows"].append({"d": d, "exact_groups": {"id": e_id, "tie_aware": e_tie, "indeg_max": e_in, "n": int(len(x_exact))},
                        "jitter_1e-3": {"id": j_id, "tie_aware": j_tie, "indeg_max": j_in},
                        "np_unique_on_exact": {"id": u_id, "tie_aware": u_tie, "indeg_max": u_in, "n": int(len(xu))}})
    print(f"  {d:<5} {e_id:.3f}  {e_tie:.3f}    {e_in:<5} | {j_id:.3f} {j_tie:.3f} {j_in:<5} | {u_id:.3f} {u_in}")
json.dump(out, open(os.path.join(OUT_DIR, "small_exact_groups.json"), "w"), indent=2)
log("saved small_exact_groups.json")
