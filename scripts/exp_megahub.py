import os, json
import numpy as np, faiss

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

N_BASE = 20000; K = 10; EF = 200; NPROBE = 64; SEED = 0
D_LIST = [32, 64, 128, 384]
SINK_FRAC = [0.0, 0.05, 0.1, 0.25, 0.5]   # sink size as fraction of N_BASE

def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return x / n

def indeg_max(index, n):
    h = index.hnsw; offs = faiss.vector_to_array(h.offsets); nbr = faiss.vector_to_array(h.neighbors)
    nb0 = h.nb_neighbors(0); deg = np.zeros(n, np.int64)
    for i in range(n):
        seg = nbr[offs[i]:offs[i]+nb0]; seg = seg[(seg>=0)&(seg<n)]; deg[seg]+=1
    return int(deg.max())

def rec(I, exact_I): return float(np.mean([len(set(a)&set(b)) for a,b in zip(I, exact_I)])/K)

print("=== mega-hub harm: clean-query recall vs (exact-dup sink size x dimension) ===")
print("    one exact sink injected into clean iid base; queries target base only.\n")
print("  D \\ sinkfrac " + "".join(f"{s:>8g}" for s in SINK_FRAC))
results = {"config": {"n_base": N_BASE, "K": K, "ef": EF, "nprobe": NPROBE,
                      "d_list": D_LIST, "sink_frac": SINK_FRAC}, "grid": {}}
for d in D_LIST:
    rng = np.random.default_rng(SEED + d)
    base = _unit(rng.standard_normal((N_BASE, d)).astype("float32"))
    q = _unit(base + rng.normal(0, 1e-3, base.shape).astype("float32")).astype("float32")
    q = np.ascontiguousarray(q)
    sink_vec = _unit(rng.standard_normal((1, d)).astype("float32"))[0]
    # ground truth on clean base ONLY (stable ids; clean rows always first)
    _, exact_I = faiss.knn(q, np.ascontiguousarray(base), K, metric=faiss.METRIC_INNER_PRODUCT)
    hn_row, iv_row, id_row = [], [], []
    for sf in SINK_FRAC:
        S = int(sf * N_BASE)
        x = base if S == 0 else np.concatenate([base, np.repeat(sink_vec[None,:], S, axis=0)]).astype("float32")
        x = np.ascontiguousarray(_unit(x))
        h = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction=200
        h.add(x); h.hnsw.efSearch = EF; _, I = h.search(q, K); hnr = rec(I, exact_I)
        nlist = max(16, int(np.sqrt(len(x))))
        iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        iv.train(x); iv.add(x); iv.nprobe = NPROBE; _, I = iv.search(q, K); ivr = rec(I, exact_I)
        im = indeg_max(h, len(x))
        results["grid"][f"d{d}_s{sf}"] = {"d": d, "sink_frac": sf, "sink": S, "n": int(len(x)),
                                          "hnsw": round(hnr,4), "ivf": round(ivr,4), "indeg_max": im}
        hn_row.append(hnr); iv_row.append(ivr); id_row.append(im)
    print(f"  d={d:<4d}      " + "".join(f"{v:>8.3f}" for v in hn_row) + "  (HNSW)")
    print(f"  {'':<9s}  " + "".join(f"{v:>8.3f}" for v in iv_row) + "  (IVF)")
    print(f"  {'':<9s}  " + "".join(f"{v:>8,d}" for v in id_row) + "  (indeg_max)\n")

json.dump(results, open(os.path.join(OUT_DIR, "megahub.json"), "w"), indent=2)
print("saved megahub.json")

