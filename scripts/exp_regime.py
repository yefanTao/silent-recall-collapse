import os, json
import numpy as np, faiss

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

N_GROUPS = 4000; R = 8; K = 10; EF = 200; NPROBE = 64; SEED = 0
D_LIST   = [32, 64, 128, 384]
JIT_LIST = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2]

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

print(f"=== regime map: {N_GROUPS} groups x R={R} near-dup copies; pure-ANN recall@10 ===")
print("    rows=dimension, cols=jitter. Looking for: HNSW collapses where IVF holds.\n")
header = "  D \\ jit  " + "".join(f"{j:>9g}" for j in JIT_LIST)
print(header)
results = {"config": {"n_groups": N_GROUPS, "R": R, "K": K, "ef": EF, "nprobe": NPROBE,
                      "d_list": D_LIST, "jit_list": JIT_LIST}, "grid": {}}
for d in D_LIST:
    rng = np.random.default_rng(SEED + d)
    proto = _unit(rng.standard_normal((N_GROUPS, d)).astype("float32"))
    hn_row, iv_row, id_row = [], [], []
    for jit in JIT_LIST:
        r2 = np.random.default_rng(1234 + d)
        x = _unit(np.repeat(proto, R, axis=0) + r2.normal(0, jit, (N_GROUPS*R, d)).astype("float32"))
        x = np.ascontiguousarray(x.astype("float32"))
        q = _unit(proto + r2.normal(0, max(jit, 1e-3), proto.shape).astype("float32")).astype("float32")
        q = np.ascontiguousarray(q)
        _, exact_I = faiss.knn(q, x, K, metric=faiss.METRIC_INNER_PRODUCT)
        h = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction=200
        h.add(x); h.hnsw.efSearch = EF; _, I = h.search(q, K); hnr = rec(I, exact_I)
        nlist = max(16, int(np.sqrt(len(x))))
        iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        iv.train(x); iv.add(x); iv.nprobe = NPROBE; _, I = iv.search(q, K); ivr = rec(I, exact_I)
        im = indeg_max(h, len(x))
        results["grid"][f"d{d}_j{jit}"] = {"d": d, "jitter": jit, "n": int(len(x)),
                                           "hnsw": round(hnr,4), "ivf": round(ivr,4), "indeg_max": im}
        hn_row.append(hnr); iv_row.append(ivr); id_row.append(im)
    print(f"  d={d:<4d}   " + "".join(f"{v:>9.3f}" for v in hn_row) + "   (HNSW)")
    print(f"  {'':<7s}  " + "".join(f"{v:>9.3f}" for v in iv_row) + "   (IVF)")
    print(f"  {'':<7s}  " + "".join(f"{v:>9,d}" for v in id_row) + "   (indeg_max)\n")

json.dump(results, open(os.path.join(OUT_DIR, "regime.json"), "w"), indent=2)
print("saved regime.json")

