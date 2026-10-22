import os, json, time
import numpy as np, faiss, hnswlib

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)

D = int(os.environ.get("D", "128")); N_BASE = int(os.environ.get("N_BASE", "20000"))
K = 10; EF = 200; M = 16; EFC = 200; SEED = int(os.environ.get("SEED", "0"))
SINK_FRAC = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype("float32")
def rec(I, exact_I): return float(np.mean([len(set(a) & set(b)) for a, b in zip(I, exact_I)]) / K)

def faiss_hnsw(x, q):
    h = faiss.IndexHNSWFlat(x.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = EFC; h.add(x); h.hnsw.efSearch = EF
    _, I = h.search(q, K)
    hn = h.hnsw
    offs = faiss.vector_to_array(hn.offsets).astype(np.int64)
    nbr = faiss.vector_to_array(hn.neighbors).astype(np.int64)
    nb0 = hn.nb_neighbors(0); n = len(x); deg = np.zeros(n, np.int64)
    idx = (offs[:n, None] + np.arange(nb0)[None, :]).ravel(); seg = nbr[idx]
    seg = seg[(seg >= 0) & (seg < n)]; np.add.at(deg, seg, 1)
    return I, int(deg.max())

def hnswlib_level0_indeg_max(index, n):
    """Parse hnswlib's saved binary layout to get the level-0 max in-degree.
    Header (96 bytes): offsetLevel0, max_elements, cur_element_count, size_data_per_element,
    label_offset, offsetData (size_t x6), maxlevel (int), enterpoint (uint), maxM, maxM0, M
    (size_t x3), mult (double), ef_construction (size_t). Then per element a block of
    size_data_per_element bytes: [uint16 link count | pad | maxM0 x uint32 links | data | label]."""
    import tempfile, struct
    fd, path = tempfile.mkstemp(suffix=".hnsw"); os.close(fd)
    index.save_index(path)
    with open(path, "rb") as f: buf = f.read()
    os.remove(path)
    (offL0, maxel, cnt, sz_el, lab_off, off_data) = struct.unpack_from("<6Q", buf, 0)
    maxlevel, enter = struct.unpack_from("<iI", buf, 48)
    maxM, maxM0, Mh = struct.unpack_from("<3Q", buf, 56)
    hdr = 96
    deg = np.zeros(n, np.int64)
    for i in range(cnt):
        base = hdr + i * sz_el
        c = struct.unpack_from("<H", buf, base)[0]
        if c:
            links = np.frombuffer(buf, dtype="<u4", count=c, offset=base + 4)
            np.add.at(deg, links[links < n], 1)
    return int(deg.max()), int(maxM0)

def hnswlib_hnsw(x, q):
    p = hnswlib.Index(space="ip", dim=x.shape[1])
    p.init_index(max_elements=len(x), ef_construction=EFC, M=M, random_seed=SEED)
    p.add_items(x, np.arange(len(x)))
    p.set_ef(EF)
    I, _ = p.knn_query(q, k=K)
    im, m0 = hnswlib_level0_indeg_max(p, len(x))
    return I.astype(np.int64), im

rng = np.random.default_rng(SEED)
base = _unit(rng.standard_normal((N_BASE, D)))
q = np.ascontiguousarray(_unit(base + rng.normal(0, 1e-3, base.shape)))
_, exact_I = faiss.knn(q, np.ascontiguousarray(base), K, metric=faiss.METRIC_INNER_PRODUCT)
sink_vec = _unit(rng.standard_normal((1, D)))[0]

out = {"config": {"D": D, "N_BASE": N_BASE, "K": K, "M": M, "efConstruction": EFC,
                  "efSearch": EF, "SEED": SEED, "sink_frac": SINK_FRAC,
                  "hnswlib_version": getattr(hnswlib, "__version__", "unknown")},
       "sweep": []}
log(f"=== Faiss-HNSW vs hnswlib on the exact-duplicate sink sweep (d={D}, N={N_BASE}) ===")
print("  sinkfrac   n      faiss   hnswlib   faiss_indeg_max")
for sf in SINK_FRAC:
    S = int(sf * N_BASE)
    x = base if S == 0 else np.concatenate([base, np.repeat(sink_vec[None, :], S, axis=0)])
    x = np.ascontiguousarray(_unit(x))
    If, im = faiss_hnsw(x, q); rf = rec(If, exact_I)
    Ih, imh = hnswlib_hnsw(x, q); rh = rec(Ih, exact_I)
    out["sweep"].append({"sink_frac": sf, "n": int(len(x)), "faiss_hnsw": round(rf, 4),
                         "hnswlib_hnsw": round(rh, 4), "faiss_indeg_max": im, "hnswlib_indeg_max": imh})
    print(f"  {sf:<8g} {len(x):<7d} {rf:.3f}   {rh:.3f}     {im:<8d} hnswlib_indeg={imh}")

# np.unique fix on hnswlib at the largest sink
S = int(SINK_FRAC[-1] * N_BASE)
x = np.ascontiguousarray(_unit(np.concatenate([base, np.repeat(sink_vec[None, :], S, axis=0)])))
uniq, uidx = np.unique(x, axis=0, return_index=True)
xu = np.ascontiguousarray(x[np.sort(uidx)])
# ground truth ids refer to the clean base, which occupies the first N_BASE rows in xu
Iu, imu = hnswlib_hnsw(xu, q); ru = rec(Iu, exact_I)
out["np_unique_fix_hnswlib"] = {"sink_frac": SINK_FRAC[-1], "n_before": int(len(x)),
                                "n_after": int(len(xu)), "hnswlib_recall": round(ru, 4), "hnswlib_indeg_max": imu}
log(f"hnswlib after np.unique at sink_frac={SINK_FRAC[-1]}: n {len(x)}->{len(xu)} recall={ru:.3f}")

# verdict
r0 = out["sweep"][0]["hnswlib_hnsw"]; r1 = out["sweep"][-1]["hnswlib_hnsw"]
out["verdict"] = {"hnswlib_drop_clean_to_max_sink": round(r0 - r1, 4),
                  "faiss_drop_clean_to_max_sink": round(out["sweep"][0]["faiss_hnsw"] - out["sweep"][-1]["faiss_hnsw"], 4),
                  "same_direction": bool((r0 - r1) > 0.05)}
json.dump(out, open(os.path.join(OUT_DIR, "hnswlib_check.json"), "w"), indent=2)
log(f"saved hnswlib_check.json  verdict={out['verdict']}")
