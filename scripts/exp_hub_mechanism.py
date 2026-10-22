import os, json
import numpy as np
import faiss

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

D = 64; K = 10; EF = [50, 200]; NPROBE = [16, 64]; SEED = 0


def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return x / n


def indeg_max_mean(index, n):
    h = index.hnsw
    offs = faiss.vector_to_array(h.offsets); nbr = faiss.vector_to_array(h.neighbors)
    nb0 = h.nb_neighbors(0); indeg = np.zeros(n, np.int64)
    for i in range(n):
        seg = nbr[offs[i]:offs[i] + nb0]; seg = seg[(seg >= 0) & (seg < n)]
        indeg[seg] += 1
    return int(indeg.max()), float(indeg.mean())


def rec(approx_I, exact_I):
    return float(np.mean([len(set(a) & set(e)) for a, e in zip(approx_I, exact_I)]) / K)


def evaluate(x, q):
    n, d = x.shape
    _, exact_I = faiss.knn(q, x, K, metric=faiss.METRIC_INNER_PRODUCT)
    h = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 200; h.add(x)
    hr = {}
    for ef in EF:
        h.hnsw.efSearch = ef; _, I = h.search(q, K); hr[ef] = round(rec(I, exact_I), 4)
    imax, imean = indeg_max_mean(h, n)
    nlist = max(16, int(np.sqrt(n)))
    iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    iv.train(x); iv.add(x); ir = {}
    for npb in NPROBE:
        iv.nprobe = npb; _, I = iv.search(q, K); ir[npb] = round(rec(I, exact_I), 4)
    return hr, ir, imax, round(imean, 1)


# =================== PART 1: jitter x one giant group ===================
print("=== PART 1: one GIANT group (size G) of near-exact dups among small groups ===")
print("    sweep jitter; does indeg explode & HNSW collapse only as jitter->0?\n")
N_SMALL = 4000; SMALL_R = 4; GIANT = 8000
rng = np.random.default_rng(SEED)
# fixed prototypes & queries (so only jitter changes between rows)
proto_small = _unit(rng.standard_normal((N_SMALL, D)).astype("float32"))
proto_giant = _unit(rng.standard_normal((1, D)).astype("float32"))
part1 = {}
for jit in [0.0, 1e-4, 1e-3, 1e-2, 5e-2]:
    r2 = np.random.default_rng(100)
    xs = [proto_small[g] + r2.normal(0, jit, (SMALL_R, D)).astype("float32") for g in range(N_SMALL)]
    xs.append(proto_giant[0] + r2.normal(0, jit, (GIANT, D)).astype("float32"))
    x = _unit(np.concatenate(xs).astype("float32"))
    # queries: fresh jittered copies of the small prototypes + the giant proto
    q = _unit(np.concatenate([proto_small + r2.normal(0, max(jit, 1e-3), proto_small.shape).astype("float32"),
                              proto_giant + r2.normal(0, max(jit, 1e-3), proto_giant.shape).astype("float32")
                              ]).astype("float32"))
    x = np.ascontiguousarray(x); q = np.ascontiguousarray(q)
    hr, ir, imax, imean = evaluate(x, q)
    part1[str(jit)] = {"n": int(len(x)), "hnsw": hr, "ivf": ir, "indeg_max": imax, "indeg_mean": imean}
    print(f"  jitter={jit:<6g} n={len(x):,}  HNSW ef50={hr[50]:.3f} ef200={hr[200]:.3f}  "
          f"IVF np64={ir[64]:.3f}  indeg_max={imax:,} (mean {imean})")

# =================== PART 2: exact-duplicate sink sweep ===================
print("\n=== PART 2: inject ONE sink of S literally-identical vectors into a clean corpus ===")
print("    (models the 21% empty-text WDC offers). sweep S; HNSW vs IVF.\n")
N_BASE = 20000
base = _unit(rng.standard_normal((N_BASE, D)).astype("float32"))
sink_vec = _unit(rng.standard_normal((1, D)).astype("float32"))[0]
q_base = _unit(base + rng.normal(0, 1e-3, base.shape).astype("float32")).astype("float32")
part2 = {}
for S in [0, 100, 500, 2000, 8000, 20000]:
    if S == 0:
        x = base.copy()
    else:
        sink = np.repeat(sink_vec[None, :], S, axis=0).astype("float32")  # EXACT duplicates
        x = np.concatenate([base, sink]).astype("float32")
    x = np.ascontiguousarray(_unit(x))
    hr, ir, imax, imean = evaluate(x, np.ascontiguousarray(q_base))  # queries target the clean base only
    part2[str(S)] = {"n": int(len(x)), "sink_frac": round(S / max(1, len(x)), 3),
                     "hnsw": hr, "ivf": ir, "indeg_max": imax, "indeg_mean": imean}
    print(f"  sink={S:<6d} ({part2[str(S)]['sink_frac']:.0%} of {len(x):,})  "
          f"HNSW ef50={hr[50]:.3f} ef200={hr[200]:.3f}  IVF np64={ir[64]:.3f}  indeg_max={imax:,}")

out = {"part1_jitter_x_giant": part1, "part2_exact_sink": part2,
       "config": {"D": D, "K": K, "small_groups": N_SMALL, "small_R": SMALL_R,
                  "giant_size": GIANT, "base_size": N_BASE}}
json.dump(out, open(os.path.join(OUT_DIR, "hub_mechanism.json"), "w"), indent=2)
print("\nsaved hub_mechanism.json")

# ---- figure ----
