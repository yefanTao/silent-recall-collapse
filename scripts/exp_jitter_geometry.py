import os, json, time
import numpy as np, faiss
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "faiss_contrib"))
from entity_index import cluster_reduce

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
FIG = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

N_BASE = 20000
HUB    = 10000
D      = 64
K      = 10
SEED   = 0

def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return x / n

def indeg_max(index, n):
    h = index.hnsw; offs = faiss.vector_to_array(h.offsets); nbr = faiss.vector_to_array(h.neighbors)
    nb0 = h.nb_neighbors(0); deg = np.zeros(n, np.int64)
    for i in range(n):
        seg = nbr[offs[i]:offs[i]+nb0]; seg = seg[(seg >= 0) & (seg < n)]; deg[seg] += 1
    return int(deg.max())

def rec(I, exact_I, k=K):
    return float(np.mean([len(set(a) & set(b)) for a, b in zip(I[:, :k], exact_I[:, :k])]) / k)

def build_hnsw(x):
    ix = faiss.IndexHNSWFlat(x.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
    ix.hnsw.efConstruction = 200; ix.add(x); ix.hnsw.efSearch = 200
    return ix

def build_ivf(x):
    nlist = max(64, int(np.sqrt(len(x))))
    ix = faiss.IndexIVFFlat(faiss.IndexFlatIP(x.shape[1]), x.shape[1], nlist,
                            faiss.METRIC_INNER_PRODUCT)
    ix.train(x); ix.add(x); ix.nprobe = 64
    return ix

rng = np.random.default_rng(SEED)
base  = _unit(rng.standard_normal((N_BASE, D)).astype("float32"))
q     = _unit(base + rng.normal(0, 1e-3, base.shape).astype("float32")).astype("float32")
q     = np.ascontiguousarray(q)
_, exact_bg = faiss.knn(q, np.ascontiguousarray(base), K, metric=faiss.METRIC_INNER_PRODUCT)
proto = _unit(rng.standard_normal((1, D)).astype("float32"))

def make_hub(geometry, eps, n=HUB):
    """Return n unit vectors clustered around `proto` with the given jitter geometry."""
    if geometry == "isotropic":
        h = proto + rng.normal(0, eps, (n, D)).astype("float32")
    elif geometry == "one_sided":
        # all variants displaced along the SAME random direction, positive amount only
        direc = _unit(rng.standard_normal((1, D)).astype("float32"))
        amt   = np.abs(rng.normal(0, eps, (n, 1))).astype("float32")
        h = proto + amt * direc
    elif geometry == "low_rank":
        # variants differ only on a few (rank r) shared coordinates: a "template" cloud
        r = 3
        dirs = _unit(rng.standard_normal((r, D)).astype("float32"))   # the few mutable fields
        coef = rng.normal(0, eps, (n, r)).astype("float32")
        h = proto + coef @ dirs
    else:
        raise ValueError(geometry)
    return _unit(h.astype("float32")).astype("float32")

GEOMS = ["isotropic", "one_sided", "low_rank"]
# pick eps per geometry to span the same cosine band; we just sweep eps and report measured cos
EPS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

print(f"d={D}  base={N_BASE:,}  hub={HUB:,}  (hub is {HUB/(N_BASE+HUB):.0%} of corpus)")
print("Goal: find a geometry where collapse persists at cos<1.0 with np.unique removing 0.\n")
hdr = f"{'geom':>10} {'eps':>7} {'meanCos':>8} {'minCos':>7} {'uniqRM':>7} {'indeg':>7} {'HNSWfull':>9} {'afterCR':>8} {'IVF':>6}"
print(hdr); print("-"*len(hdr))

rows = []
for geom in GEOMS:
    for eps in EPS:
        hub = make_hub(geom, eps)
        x = np.ascontiguousarray(np.concatenate([base, hub]).astype("float32"))

        sims = (hub @ proto.T).ravel()
        mean_cos, min_cos = float(sims.mean()), float(sims.min())

        # reviewer's free fix
        uniq = np.unique(x, axis=0)
        n_uniq_removed = len(x) - len(uniq)

        # FULL HNSW
        hf = build_hnsw(x); _, IH = hf.search(q, K)
        r_full = rec(IH, exact_bg); im = indeg_max(hf, len(x))

        # IVF control
        iv = build_ivf(x); _, II = iv.search(q, K); r_ivf = rec(II, exact_bg)

        # this module: near-dup clustering + reduce
        red, _, _, _ = cluster_reduce(x, threshold=0.9, method="centroid",
                                      metric=faiss.METRIC_INNER_PRODUCT)
        h_cr = build_hnsw(red); _, I_cr = h_cr.search(q, K)
        _, exact_red = faiss.knn(q, red, K, metric=faiss.METRIC_INNER_PRODUCT)
        r_cr = rec(I_cr, exact_red)

        rows.append(dict(geom=geom, eps=eps, mean_cos=mean_cos, min_cos=min_cos,
                         uniq_removed=int(n_uniq_removed), indeg=im, n=len(x),
                         hnsw_full=r_full, n_reduced=int(len(red)),
                         hnsw_clusterreduce=r_cr, ivf_full=r_ivf))
        print(f"{geom:>10} {eps:>7g} {mean_cos:>8.5f} {min_cos:>7.4f} {n_uniq_removed:>7,} "
              f"{im:>7,} {r_full:>9.4f} {r_cr:>8.4f} {r_ivf:>6.3f}")

json.dump({"config": dict(n_base=N_BASE, hub=HUB, d=D, k=K), "rows": rows},
          open(os.path.join(OUT, "jitter_geometry.json"), "w"), indent=2)
print(f"\nsaved {os.path.join(OUT, 'jitter_geometry.json')}")

# verdict: is there any row with cos<0.9999, uniq_removed==0, hnsw_full<0.9, and r_cr>0.95?
survivors = [r for r in rows if r["mean_cos"] < 0.9999 and r["uniq_removed"] == 0
             and r["hnsw_full"] < 0.9 and r["hnsw_clusterreduce"] > 0.95]
print("\n=== SURVIVORS (collapse at cos<1.0, np.unique blind, cluster_reduce fixes) ===")
if survivors:
    for r in survivors:
        print(f"  {r['geom']:>10} cos={r['mean_cos']:.5f} indeg={r['indeg']:,} "
              f"HNSW_full={r['hnsw_full']:.3f} -> cluster_reduce={r['hnsw_clusterreduce']:.3f} "
              f"(np.unique removed {r['uniq_removed']})")
    print("\n  => a distinct near-duplicate hub regime exists (collapse without cos==1.0).")
else:
    print("  none. Across all 3 geometries, collapse needs cos==1.0 (np.unique territory).")
    print("  => recall collapse is exact-duplicate-specific; np.unique is the complete recall fix.")
