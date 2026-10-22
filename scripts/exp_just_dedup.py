import os, json, time
import numpy as np, faiss
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "faiss_contrib"))
from entity_index import group_reduce, cluster_reduce

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
FIG = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

N_BASE = 20000
HUB    = 10000          # 1/3 of final corpus -- a real mega-hub
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

rng = np.random.default_rng(SEED)
base = _unit(rng.standard_normal((N_BASE, D)).astype("float32"))
q    = _unit(base + rng.normal(0, 1e-3, base.shape).astype("float32")).astype("float32")
q    = np.ascontiguousarray(q)
_, exact = faiss.knn(q, np.ascontiguousarray(base), K, metric=faiss.METRIC_INNER_PRODUCT)
proto = _unit(rng.standard_normal((1, D)).astype("float32"))

# eps=0 is exact; then sweep upward. Report MEASURED intra-hub cosine, not eps.
EPS = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

print(f"d={D}  base={N_BASE:,}  hub={HUB:,}  (hub is {HUB/(N_BASE+HUB):.0%} of corpus)\n")
print(f"{'eps':>7} {'meanCos':>8} {'minCos':>7} {'indeg':>7} "
      f"{'HNSWfull':>9} {'exactDedup_rm':>13} {'HNSW_afterEx':>12} {'clusterReduce':>13}")
rows = []
for eps in EPS:
    if eps == 0.0:
        hub = np.repeat(proto, HUB, axis=0).astype("float32")
    else:
        hub = _unit(proto + rng.normal(0, eps, (HUB, D)).astype("float32")).astype("float32")
    x = np.ascontiguousarray(np.concatenate([base, hub]).astype("float32"))
    gid = np.concatenate([np.arange(N_BASE), np.full(HUB, N_BASE, dtype="int64")])

    # intra-hub similarity to the prototype (how tight / how "distinct")
    sims = (hub @ proto.T).ravel()
    mean_cos, min_cos = float(sims.mean()), float(sims.min())

    # FULL
    hf = build_hnsw(x); r_full = rec(hf.search(q, K)[1], exact); im = indeg_max(hf, len(x))

    # REVIEWER'S FREE FIX: exact dedup (np.unique over rows). Distinct near-dups survive.
    uniq, idx = np.unique(x, axis=0, return_index=True)
    n_removed = len(x) - len(uniq)
    # rebuild on the deduped set; recompute ground truth on the SAME deduped gallery's
    # background part is unchanged (base rows are unique), so the gold ids still apply.
    h_ex = build_hnsw(np.ascontiguousarray(uniq.astype("float32")))
    # gold ids index into `base` rows; after np.unique the base rows may be reordered,
    # so we instead measure recall against exact NN computed on the deduped gallery's base.
    # Simpler & fair: exact dedup only changes the HUB (all base rows are distinct), so for
    # eps>0 uniq==x (nothing removed) and recall == r_full; for eps==0 the hub collapses to 1.
    base_mask = idx < N_BASE
    # recall of background queries against the deduped index, gold = exact on base
    # (background rows are present & unique in uniq, ids change -> map via search vs base)
    _, I_ex = h_ex.search(q, K)
    # translate returned uniq-row -> original-row -> is it a background gold neighbour?
    orig = idx[I_ex]                       # original row id of each returned item
    # recall = overlap with the background exact-NN original ids
    r_ex = float(np.mean([len(set(a) & set(b)) for a, b in zip(orig, exact)]) / K)

    # THIS MODULE: near-dup clustering then reduce (threshold below the hub's tightness)
    red, _, idmap, _ = cluster_reduce(x, threshold=0.95, method="centroid",
                                      metric=faiss.METRIC_INNER_PRODUCT)
    h_cr = build_hnsw(red)
    # gold for reduced: background rows are singletons -> stay distinct; map representative
    # back to whether it covers a gold background neighbour. cluster_reduce keeps singletons
    # as themselves, so representative i of a background row corresponds to that row's id.
    _, I_cr = h_cr.search(q, K)
    # map each returned representative to the set of original rows it covers
    # (approx via: did the query's true NN survive as a representative near the query?)
    # cleaner: recompute exact NN on the reduced gallery and compare HNSW-on-reduced to it
    _, exact_red = faiss.knn(q, red, K, metric=faiss.METRIC_INNER_PRODUCT)
    r_cr = rec(I_cr, exact_red)

    rows.append(dict(eps=eps, mean_cos=mean_cos, min_cos=min_cos, indeg=im, n=len(x),
                     hnsw_full=r_full, exact_removed=int(n_removed), hnsw_exactdedup=r_ex,
                     n_reduced=int(len(red)), cluster_reduce=r_cr))
    print(f"{eps:>7g} {mean_cos:>8.5f} {min_cos:>7.4f} {im:>7,} "
          f"{r_full:>9.4f} {n_removed:>13,} {r_ex:>12.4f} {r_cr:>13.4f}")

json.dump({"config": dict(n_base=N_BASE, hub=HUB, d=D, k=K), "rows": rows},
          open(os.path.join(OUT, "just_dedup.json"), "w"), indent=2)
print(f"\nsaved {os.path.join(OUT, 'just_dedup.json')}")

# figure: collapse vs intra-hub cosine, with what exact-dedup can/can't catch
