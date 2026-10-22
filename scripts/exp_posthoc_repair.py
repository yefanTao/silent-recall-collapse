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

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype("float32")
def rec(I, exact_I): return float(np.mean([len(set(a) & set(b)) for a, b in zip(I, exact_I)]) / K)

def get_graph(h):
    """Return (offsets, neighbors, nb0) numpy views of the base layer."""
    hn = h.hnsw
    offs = faiss.vector_to_array(hn.offsets).astype(np.int64)
    nbr  = faiss.vector_to_array(hn.neighbors).astype(np.int64)
    nb0  = hn.nb_neighbors(0)
    return hn, offs, nbr, nb0

def max_indeg_from(nbr, offs, nb0, n):
    deg = np.zeros(n, np.int64)
    for i in range(n):
        seg = nbr[offs[i]:offs[i]+nb0]; seg = seg[(seg >= 0) & (seg < n)]; deg[seg] += 1
    return deg

def set_neighbors(h, nbr):
    """Write a modified base-layer neighbor array back into the faiss HNSW."""
    faiss.copy_array_to_vector(nbr.astype(np.int32), h.hnsw.neighbors)

log(f"=== post-hoc HNSW repair (D={D}, N_BASE={N_BASE}, sink={SINK_FRAC}) ===")
rng = np.random.default_rng(SEED)
base = _unit(rng.standard_normal((N_BASE, D)))
S = int(SINK_FRAC * N_BASE); sink_vec = _unit(rng.standard_normal((1, D)))[0]
x = np.ascontiguousarray(np.concatenate([base, np.repeat(sink_vec[None, :], S, 0)]))
n = len(x); sink_ids = np.arange(N_BASE, n)
q = np.ascontiguousarray(_unit(base[rng.choice(N_BASE, 2000, replace=False)] + rng.normal(0, 1e-3, (2000, D))))
_, exact_I = faiss.knn(q, np.ascontiguousarray(x), K, metric=faiss.METRIC_INNER_PRODUCT)

def fresh_index():
    h = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200; h.add(x); h.hnsw.efSearch = 200; return h

results = {}
# --- baseline: collapsed built index ---
h = fresh_index()
_, I = h.search(q, K); base_recall = rec(I, exact_I)
hn, offs, nbr, nb0 = get_graph(h)
deg = max_indeg_from(nbr, offs, nb0, n)
results["collapsed"] = {"recall": round(base_recall, 4), "max_indeg": int(deg.max())}
log(f"  collapsed: recall={base_recall:.4f} max_indeg={int(deg.max())}")

# --- R2 SEVER: zero out the sink nodes' OUTGOING base-layer edges ---
h2 = fresh_index(); hn2, offs2, nbr2, nb02 = get_graph(h2)
nbr2b = nbr2.copy(); touched = 0
for s in sink_ids:
    a, b = offs2[s], offs2[s]+nb02
    touched += int((nbr2b[a:b] >= 0).sum()); nbr2b[a:b] = -1
set_neighbors(h2, nbr2b); _, I2 = h2.search(q, K)
deg2 = max_indeg_from(nbr2b, offs2, nb02, n)
results["R2_sever_sink_outedges"] = {"recall": round(rec(I2, exact_I), 4),
                                     "max_indeg": int(deg2.max()), "edges_touched": touched}
log(f"  R2 sever: recall={results['R2_sever_sink_outedges']['recall']:.4f} "
    f"max_indeg={int(deg2.max())} edges={touched}")

# --- R1 PRUNE: cap in-degree at `cap` by keeping strongest incoming edges ---
for cap in [16, 32, 64]:
    h3 = fresh_index(); hn3, offs3, nbr3, nb03 = get_graph(h3)
    nbr3b = nbr3.copy()
    # build incoming edge list: for each target t, the (source, slot) that point to it
    incoming = {}
    for i in range(n):
        a = offs3[i]
        for slot in range(nb03):
            t = nbr3b[a+slot]
            if 0 <= t < n: incoming.setdefault(t, []).append((i, a+slot))
    touched = 0
    xt = np.ascontiguousarray(x)
    for t, edges in incoming.items():
        if len(edges) <= cap: continue
        srcs = np.array([s for s, _ in edges])
        sims = xt[srcs] @ xt[t]                 # keep the `cap` strongest incoming edges
        keep = set(np.argsort(-sims)[:cap].tolist())
        for j, (s, pos) in enumerate(edges):
            if j not in keep: nbr3b[pos] = -1; touched += 1
    set_neighbors(h3, nbr3b); _, I3 = h3.search(q, K)
    deg3 = max_indeg_from(nbr3b, offs3, nb03, n)
    results[f"R1_prune_cap{cap}"] = {"recall": round(rec(I3, exact_I), 4),
                                     "max_indeg": int(deg3.max()), "edges_touched": touched}
    log(f"  R1 prune cap={cap}: recall={results[f'R1_prune_cap{cap}']['recall']:.4f} "
        f"max_indeg={int(deg3.max())} edges={touched}")

# --- upper bound: dedup + rebuild ---
xu = np.ascontiguousarray(np.unique(x, axis=0))
hu = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT)
hu.hnsw.efConstruction = 200; hu.add(xu); hu.hnsw.efSearch = 200
# recompute exact kNN on the deduped gallery for a fair recall
_, exact_u = faiss.knn(q, xu, K, metric=faiss.METRIC_INNER_PRODUCT)
_, Iu = hu.search(q, K)
results["dedup_rebuild_upperbound"] = {"recall": round(rec(Iu, exact_u), 4), "n_after": int(len(xu))}
log(f"  dedup+rebuild: recall={results['dedup_rebuild_upperbound']['recall']:.4f} n={len(xu)}")

best_repair = max(results[k]["recall"] for k in results if k.startswith(("R1", "R2")))
ub = results["dedup_rebuild_upperbound"]["recall"]; lb = results["collapsed"]["recall"]
recovered = (best_repair - lb) / (ub - lb) if ub > lb else 0.0
verdict = {
    "collapsed_recall": lb, "dedup_rebuild_recall": ub, "best_inplace_repair_recall": round(best_repair, 4),
    "fraction_of_gap_recovered_inplace": round(float(recovered), 3),
    "inplace_repair_viable": bool(recovered > 0.5),
    "interpretation": "fraction_of_gap_recovered_inplace = how much of the (rebuild - collapsed) "
                      "recall gap a cheap in-place graph edit recovers WITHOUT re-inserting data. "
                      ">0.5 would be a genuinely useful online remedy for an index you cannot rebuild; "
                      "near 0 means the collapse is baked into the built graph and only rebuild/dedup helps.",
}
out = {"config": {"D": D, "N_BASE": N_BASE, "SINK_FRAC": SINK_FRAC, "K": K, "n": n, "n_sink": int(S)},
       "results": results, "verdict": verdict}
json.dump(out, open(os.path.join(OUT_DIR, "posthoc_repair.json"), "w"), indent=2)
log("saved outputs/validation/posthoc_repair.json")
print(json.dumps(verdict, indent=2))

