import os, json, time
import numpy as np, faiss

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
D = 128; N_BASE = 20000; SINK_FRAC = 0.3; K = 10; SEED = 0
EF_LIST = [200, 500, 1000, 2000]
CELLS = [(64, 40), (32, 500), (16, 200)]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype("float32")
def rec(I, exact_I): return float(np.mean([len(set(a) & set(b)) for a, b in zip(I, exact_I)]) / K)

rng = np.random.default_rng(SEED)
base = _unit(rng.standard_normal((N_BASE, D)))
S = int(SINK_FRAC * N_BASE); sink_vec = _unit(rng.standard_normal((1, D)))[0]
q = np.ascontiguousarray(_unit(base + rng.normal(0, 1e-3, base.shape)))
_, exact_I = faiss.knn(q, np.ascontiguousarray(base), K, metric=faiss.METRIC_INNER_PRODUCT)
x = np.ascontiguousarray(_unit(np.concatenate([base, np.repeat(sink_vec[None, :], S, axis=0)])))

out = {"config": {"D": D, "N_BASE": N_BASE, "SINK_FRAC": SINK_FRAC, "K": K, "SEED": SEED,
                  "ef_list": EF_LIST, "cells": CELLS}, "collapsed": [], "clean_reference": []}
for (M, efc) in CELLS:
    h = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = efc; h.add(x)
    row = {"M": M, "efC": efc}
    for ef in EF_LIST:
        h.hnsw.efSearch = ef; _, I = h.search(q, K); row[f"ef{ef}"] = round(rec(I, exact_I), 4)
    out["collapsed"].append(row); log(f"collapsed M={M} efC={efc}: " + " ".join(f"ef{ef}={row[f'ef{ef}']:.3f}" for ef in EF_LIST))
    hc = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT); hc.hnsw.efConstruction = efc; hc.add(np.ascontiguousarray(base))
    hc.hnsw.efSearch = 200; _, Ic = hc.search(q, K)
    out["clean_reference"].append({"M": M, "efC": efc, "ef200": round(rec(Ic, exact_I), 4)})
best = max(r[f"ef{ef}"] for r in out["collapsed"] for ef in EF_LIST)
out["verdict"] = {"best_collapsed_recall_any_ef": best, "reaches_0.95": bool(best >= 0.95)}
json.dump(out, open(os.path.join(OUT_DIR, "M64_efsearch.json"), "w"), indent=2)
log(f"saved M64_efsearch.json verdict={out['verdict']}")
