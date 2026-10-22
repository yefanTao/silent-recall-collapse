import os, json, time, sys
import numpy as np, pandas as pd, faiss
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "faiss_contrib"))
from entity_index import group_reduce

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wdc_full")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42; K = 10; EF_SWEEP = [16, 32, 64, 128, 256]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("Loading WDC (clean: drop empty-text) ...")
df = pd.read_pickle(os.path.join(DATA_DIR, "wdc_full_cleaned.pkl")).reset_index(drop=True)
emb = np.load(os.path.join(DATA_DIR, "embeddings.npy")).astype(np.float32)
faiss.normalize_L2(emb)
combo = (df["title_clean"].fillna("") + df["brand_clean"].fillna("") + df["description_clean"].fillna("")).str.strip()
keep = (combo != "").to_numpy()
emb = emb[keep]; cid = df["cluster_id"].to_numpy()[keep]; d = emb.shape[1]

# ---- identical gallery construction to exp_pareto.py ----
rng = np.random.default_rng(SEED)
order = np.argsort(cid, kind="stable"); cs = cid[order]
bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); groups = {}
for i in range(len(bnd)):
    s = bnd[i]; e = bnd[i + 1] if i + 1 < len(bnd) else len(cs); groups[cs[s]] = order[s:e]
q_rows, gal_rows, gal_cids = [], [], []
N_Q = 20000
multi = [(c, p) for c, p in groups.items() if len(p) >= 2]
rng.shuffle(multi)
for c, p in multi:
    p = p.copy(); rng.shuffle(p)
    q_rows.append(p[0]); gal_rows.extend(p[1:].tolist()); gal_cids.extend([c] * (len(p) - 1))
for c, p in groups.items():
    if len(p) == 1: gal_rows.append(p[0]); gal_cids.append(c)
q_rows = np.array(q_rows[:N_Q]); Q = np.ascontiguousarray(emb[q_rows]); q_cid = cid[q_rows]
gal_rows = np.array(gal_rows); gal_cids = np.array(gal_cids)
Xfull = np.ascontiguousarray(emb[gal_rows]); n_full = len(Xfull)
log(f"  queries={len(Q):,}  FULL gallery={n_full:,} vectors ({len(np.unique(gal_cids)):,} entities)")

def entity_recall(returned_cids_2d, q_cid, k=K):
    return float(np.mean([q_cid[i] in set(returned_cids_2d[i, :k]) for i in range(len(q_cid))]))
def time_search(index, Q, k, reps=3):
    index.search(Q[:128], k); best = 1e9
    for _ in range(reps):
        t = time.time(); _, I = index.search(Q, k); best = min(best, time.time() - t)
    return I, len(Q) / best
def idx_bytes(index): return int(faiss.serialize_index(index).nbytes)
def hnsw(x):
    h = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 200
    t = time.time(); h.add(np.ascontiguousarray(x)); return h, time.time() - t

results = {"config": {"n_full": int(n_full), "n_queries": int(len(Q)), "K": K, "dim": d,
                      "omp_threads": int(faiss.omp_get_max_threads()), "ef_sweep": EF_SWEEP,
                      "hnsw": "M=16 efConstruction=200"}}

# ---------- FULL-HNSW ----------
log("FULL-HNSW ...")
hf, bt = hnsw(Xfull); by = idx_bytes(hf); rows = []
for ef in EF_SWEEP:
    hf.hnsw.efSearch = ef; I, qps = time_search(hf, Q, K); r = entity_recall(gal_cids[I], q_cid)
    rows.append({"ef": ef, "recall": round(r, 4), "qps": round(qps, 1), "bytes": by, "n": int(n_full), "build_s": round(bt, 1)})
    log(f"  ef={ef:4d} recall={r:.4f} qps={qps:9.1f} bytes={by/1e6:.1f}MB")
results["full_hnsw"] = rows
del hf

# ---------- DEDUP-HNSW (np.unique over vectors) ----------
log("DEDUP-HNSW (np.unique) ...")
codes, uniq_keys = pd.factorize(pd.Series([r.tobytes() for r in Xfull]))
n_dedup = len(uniq_keys)
first_idx = np.full(n_dedup, -1, np.int64)
for i, c in enumerate(codes):
    if first_idx[c] < 0: first_idx[c] = i
Xdedup = np.ascontiguousarray(Xfull[first_idx])
single_cid = gal_cids[first_idx]                              # naive: surviving copy keeps its own entity
multi_cids = [set() for _ in range(n_dedup)]                   # fair: credited to every entity it replaced
for i, c in enumerate(codes): multi_cids[c].add(gal_cids[i])
n_multi = int(sum(len(s) > 1 for s in multi_cids))
log(f"  {n_full:,} -> {n_dedup:,} unique vectors ({n_full/n_dedup:.2f}x fewer); "
    f"{n_multi:,} unique vectors are shared by >1 entity")
hd, bt = hnsw(Xdedup); by = idx_bytes(hd); rows = []
for ef in EF_SWEEP:
    hd.hnsw.efSearch = ef; I, qps = time_search(hd, Q, K)
    r_single = entity_recall(single_cid[I], q_cid)
    r_multi = float(np.mean([any(q_cid[i] in multi_cids[u] for u in I[i, :K] if u >= 0) for i in range(len(q_cid))]))
    rows.append({"ef": ef, "recall_multilabel": round(r_multi, 4), "recall_singlelabel": round(r_single, 4),
                 "qps": round(qps, 1), "bytes": by, "n": int(n_dedup), "build_s": round(bt, 1)})
    log(f"  ef={ef:4d} recall(multi-label)={r_multi:.4f} recall(single-label)={r_single:.4f} qps={qps:9.1f} bytes={by/1e6:.1f}MB")
results["dedup_hnsw"] = rows
results["dedup_stats"] = {"n_dedup": int(n_dedup), "reduction_vs_full": round(n_full / n_dedup, 3),
                          "unique_vectors_shared_by_multiple_entities": n_multi}
del hd

# ---------- REDUCED-HNSW ----------
log("REDUCED-HNSW (centroid) ...")
reduced, out_ids, id_map, _ = group_reduce(Xfull, gal_cids, method="centroid", metric=faiss.METRIC_INNER_PRODUCT)
n_red = len(reduced)
hr, bt = hnsw(reduced); by = idx_bytes(hr); rows = []
for ef in EF_SWEEP:
    hr.hnsw.efSearch = ef; I, qps = time_search(hr, Q, K); r = entity_recall(id_map[I], q_cid)
    rows.append({"ef": ef, "recall": round(r, 4), "qps": round(qps, 1), "bytes": by, "n": int(n_red), "build_s": round(bt, 1)})
    log(f"  ef={ef:4d} recall={r:.4f} qps={qps:9.1f} bytes={by/1e6:.1f}MB")
results["reduced_hnsw"] = rows
results["ratios"] = {
    "full_over_reduced_vectors": round(n_full / n_red, 2), "dedup_over_reduced_vectors": round(n_dedup / n_red, 2),
    "full_over_reduced_bytes": round(results["full_hnsw"][0]["bytes"] / results["reduced_hnsw"][0]["bytes"], 2),
    "dedup_over_reduced_bytes": round(results["dedup_hnsw"][0]["bytes"] / results["reduced_hnsw"][0]["bytes"], 2),
    "best_recall": {"full": max(r["recall"] for r in results["full_hnsw"]),
                    "dedup_multilabel": max(r["recall_multilabel"] for r in results["dedup_hnsw"]),
                    "dedup_singlelabel": max(r["recall_singlelabel"] for r in results["dedup_hnsw"]),
                    "reduced": max(r["recall"] for r in results["reduced_hnsw"])},
}
json.dump(results, open(os.path.join(OUT_DIR, "pareto_dedup.json"), "w"), indent=2)
log(f"saved pareto_dedup.json  ratios={json.dumps(results['ratios'])}")
