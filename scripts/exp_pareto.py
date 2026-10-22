import os, json, time
import numpy as np, pandas as pd, faiss
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "faiss_contrib"))
from entity_index import group_reduce

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wdc_full")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
SEED = 42; K = 10
EF_SWEEP   = [16, 32, 64, 128, 256]
NPROBE_SWEEP = [4, 8, 16, 32, 64]

def ts(): return time.strftime("%H:%M:%S")
def log(m): print(f"[{ts()}] {m}", flush=True)

log("Loading WDC (clean: drop empty-text) ...")
df = pd.read_pickle(os.path.join(DATA_DIR, "wdc_full_cleaned.pkl")).reset_index(drop=True)
emb = np.load(os.path.join(DATA_DIR, "embeddings.npy")).astype(np.float32)
faiss.normalize_L2(emb)
combo = (df["title_clean"].fillna("") + df["brand_clean"].fillna("")
         + df["description_clean"].fillna("")).str.strip()
keep = (combo != "").to_numpy()
emb = emb[keep]; cid = df["cluster_id"].to_numpy()[keep]
d = emb.shape[1]
log(f"  {len(emb):,} clean offers, {len(np.unique(cid)):,} entities")

# group, hold out 1 query per multi-record entity, gallery = the rest
rng = np.random.default_rng(SEED)
order = np.argsort(cid, kind="stable"); cs = cid[order]
bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); groups = {}
for i in range(len(bnd)):
    s = bnd[i]; e = bnd[i+1] if i+1 < len(bnd) else len(cs); groups[cs[s]] = order[s:e]
q_rows, gal_rows, gal_cids = [], [], []
N_Q = 20000
multi = [(c, p) for c, p in groups.items() if len(p) >= 2]
rng.shuffle(multi)
# NOTE: one row is held out of EVERY multi-record entity (not only the first
# N_Q); the query list is truncated to N_Q afterwards. This is the gallery
# construction the paper's Pareto/decompose numbers were produced with — do not
# add an early break here or n_full and the headline ratios shift.
for c, p in multi:
    p = p.copy(); rng.shuffle(p)
    q_rows.append(p[0])
    gal_rows.extend(p[1:].tolist()); gal_cids.extend([c] * (len(p) - 1))
# also add all singleton entities to the gallery (realistic catalog)
for c, p in groups.items():
    if len(p) == 1:
        gal_rows.append(p[0]); gal_cids.append(c)
q_rows = np.array(q_rows[:N_Q])
Q = np.ascontiguousarray(emb[q_rows]); q_cid = cid[q_rows]
gal_rows = np.array(gal_rows); gal_cids = np.array(gal_cids)
Xfull = np.ascontiguousarray(emb[gal_rows]); n_full = len(Xfull)
log(f"  queries={len(Q):,}  FULL gallery={n_full:,} vectors ({len(np.unique(gal_cids)):,} entities)")

def entity_recall(returned_cids_2d, q_cid, k=K):
    """fraction of queries whose own cluster_id appears in the returned top-k entity ids."""
    return float(np.mean([q_cid[i] in set(returned_cids_2d[i, :k]) for i in range(len(q_cid))]))

def time_search(index, Q, k, reps=3):
    index.search(Q[:128], k)  # warmup
    best = 1e9
    for _ in range(reps):
        t = time.time(); D, I = index.search(Q, k); best = min(best, time.time() - t)
    return I, len(Q) / best

def idx_bytes(index):
    w = faiss.serialize_index(index); return int(w.nbytes)

results = {"full_hnsw": [], "reduced_hnsw": [], "full_ivf": [], "reduced_ivfpq": [],
           "config": {"n_full": int(n_full), "n_queries": int(len(Q)), "K": K, "dim": d}}

# ---------- FULL-HNSW (returns row -> map to gal_cids) ----------
log("FULL-HNSW ...")
hf = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); hf.hnsw.efConstruction = 200
t = time.time(); hf.add(Xfull); bt = time.time() - t; by = idx_bytes(hf)
for ef in EF_SWEEP:
    hf.hnsw.efSearch = ef; I, qps = time_search(hf, Q, K)
    rec = entity_recall(gal_cids[I], q_cid)
    results["full_hnsw"].append({"ef": ef, "recall": round(rec,4), "qps": round(qps,1),
                                 "bytes": by, "n": n_full, "build_s": round(bt,1)})
    log(f"  ef={ef:4d} recall={rec:.4f} qps={qps:7.1f} bytes={by/1e6:.1f}MB")

# ---------- REDUCED-HNSW (centroid per entity; search returns entity id directly) ----------
log("REDUCED-HNSW (centroid) ...")
reduced, out_ids, id_map, _ = group_reduce(Xfull, gal_cids, method="centroid",
                                           metric=faiss.METRIC_INNER_PRODUCT)
n_red = len(reduced)
hr = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); hr.hnsw.efConstruction = 200
t = time.time(); hr.add(reduced); bt = time.time() - t; by = idx_bytes(hr)
for ef in EF_SWEEP:
    hr.hnsw.efSearch = ef; I, qps = time_search(hr, Q, K)
    rec = entity_recall(id_map[I], q_cid)
    results["reduced_hnsw"].append({"ef": ef, "recall": round(rec,4), "qps": round(qps,1),
                                    "bytes": by, "n": n_red, "build_s": round(bt,1)})
    log(f"  ef={ef:4d} recall={rec:.4f} qps={qps:7.1f} bytes={by/1e6:.1f}MB  (n={n_red:,}, {n_full/n_red:.1f}x fewer)")

# ---------- FULL-IVF (partition control) ----------
log("FULL-IVF ...")
nlist = max(64, int(np.sqrt(n_full)))
iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
t = time.time(); iv.train(Xfull); iv.add(Xfull); bt = time.time() - t; by = idx_bytes(iv)
for npb in NPROBE_SWEEP:
    iv.nprobe = npb; I, qps = time_search(iv, Q, K)
    rec = entity_recall(gal_cids[I], q_cid)
    results["full_ivf"].append({"nprobe": npb, "recall": round(rec,4), "qps": round(qps,1),
                                "bytes": by, "n": n_full, "build_s": round(bt,1)})
    log(f"  np={npb:4d} recall={rec:.4f} qps={qps:7.1f} bytes={by/1e6:.1f}MB")

# ---------- REDUCED + IVF-PQ (cardinality reduction COMPOSES with quantization) ----------
log("REDUCED-IVFPQ (centroid + IVF-PQ) ...")
m = 48  # PQ subquantizers (d=384 -> 48x8bit = 48 bytes/vec)
nlist_r = max(64, int(np.sqrt(n_red)))
ivpq = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist_r, m, 8, faiss.METRIC_INNER_PRODUCT)
ivpq_map = faiss.IndexIDMap(ivpq)
t = time.time(); ivpq.train(reduced); ivpq_map.add_with_ids(reduced, out_ids); bt = time.time() - t
by = idx_bytes(ivpq_map)
for npb in NPROBE_SWEEP:
    ivpq.nprobe = npb; I, qps = time_search(ivpq_map, Q, K)
    rec = entity_recall(id_map[I], q_cid)
    results["reduced_ivfpq"].append({"nprobe": npb, "recall": round(rec,4), "qps": round(qps,1),
                                     "bytes": by, "n": n_red, "build_s": round(bt,1)})
    log(f"  np={npb:4d} recall={rec:.4f} qps={qps:7.1f} bytes={by/1e6:.2f}MB  ({by/1e6:.2f}MB vs FULL-HNSW {results['full_hnsw'][0]['bytes']/1e6:.0f}MB)")

json.dump(results, open(os.path.join(OUT_DIR, "pareto.json"), "w"), indent=2)
log("saved pareto.json")

# ---------- figure: recall vs QPS, and the memory bar ----------
