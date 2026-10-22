import os, json, time
import numpy as np, pandas as pd, faiss

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "wdc_full")
OUT  = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
os.makedirs(OUT, exist_ok=True)
K = 10; SEED = 42; N_QUERIES = 5000
SUBSET = 200000   # cap corpus for a few-minute run; sampled by whole entities

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def indeg_max(index, n):
    h = index.hnsw; offs = faiss.vector_to_array(h.offsets); nbr = faiss.vector_to_array(h.neighbors)
    nb0 = h.nb_neighbors(0); deg = np.zeros(n, np.int64)
    for i in range(n):
        seg = nbr[offs[i]:offs[i]+nb0]; seg = seg[(seg >= 0) & (seg < n)]; deg[seg] += 1
    return int(deg.max())

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

def recall_vs_bruteforce(x, q, k=K):
    """pure-ANN recall: HNSW & IVF vs exact NN on the SAME gallery x. Returns (rH, rI, indeg)."""
    x = np.ascontiguousarray(x.astype("float32"))
    _, gold = faiss.knn(q, x, k, metric=faiss.METRIC_INNER_PRODUCT)
    h = build_hnsw(x); _, IH = h.search(q, k)
    rH = float(np.mean([len(set(a) & set(b)) for a, b in zip(IH, gold)]) / k)
    im = indeg_max(h, len(x))
    iv = build_ivf(x); _, II = iv.search(q, k)
    rI = float(np.mean([len(set(a) & set(b)) for a, b in zip(II, gold)]) / k)
    return rH, rI, im

log("loading WDC ...")
df = pd.read_pickle(os.path.join(DATA, "wdc_full_cleaned.pkl")).reset_index(drop=True)
emb = np.load(os.path.join(DATA, "embeddings.npy")).astype("float32")
faiss.normalize_L2(emb)
cid = df["cluster_id"].to_numpy()

# sample whole entities up to SUBSET (keeps duplicate structure intact)
rng = np.random.default_rng(SEED)
uniq_c = np.unique(cid); rng.shuffle(uniq_c)
keep_rows, total = [], 0
csizes = pd.Series(cid).groupby(cid).size()
for c in uniq_c:
    rows = np.flatnonzero(cid == c)
    keep_rows.append(rows); total += len(rows)
    if total >= SUBSET: break
keep = np.concatenate(keep_rows)
emb = emb[keep]; cid = cid[keep]; df = df.iloc[keep].reset_index(drop=True)
log(f"  subset: {len(emb):,} offers, {len(np.unique(cid)):,} entities")

# held-out queries = real offers (jitter-free real points), removed from gallery
qsel = rng.choice(len(emb), N_QUERIES, replace=False)
qmask = np.zeros(len(emb), bool); qmask[qsel] = True
Q = np.ascontiguousarray(emb[qsel]);
gal = emb[~qmask]; gal_cid = cid[~qmask]
log(f"  queries={len(Q):,}  gallery={len(gal):,}")

results = {}

# ---- RUNG 1: RAW ----
log("RUNG 1: RAW (all offers) ...")
rH, rI, im = recall_vs_bruteforce(gal, Q)
results["raw"] = dict(n=int(len(gal)), hnsw=round(rH,4), ivf=round(rI,4), indeg_max=im)
log(f"  n={len(gal):,} HNSW={rH:.4f} IVF={rI:.4f} indeg_max={im:,}")

# ---- RUNG 2: EXACT-DEDUP (reviewer's free fix) ----
log("RUNG 2: EXACT-DEDUP np.unique over rows ...")
uniq, uidx = np.unique(gal, axis=0, return_index=True)
n_exact_removed = len(gal) - len(uniq)
gal2 = np.ascontiguousarray(gal[np.sort(uidx)])
rH, rI, im = recall_vs_bruteforce(gal2, Q)
results["exact_dedup"] = dict(n=int(len(gal2)), removed=int(n_exact_removed),
                              hnsw=round(rH,4), ivf=round(rI,4), indeg_max=im)
log(f"  removed {n_exact_removed:,} byte-identical -> n={len(gal2):,} "
    f"HNSW={rH:.4f} IVF={rI:.4f} indeg_max={im:,}")

# ---- RUNG 3: NEAR-DEDUP among the EXACT-DEDUPED set (the distinct near-dups) ----
# How many DISTINCT (survived np.unique) vectors are still near-duplicates? cluster by
# cos>=thr via faiss range/knn graph union-find, collapse to centroid, then measure.
def near_dedup(x, thr=0.99, knn=32, block=20000):
    n = x.shape[1]; N = len(x)
    ix = faiss.IndexFlatIP(n); ix.add(x)
    parent = np.arange(N)
    def find(a):
        r=a
        while parent[r]!=r: r=parent[r]
        while parent[a]!=r: parent[a],a=r,parent[a]
        return r
    for s in range(0, N, block):
        e = min(s+block, N)
        D, I = ix.search(x[s:e], knn+1)
        for ii in range(e-s):
            gi = s+ii
            for j in range(1, knn+1):
                nb = int(I[ii,j])
                if nb < 0 or nb == gi: continue
                if D[ii,j] >= thr:
                    ra, rb = find(gi), find(nb)
                    if ra!=rb: parent[ra]=rb
        if s % (block*5) == 0: log(f"    near-dedup {e:,}/{N:,}")
    comp = np.array([find(i) for i in range(N)])
    # collapse each component to centroid
    labels, inv = np.unique(comp, return_inverse=True)
    g = len(labels); d = x.shape[1]
    red = np.zeros((g, d), "float32"); cnt = np.zeros(g)
    for i, l in enumerate(inv):
        red[l] += x[i]; cnt[l]+=1
    red /= cnt[:,None]
    faiss.normalize_L2(red)
    return np.ascontiguousarray(red), comp, int((cnt>1).sum()), int((N-g))

log("RUNG 3: NEAR-DEDUP (cluster cos>=0.99, collapse to centroid) on the exact-deduped set ...")
gal3, comp, n_multi_groups, n_near_removed = near_dedup(gal2, thr=0.99)
# of the vectors near-dedup removed, ALL are byte-distinct (they survived np.unique) -> the count
rH, rI, im = recall_vs_bruteforce(gal3, Q)
results["near_dedup"] = dict(n=int(len(gal3)),
                             removed_distinct=int(n_near_removed),
                             multi_groups=int(n_multi_groups),
                             hnsw=round(rH,4), ivf=round(rI,4), indeg_max=im)
log(f"  near-dup-removed {n_near_removed:,} DISTINCT vectors (exact-dedup was blind to these) "
    f"-> n={len(gal3):,} HNSW={rH:.4f} IVF={rI:.4f} indeg_max={im:,}")

# crux numbers
results["crux"] = {
    "exact_dedup_removed": int(n_exact_removed),
    "near_dedup_removed_DISTINCT": int(n_near_removed),
    "ratio_distinct_to_exact": round(n_near_removed / max(1, n_exact_removed), 2),
    "hnsw_raw": results["raw"]["hnsw"],
    "hnsw_after_exact_dedup": results["exact_dedup"]["hnsw"],
    "hnsw_after_near_dedup": results["near_dedup"]["hnsw"],
}
json.dump(results, open(os.path.join(OUT, "real_dedup.json"), "w"), indent=2)
log(f"saved {os.path.join(OUT,'real_dedup.json')}")
print("\n=== VERDICT INPUTS ===")
print(json.dumps(results["crux"], indent=2))
