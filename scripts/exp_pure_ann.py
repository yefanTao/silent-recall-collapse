import os, time, json
import numpy as np, pandas as pd, torch, faiss

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wdc_full")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

N_TEST = int(os.environ.get("N_TEST", "20000"))
SEED   = 42; DEVICE = "cuda:0"
K      = 10
EF_LIST = [50, 100, 200]
NPROBE_LIST = [16, 32, 64]
R_LIST = [1, 2, 3, 5, 8, 12, "all"]

def ts(): return time.strftime("%H:%M:%S")
def log(m): print(f"[{ts()}] {m}", flush=True)

log("Loading ...")
df = pd.read_pickle(os.path.join(DATA_DIR, "wdc_full_cleaned.pkl")).reset_index(drop=True)
emb = np.load(os.path.join(DATA_DIR, "embeddings.npy")).astype(np.float32)
faiss.normalize_L2(emb)
cid = df["cluster_id"].to_numpy(); title = df["title_clean"].fillna("").astype(str).to_numpy()
d = emb.shape[1]

rng = np.random.default_rng(SEED)
order = np.argsort(cid, kind="stable"); cs = cid[order]
bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); groups = {}
for i in range(len(bnd)):
    s = bnd[i]; e = bnd[i+1] if i+1 < len(bnd) else len(cs); groups[cs[s]] = order[s:e]
clean = {}
RMAX = 12
for eid, pos in groups.items():
    pos = np.array([p for p in pos if title[p].strip() != ""])
    if len(pos) >= 1 + RMAX:    # 1 held-out query + up to RMAX gallery members
        clean[eid] = pos
chosen = rng.choice(np.array(list(clean.keys()), dtype=object),
                    size=min(N_TEST, len(clean)), replace=False)
ent_q, ent_mem = {}, {}
for e in chosen:
    pos = clean[e].copy(); rng.shuffle(pos)
    ent_q[e] = pos[0]; ent_mem[e] = pos[1:]   # 1 query, rest are gallery candidates
ents = list(ent_q.keys())
q_pos = np.array([ent_q[e] for e in ents])
Q = emb[q_pos]
Qt = torch.tensor(Q, device=DEVICE)
log(f"  {len(ents):,} entities, {len(q_pos):,} queries (held out, NOT in gallery)")

def gallery_for(R):
    rows = []
    for e in ents:
        mem = ent_mem[e]
        rows.extend(mem.tolist() if R == "all" else mem[:R].tolist())
    return np.array(rows)

@torch.no_grad()
def exact_topk(gallery_vecs, k=K):
    """Brute-force exact top-k indices INTO gallery, on GPU."""
    G = torch.tensor(gallery_vecs, device=DEVICE)
    out = np.empty((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), 1024):
        sc = Qt[s:s+1024] @ G.T
        out[s:s+1024] = torch.topk(sc, k, dim=1).indices.cpu().numpy()
    return out

def intersection_recall(ann_I, exact_I):
    """Mean |ann_topk ∩ exact_topk| / k  and  recall@1 (exact NN #1 found in ann top-k)."""
    inter = np.array([len(set(a) & set(b)) for a, b in zip(ann_I, exact_I)]) / ann_I.shape[1]
    nn1_found = np.array([exact_I[i, 0] in set(ann_I[i]) for i in range(len(ann_I))])
    return float(inter.mean()), float(nn1_found.mean())

def hnsw_indegree_stats(index, n):
    """Base-layer in-degree distribution from a faiss HNSW index."""
    try:
        hnsw = index.hnsw
        offs = faiss.vector_to_array(hnsw.offsets)
        nbr  = faiss.vector_to_array(hnsw.neighbors)
        # level 0 neighbors live in [offsets[i], offsets[i] + cum_nb_neighbors(1))
        indeg = np.zeros(n, dtype=np.int64)
        for i in range(n):
            lo = offs[i]; hi = offs[i] + hnsw.nb_neighbors(0)
            seg = nbr[lo:hi]
            for v in seg:
                if 0 <= v < n:
                    indeg[v] += 1
        indeg = indeg[indeg >= 0]
        return {"indeg_mean": float(indeg.mean()), "indeg_max": int(indeg.max()),
                "indeg_p99": float(np.percentile(indeg, 99)),
                "gini": float((2*np.sum((np.arange(1,n+1))*np.sort(indeg)))/(n*np.sum(indeg)) - (n+1)/n)
                        if indeg.sum() > 0 else 0.0}
    except Exception as ex:
        return {"error": str(ex)}

results = {"config": {"n_entities": len(ents), "n_queries": int(len(Q)), "K": K,
                      "ef_list": EF_LIST, "nprobe_list": NPROBE_LIST, "R_list": [str(r) for r in R_LIST]},
           "levels": {}}

log("=== PURE ANN recall sweep (HNSW vs IVF, vs each index's OWN exact NN) ===")
for R in R_LIST:
    g = gallery_for(R)
    gv = emb[g]
    n = len(gv)
    avgR = n / len(ents)
    exact_I = exact_topk(gv)

    # HNSW
    hidx = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); hidx.hnsw.efConstruction = 200
    hidx.add(gv)
    hnsw_rec = {}
    for ef in EF_LIST:
        hidx.hnsw.efSearch = ef
        _, I = hidx.search(Q, K)
        inter, nn1 = intersection_recall(I, exact_I)
        hnsw_rec[ef] = {"ann_recall@10": round(inter, 4), "nn1_found@10": round(nn1, 4)}
    indeg = hnsw_indegree_stats(hidx, n) if avgR >= 1 else {}

    # IVF
    nlist = max(16, int(np.sqrt(n)))
    quant = faiss.IndexFlatIP(d)
    iidx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    iidx.train(gv); iidx.add(gv)
    ivf_rec = {}
    for npb in NPROBE_LIST:
        iidx.nprobe = npb
        _, I = iidx.search(Q, K)
        inter, nn1 = intersection_recall(I, exact_I)
        ivf_rec[npb] = {"ann_recall@10": round(inter, 4), "nn1_found@10": round(nn1, 4)}

    results["levels"][str(R)] = {
        "avg_vectors_per_entity": round(avgR, 3), "index_size": int(n),
        "hnsw": hnsw_rec, "ivf": ivf_rec, "hnsw_indegree": indeg,
    }
    log(f"  R={str(R):4s} n={n:7,d} (avgR={avgR:5.2f})  "
        f"HNSW ann-recall@10 ef50={hnsw_rec[50]['ann_recall@10']:.4f} ef200={hnsw_rec[200]['ann_recall@10']:.4f}  |  "
        f"IVF np16={ivf_rec[16]['ann_recall@10']:.4f} np64={ivf_rec[64]['ann_recall@10']:.4f}  |  "
        f"indeg_max={indeg.get('indeg_max','?')}")

json.dump(results, open(os.path.join(OUT_DIR, "pure_ann.json"), "w"), indent=2)
log("saved pure_ann.json")

# figure
