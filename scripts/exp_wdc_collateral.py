import os, json, time
import numpy as np, pandas as pd, torch, faiss

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wdc_full")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
SEED = 42; DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"; K = 10; EF = [50, 200]; NPROBE = [16, 64]

def ts(): return time.strftime("%H:%M:%S")
def log(m): print(f"[{ts()}] {m}", flush=True)

log("Loading WDC ...")
df = pd.read_pickle(os.path.join(DATA_DIR, "wdc_full_cleaned.pkl")).reset_index(drop=True)
emb = np.load(os.path.join(DATA_DIR, "embeddings.npy")).astype(np.float32)
faiss.normalize_L2(emb)
combo = (df["title_clean"].fillna("") + df["brand_clean"].fillna("")
         + df["description_clean"].fillna("")).str.strip()
empty = (combo == "").to_numpy()
cid = df["cluster_id"].to_numpy()
d = emb.shape[1]
log(f"  {len(df):,} offers, {empty.mean():.1%} empty-text")

rng = np.random.default_rng(SEED)

# ---- 1. characterize the empty-text sink geometry ----
emp_idx = np.flatnonzero(empty)
samp = rng.choice(emp_idx, size=min(2000, len(emp_idx)), replace=False)
V = emb[samp]
# mean pairwise cosine via centroid trick: mean_{i!=j} <vi,vj>
c = V.mean(0); mean_pair_cos = float((len(V) * (c @ c) - 1.0) / (len(V) * (len(V) - 1)) * len(V))
# simpler: average cosine to the sink centroid
cn = c / (np.linalg.norm(c) + 1e-9)
cos_to_centroid = float((V @ cn).mean())
# count exact-identical groups among empties
uniq = np.unique(np.round(V, 4), axis=0)
log(f"  empty-text sink: mean cos-to-centroid={cos_to_centroid:.4f}, "
    f"{len(uniq):,}/{len(V):,} distinct (rounded) -> {1-len(uniq)/len(V):.0%} exact dups")

# ---- 2. clean bystander set: entities with NO empty text, size in [2,8] ----
order = np.argsort(cid, kind="stable"); cs = cid[order]
bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
groups = {}
for i in range(len(bnd)):
    s = bnd[i]; e = bnd[i+1] if i+1 < len(bnd) else len(cs); groups[cs[s]] = order[s:e]
clean_ents = []
for eid, pos in groups.items():
    pos = pos[~empty[pos]]
    if 3 <= len(pos) <= 8:            # need >=1 query + >=2 gallery, modest size
        clean_ents.append((eid, pos))
rng.shuffle(clean_ents)
clean_ents = clean_ents[:12000]
log(f"  {len(clean_ents):,} clean bystander entities (no empty text, size 3-8)")

q_pos, gal_pos = [], []
for eid, pos in clean_ents:
    pos = pos.copy(); rng.shuffle(pos)
    q_pos.append(pos[0]); gal_pos.extend(pos[1:].tolist())
q_pos = np.array(q_pos); gal_pos = np.array(gal_pos)
Q = np.ascontiguousarray(emb[q_pos]); Qt = torch.tensor(Q, device=DEVICE)
clean_gallery = emb[gal_pos]
n_clean = len(clean_gallery)
log(f"  clean gallery = {n_clean:,} vectors; {len(Q):,} held-out clean queries")

@torch.no_grad()
def exact_topk(gv):
    G = torch.tensor(gv, device=DEVICE); out = np.empty((len(Q), K), np.int64)
    for s in range(0, len(Q), 1024):
        out[s:s+1024] = torch.topk(Qt[s:s+1024] @ G.T, K, 1).indices.cpu().numpy()
    return out

def rec(I, exact_I): return float(np.mean([len(set(a)&set(b)) for a,b in zip(I, exact_I)])/K)

def indeg_max(index, n):
    h = index.hnsw; offs = faiss.vector_to_array(h.offsets); nbr = faiss.vector_to_array(h.neighbors)
    nb0 = h.nb_neighbors(0); deg = np.zeros(n, np.int64)
    for i in range(n):
        seg = nbr[offs[i]:offs[i]+nb0]; seg = seg[(seg>=0)&(seg<n)]; deg[seg]+=1
    return int(deg.max())

# IMPORTANT: ground truth for clean queries is computed on the CLEAN gallery ONLY and
# we keep clean rows as the first n_clean positions in every index, so the gold ids are
# stable. Injecting the sink adds rows AFTER them; a correct ANN must still return the
# clean neighbors. Recall is intersection with the clean-only exact top-k.
exact_clean = exact_topk(clean_gallery)

# ---- 3. inject growing empty-text sink, measure clean-query recall ----
log("=== injecting empty-text sink into a FIXED clean gallery ===")
sink_all = emb[emp_idx]
results = {"sink_geometry": {"cos_to_centroid": cos_to_centroid,
                             "frac_exact_dup_rounded": round(1-len(uniq)/len(V),3)},
           "clean_only_ref": {}, "sweep": {}}
SINK_SIZES = [0, 1000, 5000, 20000, 50000, len(emp_idx)]
for S in SINK_SIZES:
    if S == 0:
        gv = clean_gallery
    else:
        sidx = sink_all if S >= len(sink_all) else sink_all[rng.choice(len(sink_all), S, replace=False)]
        gv = np.concatenate([clean_gallery, sidx]).astype("float32")
    gv = np.ascontiguousarray(gv); n = len(gv)
    h = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction=200; h.add(gv)
    hr = {}
    for ef in EF:
        h.hnsw.efSearch = ef; _, I = h.search(Q, K)
        # restrict to clean-gallery hits when intersecting (sink ids are >= n_clean)
        hr[ef] = round(rec(I, exact_clean), 4)
    im = indeg_max(h, n)
    nlist = max(16, int(np.sqrt(n)))
    iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    iv.train(gv); iv.add(gv); ir = {}
    for npb in NPROBE:
        iv.nprobe = npb; _, I = iv.search(Q, K); ir[npb] = round(rec(I, exact_clean), 4)
    results["sweep"][str(S)] = {"n_total": int(n), "sink_frac": round(S/max(1,n),3),
                                "hnsw": hr, "ivf": ir, "indeg_max": im}
    log(f"  sink={S:>6,} ({results['sweep'][str(S)]['sink_frac']:.0%} of {n:,})  "
        f"clean HNSW ef50={hr[50]:.4f} ef200={hr[200]:.4f}  IVF np64={ir[64]:.4f}  indeg_max={im:,}")

json.dump(results, open(os.path.join(OUT_DIR, "wdc_collateral.json"), "w"), indent=2)
log("saved wdc_collateral.json")

