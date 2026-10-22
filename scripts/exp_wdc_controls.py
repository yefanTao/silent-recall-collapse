import os, time, json
import numpy as np, pandas as pd, torch, faiss

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wdc_full")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42; DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"; K = 10; CAP = 50
EF_LIST = [50, 200]; NPROBE = [16, 64]

def ts(): return time.strftime("%H:%M:%S")
def log(m): print(f"[{ts()}] {m}", flush=True)

log("Loading WDC ...")
df = pd.read_pickle(os.path.join(DATA_DIR, "wdc_full_cleaned.pkl")).reset_index(drop=True)
emb_all = np.load(os.path.join(DATA_DIR, "embeddings.npy")).astype(np.float32)
faiss.normalize_L2(emb_all)
combo = (df["title_clean"].fillna("") + df["brand_clean"].fillna("")
         + df["description_clean"].fillna("")).str.strip()
nonempty = (combo != "").to_numpy()
cid_all = df["cluster_id"].to_numpy()
log(f"  {len(df):,} offers, {nonempty.mean():.1%} non-empty text")

def run_condition(name, keep_mask, cap=None):
    idx = np.flatnonzero(keep_mask)
    cid = cid_all[idx]; emb = emb_all[idx]
    # group
    order = np.argsort(cid, kind="stable"); cs = cid[order]
    bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); groups = {}
    for i in range(len(bnd)):
        s = bnd[i]; e = bnd[i+1] if i+1 < len(bnd) else len(cs); groups[cs[s]] = order[s:e]
    rng = np.random.default_rng(SEED)
    ents = [c for c, p in groups.items() if len(p) >= 2]
    rng.shuffle(ents); ents = ents[:20000]
    ent_q, ent_mem = {}, {}
    for c in ents:
        p = groups[c].copy(); rng.shuffle(p)
        ent_q[c] = p[0]
        mem = p[1:]
        if cap: mem = mem[:cap]
        ent_mem[c] = mem
    q_pos = np.array([ent_q[c] for c in ents])
    Q = emb[q_pos]; Qt = torch.tensor(Q, device=DEVICE)
    d = emb.shape[1]
    gal = np.concatenate([ent_mem[c] for c in ents])
    gv = emb[gal]; n = len(gv); avgR = n/len(ents)

    # exact
    G = torch.tensor(gv, device=DEVICE); exact_I = np.empty((len(Q), K), np.int64)
    for s in range(0, len(Q), 1024):
        sc = Qt[s:s+1024] @ G.T
        exact_I[s:s+1024] = torch.topk(sc, K, 1).indices.cpu().numpy()
    def rec(I): return float(np.mean([len(set(a)&set(b)) for a,b in zip(I, exact_I)])/K)

    h = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction=200; h.add(gv)
    hr = {}
    for ef in EF_LIST:
        h.hnsw.efSearch = ef; _, I = h.search(Q, K); hr[ef] = round(rec(I), 4)
    nlist = max(16, int(np.sqrt(n))); iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    iv.train(gv); iv.add(gv); ir = {}
    for npb in NPROBE:
        iv.nprobe = npb; _, I = iv.search(Q, K); ir[npb] = round(rec(I), 4)
    # max indegree (cheap-ish via faiss arrays)
    offs = faiss.vector_to_array(h.hnsw.offsets); nbr = faiss.vector_to_array(h.hnsw.neighbors)
    nb0 = h.hnsw.nb_neighbors(0); indeg = np.zeros(n, np.int64)
    step = max(1, n // 100000)  # subsample for speed if huge
    for i in range(0, n, 1):
        seg = nbr[offs[i]:offs[i]+nb0]
        seg = seg[(seg >= 0) & (seg < n)]
        indeg[seg] += 1
    res = {"avg_vectors_per_entity": round(avgR,2), "index_size": int(n),
           "hnsw": hr, "ivf": ir, "indeg_max": int(indeg.max())}
    log(f"  [{name}] n={n:,} avgR={avgR:.1f}  HNSW ef50={hr[50]:.4f} ef200={hr[200]:.4f}  "
        f"IVF np64={ir[64]:.4f}  indeg_max={res['indeg_max']:,}")
    return res

results = {}
log("=== WDC controls (pure-ANN, all members per entity = max redundancy) ===")
results["orig_all"]      = run_condition("orig (all)", np.ones(len(df), bool), cap=None)
results["no_empty"]      = run_condition("no-empty-text", nonempty, cap=None)
results["no_empty_cap"]  = run_condition(f"no-empty + cap{CAP}", nonempty, cap=CAP)
results["cap_only"]      = run_condition(f"cap{CAP} only", np.ones(len(df), bool), cap=CAP)

json.dump(results, open(os.path.join(OUT_DIR, "wdc_controls.json"), "w"), indent=2)
log("saved wdc_controls.json")
log("INTERPRETATION:")
o = results["orig_all"]["hnsw"]["50"]; a = results["no_empty"]["hnsw"]["50"]; b = results["no_empty_cap"]["hnsw"]["50"]
log(f"  HNSW ef50 pure-ANN recall:  orig={o:.3f}  no-empty={a:.3f}  no-empty+cap={b:.3f}")
log(f"  -> if no-empty+cap stays LOW, redundancy is the real cause; if it RECOVERS, §17 collapse was a confound.")
