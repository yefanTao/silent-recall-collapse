import os, json, time, hashlib, pickle
import numpy as np, pandas as pd, faiss
try:
    import hnswlib
except ImportError:
    hnswlib = None

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
CACHE = os.environ.get("CACHE", os.path.join(ROOT, "outputs", "cache"))
K = 10; EPS = 1e-5
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def want(name): return not ONLY or name in ONLY

def row_hashes(X):
    return np.fromiter((hash(r.tobytes()) for r in X), dtype=np.int64, count=len(X))

def indeg_max(h, n):
    hn = h.hnsw
    offs = faiss.vector_to_array(hn.offsets).astype(np.int64)
    nbr = faiss.vector_to_array(hn.neighbors).astype(np.int64)
    nb0 = hn.nb_neighbors(0); deg = np.zeros(n, np.int64)
    idx = (offs[:n, None] + np.arange(nb0)[None, :]).ravel(); seg = nbr[idx]
    seg = seg[(seg >= 0) & (seg < n)]; np.add.at(deg, seg, 1)
    return int(deg.max())

def build_hnsw(x):
    h = faiss.IndexHNSWFlat(x.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200; h.add(x); h.hnsw.efSearch = 200
    return h

def build_ivf(x, nlist_floor):
    nlist = max(nlist_floor, int(np.sqrt(len(x))))
    iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(x.shape[1]), x.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    iv.train(x); iv.add(x); iv.nprobe = 64
    return iv

def recall_variants(Dann, Iann, Dgt, Igt, has_copy):
    k = Igt.shape[1]
    idr = np.array([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(Iann, Igt)]) / k
    tie = (Dann >= (Dgt[:, k - 1] - EPS)[:, None]).sum(1) / k
    out = {"id": round(float(idr.mean()), 4), "tie_aware": round(float(tie.mean()), 4)}
    clean = ~has_copy
    out["id_clean_queries"] = round(float(idr[clean].mean()), 4) if clean.any() else None
    out["id_dup_queries"] = round(float(idr[has_copy].mean()), 4) if has_copy.any() else None
    out["tie_aware_clean_queries"] = round(float(tie[clean].mean()), 4) if clean.any() else None
    out["tie_aware_dup_queries"] = round(float(tie[has_copy].mean()), 4) if has_copy.any() else None
    return out

def measure(x, q, query_in_gallery, nlist_floor=64, with_hnswlib=False):
    x = np.ascontiguousarray(x.astype("float32")); q = np.ascontiguousarray(q.astype("float32"))
    Dgt, Igt = faiss.knn(q, x, K, metric=faiss.METRIC_INNER_PRODUCT)
    gh = row_hashes(x); qh = row_hashes(q)
    cnt = pd.Series(gh).value_counts()
    c = cnt.reindex(qh).fillna(0).to_numpy()
    has_copy = c >= (2 if query_in_gallery else 1)
    h = build_hnsw(x); Dh, Ih = h.search(q, K); im = indeg_max(h, len(x))
    iv = build_ivf(x, nlist_floor); Di, Ii = iv.search(q, K)
    res = {"n": int(len(x)), "n_queries": int(len(q)), "indeg_max": im,
           "frac_queries_with_identical_copy": round(float(has_copy.mean()), 4),
           "hnsw": recall_variants(Dh, Ih, Dgt, Igt, has_copy),
           "ivf": recall_variants(Di, Ii, Dgt, Igt, has_copy)}
    if with_hnswlib and hnswlib is not None:
        p = hnswlib.Index(space="ip", dim=x.shape[1])
        p.init_index(max_elements=len(x), ef_construction=200, M=16, random_seed=0)
        p.add_items(x, np.arange(len(x))); p.set_ef(200)
        L, Dd = p.knn_query(q, k=K)
        res["hnswlib"] = recall_variants(1.0 - Dd, L.astype(np.int64), Dgt, Igt, has_copy)
    return res

def show(tag, r):
    h, i = r["hnsw"], r["ivf"]
    log(f"  {tag:<22} n={r['n']:<8d} indeg={r['indeg_max']:<6d} q_with_copy={r['frac_queries_with_identical_copy']:.3f} | "
        f"HNSW id={h['id']:.3f} tie={h['tie_aware']:.3f} cleanq={h['id_clean_queries']} dupq={h['id_dup_queries']} | "
        f"IVF id={i['id']:.3f} tie={i['tie_aware']:.3f}"
        + (f" | hnswlib id={r['hnswlib']['id']:.3f} tie={r['hnswlib']['tie_aware']:.3f}" if "hnswlib" in r else ""))

results = {"config": {"K": K, "eps": EPS, "hnsw": "M=16 efC=200 efS=200", "ivf": "nprobe=64",
                      "definitions": {"id": "|ANN ids ∩ exact ids|/k",
                                      "tie_aware": "#{returned with score >= k-th exact score - eps}/k",
                                      "clean_queries": "queries with no byte-identical gallery copy"}}}

# ---------------------------------------------------------------- WDC redundancy sweep (Table 1)
if want("wdc_sweep") or want("real_dedup"):
    log("loading WDC ...")
    df = pd.read_pickle(os.path.join(ROOT, "data", "wdc_full", "wdc_full_cleaned.pkl")).reset_index(drop=True)
    emb = np.load(os.path.join(ROOT, "data", "wdc_full", "embeddings.npy")).astype(np.float32)
    faiss.normalize_L2(emb)
    cid = df["cluster_id"].to_numpy()

if want("wdc_sweep"):
    log("=== WDC redundancy sweep (replicates exp_pure_ann.py sampling; held-out sibling queries) ===")
    title = df["title_clean"].fillna("").astype(str).to_numpy()
    rng = np.random.default_rng(42)
    order = np.argsort(cid, kind="stable"); cs = cid[order]
    bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); groups = {}
    for i in range(len(bnd)):
        s = bnd[i]; e = bnd[i + 1] if i + 1 < len(bnd) else len(cs); groups[cs[s]] = order[s:e]
    clean = {}; RMAX = 12
    for eid, pos in groups.items():
        pos = np.array([p for p in pos if title[p].strip() != ""])
        if len(pos) >= 1 + RMAX: clean[eid] = pos
    chosen = rng.choice(np.array(list(clean.keys()), dtype=object), size=min(20000, len(clean)), replace=False)
    ent_q, ent_mem = {}, {}
    for e in chosen:
        pos = clean[e].copy(); rng.shuffle(pos); ent_q[e] = pos[0]; ent_mem[e] = pos[1:]
    ents = list(ent_q.keys()); Q = emb[np.array([ent_q[e] for e in ents])]
    out = []
    for R in [1, 2, 3, 5, 8, 12, "all"]:
        rows = []
        for e in ents:
            mem = ent_mem[e]; rows.extend(mem.tolist() if R == "all" else mem[:R].tolist())
        r = measure(emb[np.array(rows)], Q, query_in_gallery=False, nlist_floor=16)
        r["R"] = str(R); out.append(r); show(f"R={R}", r)
    results["wdc_sweep"] = out

# ---------------------------------------------------------------- WDC real dedup ladder (Table 5)
if want("real_dedup"):
    log("=== WDC dedup ladder (replicates exp_real_dedup.py sampling; held-out real-offer queries) ===")
    rng = np.random.default_rng(42)
    uniq_c = np.unique(cid); rng.shuffle(uniq_c)
    keep_rows, total = [], 0
    for c in uniq_c:
        rows = np.flatnonzero(cid == c); keep_rows.append(rows); total += len(rows)
        if total >= 200000: break
    keep = np.concatenate(keep_rows); e2 = emb[keep]
    qsel = rng.choice(len(e2), 5000, replace=False)
    qmask = np.zeros(len(e2), bool); qmask[qsel] = True
    Q = e2[qsel]; gal = e2[~qmask]
    out = []
    r = measure(gal, Q, query_in_gallery=False, with_hnswlib=True); r["treatment"] = "raw"; out.append(r); show("raw", r)
    _, uidx = np.unique(gal, axis=0, return_index=True); gal2 = gal[np.sort(uidx)]
    r = measure(gal2, Q, query_in_gallery=False, with_hnswlib=True); r["treatment"] = "exact np.unique"; out.append(r); show("exact np.unique", r)
    results["real_dedup"] = out

# ---------------------------------------------------------------- WDC-Computers (Table 6)
if want("computers"):
    log("=== WDC-Computers (replicates exp_computers.py; queries drawn from the gallery) ===")
    dfc = pickle.load(open(os.path.join(ROOT, "data", "wdc_products", "computers_cleaned.pkl"), "rb")).reset_index(drop=True)
    text = (dfc.title_clean.fillna("") + " " + dfc.description_clean.fillna("")).str.strip().tolist()
    cluster = dfc.cluster_id.to_numpy()
    X = np.ascontiguousarray(np.load(os.path.join(ROOT, "data", "wdc_products", "computers_minilm.npy")).astype("float32"))
    N = len(X)
    def qs(x):
        rng = np.random.default_rng(0); return x[rng.choice(len(x), min(5000, len(x)), replace=False)]
    by_ent = {}
    for i in range(N): by_ent.setdefault(cluster[i], []).append(i)
    out = []
    for R in [1, 8, 0]:
        keep = np.arange(N) if R == 0 else np.array([i for ids in by_ent.values() for i in ids[:R]])
        xR = X[keep]; r = measure(xR, qs(xR), query_in_gallery=True); r["R"] = "all" if R == 0 else R
        out.append(r); show(f"sweep R={r['R']}", r)
    rng = np.random.default_rng(0); sub = rng.choice(N, min(195000, N), replace=False)
    Xs = X[sub]; txt_s = [text[i] for i in sub]
    r = measure(Xs, qs(Xs), query_in_gallery=True); r["treatment"] = "raw"; out.append(r); show("triad raw", r)
    seen = set(); keep_idx = []
    for j, t in enumerate(txt_s):
        hh = hashlib.md5(t.encode("utf-8", "ignore")).hexdigest()
        if hh not in seen: seen.add(hh); keep_idx.append(j)
    Xu = Xs[np.array(keep_idx)]
    r = measure(Xu, qs(Xu), query_in_gallery=True); r["treatment"] = "exact np.unique"; out.append(r); show("triad np.unique", r)
    results["computers"] = out

# ---------------------------------------------------------------- MS MARCO (Table 7, natural part)
if want("msmarco"):
    log("=== MS MARCO natural duplication (replicates exp_msmarco.py; queries drawn from the gallery) ===")
    X = np.ascontiguousarray(np.load(os.path.join(ROOT, "data", "msmarco", "msmarco_minilm.npy")).astype("float32"))
    hashes = np.load(os.path.join(ROOT, "data", "msmarco", "msmarco_hashes.npy")); N = len(X)
    def qs(x):
        rng = np.random.default_rng(0); return x[rng.choice(len(x), min(5000, len(x)), replace=False)]
    by_grp = {}
    for i in range(N): by_grp.setdefault(int(hashes[i]), []).append(i)
    out = []
    for R in [1, 0]:
        keep = np.arange(N) if R == 0 else np.array([i for ids in by_grp.values() for i in ids[:R]])
        xR = X[keep]; r = measure(xR, qs(xR), query_in_gallery=True); r["R"] = "all" if R == 0 else R
        out.append(r); show(f"sweep R={r['R']}", r)
    rng = np.random.default_rng(0); sub = rng.choice(N, min(195000, N), replace=False)
    Xs = X[sub]; hs = hashes[sub]
    r = measure(Xs, qs(Xs), query_in_gallery=True); r["treatment"] = "raw"; out.append(r); show("triad raw", r)
    _, keep_u = np.unique(hs, return_index=True); Xu = Xs[np.sort(keep_u)]
    r = measure(Xu, qs(Xu), query_in_gallery=True); r["treatment"] = "exact np.unique"; out.append(r); show("triad np.unique", r)
    results["msmarco"] = out

# ---------------------------------------------------------------- embedding-model invariance (Table 8)
if want("embedding_models"):
    log("=== embedding-model invariance (cached embeddings of exp_embedding_interaction.py; queries from gallery) ===")
    out = []
    for name in ["minilm_l6", "bge_small", "bge_base", "e5_base", "gte_base"]:
        cf = os.path.join(CACHE, f"embint_wdc_n120000_s0_{name}.npy")
        if not os.path.exists(cf):
            raise FileNotFoundError(
                f"{cf}: run exp_embedding_interaction.py with its default "
                "N=120000 and SEED=0 before requesting embedding_models")
        X = np.ascontiguousarray(np.load(cf).astype("float32"))
        q = X[np.random.default_rng(0).choice(len(X), min(5000, len(X)), replace=False)]
        r = measure(X, q, query_in_gallery=True); r["model"] = name; r["treatment"] = "raw"; out.append(r); show(f"{name} raw", r)
        _, keep = np.unique(X, axis=0, return_index=True); Xu = X[np.sort(keep)]
        q = Xu[np.random.default_rng(0).choice(len(Xu), min(5000, len(Xu)), replace=False)]
        r = measure(Xu, q, query_in_gallery=True); r["model"] = name; r["treatment"] = "exact np.unique"; out.append(r); show(f"{name} np.unique", r)
    results["embedding_models"] = out


# ---------------------------------------------------------------- WDC controls (4 pooled conditions)
if want("wdc_controls"):
    log("=== WDC controls (replicates exp_wdc_controls.py; held-out sibling queries) ===")
    if "df" not in dir():
        df = pd.read_pickle(os.path.join(ROOT, "data", "wdc_full", "wdc_full_cleaned.pkl")).reset_index(drop=True)
        emb = np.load(os.path.join(ROOT, "data", "wdc_full", "embeddings.npy")).astype(np.float32)
        faiss.normalize_L2(emb); cid = df["cluster_id"].to_numpy()
    combo = (df["title_clean"].fillna("") + df["brand_clean"].fillna("") + df["description_clean"].fillna("")).str.strip()
    nonempty = (combo != "").to_numpy()
    def condition(keep_mask, cap):
        idx = np.flatnonzero(keep_mask); c_ = cid[idx]; e_ = emb[idx]
        order = np.argsort(c_, kind="stable"); cs = c_[order]
        bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); groups = {}
        for i in range(len(bnd)):
            s = bnd[i]; e = bnd[i + 1] if i + 1 < len(bnd) else len(cs); groups[cs[s]] = order[s:e]
        rng = np.random.default_rng(42)
        ents = [c for c, p in groups.items() if len(p) >= 2]; rng.shuffle(ents); ents = ents[:20000]
        ent_q, ent_mem = {}, {}
        for c in ents:
            p = groups[c].copy(); rng.shuffle(p); ent_q[c] = p[0]; mem = p[1:]
            if cap: mem = mem[:cap]
            ent_mem[c] = mem
        Q = e_[np.array([ent_q[c] for c in ents])]; gal = e_[np.concatenate([ent_mem[c] for c in ents])]
        return gal, Q
    out = []
    for name, mask, cap in [("orig_all", np.ones(len(df), bool), None), ("no_empty", nonempty, None),
                            ("no_empty_cap", nonempty, 50), ("cap_only", np.ones(len(df), bool), 50)]:
        gal, Q = condition(mask, cap)
        r = measure(gal, Q, query_in_gallery=False, nlist_floor=16); r["condition"] = name; out.append(r); show(name, r)
    results["wdc_controls"] = out

OUTNAME = os.environ.get("OUTNAME", "tie_robustness.json")
json.dump(results, open(os.path.join(OUT_DIR, OUTNAME), "w"), indent=2)
log(f"saved {OUTNAME}")
