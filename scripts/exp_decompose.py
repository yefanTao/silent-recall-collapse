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
EF_SWEEP = [16, 32, 64, 128, 256]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("Loading WDC-clean (same construction as exp_pareto.py) ...")
df = pd.read_pickle(os.path.join(DATA_DIR, "wdc_full_cleaned.pkl")).reset_index(drop=True)
emb = np.load(os.path.join(DATA_DIR, "embeddings.npy")).astype(np.float32)
faiss.normalize_L2(emb)
combo = (df["title_clean"].fillna("") + df["brand_clean"].fillna("")
         + df["description_clean"].fillna("")).str.strip()
keep = (combo != "").to_numpy()
emb = emb[keep]; cid = df["cluster_id"].to_numpy()[keep]
d = emb.shape[1]
log(f"  {len(emb):,} clean offers, {len(np.unique(cid)):,} entities")

rng = np.random.default_rng(SEED)
order = np.argsort(cid, kind="stable"); cs = cid[order]
bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); groups = {}
for i in range(len(bnd)):
    s = bnd[i]; e = bnd[i+1] if i+1 < len(bnd) else len(cs); groups[cs[s]] = order[s:e]
q_rows, gal_rows, gal_cids = [], [], []
N_Q = 20000
multi = [(c, p) for c, p in groups.items() if len(p) >= 2]
rng.shuffle(multi)
for c, p in multi:
    p = p.copy(); rng.shuffle(p)
    q_rows.append(p[0]); gal_rows.extend(p[1:].tolist()); gal_cids.extend([c] * (len(p) - 1))
for c, p in groups.items():
    if len(p) == 1:
        gal_rows.append(p[0]); gal_cids.append(c)
q_rows = np.array(q_rows[:N_Q])
Q = np.ascontiguousarray(emb[q_rows]); q_cid = cid[q_rows]
gal_rows = np.array(gal_rows); gal_cids = np.array(gal_cids)
Xfull = np.ascontiguousarray(emb[gal_rows]); n_full = len(Xfull)
log(f"  queries={len(Q):,} FULL gallery={n_full:,} ({len(np.unique(gal_cids)):,} entities)")

def entity_recall(returned_cids_2d, k=K):
    return float(np.mean([q_cid[i] in set(returned_cids_2d[i, :k]) for i in range(len(q_cid))]))

# ---- Reduced gallery (centroid per entity) ----
reduced, out_ids, id_map, _ = group_reduce(Xfull, gal_cids, method="centroid",
                                           metric=faiss.METRIC_INNER_PRODUCT)
n_red = len(reduced)
log(f"  REDUCED gallery={n_red:,} ({n_full/n_red:.1f}x fewer)")

# ---- R_fe: Full gallery, EXACT (perfect navigation) ----
log("R_fe Full-EXACT ...")
flat_f = faiss.IndexFlatIP(d); flat_f.add(Xfull)
_, I = flat_f.search(Q, K); R_fe = entity_recall(gal_cids[I])
log(f"  R_fe = {R_fe:.4f}")

# ---- R_re: Reduced gallery, EXACT (perfect navigation) ----
log("R_re Reduced-EXACT ...")
flat_r = faiss.IndexFlatIP(d); flat_r.add(reduced)
_, I = flat_r.search(Q, K); R_re = entity_recall(id_map[I])
log(f"  R_re = {R_re:.4f}")

# ---- R_fh: Full gallery, HNSW (sweep efSearch, report best) ----
log("R_fh Full-HNSW ...")
hf = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); hf.hnsw.efConstruction = 200
hf.add(Xfull)
fh = []
for ef in EF_SWEEP:
    hf.hnsw.efSearch = ef; _, I = hf.search(Q, K); fh.append(entity_recall(gal_cids[I]))
R_fh = max(fh); log(f"  R_fh = {R_fh:.4f} (best over ef {EF_SWEEP}; all={['%.3f'%x for x in fh]})")

# ---- R_rh: Reduced gallery, HNSW ----
log("R_rh Reduced-HNSW ...")
hr = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); hr.hnsw.efConstruction = 200
hr.add(reduced)
rh = []
for ef in EF_SWEEP:
    hr.hnsw.efSearch = ef; _, I = hr.search(Q, K); rh.append(entity_recall(id_map[I]))
R_rh = max(rh); log(f"  R_rh = {R_rh:.4f} (best over ef {EF_SWEEP}; all={['%.3f'%x for x in rh]})")

# ---- decomposition ----
nav_full = R_fe - R_fh        # navigation loss on hub-laden full gallery (the in-degree law)
nav_red  = R_re - R_rh        # navigation loss on reduced gallery (expect ~0)
slot     = R_re - R_fe        # slot efficiency at perfect navigation
total    = R_rh - R_fh        # deployed gain
out = {
    "config": {"corpus": "WDC-clean (full)", "n_full": int(n_full), "n_reduced": int(n_red),
               "n_queries": int(len(Q)), "K": K, "ef_sweep": EF_SWEEP, "metric": "entity recall@10"},
    "measurements": {"R_fh_full_hnsw": round(R_fh, 4), "R_fe_full_exact": round(R_fe, 4),
                     "R_rh_reduced_hnsw": round(R_rh, 4), "R_re_reduced_exact": round(R_re, 4)},
    "decomposition": {
        "deployed_gain_Rrh_minus_Rfh": round(total, 4),
        "navigation_loss_full_gallery": round(nav_full, 4),
        "navigation_loss_reduced_gallery": round(nav_red, 4),
        "slot_efficiency_at_perfect_nav": round(slot, 4),
        "identity_check_slot+navfull-navred": round(slot + nav_full - nav_red, 4),
        "navigation_share_of_gain": round((nav_full - nav_red) / total, 3) if abs(total) > 1e-9 else None,
        "slot_share_of_gain": round(slot / total, 3) if abs(total) > 1e-9 else None,
    },
    "interpretation": (
        "Decomposes the deployed entity-recall gain of Reduced-HNSW over Full-HNSW. "
        "navigation_loss_full is what the in-degree law costs Full-HNSW (HNSW vs exact on the "
        "hub-laden full gallery); navigation_loss_reduced should be ~0 (the reduced gallery has "
        "no mega-hub); slot_efficiency is the cardinality effect at perfect navigation. The split "
        "shows honestly how much of §7's headline gain is navigation (the paper's mechanism) vs "
        "slot/cardinality, on the real corpus -- answering the obvious reviewer question.")
}
json.dump(out, open(os.path.join(OUT_DIR, "decompose.json"), "w"), indent=2)
log("saved outputs/validation/decompose.json")
print(json.dumps(out["measurements"], indent=2))
print(json.dumps(out["decomposition"], indent=2))

