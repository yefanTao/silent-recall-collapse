import os, time, json
import numpy as np, pandas as pd, torch

DATASET  = os.environ.get("DATASET", "wdc").lower()
N_TEST   = int(os.environ.get("N_TEST", "6000"))
N_Q      = int(os.environ.get("N_Q", "3"))
MAX_POOL = int(os.environ.get("MAX_POOL", "50"))
SEED     = int(os.environ.get("SEED", "42"))
GPU      = os.environ.get("GPU", "0")
DEVICE   = f"cuda:{GPU}" if torch.cuda.is_available() else "cpu"

ROOT    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "validation")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

# (name, hf_id, query_prefix, doc_prefix) -- e5 REQUIRES query:/passage:; others none.
MODELS = [
    ("minilm_l6",  "sentence-transformers/all-MiniLM-L6-v2", "", ""),
    ("bge_small",  "BAAI/bge-small-en-v1.5",                 "", ""),
    ("bge_base",   "BAAI/bge-base-en-v1.5",                  "", ""),
    ("e5_base",    "intfloat/e5-base-v2",                    "query: ", "passage: "),
    ("gte_base",   "thenlper/gte-base",                      "", ""),
]
SPACE_NAMES = [m[0] for m in MODELS]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] [{DATASET}] {m}", flush=True)

# ---------------------------------------------------------------- load dataset
def load_dataset():
    if DATASET == "wdc":
        df = pd.read_pickle(os.path.join(ROOT, "data", "wdc_full", "wdc_full_cleaned.pkl")).reset_index(drop=True)
        text = (df["title_clean"].fillna("") + " " + df["brand_clean"].fillna("") + " "
                + df["description_clean"].fillna("")).str.strip().to_numpy()
        cid = df["cluster_id"].to_numpy()
    elif DATASET == "musicbrainz":
        df = pd.read_pickle(os.path.join(ROOT, "data", "musicbrainz", "musicbrainz_clean.pkl")).reset_index(drop=True)
        text = df["text"].fillna("").astype(str).str.strip().to_numpy()
        cid = df["cluster_id"].astype(str).to_numpy()
    else:
        raise ValueError(DATASET)
    return text, cid

log(f"loading (N_TEST={N_TEST}, N_Q={N_Q}, MAX_POOL={MAX_POOL}, device={DEVICE})")
text, cid = load_dataset()
log(f"  {len(text):,} rows, {pd.Series(cid).nunique():,} entities")

# ---------------------------------------------------------------- test split
rng = np.random.default_rng(SEED)
order = np.argsort(cid, kind="stable"); cs = cid[order]
bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
groups = {}
for i in range(len(bnd)):
    s = bnd[i]; e = bnd[i+1] if i+1 < len(bnd) else len(cs)
    groups[cs[s]] = order[s:e]
eligible = [k for k, p in groups.items() if len(p) >= 5]
log(f"  {len(eligible):,} eligible entities (>=5 rows)")
chosen = rng.choice(np.array(eligible, dtype=object), size=min(N_TEST, len(eligible)), replace=False)

ent_query, ent_pool = {}, {}
for k in chosen:
    pos = groups[k].copy(); rng.shuffle(pos)
    if len(pos) > MAX_POOL: pos = pos[:MAX_POOL]
    q, p = pos[:N_Q], pos[N_Q:]
    if len(p) == 0: continue
    ent_query[k], ent_pool[k] = q, p
ents = list(ent_query.keys()); eid_to_int = {k: i for i, k in enumerate(ents)}
Nent = len(ents)
q_pos = np.concatenate([ent_query[k] for k in ents])
q_eid = np.concatenate([np.full(len(ent_query[k]), eid_to_int[k]) for k in ents])
ent_qrows = {}; off = 0
for k in ents:
    nq = len(ent_query[k]); ent_qrows[eid_to_int[k]] = np.arange(off, off+nq); off += nq
pool_rows = np.unique(np.concatenate([ent_pool[k] for k in ents]))
row_to_pidx = {int(r): j for j, r in enumerate(pool_rows)}
log(f"  {Nent:,} test entities; {len(q_pos):,} queries; {len(pool_rows):,} pool rows")

# ---------------------------------------------------------------- encode all spaces
from sentence_transformers import SentenceTransformer
@torch.no_grad()
def encode(model, rows, prefix):
    txt = [prefix + text[i] for i in rows]
    v = model.encode(txt, batch_size=1024, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
    return v.astype("float32")

# cache keyed by the deterministic split signature -> re-runs skip re-encoding
CACHE = os.path.join(ROOT, "outputs", "cache")
os.makedirs(CACHE, exist_ok=True)
sig = f"{DATASET}_n{N_TEST}_q{N_Q}_p{MAX_POOL}_s{SEED}"
doc_emb, qry_emb = {}, {}   # space -> (npool, d) / (nQ, d)
for name, hf, qp, dp in MODELS:
    dc = os.path.join(CACHE, f"c4_{sig}_{name}_doc.npy")
    qc = os.path.join(CACHE, f"c4_{sig}_{name}_qry.npy")
    if os.path.exists(dc) and os.path.exists(qc):
        doc_emb[name] = np.load(dc); qry_emb[name] = np.load(qc)
        log(f"  cached space {name}")
        continue
    log(f"  encoding space {name} ({hf}) ...")
    m = SentenceTransformer(hf, device=DEVICE)
    doc_emb[name] = encode(m, pool_rows, dp)
    qry_emb[name] = encode(m, q_pos, qp)
    np.save(dc, doc_emb[name]); np.save(qc, qry_emb[name])
    del m; torch.cuda.empty_cache()
log("  all spaces encoded")

# ---- quantify the STRUCTURAL prerequisite for C4b: intra-entity heterogeneity ----
# (mean pairwise cosine among an entity's pool vectors; LOW dispersion -> no centrality signal possible)
def intra_entity_dispersion(space):
    disp = []
    for k in ents:
        pool = ent_pool[k]
        if len(pool) < 2: continue
        M = np.stack([doc_emb[space][row_to_pidx[int(p)]] for p in pool])
        sim = M @ M.T; iu = np.triu_indices(len(pool), 1)
        disp.append(1.0 - float(sim[iu].mean()))   # 1 - mean cosine = dispersion
    return float(np.mean(disp)), float(np.median(disp))
disp_mean, disp_med = intra_entity_dispersion("minilm_l6")
log(f"  intra-entity dispersion (1 - mean pairwise cosine, minilm space): mean={disp_mean:.4f} median={disp_med:.4f}")

# ---------------------------------------------------------------- entity-level search in a given eval space
def make_search(E):
    Vpool = torch.tensor(doc_emb[E], device=DEVICE)        # (npool, dE)
    Q     = torch.tensor(qry_emb[E], device=DEVICE)        # (nQ, dE)
    @torch.no_grad()
    def search_rows(rep_rows, qchunk=256, k=10):
        """rep_rows: list of (pool_row, eid). Build index of those vectors, entity-level max-IP."""
        idx = torch.tensor(np.stack([doc_emb[E][row_to_pidx[int(r)]] for r, _ in rep_rows]), device=DEVICE)
        eid_t = torch.tensor([e for _, e in rep_rows], device=DEVICE, dtype=torch.long)
        out = np.empty((Q.shape[0], k), np.int64)
        for s in range(0, Q.shape[0], qchunk):
            qb = Q[s:s+qchunk]
            sc = qb @ idx.T
            es = torch.full((qb.shape[0], Nent), float("-inf"), device=DEVICE)
            es.scatter_reduce_(1, eid_t.unsqueeze(0).expand(qb.shape[0], -1), sc, reduce="amax", include_self=True)
            out[s:s+qb.shape[0]] = torch.topk(es, min(k, Nent), dim=1).indices.cpu().numpy()
        return out
    return search_rows, Q, Vpool

def hit1(retr): return (retr[:, :1] == q_eid[:, None]).any(1)
def recall_at(retr, kk): return float((retr[:, :kk] == q_eid[:, None]).any(1).mean())

# ---------------------------------------------------------------- selectors (return pool_row per entity)
def medoid_row(pool, space):
    if len(pool) == 1: return pool[0]
    M = np.stack([doc_emb[space][row_to_pidx[int(p)]] for p in pool])
    sim = M @ M.T; np.fill_diagonal(sim, 0.0)
    return pool[int(sim.mean(1).argmax())]

tie = np.random.default_rng(SEED + 7)
random_rep = {eid_to_int[k]: ent_pool[k][tie.integers(len(ent_pool[k]))] for k in ents}   # fixed across eval spaces

def oracle_rows(E):
    """Per entity, holding others at minilm medoid, pick pool row winning most rank-1 in space E."""
    search_rows, Q, _ = make_search(E)
    # distractor base = medoid in minilm_l6 (fixed reference), per-query max over OTHER entities
    base_reps = [(medoid_row(ent_pool[k], "minilm_l6"), eid_to_int[k]) for k in ents]
    base_idx = torch.tensor(np.stack([doc_emb[E][row_to_pidx[int(r)]] for r, _ in base_reps]), device=DEVICE)
    base_eid = torch.tensor([e for _, e in base_reps], device=DEVICE, dtype=torch.long)
    thr = np.empty(Q.shape[0], np.float32)
    with torch.no_grad():
        for s in range(0, Q.shape[0], 256):
            qb = Q[s:s+256]
            es = torch.full((qb.shape[0], Nent), float("-inf"), device=DEVICE)
            es.scatter_reduce_(1, base_eid.unsqueeze(0).expand(qb.shape[0], -1), qb @ base_idx.T,
                               reduce="amax", include_self=True)
            for r in range(qb.shape[0]):
                ie = q_eid[s+r]; row = es[r].clone(); row[ie] = float("-inf")
                thr[s+r] = float(row.max().cpu())
    chosen_rep = {}
    for k in ents:
        ie = eid_to_int[k]; pool = ent_pool[k]; qr = ent_qrows[ie]
        P = torch.tensor(np.stack([doc_emb[E][row_to_pidx[int(p)]] for p in pool]), device=DEVICE)
        Qe = Q[qr]
        sims = (Qe @ P.T).cpu().numpy()
        wins = (sims > thr[qr][:, None]).sum(0)
        best = np.flatnonzero(wins == wins.max())
        best = best[int(np.argmax(sims[:, best].mean(0)))] if len(best) > 1 else best[0]
        chosen_rep[ie] = pool[best]
    return chosen_rep

# ---------------------------------------------------------------- run the matrix
def reps_full():     return [(p, eid_to_int[k]) for k in ents for p in ent_pool[k]]
def reps_random():   return [(random_rep[eid_to_int[k]], eid_to_int[k]) for k in ents]
def reps_medoid(S):  return [(medoid_row(ent_pool[k], S), eid_to_int[k]) for k in ents]

results = {"config": {"dataset": DATASET, "N_TEST": N_TEST, "N_Q": N_Q, "MAX_POOL": MAX_POOL,
                      "SEED": SEED, "n_entities": Nent, "n_queries": int(len(q_eid)),
                      "spaces": SPACE_NAMES, "ground_truth": "cluster_id",
                      "search": "exact entity-level max-IP"},
           "per_eval": {}, "matrix_R@1": {}, "hit1": {}}
hit1_store = {}

log("=== EVAL x SELECTION matrix ===")
for E in SPACE_NAMES:
    search_rows, _, _ = make_search(E)
    per = {}
    # references
    for name, reps in [("full", reps_full()), ("random", reps_random())]:
        retr = search_rows(reps)
        per[name] = {"R@1": recall_at(retr,1), "R@10": recall_at(retr,10), "n_index": len(reps)}
        hit1_store[(E,name)] = hit1(retr)
    # centroid (synthetic mean vector per entity, in space E)
    cen_idx, cen_eid = [], []
    for k in ents:
        v = doc_emb[E][[row_to_pidx[int(p)] for p in ent_pool[k]]].mean(0)
        n = np.linalg.norm(v); cen_idx.append(v/n if n>0 else v); cen_eid.append(eid_to_int[k])
    with torch.no_grad():
        Q = torch.tensor(qry_emb[E], device=DEVICE); idx = torch.tensor(np.stack(cen_idx), device=DEVICE)
        cretr = np.empty((Q.shape[0],10), np.int64)
        for s in range(0,Q.shape[0],256):
            sc = Q[s:s+256] @ idx.T
            cretr[s:s+256] = torch.topk(sc, min(10,len(cen_idx)), dim=1).indices.cpu().numpy()
    per["centroid"] = {"R@1": recall_at(cretr,1), "R@10": recall_at(cretr,10), "n_index": len(cen_idx)}
    hit1_store[(E,"centroid")] = hit1(cretr)
    # oracle (peeks queries in E)
    orep = oracle_rows(E); retr = search_rows([(orep[eid_to_int[k]], eid_to_int[k]) for k in ents])
    per["oracle"] = {"R@1": recall_at(retr,1), "R@10": recall_at(retr,10), "n_index": Nent}
    hit1_store[(E,"oracle")] = hit1(retr)
    # medoid in each selection space S
    row = {}
    for S in SPACE_NAMES:
        retr = search_rows(reps_medoid(S))
        r1 = recall_at(retr,1); row[S] = r1
        hit1_store[(E,f"medoid_{S}")] = hit1(retr)
    results["matrix_R@1"][E] = row
    per["medoid"] = row
    results["per_eval"][E] = per
    diag = row[E]; offs = [row[S] for S in SPACE_NAMES if S != E]
    log(f"  EVAL={E:10s} full={per['full']['R@1']:.4f} oracle={per['oracle']['R@1']:.4f} "
        f"centroid={per['centroid']['R@1']:.4f} random={per['random']['R@1']:.4f} | "
        f"medoid diag={diag:.4f} off-mean={np.mean(offs):.4f}")

# ---------------------------------------------------------------- significance: every off-diagonal medoid vs random
from scipy.stats import binomtest
def mcnemar(a, b):
    a, b = a.astype(bool), b.astype(bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum()); n = n01+n10
    if n == 0: return 0.0, 1.0
    return float(a.mean()-b.mean()), float(binomtest(n10, n, 0.5).pvalue)

offdiag_tests = []
for E in SPACE_NAMES:
    rh = hit1_store[(E,"random")]
    for S in SPACE_NAMES:
        if S == E: continue
        d, p = mcnemar(hit1_store[(E,f"medoid_{S}")], rh)
        offdiag_tests.append({"eval": E, "select": S, "dR@1": round(d,4), "p": float(f"{p:.2e}"),
                              "beats_random": bool(d > 0 and p < 1e-3)})
n_off = len(offdiag_tests); n_win = sum(t["beats_random"] for t in offdiag_tests)

# C4a per eval space: does oracle SIGNIFICANTLY exceed full? (the right test: oracle should NOT)
c4a = {}
for E in SPACE_NAMES:
    d, p = mcnemar(hit1_store[(E,"oracle")], hit1_store[(E,"full")])
    c4a[E] = {"oracle": results["per_eval"][E]["oracle"]["R@1"], "full": results["per_eval"][E]["full"]["R@1"],
              "oracle_minus_full": round(d,4), "p": float(f"{p:.2e}"),
              # Legacy field name; this comparison does not certify lossless/lossy compression.
              "oracle_sig_beats_full": bool(d > 0 and p < 0.05)}

verdict = {
    "C4a_compression_is_lossy": {
        "per_eval": c4a,
        "holds_everywhere": all(not v["oracle_sig_beats_full"] for v in c4a.values()),
        "note": "Legacy field names retained for compatibility. The query-informed selector is not "
                "an upper bound. Failure to reject a difference is not equivalence, and these "
                "comparisons do not establish that all single-representative methods are lossy.",
    },
    "C4b_noncircular_centrality": {
        "n_offdiagonal_pairs": n_off, "n_beats_random_p<1e-3": n_win,
        "fraction": round(n_win/max(n_off,1),3),
        "robust": bool(n_win == n_off),
        "offdiag_tests": offdiag_tests,
        "note": "Query-level exploratory McNemar tests. Three queries share each entity, and "
                "matrix cells share queries/models. The legacy robust field is not an entity-level "
                "uncertainty estimate or evidence of universal transfer.",
    },
}
verdict["intra_entity_dispersion"] = {"mean": round(disp_mean,4), "median": round(disp_med,4),
    "note": "1 - mean pairwise cosine among an entity's pool vectors (MiniLM space). Descriptive "
            "within-entity variation; two datasets do not isolate it as the causal mechanism."}
results["verdict"] = verdict
fp = os.path.join(OUT_DIR, f"c4_generalize_{DATASET}.json")
json.dump(results, open(fp, "w"), indent=2)
log(f"saved {fp}")
print(f"\n=== C4 GENERALIZE VERDICT [{DATASET}] ===")
print(f"  intra-entity dispersion (1 - mean pairwise cos): mean={disp_mean:.4f} median={disp_med:.4f}")
print(f"  Legacy C4a flag (query-informed heuristic comparison only): {verdict['C4a_compression_is_lossy']['holds_everywhere']}")
print(f"  C4b (non-circular centrality): {n_win}/{n_off} off-diagonal pairs beat random at p<1e-3  -> robust={verdict['C4b_noncircular_centrality']['robust']}")

# ---------------------------------------------------------------- heatmap
