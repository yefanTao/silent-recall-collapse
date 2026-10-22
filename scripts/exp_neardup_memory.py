import os, json, time
import numpy as np, faiss

D       = int(os.environ.get("D", "128"))
RANK    = int(os.environ.get("RANK", "16"))
NGROUP  = int(os.environ.get("NGROUP", "20000"))
R       = int(os.environ.get("R", "8"))
NCLEAN  = int(os.environ.get("NCLEAN", "50000"))
COS     = float(os.environ.get("COS", "0.99"))
K       = int(os.environ.get("K", "10"))
SEED    = int(os.environ.get("SEED", "0"))
NQ      = int(os.environ.get("NQ", "5000"))
ROOT    = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG_DIR, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype("float32")
def rec(I, eI): return float(np.mean([len(set(a) & set(b)) for a, b in zip(I, eI)]) / K)

def jittered_copies(base, r, target_cos, rng):
    """r near-dup copies of `base` (unit) each at ~target_cos to base, byte-distinct."""
    # x = cos*base + sqrt(1-cos^2)*perp ; perp random unit orthogonal to base
    out = np.empty((r, base.shape[0]), np.float32)
    for i in range(r):
        g = rng.standard_normal(base.shape).astype("float32")
        g -= (g @ base) * base                       # orthogonalize
        g /= max(np.linalg.norm(g), 1e-9)
        out[i] = target_cos * base + np.sqrt(max(1 - target_cos**2, 0)) * g
    return _unit(out)

log(f"=== near-dup memory test (D={D} rank={RANK} G={NGROUP} R={R} clean={NCLEAN} cos={COS}) ===")
rng = np.random.default_rng(SEED)
basis = rng.standard_normal((RANK, D)).astype("float32")
def lowrank(n): return _unit((rng.standard_normal((n, RANK)).astype("float32")) @ basis)

clean = lowrank(NCLEAN)
group_bases = lowrank(NGROUP)
# full set: clean + for each group its base + (R-1) near-dup jittered copies
blocks = [clean]
rep_idx = [np.arange(NCLEAN)]            # reduced set = clean + one rep per group
cursor = NCLEAN
for gb in group_bases:
    copies = np.concatenate([gb[None, :], jittered_copies(gb, R - 1, COS, rng)])
    blocks.append(copies)
    rep_idx.append(np.array([cursor]))   # keep the group base only
    cursor += R
X = np.ascontiguousarray(np.concatenate(blocks))
reduced_keep = np.concatenate(rep_idx)
Xr = np.ascontiguousarray(X[reduced_keep])
N = len(X)
log(f"full N={N}  reduced N={len(Xr)}  ({N/len(Xr):.1f}x)")

# np.unique is a no-op? (must be: zero exact dups)
n_exact = N - len(np.unique(X, axis=0))
log(f"exact-dup rows (np.unique would remove): {n_exact}  <-- must be ~0")

# queries: perturbed clean points (ground truth on FULL set, the real gallery)
qidx = rng.choice(NCLEAN, NQ, replace=False)
q = np.ascontiguousarray(_unit(X[qidx] + rng.normal(0, 0.02, (NQ, D)).astype("float32")))
_, exact_full = faiss.knn(q, X, K, metric=faiss.METRIC_INNER_PRODUCT)
_, exact_red  = faiss.knn(q, Xr, K, metric=faiss.METRIC_INNER_PRODUCT)

def hnsw_recall(x, exact):
    h = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200; h.add(x); h.hnsw.efSearch = 200
    _, I = h.search(q, K); return rec(I, exact)
def ivfpq_recall(x, exact, m=16, nbits=8):
    nlist = max(64, int(np.sqrt(len(x))))
    iv = faiss.IndexIVFPQ(faiss.IndexFlatIP(D), D, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    iv.train(x); iv.add(x); iv.nprobe = 64
    _, I = iv.search(q, K); return rec(I, exact)

PQ_M, PQ_BITS = 16, 8
def bytes_flat(n): return n * D * 4
def bytes_pq(n):   return n * PQ_M            # nbits=8 -> 1 byte/subq

points = []
# (1) Full-HNSW (uncompressed)
r1 = hnsw_recall(X, exact_full); points.append(
    {"label": "Full-HNSW", "n": N, "recall": round(r1, 4), "bytes": bytes_flat(N), "compress": "none"})
log(f"  Full-HNSW       recall={r1:.4f}  mem={bytes_flat(N)/1e6:.1f}MB")
# (2) PQ-only  <-- the competitor np.unique can't beat and reduction must beat
r2 = ivfpq_recall(X, exact_full); points.append(
    {"label": "PQ-only(IVFPQ)", "n": N, "recall": round(r2, 4), "bytes": bytes_pq(N), "compress": "pq"})
log(f"  PQ-only         recall={r2:.4f}  mem={bytes_pq(N)/1e6:.1f}MB")
# (3) Reduce (near-dup groups -> 1 rep)  [recall measured on REDUCED gallery = honest pure-ANN]
r3 = hnsw_recall(Xr, exact_red); points.append(
    {"label": "Reduce-HNSW", "n": len(Xr), "recall": round(r3, 4), "bytes": bytes_flat(len(Xr)), "compress": "reduce"})
log(f"  Reduce-HNSW     recall={r3:.4f}  mem={bytes_flat(len(Xr))/1e6:.1f}MB")
# (4) Reduce + PQ
r4 = ivfpq_recall(Xr, exact_red); points.append(
    {"label": "Reduce+PQ", "n": len(Xr), "recall": round(r4, 4), "bytes": bytes_pq(len(Xr)), "compress": "reduce+pq"})
log(f"  Reduce+PQ       recall={r4:.4f}  mem={bytes_pq(len(Xr))/1e6:.1f}MB")

# verdict: at matched recall, does reduction beat PQ-only on memory?
pq = next(p for p in points if p["compress"] == "pq")
red = next(p for p in points if p["compress"] == "reduce")
redpq = next(p for p in points if p["compress"] == "reduce+pq")
out = {"config": {"D": D, "RANK": RANK, "NGROUP": NGROUP, "R": R, "NCLEAN": NCLEAN,
                  "COS": COS, "K": K, "NQ": NQ, "SEED": SEED, "PQ_M": PQ_M, "PQ_BITS": PQ_BITS,
                  "exact_dup_rows_in_full": int(n_exact)},
       "points": points,
       "verdict": {
           "np_unique_is_noop": bool(n_exact < 10),
           "pq_only_recall": pq["recall"], "pq_only_MB": round(pq["bytes"]/1e6, 3),
           "reduce_recall": red["recall"], "reduce_MB": round(red["bytes"]/1e6, 3),
           "reduce_pq_recall": redpq["recall"], "reduce_pq_MB": round(redpq["bytes"]/1e6, 3),
           "reduce_vs_pq_mem_ratio": round(pq["bytes"]/max(red["bytes"], 1), 2),
           "reduce_pq_vs_pq_mem_ratio": round(pq["bytes"]/max(redpq["bytes"], 1), 2),
           "interpretation": (
               "Near-dups (cos<1.0) are np.unique-blind. Recall is NOT the axis (settled: §22). "
               "If Reduce/Reduce+PQ reaches >= PQ-only recall at materially less memory, near-dup "
               "reduction has a real MEMORY value beyond both np.unique and PQ -> nucleus survives "
               "as a compression axis. If PQ-only already matches on memory, PQ eats the near-dup "
               "redundancy and reduction adds nothing -> the direction is closed.")}}
json.dump(out, open(os.path.join(OUT_DIR, "neardup_memory.json"), "w"), indent=2)
log("saved outputs/validation/neardup_memory.json")
print(json.dumps(out["verdict"], indent=2))

