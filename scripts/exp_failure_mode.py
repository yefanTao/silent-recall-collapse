import os, json, time
import numpy as np, faiss

D        = int(os.environ.get("D", "128"))
N_BASE   = int(os.environ.get("N_BASE", "20000"))
SINK_FRAC= float(os.environ.get("SINK_FRAC", "0.3"))
K        = int(os.environ.get("K", "10"))
SEED     = int(os.environ.get("SEED", "0"))
ROOT     = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR  = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG_DIR, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype("float32")

log(f"=== failure-mode dissection (D={D}, N_BASE={N_BASE}, sink={SINK_FRAC}) ===")
rng = np.random.default_rng(SEED)
base = _unit(rng.standard_normal((N_BASE, D)))
S = int(SINK_FRAC * N_BASE)
sink_vec = _unit(rng.standard_normal((1, D)))[0]
x = np.ascontiguousarray(np.concatenate([base, np.repeat(sink_vec[None, :], S, 0)]))
n = len(x)
sink_ids = set(range(N_BASE, n))           # the duplicate-sink node ids
# queries: perturbed copies of CLEAN base points only (we want to find clean neighbors)
qsel = rng.choice(N_BASE, size=2000, replace=False)
q = np.ascontiguousarray(_unit(base[qsel] + rng.normal(0, 1e-3, (len(qsel), D))))
_, exact_I = faiss.knn(q, np.ascontiguousarray(x), K, metric=faiss.METRIC_INNER_PRODUCT)
truth1 = exact_I[:, 0]

h = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT)
h.hnsw.efConstruction = 200; h.add(x)

def run(ef):
    h.hnsw.efSearch = ef
    Dh, Ih = h.search(q, K)
    return Dh, Ih

# brute-force similarity of true top-1 (for distance-gap comparison)
xt = np.ascontiguousarray(x)
truth_sim = np.einsum("ij,ij->i", q, xt[truth1])

results = {}
for ef in [50, 200, 500]:
    Dh, Ih = run(ef)
    hit = (Ih == truth1[:, None]).any(1)
    miss = ~hit
    top1 = Ih[:, 0]
    top1_sim = np.einsum("ij,ij->i", q, xt[top1])
    hub_in_cand = np.array([len(set(row.tolist()) & sink_ids) > 0 for row in Ih])
    results[ef] = {
        "recall@1": float((Ih[:, :1] == truth1[:, None]).any(1).mean()),
        "recall@10": float(hit.mean()),
        "miss_rate": float(miss.mean()),
        # of the MISSED queries, what fraction returned a sink node? (TRAP signature)
        "miss_with_hub_in_topk": float(hub_in_cand[miss].mean()) if miss.any() else 0.0,
        # near-miss vs lost: among misses, how close was HNSW's top1 to the truth?
        "miss_mean_sim_gap": float((truth_sim[miss] - top1_sim[miss]).mean()) if miss.any() else 0.0,
        "miss_frac_nearmiss(gap<0.01)": float(((truth_sim[miss] - top1_sim[miss]) < 0.01).mean()) if miss.any() else 0.0,
        "overall_hub_in_topk": float(hub_in_cand.mean()),
    }
    log(f"  ef={ef:<4d} R@1={results[ef]['recall@1']:.3f} miss={results[ef]['miss_rate']:.3f} "
        f"hub_in_missed_topk={results[ef]['miss_with_hub_in_topk']:.3f} "
        f"nearmiss_frac={results[ef]['miss_frac_nearmiss(gap<0.01)']:.3f}")

# ef-sensitivity verdict: does 10x budget rescue? (TRAP) or not (UNREACHABLE)?
r_lo, r_hi = results[50]["recall@1"], results[500]["recall@1"]
ef_gain = r_hi - r_lo
hub_trap = results[200]["miss_with_hub_in_topk"]
verdict = {
    "ef_budget_gain_50to500": round(ef_gain, 4),
    "ef_rescues_substantially": bool(ef_gain > 0.10),
    "fraction_missed_queries_returning_hub_node": round(hub_trap, 4),
    "dominant_mechanism": (
        "TRAP (hub captures candidate budget; runtime cap could help)" if hub_trap > 0.3
        else "UNREACHABLE / near-miss (graph rewired; only rebuild/dedup helps)"),
    "interpretation": "If missed queries frequently return a sink node (hub_trap high) and a "
                      "larger efSearch rescues recall, the failure is a runtime TRAP -> a hub-cap "
                      "remedy is plausible. If not, the hub has rewired the graph so the true NN is "
                      "effectively unreachable -> only dedup/rebuild restores recall.",
}
out = {"config": {"D": D, "N_BASE": N_BASE, "SINK_FRAC": SINK_FRAC, "K": K, "n": n,
                  "n_sink": S, "n_queries": int(len(q))},
       "per_ef": results, "verdict": verdict}
json.dump(out, open(os.path.join(OUT_DIR, "failure_mode.json"), "w"), indent=2)
log("saved outputs/validation/failure_mode.json")
print(json.dumps(verdict, indent=2))

