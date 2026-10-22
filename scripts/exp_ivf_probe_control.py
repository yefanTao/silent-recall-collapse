import os, json, time, math
import numpy as np, faiss

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "outputs", "validation"); os.makedirs(OUT_DIR, exist_ok=True)
DFIX = 128; N_BASE = 20000; K = 10; SEED = 0
SINK_FRAC = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
    return (x / n).astype("float32")
def rec(I, exact_I): return float(np.mean([len(set(a) & set(b)) for a, b in zip(I, exact_I)]) / K)

rng = np.random.default_rng(SEED)
base = _unit(rng.standard_normal((N_BASE, DFIX)))
q = np.ascontiguousarray(_unit(base + rng.normal(0, 1e-3, base.shape)))
_, exact_I = faiss.knn(q, np.ascontiguousarray(base), K, metric=faiss.METRIC_INNER_PRODUCT)
sink_vec = _unit(rng.standard_normal((1, DFIX)))[0]
nlist_clean = max(16, int(np.sqrt(N_BASE)))

def ivf(x, nlist, nprobe, train_on=None):
    iv = faiss.IndexIVFFlat(faiss.IndexFlatIP(DFIX), DFIX, nlist, faiss.METRIC_INNER_PRODUCT)
    iv.train(x if train_on is None else train_on); iv.add(x); iv.nprobe = nprobe
    _, I = iv.search(q, K); return rec(I, exact_I)

out = {"config": {"D": DFIX, "N_BASE": N_BASE, "K": K, "SEED": SEED, "sink_frac": SINK_FRAC,
                  "nlist_clean": nlist_clean}, "rows": []}
print("  frac     n      (a)paper  (b)fixed-frac  (c)fixed-nlist  (d)clean-trained")
for sf in SINK_FRAC:
    S = int(sf * N_BASE)
    x = base if S == 0 else np.concatenate([base, np.repeat(sink_vec[None, :], S, axis=0)])
    x = np.ascontiguousarray(_unit(x)); n = len(x)
    nl = max(16, int(np.sqrt(n)))
    a = ivf(x, nl, 64)
    b = ivf(x, nl, int(math.ceil(0.45 * nl)))
    c = ivf(x, nlist_clean, 64)
    d = ivf(x, nlist_clean, 64, train_on=np.ascontiguousarray(base))
    out["rows"].append({"sink_frac": sf, "n": int(n), "nlist_sqrt_n": nl,
                        "a_paper_nprobe64": round(a, 4), "b_fixed_probe_frac": round(b, 4),
                        "c_fixed_nlist": round(c, 4), "d_fixed_nlist_clean_trained": round(d, 4)})
    print(f"  {sf:<7g} {n:<7d} {a:.3f}     {b:.3f}          {c:.3f}           {d:.3f}")
r = out["rows"]
out["verdict"] = {k: round(r[0][k] - r[-1][k], 4) for k in
                  ["a_paper_nprobe64", "b_fixed_probe_frac", "c_fixed_nlist", "d_fixed_nlist_clean_trained"]}
out["verdict"]["note"] = "drop from sink_frac 0 to 0.5 under each budget rule; a hub effect would survive all four"
json.dump(out, open(os.path.join(OUT_DIR, "ivf_probe_control.json"), "w"), indent=2)
log(f"saved ivf_probe_control.json  drops={out['verdict']}")
