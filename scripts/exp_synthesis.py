import os, json
import numpy as np
from scipy.stats import spearmanr

VDIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
OUT_DIR = VDIR
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def load(name): return json.load(open(os.path.join(VDIR, name)))
def have(name): return os.path.exists(os.path.join(VDIR, name))

tie_ok = have("tie_robustness.json") and have("tie_wdc_controls.json") and have("small_exact_groups.json")
if tie_ok:
    tie = load("tie_robustness.json"); ctl = load("tie_wdc_controls.json"); seg = load("small_exact_groups.json")
    tie_sweep = {r["R"]: r for r in tie["wdc_sweep"]}
    tie_ctl = {r["condition"]: r for r in ctl["wdc_controls"]}
    tie_seg = {r["d"]: r for r in seg["rows"]}

def build(tie_aware):
    """Return list of (indeg_max, hnsw_recall, ivf_recall, source, sweep)."""
    pts = []
    r = load("regime.json")
    for k, v in r["grid"].items():
        h, im = v["hnsw"], v["indeg_max"]
        if tie_aware and tie_ok and float(v["jitter"]) == 0.0 and v["d"] in tie_seg:
            h = tie_seg[v["d"]]["exact_groups"]["tie_aware"]; im = tie_seg[v["d"]]["exact_groups"]["indeg_max"]
        pts.append((im, h, v["ivf"], "synthetic: small groups", "regime"))
    m = load("megahub.json")
    for k, v in m["grid"].items():
        pts.append((v["indeg_max"], v["hnsw"], v["ivf"], "synthetic: exact sink", "megahub"))
    hm = load("hub_mechanism.json")
    for k, v in hm["part1_jitter_x_giant"].items():
        pts.append((v["indeg_max"], v["hnsw"]["200"], v["ivf"]["64"], "synthetic: giant group", "hub_mechanism_giant"))
    for k, v in hm["part2_exact_sink"].items():
        pts.append((v["indeg_max"], v["hnsw"]["200"], v["ivf"]["64"], "synthetic: exact sink", "hub_mechanism_sink"))
    wc = load("wdc_controls.json")
    for k, v in wc.items():
        if tie_aware and tie_ok and k in tie_ctl:
            t = tie_ctl[k]; pts.append((t["indeg_max"], t["hnsw"]["tie_aware"], t["ivf"]["tie_aware"], "real WDC", "wdc_controls"))
        else:
            pts.append((v["indeg_max"], v["hnsw"]["200"], v["ivf"]["64"], "real WDC", "wdc_controls"))
    cl = load("wdc_collateral.json")
    for k, v in cl["sweep"].items():
        pts.append((v["indeg_max"], v["hnsw"]["200"], v["ivf"]["64"], "real WDC", "wdc_collateral"))
    pa = load("pure_ann.json")
    for k, v in pa["levels"].items():
        im = v.get("hnsw_indegree", {}).get("indeg_max")
        if im is None: continue
        if tie_aware and tie_ok and k in tie_sweep:
            t = tie_sweep[k]; pts.append((t["indeg_max"], t["hnsw"]["tie_aware"], t["ivf"]["tie_aware"], "real WDC", "pure_ann"))
        else:
            pts.append((im, v["hnsw"]["200"]["ann_recall@10"], v["ivf"]["64"]["ann_recall@10"], "real WDC", "pure_ann"))
    return [p for p in pts if p[0] and p[0] > 0]

def stats(pts):
    indeg = np.array([p[0] for p in pts], float); hnsw = np.array([p[1] for p in pts], float)
    ivf = np.array([p[2] for p in pts], float); sweeps = np.array([p[4] for p in pts])
    rho_h, ph = spearmanr(np.log10(indeg), hnsw); rho_i, pi = spearmanr(np.log10(indeg), ivf)
    buckets = []
    for lo, hi in [(0, 200), (200, 2000), (2000, 20000), (20000, 1e9)]:
        mask = (indeg >= lo) & (indeg < hi)
        if mask.sum():
            buckets.append({"indeg_bucket": [lo, hi], "n": int(mask.sum()),
                            "median_hnsw": round(float(np.median(hnsw[mask])), 3),
                            "median_ivf": round(float(np.median(ivf[mask])), 3),
                            "min_hnsw": round(float(hnsw[mask].min()), 3), "max_hnsw": round(float(hnsw[mask].max()), 3)})
    per_sweep = {}
    for s in sorted(set(sweeps)):
        mk = sweeps == s
        if mk.sum() >= 4:
            rs, ps = spearmanr(np.log10(indeg[mk]), hnsw[mk])
            per_sweep[s] = {"n": int(mk.sum()), "spearman_hnsw": round(float(rs), 3) if np.isfinite(rs) else None}
        else:
            per_sweep[s] = {"n": int(mk.sum()), "spearman_hnsw": None}
    return {"n_conditions": len(pts), "spearman_logindeg_hnsw": round(float(rho_h), 3),
            "spearman_logindeg_ivf": round(float(rho_i), 3), "nominal_p_hnsw": float(f"{ph:.2e}"),
            "nominal_p_ivf": float(f"{pi:.2e}"), "bucket_medians": buckets, "per_sweep": per_sweep}

pts_tie = build(tie_aware=True); pts_id = build(tie_aware=False)
S_tie = stats(pts_tie); S_id = stats(pts_id)
primary = S_tie if tie_ok else S_id
print(f"N points = {primary['n_conditions']}   scoring = {'tie-aware' if tie_ok else 'id-based'}")
print(f"Spearman rho(log indeg_max, HNSW recall) = {primary['spearman_logindeg_hnsw']}")
print(f"Spearman rho(log indeg_max, IVF  recall) = {primary['spearman_logindeg_ivf']}")
print("per-sweep rho:", {k: v["spearman_hnsw"] for k, v in primary["per_sweep"].items()})
print("\nindeg_max bucket      median HNSW  [min,max]      median IVF   n")
for b in primary["bucket_medians"]:
    print(f"  [{b['indeg_bucket'][0]:>6.0f},{b['indeg_bucket'][1]:<8.0f})  {b['median_hnsw']:.3f}  [{b['min_hnsw']:.3f},{b['max_hnsw']:.3f}]   {b['median_ivf']:.3f}   {b['n']}")
if tie_ok:
    print(f"\nlegacy id-based: rho_hnsw={S_id['spearman_logindeg_hnsw']}  rho_ivf={S_id['spearman_logindeg_ivf']}")

out = dict(primary); out["scoring"] = "tie-aware" if tie_ok else "id-based"
out["sources"] = sorted(set(p[3] for p in pts_tie))
out["points"] = [{"indeg_max": float(a), "hnsw": float(b), "ivf": float(c), "source": s, "sweep": w} for (a, b, c, s, w) in pts_tie]
if tie_ok: out["legacy_id_based"] = S_id
json.dump(out, open(os.path.join(OUT_DIR, "synthesis_indegree.json"), "w"), indent=2)
print(f"\nsaved {os.path.join(OUT_DIR, 'synthesis_indegree.json')}")

