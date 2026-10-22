"""Exploratory numerical diagnostic for the prespecified jitter audit."""
import json
from pathlib import Path
import numpy as np
from audit_pvldb import unit

root = Path(__file__).resolve().parents[1]
output = root / "outputs/pvldb_audit/2026-10-08/jitter_precision.json"
rows = []
for seed in [0, 1, 2]:
    rng = np.random.default_rng(seed)
    base = unit(rng.standard_normal((20000, 64)))
    qids = rng.choice(20000, 2000, replace=False)
    q = unit(base[qids] + rng.normal(0, 1e-3, (2000, 64)))
    proto = unit(rng.standard_normal((1, 64)))
    noise = rng.standard_normal((10000, 64)).astype(np.float32)
    dirs = unit(rng.standard_normal((3, 64)))
    coeff = rng.standard_normal((10000, 3)).astype(np.float32)
    for geom in ["isotropic", "low_rank"]:
        perturb = noise if geom == "isotropic" else coeff @ dirs
        for eps in [1e-7, 1e-5, 1e-3]:
            hub = unit(proto + eps * perturb)
            a, b = hub[:1000], hub[1000:2000]
            ip32 = np.einsum("ij,ij->i", a, b)
            a64, b64 = a.astype(np.float64), b.astype(np.float64)
            cos64 = np.einsum("ij,ij->i", a64, b64) / (
                np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1))
            rows.append({"seed": seed, "geometry": geom, "eps": eps,
                         "distinct_vectors": len(np.unique(hub, axis=0)),
                         "numpy_float32_ip_equals_one_fraction": float(np.mean(ip32 == 1.0)),
                         "float64_cosine_min": float(cos64.min()),
                         "float64_cosine_mean": float(cos64.mean()),
                         "nonidentical_pair_fraction": float(np.mean(np.any(a != b, axis=1)))})
result = {"purpose": "exploratory precision diagnostic, not instrumented Faiss prune comparisons",
          "rows": rows}
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(output)
print([(r["geometry"], r["eps"], r["numpy_float32_ip_equals_one_fraction"],
        r["float64_cosine_mean"]) for r in rows if r["seed"] == 0])
