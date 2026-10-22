"""Paired entity-cluster bootstrap for independently replayed representation matrices.

All queries of an entity stay together. With two splits, the 80 off-diagonal
comparisons across two datasets form one family; intervals allow dependent cells.
These are bootstrap intervals for the sampled entity population, conditional on
the fitted models and split, not uncertainty over training or arbitrary datasets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--run-label", default="c4_verified")
parser.add_argument("--bootstrap", type=int, default=100000)
parser.add_argument("--family-size", type=int, default=80)
args = parser.parse_args()
base = ROOT / "outputs/pvldb_audit/2026-10-10" / args.run_label
rng = np.random.default_rng(20261010)
report = {"method": "paired entity-cluster percentile bootstrap", "bootstrap_replicates": args.bootstrap,
          "rng_seed": 20261010, "primary_family_size": args.family_size,
          "primary_family": "20 off-diagonal contrasts x 2 datasets x 2 splits (default 80)",
          "multiplicity": f"Bonferroni 95% family intervals across {args.family_size} primary comparisons",
          "limitations": ["Bootstrap coverage is approximate", "Conditional on the sampled split and model revisions",
                          "A positive interval does not imply transfer to other corpora or models"],
          "datasets": {}}

def interval(a, b, family_size):
    delta = a - b
    # Resampling a whole entity is equivalent to resampling the empirical
    # categories of its paired mean difference; this avoids a huge B x N array.
    values, counts = np.unique(np.round(delta, 12), return_counts=True)
    samples = rng.multinomial(len(delta), counts / len(delta), size=args.bootstrap)
    estimates = samples @ values / len(delta)
    tail = .05 / (2 * family_size)
    pointwise = np.quantile(estimates, [.025, .975]).tolist()
    adjusted = np.quantile(estimates, [tail, 1-tail]).tolist()
    return {"delta_R@1": float(delta.mean()), "ci95": pointwise,
            "family_ci": adjusted, "family_size": family_size,
            "positive_family_interval": bool(adjusted[0] > 0),
            "entities_improved": int((delta > 0).sum()),
            "entities_worsened": int((delta < 0).sum()),
            "entities_tied": int((delta == 0).sum())}

for dataset in ["wdc", "musicbrainz"]:
    folder = base / dataset
    runtime = json.loads((folder / "runtime.json").read_text())
    assert runtime["status"] == "complete", (dataset, runtime["status"])
    results = json.loads((folder / f"c4_generalize_{dataset}.json").read_text())
    paired = folder / "paired_hits.npz"
    with np.load(paired) as data:
        ids, inverse = np.unique(data["q_eid"], return_inverse=True)
        sizes = np.bincount(inverse)
        assert len(ids) == results["config"]["n_entities"]
        assert np.all(sizes == 3), "Paper compares equal-weight entities with three queries each"
        means = {key: np.bincount(inverse, weights=data[key].astype(float)) / sizes
                 for key in data.files if "__" in key}
    primary, secondary = [], []
    for evaluation in results["config"]["spaces"]:
        for selection in results["config"]["spaces"]:
            actual = means[f"{evaluation}__medoid_{selection}"]
            assert np.isclose(actual.mean(), results["matrix_R@1"][evaluation][selection])
            if selection != evaluation:
                primary.append({"eval": evaluation, "selection": selection,
                                **interval(actual, means[f"{evaluation}__random"], args.family_size)})
        for method in ["oracle", "centroid"]:
            secondary.append({"eval": evaluation, "method": method,
                              **interval(means[f"{evaluation}__{method}"], means[f"{evaluation}__full"], args.family_size // 2)})
    report["datasets"][dataset] = {
        "seed": results["config"]["SEED"], "entities": len(ids), "queries": int(sizes.sum()),
        "paired_hits_sha256": hashlib.sha256(paired.read_bytes()).hexdigest(),
        "offdiagonal": primary, "secondary_vs_full": secondary,
        "n_positive_family_intervals": sum(row["positive_family_interval"] for row in primary),
        "offdiagonal_gain_range": [min(row["delta_R@1"] for row in primary), max(row["delta_R@1"] for row in primary)],
        "headroom_mean": float(np.mean([means[f"{e}__full"].mean() - means[f"{e}__random"].mean()
                                        for e in results["config"]["spaces"]]))}
report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(base / "entity-bootstrap.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({d: {k: v for k, v in rows.items() if k not in ["offdiagonal", "secondary_vs_full"]}
                  for d, rows in report["datasets"].items()}, indent=2))
