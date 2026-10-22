"""Summarize three construction seeds without overwriting archived results."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/pvldb_audit/2026-10-09"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs", nargs=3, type=Path,
                    default=[BASE / "pareto", BASE / "pareto_seed1", BASE / "pareto_seed2"])
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
paths = [(ROOT / p).resolve() for p in args.runs]
output = ROOT / args.output
if output.exists():
    raise FileExistsError(f"Refusing to overwrite {output}")
runs = [json.loads((p / "pareto_dedup.json").read_text()) for p in paths]
metas = [json.loads((p / "runtime.json").read_text()) for p in paths]
assert all(m["status"] == "complete" and m["threads"] == 8 for m in metas)
assert [m["construction_seed"] for m in metas] == [0, 1, 2]
assert all(r["config"]["n_full"] == 947584 and r["config"]["n_queries"] == 20000 for r in runs)
def span(values):
    values = list(values)
    return [float(min(values)), float(max(values))]
summary = {"seeds": [0, 1, 2], "threads": 8, "timing_statistic": "median_of_three",
           "total_seconds": sum(m["elapsed_seconds"] for m in metas),
           "peak_rss_gib": max(m["peak_process_rss_bytes"] for m in metas) / 2**30,
           "conditions": {}, "matched_ef_reduced_over_full_qps": [],
           "lowest_reduced_over_highest_full_qps": [], "raw_checksums": {}}
for key, metric in [("full_hnsw", "recall"), ("dedup_hnsw", "recall_multilabel"), ("reduced_hnsw", "recall")]:
    summary["conditions"][key] = {"n": runs[0][key][0]["n"], "index_mb": runs[0][key][0]["bytes"] / 1e6,
                                  "hit10_at_ef256": span(r[key][-1][metric] for r in runs),
                                  "qps_at_ef256": span(r[key][-1]["qps"] for r in runs),
                                  "hit10_at_ef16": span(r[key][0][metric] for r in runs),
                                  "qps_at_ef16": span(r[key][0]["qps"] for r in runs)}
for r in runs:
    summary["matched_ef_reduced_over_full_qps"].extend(b["qps"] / a["qps"] for a, b in zip(r["full_hnsw"], r["reduced_hnsw"]))
    summary["lowest_reduced_over_highest_full_qps"].append(r["reduced_hnsw"][0]["qps"] / r["full_hnsw"][-1]["qps"])
for key in ["matched_ef_reduced_over_full_qps", "lowest_reduced_over_highest_full_qps"]:
    summary[key] = span(summary[key])
for p in paths:
    for name in ["pareto_dedup.json", "runtime.json"]:
        summary["raw_checksums"][(p / name).relative_to(ROOT).as_posix()] = hashlib.sha256((p / name).read_bytes()).hexdigest()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
