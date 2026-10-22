"""Replay the archived three-HNSW frontier on the restored WDC cache.

Preserves its gallery, held-out queries and entity evaluators. Redirects output,
uses eight threads and an explicit construction seed, and records all three
timings plus their median instead of selecting the fastest run. This is a new
runtime/cache replication, not a replacement of the archived experiment.
"""
import ast
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import faiss
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="outputs/pvldb_audit/2026-10-09/pareto")
args = parser.parse_args()
OUT = ROOT / args.output
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "pareto_dedup.json").exists():
    raise FileExistsError("Refusing to overwrite an existing replication")
SOURCE = ROOT / "scripts/exp_pareto_dedup.py"
faiss.omp_set_num_threads(8)
events = []
peak = [0]
stop = threading.Event()
def monitor():
    process = psutil.Process()
    while not stop.wait(0.1):
        peak[0] = max(peak[0], process.memory_info().rss)
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
def time_search(index, queries, k, reps=3):
    index.search(queries[:128], k)
    timings = []
    for _ in range(reps):
        start = time.perf_counter()
        _, ids = index.search(queries, k)
        timings.append(time.perf_counter() - start)
    events.append({"index_n": int(index.ntotal), "efSearch": int(index.hnsw.efSearch),
                   "seconds": timings, "median_qps": len(queries) / float(np.median(timings))})
    return ids, events[-1]["median_qps"]

tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
class Replay(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "OUT_DIR" for t in node.targets):
            node.value = ast.Constant(str(OUT))
        return self.generic_visit(node)
    def visit_FunctionDef(self, node):
        if node.name == "time_search":
            return None  # Use the timing function supplied in the execution namespace.
        if node.name == "hnsw":
            seed = ast.parse(f"h.hnsw.rng = faiss.RandomGenerator({args.seed})").body[0]
            node.body.insert(1, seed)
        return self.generic_visit(node)
tree = ast.fix_missing_locations(Replay().visit(tree))
namespace = {"__file__": str(SOURCE), "__name__": "__main__", "time_search": time_search}
start = time.perf_counter()
try:
    exec(compile(tree, str(SOURCE), "exec"), namespace)
finally:
    stop.set()
    thread.join()
metadata = {"status": "complete", "elapsed_seconds": time.perf_counter() - start,
            "peak_process_rss_bytes": peak[0], "threads": faiss.omp_get_max_threads(),
            "construction_seed": args.seed, "timing_repetitions": 3, "statistic": "median",
            "query_batch": 20000, "warmup_queries": 128,
            "faiss": faiss.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cpu": "Intel Core Ultra 9 285",
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cache_provenance": json.loads((ROOT / "data/wdc_full/audit_embedding.json").read_text()),
            "timings": events,
            "limitations": "One construction per representation; three timing repeats, not three builds. New embeddings/runtime; excludes label-map evaluation time from QPS as in the archive."}
(OUT / "runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in metadata.items() if k not in ["timings", "cache_provenance"]}, indent=2))
