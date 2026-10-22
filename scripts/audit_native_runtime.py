"""Replay native graph implementations into isolated, provenance-tracked outputs."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading
import time
import subprocess
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
TEMP = (ROOT / "tmp/native-temp").resolve()
TEMP.mkdir(parents=True, exist_ok=True)
import tempfile
tempfile.tempdir = str(TEMP)
os.environ.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                  MPLCONFIGDIR=str(ROOT / "tmp/matplotlib-native"))
import faiss
import psutil
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", choices=["hnswlib_check", "diskann", "diskann_deep"], required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-label", default="native", help="New output family; original archive is native")
args = parser.parse_args()
if not args.run_label.replace("_", "").replace("-", "").isalnum():
    parser.error("run-label must contain only letters, numbers, underscores or hyphens")
os.environ["SEED"] = str(args.seed)
OUT = ROOT / f"outputs/pvldb_audit/2026-10-10/{args.run_label}/{args.experiment}_seed{args.seed}"
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "runtime.json").exists():
    previous = json.loads((OUT / "runtime.json").read_text())
    if previous["status"] == "complete":
        raise FileExistsError("Refusing to overwrite a completed replication")
    (OUT / "runtime.json").rename(OUT / f"failed-runtime-{time.time_ns()}.json")
SOURCE = ROOT / f"scripts/exp_{args.experiment}.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
class Replay(ast.NodeTransformer):
    def visit_Import(self, node):
        if any(n.name == "diskannpy" for n in node.names):
            return None  # Supplied proxy runs the unchanged DiskANN calls in a worker.
        return node
    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id in ["OUT_DIR", "FIG_DIR"] for t in node.targets):
            node.value = ast.Constant(str(OUT))
        return self.generic_visit(node)
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr in ["add_items", "knn_query"]:
            node.keywords.append(ast.keyword(arg="num_threads", value=ast.Constant(8)))
        return self.generic_visit(node)
tree = ast.fix_missing_locations(Replay().visit(tree))
(OUT / "executed.py").write_text(ast.unparse(tree), encoding="utf-8")
faiss.omp_set_num_threads(8)
stop = threading.Event()
peak = [0]
def monitor():
    process = psutil.Process()
    while not stop.wait(0.1):
        peak[0] = max(peak[0], process.memory_info().rss)
worker = threading.Thread(target=monitor, daemon=True)
worker.start()
start = time.perf_counter()
status = "failed"
def diskann_job(operation, kwargs, array, index=None):
    import numpy as np
    with tempfile.TemporaryDirectory(dir=TEMP) as directory:
        directory = Path(directory)
        np.save(directory / "input.npy", array)
        job = {"operation": operation, "kwargs": kwargs, "input": str(directory / "input.npy"),
               "output": str(directory / "result.npz"), "index": index}
        (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/audit_diskann_worker.py"), str(directory / "job.json")],
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError((directory / "worker.log").read_text(encoding="utf-8"))
        if operation == "search":
            with np.load(directory / "result.npz") as result:
                return SimpleNamespace(identifiers=result["identifiers"], distances=result["distances"])
def build_memory_index(**kwargs):
    return diskann_job("build", kwargs, kwargs.pop("data"))
class StaticMemoryIndex:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def batch_search(self, queries, **kwargs):
        return diskann_job("search", kwargs, queries, self.kwargs)
proxy = SimpleNamespace(build_memory_index=build_memory_index, StaticMemoryIndex=StaticMemoryIndex)
try:
    exec(compile(tree, str(SOURCE), "exec"), {"__file__": str(SOURCE), "__name__": "__main__", "diskannpy": proxy})
    status = "complete"
finally:
    stop.set()
    worker.join()
    result = {"status": status, "seconds": time.perf_counter() - start, "peak_rss_bytes": peak[0],
              "seed": args.seed, "threads": 8,
              "packages": {p: importlib.metadata.version(p) for p in ["faiss-cpu", "hnswlib", "diskannpy", "numpy", "scipy"]},
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "executed_sha256": hashlib.sha256((OUT / "executed.py").read_bytes()).hexdigest(),
              "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "changes": ["Isolated outputs", "Explicit eight threads including hnswlib calls", "Seed supplied by environment",
                          "DiskANN calls isolated in workers to avoid conflicting OpenMP runtimes; arguments unchanged"],
              "limitations": ["New runtime; archived outputs retained", "Parallel graph construction is not bitwise deterministic"]}
    (OUT / "runtime.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
