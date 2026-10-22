"""Replay the five-space representation experiment without replacing its archive.

The archived `oracle` is a query-informed heuristic, not a global upper bound.
Export paired query outcomes so uncertainty can be assessed at entity level.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                  HF_HOME=str(ROOT / "data/huggingface"), MPLCONFIGDIR=str(ROOT / "tmp/matplotlib"))
import numpy as np
import psutil
import torch
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["wdc", "musicbrainz"], required=True)
parser.add_argument("--run-label", default="c4")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--revisions-file", type=Path)
args = parser.parse_args()
if not args.run_label.replace("_", "").replace("-", "").isalnum():
    parser.error("run-label must contain only letters, numbers, underscores or hyphens")
os.environ.update(DATASET=args.dataset, N_TEST="6000", N_Q="3", MAX_POOL="50", SEED=str(args.seed), GPU="0")
OUT = ROOT / "outputs/pvldb_audit/2026-10-10" / args.run_label / args.dataset
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "runtime.json").exists():
    raise FileExistsError("Refusing to overwrite a completed replication")
SOURCE = ROOT / "scripts/exp_c4_generalize.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
models = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "MODELS" for t in n.targets)))
revisions = (json.loads(args.revisions_file.read_text()) if args.revisions_file else
             {hf: HfApi().model_info(hf).sha for _, hf, _, _ in models})
assert all(hf in revisions for _, hf, _, _ in models)
(OUT / "model-revisions.json").write_text(json.dumps(revisions, indent=2), encoding="utf-8")

class Replay(ast.NodeTransformer):
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ["OUT_DIR", "FIG_DIR", "CACHE"]:
                node.value = ast.Constant(str(OUT / "cache") if target.id == "CACHE" else str(OUT))
        return self.generic_visit(node)
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "SentenceTransformer":
            node.keywords.append(ast.keyword(arg="revision", value=ast.Subscript(
                value=ast.Name(id="revisions", ctx=ast.Load()), slice=ast.Name(id="hf", ctx=ast.Load()), ctx=ast.Load())))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "encode":
            for kw in node.keywords:
                if kw.arg == "batch_size":
                    kw.value = ast.Constant(128)
        return self.generic_visit(node)

tree = ast.fix_missing_locations(Replay().visit(tree))
(OUT / "executed.py").write_text(ast.unparse(tree), encoding="utf-8")
torch.set_num_threads(8)
peak = [0]
stop = threading.Event()
def monitor():
    process = psutil.Process()
    while not stop.wait(0.2):
        peak[0] = max(peak[0], process.memory_info().rss)
worker = threading.Thread(target=monitor, daemon=True)
worker.start()
namespace = {"__file__": str(SOURCE), "__name__": "__main__", "revisions": revisions}
start = time.perf_counter()
status = "failed"
try:
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    np.savez_compressed(OUT / "paired_hits.npz", q_eid=namespace["q_eid"],
                        q_pos=namespace["q_pos"], pool_rows=namespace["pool_rows"],
                        **{f"{e}__{name}": value for (e, name), value in namespace["hit1_store"].items()})
    status = "complete"
finally:
    stop.set()
    worker.join()
    data = ROOT / ("data/wdc_full/wdc_full_cleaned.pkl" if args.dataset == "wdc" else "data/musicbrainz/musicbrainz_clean.pkl")
    metadata = {"status": status, "seconds": time.perf_counter() - start, "peak_rss_bytes": peak[0],
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "executed_sha256": hashlib.sha256((OUT / "executed.py").read_bytes()).hexdigest(),
                "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "revisions": revisions,
                "seed": args.seed,
                "packages": {p: importlib.metadata.version(p) for p in ["numpy", "torch", "sentence-transformers", "scipy"]},
                "changes": ["Separate outputs and caches", "WDC score-free cleaned pickle; same consumed columns",
                            "Pinned current model revisions", "Encoding batch 128 instead of 1024; eight CPU threads"],
                "limitations": [f"One entity split (seed {args.seed})", "Archived oracle and query-level p-values are not certified conclusions",
                                "Re-encoded current model/runtime replication, not byte-identical archive reproduction"]}
    (OUT / "runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
