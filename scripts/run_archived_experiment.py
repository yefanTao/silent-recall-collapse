"""Run one legacy experiment with isolated results, keeping archived JSONs intact.

Only top-level OUT_DIR/FIG_DIR assignments are redirected. Data/model caches
retain their original paths. Numerical settings and scoring stay unchanged.
Use README.md to prepare inputs and choose tie-aware scoring where required.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(p.stem for p in (ROOT / "scripts").glob("exp_*.py")))
    parser.add_argument("--output", type=Path, required=True,
                        help="New result directory relative to artifact root, or an absolute path")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="Explicit original-script environment setting, recorded in runtime.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    settings = {}
    for item in args.set:
        name, sep, value = item.partition("=")
        if not sep or not name.isidentifier():
            parser.error("--set must be NAME=VALUE")
        settings[name] = value
    source = ROOT / "scripts" / (args.experiment + ".py")
    data = source.read_bytes()
    tree = ast.parse(data, filename=str(source))
    redirected = []

    class Redirect(ast.NodeTransformer):
        def visit_Assign(self, node):
            # Only module-level assignments are visited below.
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in names:
                if name in {"OUT_DIR", "FIG_DIR"}:
                    destination = output / ("figures" if name == "FIG_DIR" else "results")
                    node.value = ast.copy_location(ast.Constant(str(destination)), node.value)
                    redirected.append(name)
            return node

    redirect = Redirect()
    tree.body = [redirect.visit(node) if isinstance(node, ast.Assign) else node for node in tree.body]
    if "OUT_DIR" not in redirected:
        parser.error("This producer has different I/O conventions; run its documented command in a working copy")
    record = {"experiment": args.experiment, "source_sha256": hashlib.sha256(data).hexdigest(),
              "output": str(output), "modified_assignments": redirected, "environment": settings,
              "note": "Original numerical code/scoring; data and model caches retain original paths"}
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(settings)
    start = time.perf_counter()
    record["status"] = "running"
    (output / "runtime.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        exec(compile(ast.fix_missing_locations(tree), str(source), "exec"),
             {"__file__": str(source), "__name__": "__main__"})
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        raise
    finally:
        record["seconds"] = time.perf_counter() - start
        (output / "runtime.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
