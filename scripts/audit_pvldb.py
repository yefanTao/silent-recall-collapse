"""Non-destructive PVLDB audit. Outputs never replace outputs/validation/.

Run with python scripts/audit_pvldb.py --suite controls
See README.md for data preparation and scoring protocols.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import threading
import time
from pathlib import Path

import faiss
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
K = 10
EPSILONS = [0.0, 1e-7, 1e-6, 1e-5]


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30))


def exact(x, q, k=K):
    ix = faiss.IndexFlatIP(x.shape[1])
    ix.add(x)
    scores, ids = [], []
    for start in range(0, len(q), 256):
        d, i = ix.search(q[start:start + 256], k)
        scores.append(d)
        ids.append(i)
    return np.concatenate(scores), np.concatenate(ids)


def id_recall(got, gold):
    return float(np.mean([len(set(a.tolist()) & set(b.tolist())) / gold.shape[1]
                          for a, b in zip(got, gold)]))


def metrics(scores, ids, gold_scores, gold_ids, clean=None):
    result = {"id_recall": id_recall(ids, gold_ids)}
    valid = ids >= 0
    for eps in EPSILONS:
        hits = (valid & (scores >= gold_scores[:, -1, None] - eps)).mean(axis=1)
        result[f"tie_recall_eps_{eps:g}"] = float(hits.mean())
        if clean is not None and np.any(clean):
            result[f"clean_tie_recall_eps_{eps:g}"] = float(hits[clean].mean())
    if clean is not None:
        result["n_clean_queries"] = int(clean.sum())
        if clean.any():
            result["clean_id_recall"] = id_recall(ids[clean], gold_ids[clean])
    return result


def indegree(ix):
    h = ix.hnsw
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
    neighbors = faiss.vector_to_array(h.neighbors)
    width = h.nb_neighbors(0)
    degrees = np.zeros(ix.ntotal, np.int64)
    for start in range(0, ix.ntotal, 8192):
        positions = offsets[start:min(start + 8192, ix.ntotal), None] + np.arange(width)
        ids = neighbors[positions.ravel()]
        np.add.at(degrees, ids[ids >= 0], 1)
    return int(degrees.max()), int(degrees.argmax())


def build(x, kind, seed):
    if kind == "hnsw":
        ix = faiss.IndexHNSWFlat(x.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
        ix.hnsw.efConstruction = 200
        ix.hnsw.efSearch = 200
        ix.hnsw.rng = faiss.RandomGenerator(seed)
    else:
        nlist = max(16, int(math.sqrt(len(x))))
        ix = faiss.IndexIVFFlat(faiss.IndexFlatIP(x.shape[1]), x.shape[1], nlist,
                                faiss.METRIC_INNER_PRODUCT)
        ix.cp.seed = seed
        ix.train(x)
        ix.nprobe = min(64, nlist)
    ix.add(x)
    return ix


class Audit:
    def __init__(self, args):
        self.args = args
        self.directory = Path(args.output)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{args.suite}.json"
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite {self.path}; select a new --output directory")
        self.start = time.perf_counter()
        self.peak_rss = 0
        self.stop = threading.Event()
        self.monitor = threading.Thread(target=self.monitor_memory, daemon=True)
        self.monitor.start()
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except Exception:
            commit = None
        self.data = {"config": vars(args), "k": K, "epsilons": EPSILONS,
                     "environment": {"platform": platform.platform(), "python": platform.python_version(),
                                     "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
                                     "ram_bytes": psutil.virtual_memory().total,
                                     "packages": {name: importlib.metadata.version(name)
                                                  for name in ["numpy", "scipy", "faiss-cpu", "pandas", "psutil"]},
                                     "faiss_compile_options": faiss.get_compile_options()},
                     "base_commit": commit,
                     "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "status": "running", "rows": []}
        self.write()

    def monitor_memory(self):
        process = psutil.Process()
        while not self.stop.wait(0.1):
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)

    def write(self):
        self.data["elapsed_seconds"] = time.perf_counter() - self.start
        self.data["peak_process_rss_bytes"] = self.peak_rss
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def record(self, row):
        self.data["rows"].append(row)
        self.write()
        r = row.get("full_gallery", row.get("metrics", {}))
        print(f"[{time.strftime('%H:%M:%S')}] {row.get('case')} seed={row.get('seed')} "
              f"d={row.get('d')} index={row.get('index')} n={row.get('n')} "
              f"recall={r.get('tie_recall_eps_1e-05')} indeg={row.get('max_indegree')} "
              f"time={row.get('seconds', 0):.2f}s", flush=True)

    def finish(self):
        self.stop.set()
        self.monitor.join()
        self.data["status"] = "complete"
        self.write()
        print(f"COMPLETE {self.path} rows={len(self.data['rows'])} "
              f"elapsed={self.data['elapsed_seconds']:.1f}s "
              f"peak_RSS={self.peak_rss / 2**30:.2f}GiB", flush=True)


def controls(audit, pilot=False):
    args = audit.args
    n = 2000 if pilot else 20000
    seeds = [0] if pilot else args.seeds
    dims = [64] if pilot else [64, 128]
    for seed in seeds:
        for d in dims:
            rng = np.random.default_rng(seed)
            base = unit(rng.standard_normal((n, d)))
            qids = rng.choice(n, min(args.nq, n), replace=False)
            q = unit(base[qids] + rng.normal(0, 1e-3, (len(qids), d)))
            sink = unit(rng.standard_normal((1, d)))
            different = unit(rng.standard_normal((n // 2, d)))
            old_d, old_i = exact(base, q)
            for tag in ["clean", "exact_sink", "distinct_addition"]:
                x = base if tag == "clean" else np.concatenate(
                    [base, np.repeat(sink, n // 2, axis=0) if tag == "exact_sink" else different])
                gold_d, gold_i = exact(x, q)
                # Check the kth boundary using exact scores over the AUGMENTED gallery.
                sink_scores = (q @ sink.T).ravel()
                sink_eligible = sink_scores >= old_d[:, -1] - 1e-5
                for kind in ["hnsw", "ivf"]:
                    t0 = time.perf_counter()
                    ix = build(x, kind, seed)
                    found_d, found_i = ix.search(q, K)
                    row = {"case": tag, "seed": seed, "d": d, "n": len(x),
                           "index": kind, "nq": len(q), "sink_fraction_final": (n // 2) / len(x)
                           if tag == "exact_sink" else 0,
                           "sink_eligible_query_fraction": float(sink_eligible.mean()),
                           "full_gallery": metrics(found_d, found_i, gold_d, gold_i),
                           "legacy_clean_base": metrics(found_d, found_i, old_d, old_i),
                           "gold_contains_added_fraction": float((gold_i >= n).mean()),
                           "seconds": time.perf_counter() - t0}
                    if kind == "hnsw":
                        row["max_indegree"], _ = indegree(ix)
                    audit.record(row)
                    del ix
                    gc.collect()


def jitter(audit):
    args = audit.args
    n, s, d = 20000, 10000, 64
    epsilons = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        base = unit(rng.standard_normal((n, d)))
        qids = rng.choice(n, args.nq, replace=False)
        q = unit(base[qids] + rng.normal(0, 1e-3, (args.nq, d)))
        proto = unit(rng.standard_normal((1, d)))
        noise = rng.standard_normal((s, d)).astype(np.float32)
        directions = unit(rng.standard_normal((3, d)))
        coefficients = rng.standard_normal((s, 3)).astype(np.float32)
        old_d, old_i = exact(base, q)
        for geometry in ["isotropic", "low_rank"]:
            perturb = noise if geometry == "isotropic" else coefficients @ directions
            for eps in epsilons:
                t0 = time.perf_counter()
                hub = np.repeat(proto, s, axis=0) if eps == 0 else unit(proto + eps * perturb)
                x = np.ascontiguousarray(np.concatenate([base, hub]))
                gold_d, gold_i = exact(x, q)
                ix = build(x, "hnsw", seed)
                found_d, found_i = ix.search(q, K)
                maximum, argmax = indegree(ix)
                row = {"case": geometry, "eps": eps, "seed": seed, "d": d, "n": len(x),
                       "index": "hnsw", "nq": len(q), "max_indegree": maximum,
                       "argmax_node": argmax,
                       "byte_distinct_hub_vectors": int(len(np.unique(hub, axis=0))),
                       "hub_cos_to_proto_min": float((hub @ proto.T).min()),
                       "hub_cos_to_proto_max": float((hub @ proto.T).max()),
                       "full_gallery": metrics(found_d, found_i, gold_d, gold_i),
                       "legacy_clean_base": metrics(found_d, found_i, old_d, old_i),
                       "seconds": time.perf_counter() - t0}
                audit.record(row)
                del ix, x
                gc.collect()


def compression(audit):
    # Match the published synthetic generator and query construction; additionally
    # score kept representative IDs against the ORIGINAL full-gallery task.
    d, rank, nclean, ngroups, copies, cosine = 128, 16, 50000, 20000, 8, 0.99
    for seed in audit.args.seeds:
        rng = np.random.default_rng(seed)
        basis = rng.standard_normal((rank, d)).astype(np.float32)
        clean = unit(rng.standard_normal((nclean, rank)).astype(np.float32) @ basis)
        group_bases = unit(rng.standard_normal((ngroups, rank)).astype(np.float32) @ basis)
        x = np.empty((nclean + ngroups * copies, d), np.float32)
        x[:nclean] = clean
        for g, gb in enumerate(group_bases):
            start = nclean + g * copies
            x[start] = gb
            for j in range(1, copies):
                noise = rng.standard_normal(d).astype(np.float32)
                noise -= (noise @ gb) * gb
                noise /= max(np.linalg.norm(noise), 1e-9)
                x[start + j] = unit((cosine * gb + math.sqrt(1 - cosine ** 2) * noise)[None])[0]
        keep = np.r_[np.arange(nclean), np.arange(nclean, len(x), copies)]
        xr = np.ascontiguousarray(x[keep])
        qids = rng.choice(nclean, audit.args.nq, replace=False)
        q = unit(x[qids] + rng.normal(0, 0.02, (len(qids), d)).astype(np.float32))
        full_d, full_i = exact(x, q, K * copies)
        red_d, red_i = exact(xr, q)
        group_ids = np.r_[np.arange(nclean), np.repeat(np.arange(nclean, nclean + ngroups), copies)]
        gold_entities = []
        for row in group_ids[full_i]:
            ordered = list(dict.fromkeys(row.tolist()))[:K]
            assert len(ordered) == K
            gold_entities.append(ordered)
        gold_entities = np.asarray(gold_entities, np.int64)
        for tag, gallery, local_d, local_i, original_ids in [
            ("full", x, full_d[:, :K], full_i[:, :K], np.arange(len(x))),
            ("reduced", xr, red_d, red_i, keep),
        ]:
            t0 = time.perf_counter()
            ix = build(gallery, "hnsw", seed)
            found_d, found_i = ix.search(q, K)
            mapped_i = original_ids[found_i]
            # Get distinct entities fairly (up to R*K items suffices since group size<=R).
            _, more_i = ix.search(q, K * copies)
            got_entities = []
            for row in group_ids[original_ids[more_i]]:
                got_entities.append(list(dict.fromkeys(row.tolist()))[:K])
            got_entities = np.asarray(got_entities, np.int64)
            exact_gallery_entities = group_ids[original_ids[local_i]]
            row = {"case": tag, "seed": seed, "d": d, "n": len(gallery), "index": "hnsw",
                   "nq": len(q), "metrics": metrics(found_d, found_i, local_d, local_i),
                   "original_full_vector_task": metrics(found_d, mapped_i, full_d[:, :K], full_i[:, :K]),
                   "original_full_entity_top10_recall": id_recall(got_entities, gold_entities),
                   "query_own_entity_hit_at10": float(np.mean([
                       int(qids[i]) in set(row.tolist())
                       for i, row in enumerate(group_ids[mapped_i])])),
                   "reduced_exact_original_entity_top10_recall": id_recall(group_ids[keep[red_i]], gold_entities),
                   "reduced_exact_original_vector_task": metrics(red_d, keep[red_i], full_d[:, :K], full_i[:, :K]),
                   "vector_payload_bytes": int(gallery.nbytes),
                   "serialized_index_bytes": int(faiss.serialize_index(ix).nbytes),
                   "representative_original_id_mapping_bytes": int(keep.nbytes) if tag == "reduced" else 0,
                   "seconds": time.perf_counter() - t0}
            audit.record(row)
            del ix
            gc.collect()


def wdc(audit):
    import pandas as pd
    data = Path(audit.args.data)
    pkl = data / "wdc_full_cleaned.pkl"
    if not pkl.exists():
        pkl = data / "wdc_full_cleaned.pkl"
    if not pkl.exists() or not (data / "embeddings.npy").exists():
        raise FileNotFoundError(f"WDC rows/embeddings missing in {data}")
    df = pd.read_pickle(pkl).reset_index(drop=True)
    emb = np.ascontiguousarray(np.load(data / "embeddings.npy"), dtype=np.float32)
    assert len(emb) == len(df), "Embedding row alignment must match dataframe"
    faiss.normalize_L2(emb)
    cid = df.cluster_id.to_numpy()
    title = df.title_clean.fillna("").astype(str).to_numpy()
    rng = np.random.default_rng(42)
    order = np.argsort(cid, kind="stable")
    cs = cid[order]
    boundaries = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    eligible = {}
    for a, b in zip(boundaries, np.r_[boundaries[1:], len(cs)]):
        positions = np.array([p for p in order[a:b] if title[p].strip() != ""])
        if len(positions) >= 13:
            eligible[cs[a]] = positions
    chosen = rng.choice(np.array(list(eligible), dtype=object), min(20000, len(eligible)), replace=False)
    qs, members = [], []
    for entity in chosen:
        positions = eligible[entity].copy()
        rng.shuffle(positions)
        qs.append(positions[0])
        members.append(positions[1:])
    q = emb[np.asarray(qs)]
    audit.data["wdc_provenance"] = {"row_path": str(pkl), "embedding_path": str(data / "embeddings.npy"),
                                     "n_rows": len(emb), "n_queries": len(q), "dimension": emb.shape[1]}
    for r in [1, 12, "all"]:
        rows = np.concatenate([m if r == "all" else m[:r] for m in members])
        x = np.ascontiguousarray(emb[rows])
        gold_d, gold_i = exact(x, q, K + 1)
        hashes = {hashlib.sha256(row.tobytes()).digest() for row in x}
        clean = np.array([hashlib.sha256(row.tobytes()).digest() not in hashes for row in q])
        # Distinguish exact boundary ties from epsilon relaxing the metric.
        boundary = {f"gap_le_{eps:g}": float(np.mean(gold_d[:, K - 1] - gold_d[:, K] <= eps))
                    for eps in EPSILONS}
        for seed in audit.args.seeds:
            for kind in ["hnsw", "ivf"]:
                t0 = time.perf_counter()
                ix = build(x, kind, seed)
                found_d, found_i = ix.search(q, K)
                row = {"case": str(r), "seed": seed, "d": x.shape[1], "n": len(x), "index": kind,
                       "nq": len(q), "metrics": metrics(found_d, found_i, gold_d[:, :K], gold_i[:, :K], clean),
                       "exact_topk_boundary": boundary, "seconds": time.perf_counter() - t0}
                if kind == "hnsw":
                    row["max_indegree"], _ = indegree(ix)
                audit.record(row)
                np.savez_compressed(audit.directory / f"wdc_R{r}_{kind}_seed{seed}_scores.npz",
                                    found_scores=found_d, found_ids=found_i, gold_scores=gold_d,
                                    gold_ids=gold_i, clean_query_mask=clean)
                del ix
                gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["pilot", "controls", "jitter", "compression", "wdc"], required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nq", type=int, default=2000)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "pvldb_audit" / "2026-10-08"))
    parser.add_argument("--data", default=str(ROOT / "data" / "wdc_full"))
    args = parser.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    faiss.omp_set_num_threads(args.threads)
    audit = Audit(args)
    try:
        if args.suite in ["pilot", "controls"]:
            controls(audit, pilot=args.suite == "pilot")
        else:
            globals()[args.suite](audit)
        audit.finish()
    except BaseException as exc:
        audit.stop.set()
        audit.data["status"] = "failed"
        audit.data["error"] = repr(exc)
        audit.write()
        raise


if __name__ == "__main__":
    main()
