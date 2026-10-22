# Copyright (c) 2026 the faiss-entity-index authors.
# SPDX-License-Identifier: MIT
# Licensed under the MIT license found in the LICENSE file in this directory.

"""
Public-data demonstration of duplicate injection and group reduction.

Downloads ANN-benchmark vectors from the Hugging Face open-vdb mirror and
injects one exact-duplicate group. Reports HNSW, reduced HNSW and IVF recall
under the tested settings. Results vary with runtime and graph construction.
Recall is top-k ID overlap against exact search on the clean base, excluding
injected copies. This differs from full-gallery tie-aware recall.

The duplication rate is the sink's fraction of the augmented gallery.
Group reduction changes the representation; it is not lossless compression
or a guarantee of improved nearest-neighbor ranking on arbitrary corpora.

Run: python demo_public.py --all
Use ENTITY_INDEX_CACHE to select the download/cache directory.
"""
import argparse
import os
import time
import urllib.request

import numpy as np
import faiss

from entity_index import group_reduce


HF = "https://huggingface.co/datasets/open-vdb/{repo}/resolve/main/train/train-00001-of-00001.parquet"
DATASETS = {
    # tag            : (hf repo,                         metric, faiss metric)
    "fashion-mnist": ("fashion-mnist-784-euclidean", "l2",  faiss.METRIC_L2),
    "glove-25":      ("glove-25-angular",            "cos", faiss.METRIC_INNER_PRODUCT),
    "glove-100":     ("glove-100-angular",           "cos", faiss.METRIC_INNER_PRODUCT),
    "sift-128":      ("sift-128-euclidean",          "l2",  faiss.METRIC_L2),
}


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _default_cache():
    """Stable cache dir independent of the working directory."""
    return os.environ.get(
        "ENTITY_INDEX_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "faiss_entity_index"))


def load_public(tag, cache=None):
    """Download (once) and load a public ANN-benchmark train split as float32."""
    import pyarrow.parquet as pq
    cache = cache or _default_cache()
    repo, metric, faiss_metric = DATASETS[tag]
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{tag}_train.parquet")
    if not os.path.exists(path):
        url = HF.format(repo=repo)
        _log(f"downloading {tag} from HuggingFace open-vdb mirror ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        # Download to a temp file and rename so an interrupted download never
        # leaves a truncated parquet that poisons the cache.
        tmp = path + ".tmp"
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, path)
        _log(f"  saved {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    x = np.asarray(pq.read_table(path).column("emb").to_pylist(), dtype="float32")
    if metric == "cos":
        faiss.normalize_L2(x)
    return np.ascontiguousarray(x), faiss_metric


def hnsw_indeg_max(index, n):
    """Maximum base-layer in-degree of the supplied HNSW graph."""
    h = index.hnsw
    offs = faiss.vector_to_array(h.offsets)
    nbr = faiss.vector_to_array(h.neighbors)
    nb0 = h.nb_neighbors(0)
    deg = np.zeros(n, dtype=np.int64)
    for i in range(n):
        seg = nbr[offs[i]:offs[i] + nb0]
        seg = seg[(seg >= 0) & (seg < n)]
        deg[seg] += 1
    return int(deg.max())


def ann_recall(index, q, exact_I, k=10):
    """ID overlap against the supplied clean-base ground truth."""
    _, I = index.search(q, k)
    return float(np.mean([len(set(a) & set(b)) for a, b in zip(I[:, :k], exact_I[:, :k])]) / k)


def idx_mb(index):
    return faiss.serialize_index(index).nbytes / 1e6


def build_hnsw(x, metric, ef=200):
    ix = faiss.IndexHNSWFlat(x.shape[1], 16, metric)
    ix.hnsw.efConstruction = 200
    ix.add(x)
    ix.hnsw.efSearch = ef
    return ix


def build_ivf(x, metric, nprobe=64):
    nlist = max(16, int(np.sqrt(len(x))))
    ix = faiss.IndexIVFFlat(faiss.IndexFlat(x.shape[1], metric), x.shape[1], nlist, metric)
    ix.train(x)
    ix.add(x)
    ix.nprobe = nprobe
    return ix


def run(tag, dup_rates, n_base, n_queries, seed=0):
    if n_base < 10 or n_queries < 1:
        raise ValueError("Need at least 10 base vectors and one query")
    if any(not 0 <= rate < 1 for rate in dup_rates):
        raise ValueError("Duplication rates must be in [0, 1)")
    x, metric = load_public(tag)
    if n_base + n_queries > len(x):
        raise ValueError("Requested base and queries exceed the dataset size")
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(x), min(n_base + n_queries, len(x)), replace=False)
    base = np.ascontiguousarray(x[sel[:n_base]])
    q = np.ascontiguousarray(x[sel[n_base:n_base + n_queries]])
    d = base.shape[1]

    # Ground truth on the CLEAN base only (stable ids; clean rows stay first as the hub grows).
    _, exact = faiss.knn(q, base, 10, metric=metric)

    # The mega-hub prototype is a REAL corpus vector (an actual record duplicated verbatim,
    # the way a copy-paste listing or boilerplate template appears in production).
    proto = base[0:1].copy()

    metric_name = "cosine" if metric == faiss.METRIC_INNER_PRODUCT else "L2"
    print(f"\n{'='*78}")
    print(f"  {tag}  (REAL public embeddings, d={d}, metric={metric_name})")
    print(f"  clean base = {n_base:,} vectors   |   held-out queries = {n_queries:,}")
    print(f"  mega-hub = an actual base vector duplicated verbatim (jitter=0), as in")
    print(f"  copy-paste catalog listings / empty-text offers / template records.")
    print(f"{'='*78}")
    print(f"{'dup_rate':>9} {'hub_size':>9} {'|index|':>9} {'indeg_max':>10} "
          f"{'HNSW_full':>10} {'HNSW_reduced':>13} {'IVF_full':>9} {'shrink':>7}")

    rows = []
    for dr in dup_rates:
        hub = int(round(dr / (1 - dr) * n_base)) if dr > 0 else 0  # hub as frac of FINAL corpus
        if hub == 0:
            xx, gid = base, np.arange(n_base, dtype="int64")
        else:
            hubv = np.repeat(proto, hub, axis=0).astype("float32")
            xx = np.ascontiguousarray(np.concatenate([base, hubv]))
            gid = np.concatenate([np.arange(n_base), np.zeros(hub, dtype="int64")])

        hf = build_hnsw(xx, metric)
        r_full = ann_recall(hf, q, exact)
        indeg = hnsw_indeg_max(hf, len(xx))
        mb_full = idx_mb(hf)

        # ---- THE FIX: one call collapses every group to a representative ----
        reduced, _, _, _ = group_reduce(xx, gid, method="centroid", metric=metric)
        hr = build_hnsw(reduced, metric)
        r_red = ann_recall(hr, q, exact)
        mb_red = idx_mb(hr)

        iv = build_ivf(xx, metric)
        r_ivf = ann_recall(iv, q, exact)

        shrink = len(xx) / len(reduced)
        rows.append(dict(dup_rate=dr, hub=hub, n=len(xx), indeg=indeg, mb_full=mb_full,
                         mb_red=mb_red, hnsw_full=r_full, hnsw_red=r_red, ivf=r_ivf,
                         shrink=shrink))
        print(f"{dr:>9.2f} {hub:>9,} {len(xx):>9,} {indeg:>10,} "
              f"{r_full:>10.4f} {r_red:>13.4f} {r_ivf:>9.4f} {shrink:>6.1f}x")
    return tag, d, metric_name, rows



def plot(results, out=None):
    try:
        import matplotlib
    except ImportError:
        _log("matplotlib not installed; skipping the comparison figure "
             "(pip install 'faiss-entity-index[plot]' to enable it)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if out is None:
        out = os.path.join(_default_cache(), "demo_public.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5), squeeze=False)
    for ax, (tag, d, mname, rows) in zip(axes[0], results):
        dr = [r["dup_rate"] for r in rows]
        ax.plot(dr, [r["hnsw_full"] for r in rows], "o-", color="tomato", label="HNSW")
        ax.plot(dr, [r["hnsw_red"] for r in rows], "o-", color="seagreen", markersize=9, label="Reduced HNSW")
        ax.plot(dr, [r["ivf"] for r in rows], "s--", color="steelblue", alpha=.8, markersize=5, label="IVF")
        ax.set_xlabel("mega-hub duplication rate")
        ax.set_ylabel("Clean-base ID recall@10")
        ax.set_ylim(0.5, 1.03)
        ax.grid(alpha=.3)
        ax.set_title(f"{tag} (real, d={d}, {mname})")
        ax.legend(fontsize=9, loc="lower left")
    fig.suptitle("Duplicate injection and group reduction on public vectors", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out, dpi=150)
    _log(f"saved figure {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fashion-mnist", choices=list(DATASETS),
                    help="public ann-benchmark dataset (auto-downloaded, cached)")
    ap.add_argument("--dup-rate", type=float, default=None,
                    help="single mega-hub rate to narrate (e.g. 0.2 ~ WDC empty-text); "
                         "omit to sweep")
    ap.add_argument("--n-base", type=int, default=20000, help="clean base size")
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--all", action="store_true",
                    help="run fashion-mnist + glove-25 and save the comparison figure")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.dup_rate is not None and not (0.0 <= args.dup_rate < 1.0):
        ap.error("--dup-rate must be in [0, 1): it is the hub's fraction of the final corpus")

    t0 = time.time()
    if args.all:
        results = [run(tag, [0.0, 0.1, 0.33, 0.5], args.n_base, args.n_queries)
                   for tag in ("fashion-mnist", "glove-25")]
        if not args.no_plot:
            plot(results)
    elif args.dup_rate is not None:
        results = [run(args.dataset, [0.0, args.dup_rate], args.n_base, args.n_queries)]
        if not args.no_plot:
            plot(results)
    else:
        results = [run(args.dataset, [0.0, 0.1, 0.33, 0.5], args.n_base, args.n_queries)]
        if not args.no_plot:
            plot(results, out=os.path.join(_default_cache(), f"demo_{args.dataset}.png"))

    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
