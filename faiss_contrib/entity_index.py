# Copyright (c) 2026 the faiss-entity-index authors.
# SPDX-License-Identifier: MIT
# Licensed under the MIT license found in the LICENSE file in this directory.

"""
This contrib module builds *entity-reduced* indexes: when a corpus contains many
near-duplicate vectors that describe the same real-world entity (multi-source
product catalogs, deduplicated document collections, etc.), indexing every record
inflates the index with redundant points. Concentrated duplicates can also degrade
graph-ANN navigation recall in the tested configurations. This effect depends on
geometry, numerical precision, search budgets and implementation.

This module collapses each group of vectors to a single representative before
indexing, which shrinks the index in proportion to average group size and changes
the retrieval representation. It can improve navigation while losing original
vector or entity-ranking fidelity. It composes with per-vector
quantization (PQ/SQ): cardinality reduction removes vectors, quantization shrinks
each vector.

The group assignment is supplied by the caller (e.g. an entity-resolution
cluster id). When labels are unavailable, cluster_reduce uses a thresholded
k-nearest-neighbor graph to derive connected components.

Functions
---------
group_reduce          : collapse vectors to one representative per group.
build_entity_index    : reduce + wrap any faiss index so search() returns group ids.
"""

import numpy as np
import faiss


def _normalize_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def group_reduce(x, group_ids, method="centroid",
                 metric=faiss.METRIC_INNER_PRODUCT):
    """
    Collapse each group of vectors to a single representative.

    Parameters
    ----------
    x : array (n, d) float32
        Input vectors.
    group_ids : array (n,)
        Numeric or string group label per row (e.g. entity / cluster id). When labels are unavailable, cluster_reduce uses a thresholded
k-nearest-neighbor graph to derive connected components.
        Rows sharing a label are collapsed to one representative.
    method : {'centroid', 'medoid'}
        'centroid' -- mean of the group's vectors (re-normalized for an
        inner-product metric so it stays comparable to unit-norm data).
        'medoid'   -- the actual group member with the highest summed similarity
        to the rest of the group (a real data point, no synthesis).
    metric : faiss metric
        METRIC_INNER_PRODUCT (centroid is re-normalized) or METRIC_L2.

    Returns
    -------
    reduced : array (g, d) float32
        One representative vector per group.
    out_ids : array (g,) int64
        Integer group id assigned to each representative (0..g-1), in the order
        groups first appear. Use ``id_map`` to recover the original labels.
    id_map : array (g,)
        id_map[i] is the original group label of representative i.
    members : list of arrays
        members[i] holds the row indices of x that were collapsed into
        representative i (for optional expansion of a hit back to its records).
    """
    x = np.ascontiguousarray(x, dtype="float32")
    n, d = x.shape
    group_ids = np.asarray(group_ids)
    if group_ids.shape != (n,):
        raise ValueError("group_ids must have one entry per row")
    if method not in ("centroid", "medoid"):
        raise ValueError("method must be centroid or medoid")
    if metric not in (faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2):
        raise ValueError("metric must be inner product or squared L2")

    # Stable grouping by first appearance.
    labels, first_pos, inverse = np.unique(
        group_ids, return_index=True, return_inverse=True)
    order = np.argsort(first_pos)              # groups in order of first appearance
    relabel = np.empty(len(labels), dtype="int64")
    relabel[order] = np.arange(len(labels))
    g = len(labels)

    members = [[] for _ in range(g)]
    for row, lab in enumerate(inverse):
        members[relabel[lab]].append(row)
    members = [np.array(m, dtype="int64") for m in members]

    reduced = np.empty((g, d), dtype="float32")
    if method == "centroid":
        for i, m in enumerate(members):
            reduced[i] = x[m].mean(axis=0)
        if metric == faiss.METRIC_INNER_PRODUCT:
            reduced = _normalize_rows(reduced)
    else:  # medoid
        for i, m in enumerate(members):
            if len(m) == 1:
                reduced[i] = x[m[0]]
                continue
            sub = x[m].astype(np.float64)
            total = sub.sum(axis=0)
            norm2 = np.einsum("ij,ij->i", sub, sub)
            if metric == faiss.METRIC_INNER_PRODUCT:
                scores = sub @ total - norm2  # summed similarity to other members
            else:
                scores = 2.0 * (sub @ total) - len(sub) * norm2
            reduced[i] = x[m[int(scores.argmax())]]

    out_ids = np.arange(g, dtype="int64")
    id_map = labels[order]
    return reduced, out_ids, id_map, members


def build_entity_index(inner_index, x, group_ids, method="centroid",
                       train_on="reduced"):
    """
    Build an entity-reduced index: collapse groups, then add the representatives
    to ``inner_index`` wrapped so that ``search`` returns the integer group id of
    the matched entity rather than a per-vector row id.

    Parameters
    ----------
    inner_index : faiss.Index
        Any (empty) faiss index over dimension d, e.g.
        ``faiss.index_factory(d, "HNSW32", faiss.METRIC_INNER_PRODUCT)``.
    x, group_ids, method : see ``group_reduce``.
    train_on : {'reduced', 'full'}
        Which vectors to train the inner index on (matters for IVF/PQ). 'reduced'
        trains on the representatives (faster, usually fine); 'full' trains on all
        input vectors before reduction.

    Returns
    -------
    index : faiss.IndexIDMap
        search() yields integer group ids (0..g-1).
    id_map : array (g,)
        Maps the integer group id back to the caller's original group label.
    members : list of arrays
        Original row indices collapsed into each group (for hit expansion).
    """
    if train_on not in ("reduced", "full"):
        raise ValueError("train_on must be reduced or full")
    metric = inner_index.metric_type
    reduced, out_ids, id_map, members = group_reduce(
        x, group_ids, method=method, metric=metric)

    if not inner_index.is_trained:
        train_vecs = reduced if train_on == "reduced" else \
            np.ascontiguousarray(x, dtype="float32")
        inner_index.train(train_vecs)

    index = faiss.IndexIDMap(inner_index)
    index.add_with_ids(reduced, out_ids)
    return index, id_map, members


def cluster_reduce(x, threshold=0.9, method="centroid",
                   metric=faiss.METRIC_INNER_PRODUCT, knn=16):
    """
    Derive groups automatically by near-duplicate clustering (single-linkage over
    edges above ``threshold`` similarity), then reduce. Use when no external group
    labels are available.

    Groups are the connected components of the graph that links each vector to its
    k nearest neighbors whose similarity exceeds ``threshold``. For
    METRIC_INNER_PRODUCT, similarity is the inner product (use unit-norm vectors so
    it is cosine); for METRIC_L2, ``threshold`` is interpreted as a maximum squared
    distance and the comparison is inverted.

    Returns the same tuple as ``group_reduce``: (reduced, out_ids, id_map, members)
    where id_map[i] is the synthetic component id of representative i.
    """
    x = np.ascontiguousarray(x, dtype="float32")
    n, d = x.shape
    if not isinstance(knn, (int, np.integer)) or knn < 1:
        raise ValueError("knn must be a positive integer")

    tmp = faiss.IndexFlat(d, metric)
    tmp.add(x)
    sims, nbrs = tmp.search(x, knn + 1)

    # Union-Find over near-duplicate edges.
    parent = np.arange(n)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for i in range(n):
        remaining = knn
        for j, nb in enumerate(nbrs[i]):
            if nb < 0 or nb == i:
                continue
            if remaining == 0:
                break
            remaining -= 1
            s = sims[i, j]
            link = s >= threshold if metric == faiss.METRIC_INNER_PRODUCT \
                else s <= threshold
            if link:
                ri, rn = find(i), find(int(nb))
                if ri != rn:
                    parent[ri] = rn

    comp = np.array([find(i) for i in range(n)])
    return group_reduce(x, comp, method=method, metric=metric)
