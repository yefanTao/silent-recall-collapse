# Copyright (c) 2026 the faiss-entity-index authors.
# SPDX-License-Identifier: MIT
# Licensed under the MIT license found in the LICENSE file in this directory.

import unittest
import numpy as np
import faiss

from entity_index import group_reduce, build_entity_index, cluster_reduce
from unittest.mock import patch
import demo_public


def make_grouped_data(n_groups=200, group_size=8, d=32, jitter=0.05, seed=123):
    """n_groups prototypes, each replicated group_size times with small jitter.
    Returns (x unit-norm float32, group_ids int64, prototypes)."""
    rng = np.random.default_rng(seed)
    proto = rng.standard_normal((n_groups, d)).astype("float32")
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    xs, gids = [], []
    for g in range(n_groups):
        for _ in range(group_size):
            v = proto[g] + rng.normal(0, jitter, size=d).astype("float32")
            xs.append(v / np.linalg.norm(v))
            gids.append(g)
    x = np.ascontiguousarray(np.stack(xs), dtype="float32")
    gid = np.array(gids, dtype="int64")
    # shuffle so groups are interleaved (don't rely on contiguous layout)
    perm = rng.permutation(len(x))
    return x[perm], gid[perm], proto


class TestGroupReduce(unittest.TestCase):

    def test_centroid_shapes_and_ids(self):
        x, gid, _ = make_grouped_data(n_groups=50, group_size=6, d=16)
        reduced, out_ids, id_map, members = group_reduce(x, gid, method="centroid")
        self.assertEqual(reduced.shape, (50, 16))
        self.assertEqual(reduced.dtype, np.float32)
        self.assertEqual(out_ids.shape, (50,))
        np.testing.assert_array_equal(out_ids, np.arange(50))
        # every input row is accounted for exactly once
        all_rows = np.concatenate(members)
        np.testing.assert_array_equal(np.sort(all_rows), np.arange(len(x)))
        # id_map recovers the original labels
        self.assertEqual(set(id_map.tolist()), set(gid.tolist()))

    def test_centroid_renormalized_for_ip(self):
        x, gid, _ = make_grouped_data(n_groups=30, group_size=5, d=24)
        reduced, _, _, _ = group_reduce(x, gid, method="centroid",
                                        metric=faiss.METRIC_INNER_PRODUCT)
        norms = np.linalg.norm(reduced, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_medoid_is_real_member(self):
        x, gid, _ = make_grouped_data(n_groups=20, group_size=7, d=16)
        reduced, _, id_map, members = group_reduce(x, gid, method="medoid")
        for i in range(len(reduced)):
            # the medoid must equal one of the actual input rows of that group
            sub = x[members[i]]
            self.assertTrue(np.any(np.all(np.isclose(sub, reduced[i]), axis=1)))

    def test_singleton_group(self):
        x = np.eye(4, dtype="float32")
        gid = np.array([0, 0, 1, 2], dtype="int64")
        reduced, out_ids, id_map, members = group_reduce(x, gid, method="medoid")
        self.assertEqual(reduced.shape, (3, 4))
        self.assertEqual(len(members[1]), 1)


class TestBuildEntityIndex(unittest.TestCase):

    def _recall_at_1(self, index, id_map, queries, true_gids):
        _, I = index.search(queries, 1)
        got = id_map[I[:, 0]]
        return float((got == true_gids).mean())

    def test_ntotal_equals_n_groups(self):
        x, gid, _ = make_grouped_data(n_groups=100, group_size=8, d=32)
        inner = faiss.index_factory(32, "Flat", faiss.METRIC_INNER_PRODUCT)
        index, id_map, members = build_entity_index(inner, x, gid, method="centroid")
        self.assertEqual(index.ntotal, 100)
        self.assertEqual(len(id_map), 100)

    def test_search_returns_group_ids_and_recovers_entity(self):
        # query = a fresh jittered copy of each prototype; must map to its group
        x, gid, proto = make_grouped_data(n_groups=120, group_size=8, d=32)
        rng = np.random.default_rng(7)
        q = proto + rng.normal(0, 0.05, size=proto.shape).astype("float32")
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        q = np.ascontiguousarray(q, dtype="float32")
        true_gids = np.arange(120)

        for factory in ["Flat", "HNSW32", "IVF64,Flat"]:
            inner = faiss.index_factory(32, factory, faiss.METRIC_INNER_PRODUCT)
            index, id_map, _ = build_entity_index(inner, x, gid, method="centroid")
            self.assertEqual(index.ntotal, 120)
            r = self._recall_at_1(index, id_map, q, true_gids)
            self.assertGreaterEqual(r, 0.8, f"{factory} recall@1={r}")

    def test_medoid_also_builds(self):
        x, gid, _ = make_grouped_data(n_groups=60, group_size=6, d=16)
        inner = faiss.index_factory(16, "Flat", faiss.METRIC_INNER_PRODUCT)
        index, id_map, members = build_entity_index(inner, x, gid, method="medoid")
        self.assertEqual(index.ntotal, 60)



class TestReleaseBehavior(unittest.TestCase):
    def test_medoid_matches_pairwise_objective(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(19, 5)).astype("float32")
        for metric in [faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2]:
            got, _, _, _ = group_reduce(x, np.zeros(len(x)), method="medoid", metric=metric)
            if metric == faiss.METRIC_INNER_PRODUCT:
                objective = x.astype(float) @ x.astype(float).T
                np.fill_diagonal(objective, 0)
                winner = objective.sum(1).argmax()
            else:
                distances = ((x[:, None].astype(float)-x[None, :].astype(float))**2).sum(2)
                winner = distances.sum(1).argmin()
            np.testing.assert_array_equal(got[0], x[winner])

    def test_invalid_reduction_settings_are_rejected(self):
        x = np.eye(3, dtype="float32")
        for kwargs in [{"method": "mean_typo"}, {"metric": faiss.METRIC_L1}]:
            with self.assertRaises(ValueError):
                group_reduce(x, np.arange(3), **kwargs)

    def test_invalid_training_choice_is_rejected(self):
        with self.assertRaises(ValueError):
            build_entity_index(faiss.IndexFlatL2(3), np.eye(3, dtype="float32"), np.arange(3), train_on="typo")

    def test_demo_copies_keep_original_group_and_ids(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(120, 8)).astype("float32")
        with patch.object(demo_public, "load_public", return_value=(x, faiss.METRIC_L2)):
            with patch.object(demo_public, "group_reduce", wraps=group_reduce) as reduce:
                demo_public.run("fashion-mnist", [0.5], 80, 20)
        vectors, groups = reduce.call_args.args
        np.testing.assert_array_equal(groups[80:], np.zeros(80, dtype="int64"))
        representatives, _, mapping, members = group_reduce(vectors, groups, metric=faiss.METRIC_L2)
        self.assertEqual(len(representatives), 80)
        np.testing.assert_array_equal(mapping, np.arange(80))
        self.assertEqual(len(members[0]), 81)

    def test_demo_rejects_impossible_query_split(self):
        with patch.object(demo_public, "load_public", return_value=(np.eye(20, dtype="float32"), faiss.METRIC_L2)):
            with self.assertRaises(ValueError):
                demo_public.run("fashion-mnist", [0.5], 20, 5)



class TestClusteringDependency(unittest.TestCase):
    def test_recovers_planted_groups(self):
        x, _, _ = make_grouped_data(n_groups=40, group_size=10, d=32, jitter=0.02)
        reduced, _, _, members = cluster_reduce(x, threshold=0.9, knn=16)
        self.assertGreaterEqual(len(reduced), 35)
        self.assertLessEqual(len(reduced), 45)
        np.testing.assert_array_equal(np.sort(np.concatenate(members)), np.arange(len(x)))

    def test_top_neighbor_need_not_be_self(self):
        # Different norms can put another point above self in inner-product order.
        x = np.array([[1., 0.], [2., 0.], [0., 1.]], dtype="float32")
        _, _, _, members = cluster_reduce(x, threshold=1.5, knn=1)
        groups = {frozenset(m.tolist()) for m in members}
        self.assertEqual(groups, {frozenset([0, 1]), frozenset([2])})

if __name__ == "__main__":
    unittest.main()
