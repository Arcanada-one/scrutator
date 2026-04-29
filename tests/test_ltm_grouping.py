"""Tests for LTM-0018 cosine-grouping primitive."""

from __future__ import annotations

import numpy as np

from scrutator.ltm.grouping import cluster_by_cosine


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class TestClusterByCosine:
    def test_n_less_than_2_returns_empty(self):
        assert cluster_by_cosine(np.zeros((0, 4), dtype=np.float32), threshold=0.85) == {}
        single = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        assert cluster_by_cosine(single, threshold=0.85) == {}

    def test_two_pair_above_threshold_one_group(self):
        v1 = _unit(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        orth = _unit(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        v2 = _unit(0.9 * v1 + 0.1 * orth)
        vectors = np.stack([v1, v2])
        groups = cluster_by_cosine(vectors, threshold=0.85)
        assert len(groups) == 1
        (members,) = groups.values()
        assert sorted(members) == [0, 1]

    def test_below_threshold_filtered(self):
        v1 = _unit(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        v2 = _unit(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        vectors = np.stack([v1, v2])
        assert cluster_by_cosine(vectors, threshold=0.85) == {}

    def test_transitivity_chain(self):
        v1 = _unit(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        orth1 = _unit(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        orth2 = _unit(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
        # a~b high, b~c high, a~c lower (but transitive union should bind a-c)
        v2 = _unit(0.9 * v1 + 0.1 * orth1)
        v3 = _unit(0.9 * v2 + 0.1 * orth2)
        vectors = np.stack([v1, v2, v3])
        groups = cluster_by_cosine(vectors, threshold=0.85)
        assert len(groups) == 1
        (members,) = groups.values()
        assert sorted(members) == [0, 1, 2]

    def test_unit_norm_invariant(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(5, 8)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        unit = raw / norms
        assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)
        # Should not raise; result is a dict (possibly empty, depending on chance)
        groups = cluster_by_cosine(unit, threshold=0.85)
        assert isinstance(groups, dict)
        # All emitted groups have ≥2 members
        assert all(len(idx) >= 2 for idx in groups.values())
