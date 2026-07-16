"""Regression test for LTM-0026: pgvector.Vector decode in the cosine reflect path."""

from __future__ import annotations

import numpy as np
from pgvector import Vector

from scrutator.db.repository import dense_to_float32


class TestDenseToFloat32:
    def test_pgvector_vector(self):
        out = dense_to_float32(Vector([1.0, 2.0, 3.0]))
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_plain_list(self):
        out = dense_to_float32([0.5, -0.5])
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [0.5, -0.5])

    def test_ndarray_passthrough(self):
        out = dense_to_float32(np.asarray([1.0, 0.0], dtype=np.float64))
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [1.0, 0.0])

    def test_stacks_into_2d_float32_matrix(self):
        rows = [Vector([1.0, 0.0]), Vector([0.0, 1.0])]
        matrix = np.asarray([dense_to_float32(v) for v in rows], dtype=np.float32)
        assert matrix.shape == (2, 2)
        assert matrix.dtype == np.float32
