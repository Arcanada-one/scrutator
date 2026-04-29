"""LTM-0018 — content-based grouping primitives for ReflectJob.

Pure functions, no I/O. Determinism guaranteed when caller passes vectors in
stable order (e.g. ``ORDER BY chunk_id``) and pins numpy version.
"""

from __future__ import annotations

import numpy as np


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]  # path compression
        i = parent[i]
    return i


def _union(parent: list[int], i: int, j: int) -> None:
    ri, rj = _find(parent, i), _find(parent, j)
    if ri != rj:
        parent[max(ri, rj)] = min(ri, rj)


def cluster_by_cosine(
    vectors: np.ndarray,
    threshold: float,
) -> dict[int, list[int]]:
    """Group row indices by cosine similarity ≥ ``threshold`` via union-find.

    Args:
        vectors: ``(n, dim)`` array of unit-normalized embeddings. Caller MUST
            normalize; the inner product ``V @ V.T`` then equals cosine.
        threshold: edge threshold; recommended ``[0.7, 0.95]``.

    Returns:
        ``{cluster_root_index: sorted([row_indices])}`` for groups of size ≥ 2.
        Singletons are filtered. Empty input or n<2 → ``{}``.
    """
    n = int(vectors.shape[0]) if vectors.ndim == 2 else 0
    if n < 2:
        return {}
    sims = vectors @ vectors.T  # assumes unit-norm rows
    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                _union(parent, i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(parent, i)
        groups.setdefault(root, []).append(i)
    return {k: sorted(v) for k, v in groups.items() if len(v) >= 2}
