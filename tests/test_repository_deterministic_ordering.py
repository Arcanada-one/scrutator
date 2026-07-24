"""Regression coverage for total ordering at every hybrid-search LIMIT boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _pool(connection):
    pool = MagicMock()
    context = AsyncMock()
    context.__aenter__ = AsyncMock(return_value=connection)
    context.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = context
    return pool


async def _hybrid_sql(*, sparse: bool) -> str:
    from scrutator.db.repository import hybrid_search

    connection = AsyncMock()
    connection.fetch.return_value = []
    with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=_pool(connection)):
        await hybrid_search(
            query_embedding=[0.1] * 1024,
            query_text="tied query",
            namespace_id=5,
            query_sparse={"token": 1.0} if sparse else None,
        )
    return connection.fetch.call_args[0][0]


def _assert_dense_and_fts_total_order(sql: str) -> None:
    normalized = " ".join(sql.split())
    assert normalized.count("ORDER BY c.embedding_dense <=> $1, c.id ASC") == 2
    assert normalized.count("plainto_tsquery('english', $4)) DESC, c.id ASC") == 2
    assert "ORDER BY rrf_score DESC, chunk_id ASC" in normalized


@pytest.mark.asyncio
async def test_three_way_hybrid_has_total_order_at_every_limit() -> None:
    sql = await _hybrid_sql(sparse=True)
    normalized = " ".join(sql.split())

    _assert_dense_and_fts_total_order(sql)
    assert "ORDER BY r.rrf_score DESC, r.chunk_id ASC" in normalized
    assert normalized.count("FROM jsonb_each_text($6::jsonb) AS q(key, value)") == 2
    assert normalized.count(") DESC, sv.chunk_id ASC") == 2


@pytest.mark.asyncio
async def test_two_way_hybrid_has_total_order_at_every_limit() -> None:
    sql = await _hybrid_sql(sparse=False)
    _assert_dense_and_fts_total_order(sql)
    assert "ORDER BY r.rrf_score DESC, r.chunk_id ASC" in " ".join(sql.split())


@pytest.mark.asyncio
async def test_filtered_hybrid_has_total_order_at_every_limit() -> None:
    from scrutator.db.repository import search_with_filters

    connection = AsyncMock()
    connection.fetch.return_value = []
    with (
        patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=_pool(connection)),
        patch("scrutator.search.embedder.embed_single", new_callable=AsyncMock, return_value=[0.1] * 1024),
    ):
        await search_with_filters(query_text="tied query", namespace_id=5)

    sql = connection.fetch.call_args[0][0]
    _assert_dense_and_fts_total_order(sql)
    assert "ORDER BY score DESC, r.chunk_id ASC" in " ".join(sql.split())
