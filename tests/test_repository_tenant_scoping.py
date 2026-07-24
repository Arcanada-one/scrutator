"""Tests for SRCH-0023 Step 3 (the crux): repository read queries reject a NULL/absent
namespace_id — the `($2::int IS NULL OR c.namespace_id = $2)` escape hatch that caused the
confirmed full-corpus cross-tenant leak is deleted.

V-AC-1, V-AC-3, V-AC-4.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_pool_mock(mock_conn):
    transaction = AsyncMock()
    transaction.__aenter__ = AsyncMock(return_value=None)
    transaction.__aexit__ = AsyncMock(return_value=False)
    mock_conn.transaction = MagicMock(return_value=transaction)
    mock_conn.fetchval = AsyncMock(return_value="force_custom_plan")
    mock_pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx
    return mock_pool


class TestHybridSearchMandatoryNamespaceId:
    def test_namespace_id_has_no_default(self):
        from scrutator.db.repository import hybrid_search

        sig = inspect.signature(hybrid_search)
        assert sig.parameters["namespace_id"].default is inspect.Parameter.empty, (
            "hybrid_search.namespace_id must be mandatory (no default) — "
            "an optional namespace_id is how the NULL-namespace full-corpus leak re-enters"
        )

    @pytest.mark.asyncio
    async def test_call_without_namespace_id_raises_type_error(self):
        from scrutator.db.repository import hybrid_search

        with pytest.raises(TypeError):
            await hybrid_search(query_embedding=[0.1] * 1024, query_text="q")

    @pytest.mark.asyncio
    async def test_3way_sql_has_no_unscoped_escape(self):
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                query_sparse={"tok": 1.0},
            )

        sql = mock_conn.fetch.call_args[0][0]
        assert "IS NULL OR" not in sql, "unscoped-read escape hatch must be deleted from the 3-way RRF path"
        assert "c.namespace_id = $2" in sql
        # positional params: (vector, namespace_id, fetch_limit, query_text, sql_final_limit, sparse_json)
        namespace_id_param = mock_conn.fetch.call_args[0][2]
        assert namespace_id_param == 5

    @pytest.mark.asyncio
    async def test_2way_sql_has_no_unscoped_escape(self):
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(query_embedding=[0.1] * 1024, query_text="q", namespace_id=5)

        sql = mock_conn.fetch.call_args[0][0]
        assert "IS NULL OR" not in sql, "unscoped-read escape hatch must be deleted from the 2-way RRF path"
        assert "c.namespace_id = $2" in sql


class TestSearchWithFiltersMandatoryNamespaceId:
    def test_namespace_id_has_no_default(self):
        from scrutator.db.repository import search_with_filters

        sig = inspect.signature(search_with_filters)
        assert sig.parameters["namespace_id"].default is inspect.Parameter.empty

    @pytest.mark.asyncio
    async def test_call_without_namespace_id_raises_type_error(self):
        from scrutator.db.repository import search_with_filters

        with (
            patch("scrutator.search.embedder.embed_single", new_callable=AsyncMock, return_value=[0.1] * 1024),
            pytest.raises(TypeError),
        ):
            await search_with_filters(query_text="q")

    @pytest.mark.asyncio
    async def test_sql_has_no_unscoped_escape(self):
        from scrutator.db.repository import search_with_filters

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool),
            patch("scrutator.search.embedder.embed_single", new_callable=AsyncMock, return_value=[0.1] * 1024),
        ):
            await search_with_filters(query_text="q", namespace_id=5, include_expired=True)

        sql = mock_conn.fetch.call_args[0][0]
        # include_expired=True suppresses the unrelated valid_until filter so this assertion
        # isolates the namespace-scoping escape hatch specifically.
        assert "IS NULL OR" not in sql, "unscoped-read escape hatch must be deleted from search_with_filters"
        assert "c.namespace_id = $2" in sql
