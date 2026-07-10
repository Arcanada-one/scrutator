"""Tests for SRCH-0023 Step 4: scoped enumeration endpoints (V-AC-6).

get_namespaces() / get_stats() must accept a namespace_ids filter and never enumerate
namespaces outside it — closes the tenant-label enumeration oracle.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_pool_mock(mock_conn):
    mock_pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx
    return mock_pool


class TestGetNamespacesScoped:
    def test_namespace_ids_has_no_default(self):
        from scrutator.db.repository import get_namespaces

        sig = inspect.signature(get_namespaces)
        assert "namespace_ids" in sig.parameters
        assert sig.parameters["namespace_ids"].default is inspect.Parameter.empty

    @pytest.mark.asyncio
    async def test_filters_by_namespace_ids(self):
        from scrutator.db.repository import get_namespaces

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"id": 1, "name": "arcanada", "description": None, "chunk_count": 3}]
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            result = await get_namespaces(namespace_ids=frozenset({1}))

        assert len(result) == 1
        sql = mock_conn.fetch.call_args[0][0]
        assert "n.id = ANY" in sql

    @pytest.mark.asyncio
    async def test_empty_allowed_set_returns_no_rows(self):
        """A principal with zero grants must see zero namespaces, not everything."""
        from scrutator.db.repository import get_namespaces

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            result = await get_namespaces(namespace_ids=frozenset())

        assert result == []
        params = mock_conn.fetch.call_args[0][1:]
        assert params[0] == []


class TestGetStatsScoped:
    def test_namespace_ids_has_no_default(self):
        from scrutator.db.repository import get_stats

        sig = inspect.signature(get_stats)
        assert "namespace_ids" in sig.parameters
        assert sig.parameters["namespace_ids"].default is inspect.Parameter.empty

    @pytest.mark.asyncio
    async def test_filters_totals_by_namespace_ids(self):
        from scrutator.db.repository import get_stats

        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = [3, 1, 1]
        mock_conn.fetch.return_value = [{"name": "arcanada", "chunk_count": 3, "project_count": 1}]
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            result = await get_stats(namespace_ids=frozenset({1}))

        assert result["total_chunks"] == 3
        assert len(result["namespaces"]) == 1
        # every fetchval / fetch call must be scoped by namespace_ids, not global counts
        for call in mock_conn.fetchval.call_args_list:
            sql = call[0][0]
            assert "id = ANY" in sql or "namespace_id = ANY" in sql
