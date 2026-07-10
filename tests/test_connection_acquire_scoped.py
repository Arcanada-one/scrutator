"""Tests for SRCH-0023 Step 5: acquire_scoped() pooled-connection tenant-GUC isolation.

V-AC-8: two tenants served over one pooled asyncpg connection must never leak the prior
tenant's `app.tenant_id` GUC. `SET LOCAL` (via `set_config(..., true)`) is transaction-scoped
by Postgres semantics — acquire_scoped MUST wrap the whole borrow in one explicit transaction
so the GUC reverts automatically when the transaction ends, before the connection returns to
the pool for the next borrower.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_pool_mock(mock_conn):
    mock_pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx
    return mock_pool


def _make_conn_with_transaction():
    conn = AsyncMock()
    txn_ctx = AsyncMock()
    txn_ctx.__aenter__ = AsyncMock(return_value=None)
    txn_ctx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn_ctx)
    return conn


class TestAcquireScoped:
    @pytest.mark.asyncio
    async def test_sets_tenant_guc_via_set_config(self):
        from scrutator.db.connection import acquire_scoped

        conn = _make_conn_with_transaction()
        mock_pool = _make_pool_mock(conn)

        with patch("scrutator.db.connection.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            async with acquire_scoped(5) as acquired_conn:
                assert acquired_conn is conn

        conn.transaction.assert_called_once()
        conn.execute.assert_called_once()
        sql, *params = conn.execute.call_args[0]
        assert "set_config" in sql
        assert "app.tenant_id" in sql
        assert params[0] == "5"

    @pytest.mark.asyncio
    async def test_uses_explicit_transaction_wrapper(self):
        """SET LOCAL only reverts at transaction end — acquire_scoped MUST open one."""
        from scrutator.db.connection import acquire_scoped

        conn = _make_conn_with_transaction()
        mock_pool = _make_pool_mock(conn)

        with patch("scrutator.db.connection.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            async with acquire_scoped(1):
                pass

        conn.transaction.return_value.__aenter__.assert_awaited_once()
        conn.transaction.return_value.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_two_sequential_tenants_on_same_pooled_conn_do_not_leak(self):
        """Simulate a pool with a single physical connection serving two tenants in sequence —
        each acquire_scoped call must independently set its own tenant_id; the transaction
        wrapper is what guarantees no leakage between them (asserted per-call below)."""
        from scrutator.db.connection import acquire_scoped

        conn = _make_conn_with_transaction()
        mock_pool = _make_pool_mock(conn)

        with patch("scrutator.db.connection.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            async with acquire_scoped(10):
                pass
            async with acquire_scoped(20):
                pass

        assert conn.execute.call_count == 2
        first_call_params = conn.execute.call_args_list[0][0]
        second_call_params = conn.execute.call_args_list[1][0]
        assert first_call_params[1] == "10"
        assert second_call_params[1] == "20"
        assert conn.transaction.call_count == 2
