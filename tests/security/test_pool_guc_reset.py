"""SRCH-0023 V-AC-8 — pooled-connection tenant isolation.

Two tenants served over one pooled asyncpg connection must never leak the prior tenant's
`app.tenant_id` GUC. Unit coverage of the mechanism lives in
tests/test_connection_acquire_scoped.py; this file is the V-AC-8-labelled spec-graph
evidence location per the plan's Step 7 file list.
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


class TestPoolGucIsolation:
    @pytest.mark.asyncio
    async def test_second_tenant_never_sees_first_tenants_guc(self):
        """Two acquire_scoped() calls over the SAME physical pooled connection each set
        their own app.tenant_id inside their own transaction — proving isolation."""
        from scrutator.db.connection import acquire_scoped

        conn = _make_conn_with_transaction()
        mock_pool = _make_pool_mock(conn)

        with patch("scrutator.db.connection.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            async with acquire_scoped(1):
                pass
            async with acquire_scoped(2):
                pass

        first_guc = conn.execute.call_args_list[0][0][1]
        second_guc = conn.execute.call_args_list[1][0][1]
        assert first_guc == "1"
        assert second_guc == "2"
        assert first_guc != second_guc
        # each borrow opened its own transaction — the mechanism SET LOCAL relies on to revert
        assert conn.transaction.call_count == 2

    @pytest.mark.asyncio
    async def test_guc_set_inside_transaction_not_outside(self):
        """SET LOCAL only reverts at transaction end — acquire_scoped MUST execute
        set_config() after entering the transaction, never before/outside it."""
        from scrutator.db.connection import acquire_scoped

        call_order = []
        conn = _make_conn_with_transaction()
        conn.transaction.return_value.__aenter__.side_effect = lambda: call_order.append("txn_enter") or None
        conn.execute.side_effect = lambda *a, **kw: call_order.append("set_guc")

        mock_pool = _make_pool_mock(conn)

        with patch("scrutator.db.connection.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            async with acquire_scoped(1):
                pass

        assert call_order.index("txn_enter") < call_order.index("set_guc")
