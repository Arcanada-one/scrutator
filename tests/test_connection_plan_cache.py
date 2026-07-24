"""Regression coverage for the PostgreSQL planner invariant used by retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _pool_and_connection(*, mode: str = "force_custom_plan"):
    connection = AsyncMock()
    connection.fetchval = AsyncMock(return_value=mode)
    transaction = AsyncMock()
    transaction.__aenter__ = AsyncMock(return_value=None)
    transaction.__aexit__ = AsyncMock(return_value=False)
    connection.transaction = MagicMock(return_value=transaction)
    acquisition = AsyncMock()
    acquisition.__aenter__ = AsyncMock(return_value=connection)
    acquisition.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquisition
    return pool, connection, transaction


@pytest.mark.asyncio
async def test_search_connection_forces_custom_plan_inside_read_only_transaction():
    from scrutator.db.connection import acquire_search_connection

    pool, connection, transaction = _pool_and_connection()

    async with acquire_search_connection(pool) as acquired:
        assert acquired is connection

    connection.transaction.assert_called_once_with(readonly=True)
    connection.execute.assert_awaited_once_with("SET LOCAL plan_cache_mode = force_custom_plan")
    connection.fetchval.assert_awaited_once_with("SHOW plan_cache_mode")
    transaction.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_connection_fails_closed_on_wrong_plan_mode():
    from scrutator.db.connection import acquire_search_connection

    pool, _connection, transaction = _pool_and_connection(mode="auto")

    with pytest.raises(RuntimeError, match="custom query plans"):
        async with acquire_search_connection(pool):
            pytest.fail("wrong plan mode must fail before yielding")

    transaction.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_connection_rolls_back_on_caller_error():
    from scrutator.db.connection import acquire_search_connection

    pool, _connection, transaction = _pool_and_connection()

    with pytest.raises(ValueError, match="caller failed"):
        async with acquire_search_connection(pool):
            raise ValueError("caller failed")

    transaction.__aexit__.assert_awaited_once()
