"""Asyncpg connection pool with pgvector support."""

from __future__ import annotations

from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from scrutator.config import settings

_pool: asyncpg.Pool | None = None
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register pgvector type on each new connection."""
    await register_vector(conn)


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool (singleton)."""
    global _pool  # noqa: PLW0603
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=settings.database_pool_min,
            max_size=settings.database_pool_max,
            init=_init_connection,
        )
    return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        await _pool.close()
        _pool = None


async def apply_schema() -> None:
    """Apply database schema (idempotent — uses IF NOT EXISTS)."""
    pool = await get_pool()
    sql = _SCHEMA_PATH.read_text()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(sql)
