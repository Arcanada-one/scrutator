"""Asyncpg connection pool with pgvector support."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def acquire_search_connection(pool: asyncpg.Pool) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a read-only connection that always uses parameter-aware search plans.

    PostgreSQL's automatic prepared-statement policy may switch from custom to generic
    plans after five executions. On the production corpus that changed the candidate
    set for identical hybrid-search inputs. Keep the override transaction-local so it
    cannot affect ingestion, graph, provenance, or maintenance queries using the pool.
    """
    async with pool.acquire() as conn, conn.transaction(readonly=True):
        await conn.execute("SET LOCAL plan_cache_mode = force_custom_plan")
        if await conn.fetchval("SHOW plan_cache_mode") != "force_custom_plan":
            raise RuntimeError("hybrid retrieval requires PostgreSQL custom query plans")
        yield conn


@asynccontextmanager
async def acquire_scoped(namespace_id: int) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a pooled connection scoped to one tenant for the RLS (SRCH-0023 B2) GUC.

    `SET LOCAL` — set here via the parameterized `set_config(..., true)` form — is
    transaction-scoped by Postgres semantics: it automatically reverts when the transaction
    ends. Wrapping the entire borrow in one explicit transaction is what prevents a pooled
    connection from leaking one tenant's `app.tenant_id` into the next borrower (V-AC-8) —
    `SET LOCAL` executed outside a transaction behaves like session-wide `SET`, which
    asyncpg's pool does not reset on release.
    """
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(namespace_id))
        yield conn
