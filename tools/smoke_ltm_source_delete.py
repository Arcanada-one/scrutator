#!/usr/bin/env python3
"""Disposable PostgreSQL behavior smoke for LTM source deletion.

Set SCRUTATOR_SOURCE_DELETE_SMOKE_DSN to a disposable PostgreSQL database.
The runner creates and drops a unique schema and never touches public tables.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from contextlib import AbstractAsyncContextManager

import asyncpg

from scrutator.db import repository


class _Acquire(AbstractAsyncContextManager):
    def __init__(self, conn) -> None:
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _Pool:
    def __init__(self, conn) -> None:
        self.conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self.conn)


class _FailingConnection:
    """Delegate to a real connection but fail after early delete mutations."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self.conn = conn

    def transaction(self):
        return self.conn.transaction()

    async def fetch(self, query, *args):
        return await self.conn.fetch(query, *args)

    async def execute(self, query, *args):
        if "DELETE FROM entity_edges" in " ".join(query.split()):
            raise RuntimeError("injected mid-transaction failure")
        return await self.conn.execute(query, *args)


async def _call_delete(conn, namespace_id: int, source_path: str, *, fail: bool = False):
    original_get_pool = repository.get_pool
    selected_conn = _FailingConnection(conn) if fail else conn

    async def smoke_pool() -> _Pool:
        return _Pool(selected_conn)

    repository.get_pool = smoke_pool
    try:
        return await repository.delete_ltm_source(namespace_id, source_path)
    finally:
        repository.get_pool = original_get_pool


async def _snapshot(conn: asyncpg.Connection) -> dict[str, list[tuple]]:
    queries = {
        "chunks": "SELECT id::text, namespace_id, source_path FROM chunks ORDER BY id",
        "entities": "SELECT id::text, source_chunk_id::text FROM entities ORDER BY id",
        "edges": (
            "SELECT id, source_entity_id::text, target_entity_id::text, source_chunk_id::text "
            "FROM entity_edges ORDER BY id"
        ),
        "entity_sources": (
            "SELECT entity_id::text, namespace_id, source_path FROM entity_sources "
            "ORDER BY entity_id, namespace_id, source_path"
        ),
        "edge_sources": (
            "SELECT edge_id, namespace_id, source_path FROM entity_edge_sources "
            "ORDER BY edge_id, namespace_id, source_path"
        ),
        "hashes": (
            "SELECT namespace_id, source_path, content_hash FROM structured_graph_sources "
            "ORDER BY namespace_id, source_path"
        ),
    }
    return {name: [tuple(row) for row in await conn.fetch(query)] for name, query in queries.items()}


async def _seed_last_source(
    conn: asyncpg.Connection,
    namespace_id: int,
    source_path: str,
    edge_id: int,
) -> tuple[uuid.UUID, uuid.UUID]:
    chunk_id, source_id, target_id = [uuid.uuid4() for _ in range(3)]
    await conn.execute("INSERT INTO chunks VALUES ($1, $2, $3)", chunk_id, namespace_id, source_path)
    await conn.executemany(
        "INSERT INTO entities VALUES ($1, $2)",
        [(source_id, chunk_id), (target_id, chunk_id)],
    )
    await conn.execute("INSERT INTO entity_edges VALUES ($1, $2, $3, $4)", edge_id, source_id, target_id, chunk_id)
    await conn.executemany(
        "INSERT INTO entity_sources VALUES ($1, $2, $3)",
        [(source_id, namespace_id, source_path), (target_id, namespace_id, source_path)],
    )
    await conn.execute("INSERT INTO entity_edge_sources VALUES ($1, $2, $3)", edge_id, namespace_id, source_path)
    await conn.execute("INSERT INTO structured_graph_sources VALUES ($1, $2, $3)", namespace_id, source_path, "a" * 64)
    return source_id, target_id


async def _run() -> None:
    dsn = os.environ.get("SCRUTATOR_SOURCE_DELETE_SMOKE_DSN")
    if not dsn:
        raise SystemExit("SCRUTATOR_SOURCE_DELETE_SMOKE_DSN is required")
    schema = f"ltm_source_delete_smoke_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute(
            """
            CREATE TABLE chunks (id UUID PRIMARY KEY, namespace_id INT NOT NULL, source_path TEXT NOT NULL);
            CREATE TABLE entities (
                id UUID PRIMARY KEY, source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL
            );
            CREATE TABLE entity_edges (
                id INT PRIMARY KEY,
                source_entity_id UUID NOT NULL,
                target_entity_id UUID NOT NULL,
                source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL
            );
            CREATE TABLE entity_sources (
                entity_id UUID NOT NULL, namespace_id INT NOT NULL, source_path TEXT NOT NULL
            );
            CREATE TABLE entity_edge_sources (
                edge_id INT NOT NULL, namespace_id INT NOT NULL, source_path TEXT NOT NULL
            );
            CREATE TABLE structured_graph_sources (
                namespace_id INT NOT NULL, source_path TEXT NOT NULL, content_hash TEXT NOT NULL
            );
            """
        )
        source_a = "muneral://task/a"
        source_b = "muneral://task/b"
        chunk_id, task_id, shared_id, lonely_id = [uuid.uuid4() for _ in range(4)]
        await conn.execute("INSERT INTO chunks VALUES ($1, 7, $2)", chunk_id, source_a)
        await conn.executemany(
            "INSERT INTO entities VALUES ($1, $2)",
            [(task_id, chunk_id), (shared_id, chunk_id), (lonely_id, chunk_id)],
        )
        await conn.execute("INSERT INTO entity_edges VALUES (11, $1, $2, $3)", task_id, shared_id, chunk_id)
        await conn.executemany(
            "INSERT INTO entity_sources VALUES ($1, 7, $2)",
            [(task_id, source_a), (shared_id, source_a), (shared_id, source_b), (lonely_id, source_a)],
        )
        await conn.executemany("INSERT INTO entity_edge_sources VALUES (11, 7, $1)", [(source_a,), (source_b,)])
        await conn.execute("INSERT INTO structured_graph_sources VALUES (7, $1, $2)", source_a, "a" * 64)

        first = await _call_delete(conn, 7, source_a)
        second = await _call_delete(conn, 7, source_a)

        expected = {
            "chunks_deleted": 1,
            "entity_sources_deleted": 3,
            "edge_sources_deleted": 1,
            "edges_deleted": 0,
            "entities_deleted": 1,
            "idempotent_noop": False,
        }
        assert first == expected, first
        assert second["idempotent_noop"] is True and not any(
            value for key, value in second.items() if key != "idempotent_noop"
        ), second
        assert await conn.fetchval("SELECT EXISTS(SELECT 1 FROM entities WHERE id=$1)", shared_id)
        assert await conn.fetchval("SELECT EXISTS(SELECT 1 FROM entity_edges WHERE id=11)")
        assert await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM entity_sources WHERE entity_id=$1 AND source_path=$2)", shared_id, source_b
        )

        rollback_source = "muneral://task/rollback"
        await _seed_last_source(conn, 7, rollback_source, 21)
        before_failure = await _snapshot(conn)
        try:
            await _call_delete(conn, 7, rollback_source, fail=True)
        except RuntimeError as exc:
            assert str(exc) == "injected mid-transaction failure"
        else:
            raise AssertionError("expected injected deletion failure")
        assert await _snapshot(conn) == before_failure

        last_source = "muneral://task/last-source"
        last_entities = await _seed_last_source(conn, 7, last_source, 31)
        last_result = await _call_delete(conn, 7, last_source)
        assert last_result == {
            "chunks_deleted": 1,
            "entity_sources_deleted": 2,
            "edge_sources_deleted": 1,
            "edges_deleted": 1,
            "entities_deleted": 2,
            "idempotent_noop": False,
        }, last_result
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM entities WHERE id=ANY($1::uuid[]))", last_entities)

        same_source = "muneral://task/same-path"
        ns7_entities = await _seed_last_source(conn, 7, same_source, 41)
        ns8_entities = await _seed_last_source(conn, 8, same_source, 42)
        cross_result = await _call_delete(conn, 7, same_source)
        assert cross_result["chunks_deleted"] == 1
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM entities WHERE id=ANY($1::uuid[]))", ns7_entities)
        assert await conn.fetchval("SELECT COUNT(*) = 2 FROM entities WHERE id=ANY($1::uuid[])", ns8_entities)
        assert await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM structured_graph_sources WHERE namespace_id=8 AND source_path=$1)",
            same_source,
        )
    finally:
        await conn.execute("RESET search_path")
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("PASS: LTM source deletion preserves provenance, rollback, cleanup, and tenant isolation")
