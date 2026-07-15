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
    def __init__(self, conn: asyncpg.Connection) -> None:
        self.conn = conn

    async def __aenter__(self) -> asyncpg.Connection:
        return self.conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _Pool:
    def __init__(self, conn: asyncpg.Connection) -> None:
        self.conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self.conn)


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
                id INT PRIMARY KEY, source_entity_id UUID NOT NULL, target_entity_id UUID NOT NULL
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
        await conn.execute("INSERT INTO entity_edges VALUES (11, $1, $2)", task_id, shared_id)
        await conn.executemany(
            "INSERT INTO entity_sources VALUES ($1, 7, $2)",
            [(task_id, source_a), (shared_id, source_a), (shared_id, source_b), (lonely_id, source_a)],
        )
        await conn.executemany("INSERT INTO entity_edge_sources VALUES (11, 7, $1)", [(source_a,), (source_b,)])
        await conn.execute("INSERT INTO structured_graph_sources VALUES (7, $1, $2)", source_a, "a" * 64)

        original_get_pool = repository.get_pool

        async def smoke_pool() -> _Pool:
            return _Pool(conn)

        repository.get_pool = smoke_pool
        try:
            first = await repository.delete_ltm_source(7, source_a)
            second = await repository.delete_ltm_source(7, source_a)
        finally:
            repository.get_pool = original_get_pool

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
    print("PASS: LTM source deletion preserves shared provenance and is idempotent")
