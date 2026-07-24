"""Real-PostgreSQL proof for deterministic hybrid-search tie handling.

CI provides an ephemeral pgvector database through
``SCRUTATOR_DETERMINISM_TEST_DATABASE_URL``. Developer runs skip safely when
that explicit disposable target is absent.
"""

from __future__ import annotations

import os
import random
import uuid

import asyncpg
import pytest
from pgvector.asyncpg import register_vector


@pytest.fixture
async def deterministic_search_database():
    dsn = os.environ.get("SCRUTATOR_DETERMINISM_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("no disposable PostgreSQL in SCRUTATOR_DETERMINISM_TEST_DATABASE_URL")

    schema = f"determinism_{uuid.uuid4().hex}"
    admin = await asyncpg.connect(dsn)
    await admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    await admin.execute(
        f"""
        CREATE TABLE "{schema}".namespaces (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE "{schema}".projects (
            id SERIAL PRIMARY KEY,
            namespace_id INT REFERENCES "{schema}".namespaces(id),
            name TEXT NOT NULL
        );
        CREATE TABLE "{schema}".chunks (
            id UUID PRIMARY KEY,
            namespace_id INT REFERENCES "{schema}".namespaces(id) NOT NULL,
            project_id INT REFERENCES "{schema}".projects(id),
            source_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            chunk_index INT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding_dense vector(1024),
            textsearch_ru tsvector GENERATED ALWAYS AS (
                to_tsvector('russian', content)
            ) STORED,
            textsearch_en tsvector GENERATED ALWAYS AS (
                to_tsvector('english', content)
            ) STORED,
            metadata JSONB DEFAULT '{{}}'
        );
        CREATE TABLE "{schema}".sparse_vectors (
            chunk_id UUID REFERENCES "{schema}".chunks(id) ON DELETE CASCADE PRIMARY KEY,
            token_weights JSONB NOT NULL
        );
        """
    )

    async def initialize(connection: asyncpg.Connection) -> None:
        await register_vector(connection)

    async def configure(connection: asyncpg.Connection) -> None:
        await connection.execute(f'SET search_path TO "{schema}", public')

    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=1,
        init=initialize,
        setup=configure,
    )
    try:
        namespace_id = await pool.fetchval("INSERT INTO namespaces (name) VALUES ('determinism') RETURNING id")
        yield pool, namespace_id
    finally:
        await pool.close()
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("plan_cache_mode", ["force_custom_plan", "force_generic_plan"])
async def test_hybrid_search_repeats_exact_tie_order_on_postgresql(
    deterministic_search_database, monkeypatch, plan_cache_mode
) -> None:
    from scrutator.db.repository import hybrid_search

    pool, namespace_id = deterministic_search_database
    identifiers = [uuid.UUID(int=value) for value in range(1, 7)]
    insertion_order = identifiers.copy()
    random.Random(20260724).shuffle(insertion_order)
    vector = [1.0] + ([0.0] * 1023)

    for chunk_id in insertion_order:
        await pool.execute(
            """
            INSERT INTO chunks (
                id, namespace_id, source_path, source_type, chunk_index,
                content, content_hash, embedding_dense, metadata
            )
            VALUES (
                $1, $2, $3, 'md', 0, 'deterministic tie', $4, $5,
                jsonb_build_object(
                    'actor', CASE WHEN $6 THEN 'keep' ELSE 'exclude' END,
                    'importance', 0.5
                )
            )
            """,
            chunk_id,
            namespace_id,
            f"{chunk_id}.md",
            chunk_id.hex,
            vector,
            chunk_id.int % 2 == 1,
        )
        await pool.execute(
            "INSERT INTO sparse_vectors (chunk_id, token_weights) VALUES ($1, $2::jsonb)",
            chunk_id,
            '{"deterministic": 1.0}',
        )

    await pool.execute(f"SET plan_cache_mode = {plan_cache_mode}")
    monkeypatch.setattr("scrutator.db.repository.get_pool", lambda: _awaitable(pool))
    expected = [str(chunk_id) for chunk_id in identifiers[:2]]

    for sparse_query in (None, {"deterministic": 1.0}):
        observed = []
        for _ in range(30):
            rows = await hybrid_search(
                query_embedding=vector,
                query_text="deterministic tie",
                namespace_id=namespace_id,
                limit=2,
                fetch_multiplier=2,
                query_sparse=sparse_query,
            )
            observed.append([row.chunk_id for row in rows])
        assert observed == [expected] * 30


@pytest.mark.asyncio
async def test_filtered_hybrid_repeats_exact_tie_order_on_postgresql(
    deterministic_search_database, monkeypatch
) -> None:
    from scrutator.db.repository import search_with_filters

    pool, namespace_id = deterministic_search_database
    identifiers = [uuid.UUID(int=value) for value in range(1, 7)]
    insertion_order = identifiers.copy()
    random.Random(20260724).shuffle(insertion_order)
    vector = [1.0] + ([0.0] * 1023)

    for chunk_id in insertion_order:
        await pool.execute(
            """
            INSERT INTO chunks (
                id, namespace_id, source_path, source_type, chunk_index,
                content, content_hash, embedding_dense, metadata
            )
            VALUES (
                $1, $2, $3, 'md', 0, 'deterministic tie', $4, $5,
                jsonb_build_object(
                    'actor', CASE WHEN $6 THEN 'keep' ELSE 'exclude' END,
                    'importance', 0.5
                )
            )
            """,
            chunk_id,
            namespace_id,
            f"{chunk_id}.md",
            chunk_id.hex,
            vector,
            chunk_id.int % 2 == 1,
        )

    await pool.execute("SET plan_cache_mode = force_generic_plan")
    monkeypatch.setattr("scrutator.db.repository.get_pool", lambda: _awaitable(pool))
    monkeypatch.setattr("scrutator.search.embedder.embed_single", lambda _query: _awaitable(vector))
    expected = [str(identifiers[0]), str(identifiers[2])]

    observed = []
    for _ in range(30):
        rows = await search_with_filters(
            query_text="deterministic tie",
            namespace_id=namespace_id,
            actor="keep",
            importance_boost=True,
            limit=2,
        )
        observed.append([row["chunk_id"] for row in rows])
    assert observed == [expected] * 30


async def _awaitable(value):
    return value
