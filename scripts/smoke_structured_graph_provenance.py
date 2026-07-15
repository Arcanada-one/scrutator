"""Live PostgreSQL smoke for structured graph provenance.

Usage:
    SCRUTATOR_TEST_DATABASE_URL=postgresql://... PYTHONPATH=src \
      python scripts/smoke_structured_graph_provenance.py

The runner creates unique namespaces and removes them with cascading cleanup.
It refuses to run without the explicit test DSN environment variable.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg

from scrutator.db import repository


async def _main() -> None:
    dsn = os.environ.get("SCRUTATOR_TEST_DATABASE_URL")
    if not dsn:
        raise SystemExit("SCRUTATOR_TEST_DATABASE_URL is required")

    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4)
    suffix = uuid.uuid4().hex
    namespace_names = [f"ltm-smoke-{suffix}-a", f"ltm-smoke-{suffix}-b"]
    source_a = f"smoke://{suffix}/a"
    source_b = f"smoke://{suffix}/b"
    entities = [
        {"name": f"SMOKE:{suffix}:1", "entity_type": "test", "properties": {}},
        {"name": f"SMOKE:{suffix}:2", "entity_type": "test", "properties": {}},
    ]
    edge = [{"source": entities[0]["name"], "target": entities[1]["name"], "relation": "depends_on"}]

    async def smoke_pool():
        return pool

    original_get_pool = repository.get_pool
    repository.get_pool = smoke_pool
    namespace_ids: list[int] = []
    try:
        async with pool.acquire() as conn:
            tables = await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_name = ANY($1::text[])
                """,
                ["entity_sources", "entity_edge_sources", "structured_graph_sources"],
            )
            assert {row["table_name"] for row in tables} == {
                "entity_sources",
                "entity_edge_sources",
                "structured_graph_sources",
            }
            for name in namespace_names:
                namespace_ids.append(await conn.fetchval("INSERT INTO namespaces(name) VALUES($1) RETURNING id", name))

        ns_a, ns_b = namespace_ids
        await repository.apply_structured_graph(ns_a, source_a, "a" * 64, entities, edge)
        await repository.apply_structured_graph(ns_a, source_b, "b" * 64, entities, edge)
        await repository.apply_structured_graph(ns_b, source_a, "c" * 64, entities, edge)

        await repository.apply_structured_graph(ns_a, source_a, "d" * 64, entities, [])
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM entity_edge_sources WHERE namespace_id=$1", ns_a) == 1

        await repository.apply_structured_graph(ns_a, source_b, "e" * 64, entities, [])
        async with pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    """
                SELECT count(*) FROM entity_edges ee
                JOIN entities e ON e.id=ee.source_entity_id
                WHERE e.namespace_id=$1
                """,
                    ns_a,
                )
                == 0
            )
            assert await conn.fetchval("SELECT count(*) FROM entities WHERE namespace_id=$1", ns_a) == 2
            assert (
                await conn.fetchval(
                    """
                SELECT count(*) FROM entity_edges ee
                JOIN entities e ON e.id=ee.source_entity_id
                WHERE e.namespace_id=$1
                """,
                    ns_b,
                )
                == 1
            )

        before_hash = "e" * 64
        failing_entities = [*entities, {"name": f"SMOKE:{suffix}:extra", "entity_type": "test", "properties": {}}]
        try:
            await repository.apply_structured_graph(
                ns_a,
                source_b,
                "f" * 64,
                failing_entities,
                [{"source": "missing", "target": entities[0]["name"], "relation": "invalid"}],
            )
        except KeyError:
            pass
        else:
            raise AssertionError("rollback probe did not fail")
        async with pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT content_hash FROM structured_graph_sources WHERE namespace_id=$1 AND source_path=$2",
                    ns_a,
                    source_b,
                )
                == before_hash
            )
            assert not await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM entities WHERE namespace_id=$1 AND name=$2)",
                ns_a,
                failing_entities[-1]["name"],
            )
    finally:
        repository.get_pool = original_get_pool
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM namespaces WHERE name = ANY($1::text[])", namespace_names)
        await pool.close()

    print("structured graph provenance smoke: PASS")


if __name__ == "__main__":
    asyncio.run(_main())
