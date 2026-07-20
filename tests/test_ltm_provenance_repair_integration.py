"""Disposable-PostgreSQL integration coverage for the LTM-0014 repair.

Set ``SCRUTATOR_LTM0014_TEST_DATABASE_URL`` to a disposable Scrutator-schema
database. The suite never falls back to the configured application database.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid

import asyncpg
import pytest

from scrutator.config import settings
from scrutator.tools.ltm_provenance_repair import Repair, RepairPlan, apply_plan, read_snapshot, rollback_plan


@pytest.fixture
async def repair_database():
    dsn = os.environ.get("SCRUTATOR_LTM0014_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("no disposable PostgreSQL in SCRUTATOR_LTM0014_TEST_DATABASE_URL")
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4)
    database_name = await pool.fetchval("SELECT current_database()")
    approval = os.environ.get("LTM0014_TEST_DB_GO")
    if not database_name.endswith("_test") or approval != database_name or dsn == settings.database_url:
        await pool.close()
        pytest.fail("LTM-0014 integration DB must be a separately approved *_test database")
    if await pool.fetchval("SELECT to_regclass('public.entity_sources')") is None:
        await pool.close()
        pytest.fail("disposable PostgreSQL does not have the Scrutator schema")
    namespace = f"ltm0014-test-{uuid.uuid4()}"
    namespace_id = await pool.fetchval("INSERT INTO namespaces (name) VALUES ($1) RETURNING id", namespace)
    try:
        yield pool, namespace, namespace_id
    finally:
        await pool.execute("DELETE FROM entity_events WHERE namespace_id = $1", namespace_id)
        await pool.execute("DELETE FROM entity_sources WHERE namespace_id = $1", namespace_id)
        await pool.execute("DELETE FROM entities WHERE namespace_id = $1", namespace_id)
        await pool.execute("DELETE FROM chunks WHERE namespace_id = $1", namespace_id)
        await pool.execute("DELETE FROM namespaces WHERE id = $1", namespace_id)
        await pool.close()


async def _chunk(pool, namespace_id: int, source_path: str, content: str) -> tuple[str, str]:
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    chunk_id = await pool.fetchval(
        """
        INSERT INTO chunks (namespace_id, source_path, source_type, chunk_index, content, content_hash)
        VALUES ($1, $2, 'md', 0, $3, $4) RETURNING id::text
        """,
        namespace_id,
        source_path,
        content,
        content_hash,
    )
    return chunk_id, content_hash


@pytest.mark.asyncio
async def test_apply_reapply_rollback_preserve_healthy_graph_rows(repair_database, monkeypatch):
    pool, namespace, namespace_id = repair_database
    target_chunk, target_hash = await _chunk(pool, namespace_id, "target.md", "SRCH-0025 is complete.")
    healthy_chunk, _ = await _chunk(pool, namespace_id, "healthy.md", "Healthy entity remains linked.")
    target_entity = await pool.fetchval(
        "INSERT INTO entities (namespace_id, name, entity_type) VALUES ($1, 'SRCH-0025', 'task') RETURNING id::text",
        namespace_id,
    )
    healthy_entity = await pool.fetchval(
        """
        INSERT INTO entities (namespace_id, name, entity_type, source_chunk_id)
        VALUES ($1, 'Healthy', 'project', $2::uuid) RETURNING id::text
        """,
        namespace_id,
        healthy_chunk,
    )
    edge_id = await pool.fetchval(
        """
        INSERT INTO entity_edges (source_entity_id, target_entity_id, relation, source_chunk_id)
        VALUES ($1::uuid, $2::uuid, 'references', $3::uuid) RETURNING id
        """,
        healthy_entity,
        target_entity,
        healthy_chunk,
    )
    event_id = await pool.fetchval(
        """
        INSERT INTO entity_events (namespace_id, entity_id, event_type, source_chunk_id)
        VALUES ($1, $2::uuid, 'observed', $3::uuid) RETURNING id::text
        """,
        namespace_id,
        healthy_entity,
        healthy_chunk,
    )
    snapshot = await read_snapshot(pool, namespace)
    repair = Repair(target_entity, "SRCH-0025", "task", target_chunk, "target.md", target_hash)
    plan = RepairPlan.from_snapshot(snapshot, "integration", [repair])
    monkeypatch.setenv("LTM0014_APPLY_GO", plan.plan_sha256)
    monkeypatch.setenv("LTM0014_ROLLBACK_GO", plan.plan_sha256)

    assert await apply_plan(pool, plan) == {"applied": 1, "already_applied": 0}
    assert await apply_plan(pool, plan) == {"applied": 0, "already_applied": 1}
    assert (
        await pool.fetchval("SELECT source_chunk_id::text FROM entities WHERE id = $1::uuid", target_entity)
        == target_chunk
    )
    assert (
        await pool.fetchval("SELECT source_chunk_id::text FROM entities WHERE id = $1::uuid", healthy_entity)
        == healthy_chunk
    )
    assert await pool.fetchval("SELECT source_chunk_id::text FROM entity_edges WHERE id = $1", edge_id) == healthy_chunk
    assert (
        await pool.fetchval("SELECT source_chunk_id::text FROM entity_events WHERE id = $1::uuid", event_id)
        == healthy_chunk
    )

    assert await rollback_plan(pool, plan) == {"rolled_back": 1, "already_rolled_back": 0}
    assert await rollback_plan(pool, plan) == {"rolled_back": 0, "already_rolled_back": 1}
    assert await pool.fetchval("SELECT source_chunk_id FROM entities WHERE id = $1::uuid", target_entity) is None
    assert await pool.fetchval("SELECT count(*) FROM entity_sources WHERE entity_id = $1::uuid", target_entity) == 0


@pytest.mark.asyncio
async def test_insert_conflict_rolls_back_every_entity_update(repair_database, monkeypatch):
    pool, namespace, namespace_id = repair_database
    first_chunk, first_hash = await _chunk(pool, namespace_id, "first.md", "First Entity")
    second_chunk, second_hash = await _chunk(pool, namespace_id, "second.md", "Second Entity")
    first_entity = await pool.fetchval(
        """
        INSERT INTO entities (namespace_id, name, entity_type)
        VALUES ($1, 'First Entity', 'project') RETURNING id::text
        """,
        namespace_id,
    )
    second_entity = await pool.fetchval(
        """
        INSERT INTO entities (namespace_id, name, entity_type)
        VALUES ($1, 'Second Entity', 'project') RETURNING id::text
        """,
        namespace_id,
    )
    await pool.execute(
        """
        INSERT INTO entity_sources (entity_id, namespace_id, source_path, content_hash, source_chunk_id)
        VALUES ($1::uuid, $2, 'second.md', $3, NULL)
        """,
        second_entity,
        namespace_id,
        second_hash,
    )
    snapshot = await read_snapshot(pool, namespace)
    plan = RepairPlan.from_snapshot(
        snapshot,
        "atomicity",
        [
            Repair(first_entity, "First Entity", "project", first_chunk, "first.md", first_hash),
            Repair(second_entity, "Second Entity", "project", second_chunk, "second.md", second_hash),
        ],
    )
    monkeypatch.setenv("LTM0014_APPLY_GO", plan.plan_sha256)

    with pytest.raises(asyncpg.UniqueViolationError):
        await apply_plan(pool, plan)

    assert (
        await pool.fetchval(
            "SELECT count(*) FROM entities WHERE id = ANY($1::uuid[]) AND source_chunk_id IS NOT NULL",
            [first_entity, second_entity],
        )
        == 0
    )
    assert await pool.fetchval("SELECT count(*) FROM entity_sources WHERE entity_id = $1::uuid", first_entity) == 0


@pytest.mark.asyncio
async def test_waiting_rollback_reads_snapshot_after_session_lock(repair_database, monkeypatch):
    pool, namespace, namespace_id = repair_database
    chunk_id, content_hash = await _chunk(pool, namespace_id, "race.md", "Race Entity")
    entity_id = await pool.fetchval(
        """
        INSERT INTO entities (namespace_id, name, entity_type)
        VALUES ($1, 'Race Entity', 'project') RETURNING id::text
        """,
        namespace_id,
    )
    snapshot = await read_snapshot(pool, namespace)
    repair = Repair(entity_id, "Race Entity", "project", chunk_id, "race.md", content_hash)
    plan = RepairPlan.from_snapshot(snapshot, "race", [repair])
    monkeypatch.setenv("LTM0014_ROLLBACK_GO", plan.plan_sha256)
    lock_key = f"ltm0014:{namespace_id}"

    async with pool.acquire() as blocker:
        await blocker.execute("SELECT pg_advisory_lock(hashtextextended($1, 0))", lock_key)
        rollback = asyncio.create_task(rollback_plan(pool, plan))
        await asyncio.sleep(0.1)
        assert not rollback.done()
        async with blocker.transaction():
            await blocker.execute(
                "UPDATE entities SET source_chunk_id = $2::uuid WHERE id = $1::uuid",
                entity_id,
                chunk_id,
            )
            await blocker.execute(
                """
                INSERT INTO entity_sources (entity_id, namespace_id, source_path, content_hash, source_chunk_id)
                VALUES ($1::uuid, $2, 'race.md', $3, $4::uuid)
                """,
                entity_id,
                namespace_id,
                content_hash,
                chunk_id,
            )
        await blocker.execute("SELECT pg_advisory_unlock(hashtextextended($1, 0))", lock_key)

    assert await asyncio.wait_for(rollback, timeout=5) == {"rolled_back": 1, "already_rolled_back": 0}
    assert await pool.fetchval("SELECT source_chunk_id FROM entities WHERE id = $1::uuid", entity_id) is None
