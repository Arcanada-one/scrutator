"""Regression tests for LTM-0019 — entity source_chunk_id NULL-persistence bug.

Root cause: `delete_by_source()` (indexer.py, called on every re-ingest) deletes
rows in `chunks`. Schema has `entities.source_chunk_id UUID REFERENCES
chunks(id) ON DELETE SET NULL` (same for entity_edges, entity_events), so the
chunk delete NULLs any entity/edge/event pointing at that chunk via FK
cascade. The subsequent re-`upsert_*` call on re-extraction hits the
`ON CONFLICT ... DO UPDATE SET` branch, which historically never repaired
`source_chunk_id` — leaving it permanently NULL and making the entity
invisible to recall-time enrichment (`get_entities_for_chunks` joins on
`e.source_chunk_id = ANY(chunk_ids)`).

Layer choice: there is no live/test Postgres harness in this repo (verified —
`tests/conftest.py` does not exist; every repository-touching test in
`tests/test_chunk_lookup.py` / `tests/test_edges_by_path.py` mocks
`scrutator.db.repository.get_pool` and inspects `conn.fetch(rise).call_args`
rather than hitting a real DB). Exercising the actual `ON CONFLICT` merge
semantics would require a live Postgres to evaluate the SQL. The closest
honest test at this layer is to assert the *generated SQL text* passed to
`conn.fetchrow` contains the `source_chunk_id = COALESCE(EXCLUDED.source_chunk_id,
<table>.source_chunk_id)` repair clause in each `DO UPDATE SET` — mirroring the
existing convention of inspecting `conn.fetch.call_args` (see
`test_chunk_lookup.py::TestGetChunksBySourcePath::test_with_namespace`).
This test cannot prove the Postgres merge behaves correctly (that requires an
integration/live-DB test), but it does prove the SQL statement itself carries
the repair — which is exactly the surgical fix this task ships.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_pool(row: dict):
    """Create a mock asyncpg pool; conn.fetchrow returns `row`."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetchrow.return_value = row
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


class TestUpsertEntitySourceChunkIdRepair:
    @pytest.mark.asyncio
    async def test_do_update_set_repairs_source_chunk_id(self):
        pool, conn = _mock_pool({"id": "entity-uuid"})
        with patch("scrutator.db.repository.get_pool", return_value=pool):
            from scrutator.db.repository import upsert_entity

            await upsert_entity(
                namespace_id=1,
                name="Auth Arcana",
                entity_type="project",
                source_chunk_id="chunk-uuid-1",
            )

        query = conn.fetchrow.call_args[0][0]
        assert "DO UPDATE SET" in query
        assert "source_chunk_id = COALESCE(EXCLUDED.source_chunk_id, entities.source_chunk_id)" in query, (
            "upsert_entity's ON CONFLICT clause must repair source_chunk_id (LTM-0019)"
        )


class TestUpsertEntityEdgeSourceChunkIdRepair:
    @pytest.mark.asyncio
    async def test_do_update_set_repairs_source_chunk_id(self):
        pool, conn = _mock_pool({"id": 42})
        with patch("scrutator.db.repository.get_pool", return_value=pool):
            from scrutator.db.repository import upsert_entity_edge

            await upsert_entity_edge(
                source_entity_id="entity-a",
                target_entity_id="entity-b",
                relation="related_to",
                source_chunk_id="chunk-uuid-2",
            )

        query = conn.fetchrow.call_args[0][0]
        assert "DO UPDATE SET" in query
        assert "source_chunk_id = COALESCE(EXCLUDED.source_chunk_id, entity_edges.source_chunk_id)" in query, (
            "upsert_entity_edge's ON CONFLICT clause must repair source_chunk_id (LTM-0019)"
        )


class TestUpsertEntityEventSourceChunkIdRepair:
    @pytest.mark.asyncio
    async def test_do_update_set_repairs_source_chunk_id(self):
        pool, conn = _mock_pool({"id": "event-uuid"})
        with patch("scrutator.db.repository.get_pool", return_value=pool):
            from scrutator.db.repository import upsert_entity_event

            await upsert_entity_event(
                namespace_id=1,
                entity_id="entity-uuid",
                event_type="status_change",
                source_chunk_id="chunk-uuid-3",
            )

        query = conn.fetchrow.call_args[0][0]
        assert "DO UPDATE SET" in query
        assert "source_chunk_id = COALESCE(EXCLUDED.source_chunk_id, entity_events.source_chunk_id)" in query, (
            "upsert_entity_event's ON CONFLICT clause must repair source_chunk_id (LTM-0019)"
        )
