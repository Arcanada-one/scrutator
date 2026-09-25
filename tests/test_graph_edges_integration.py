"""A2-308 — graph_edges behaviour measured against a REAL Postgres, not a mock.

Every other test in this suite mocks asyncpg. That is fine for routing and scoping, but the
three defects fixed here are all defects of what the DATABASE does: whether `xmax = 0`
really distinguishes an insert from an upsert, whether a `DO UPDATE ... WHERE` really
declines the row, and whether the column can hold 2^24+1 without rounding. A mock written
by the same hand as the fix would simply agree with it (A2-287), so these run on a live
server or they do not run at all — and when they do not run, they say so rather than pass.

Point the suite at a server with SCRUTATOR_TEST_DSN; without it the module is skipped.
"""

from __future__ import annotations

import os
import pathlib
import re
import uuid

import asyncpg
import pytest

DSN = os.environ.get("SCRUTATOR_TEST_DSN")

pytestmark = [
    pytest.mark.skipif(not DSN, reason="SCRUTATOR_TEST_DSN not set — live Postgres required"),
    pytest.mark.asyncio,
]

SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "src" / "scrutator" / "db" / "schema.sql"
MIGRATION_006 = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "scrutator"
    / "db"
    / "migrations"
    / "006_graph_edges_weight_numeric.sql"
)

# 2^24 + 1 — the first integer a float32 cannot represent. It rounds to 2^24.
UNREPRESENTABLE_IN_REAL = 16_777_217


def _graph_edges_ddl(source: str) -> str:
    """Lift just the graph_edges table out of schema.sql (the rest needs pgvector)."""
    match = re.search(r"CREATE TABLE IF NOT EXISTS graph_edges \(.*?\);", source, re.S)
    assert match, "graph_edges table not found in schema.sql"
    return match.group(0)


async def _fresh_table(conn, *, weight_type: str = "from-schema"):
    await conn.execute("DROP TABLE IF EXISTS graph_edges")
    await conn.execute("DROP TABLE IF EXISTS chunks")
    # graph_edges FKs chunks(id); a minimal stand-in keeps pgvector out of these tests.
    await conn.execute("CREATE TABLE chunks (id UUID PRIMARY KEY, namespace_id INT)")
    ddl = _graph_edges_ddl(SCHEMA.read_text())
    if weight_type == "real":
        ddl = ddl.replace("weight NUMERIC DEFAULT 1.0", "weight REAL DEFAULT 1.0")
    await conn.execute(ddl)


async def _chunk(conn, namespace_id: int = 1) -> str:
    chunk_id = str(uuid.uuid4())
    await conn.execute("INSERT INTO chunks (id, namespace_id) VALUES ($1::uuid, $2)", chunk_id, namespace_id)
    return chunk_id


@pytest.fixture
async def conn():
    connection = await asyncpg.connect(DSN)
    try:
        yield connection
    finally:
        await connection.close()


class TestUpsertAccounting:
    """`created` must be rows that did not exist, not edges handed to the statement."""

    @staticmethod
    async def _upsert(conn, source, target, *, weight, created_by):
        return await conn.fetchrow(
            """
            INSERT INTO graph_edges (source_chunk_id, target_chunk_id, edge_type, weight, created_by)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5)
            ON CONFLICT (source_chunk_id, target_chunk_id, edge_type)
            DO UPDATE SET weight = EXCLUDED.weight
            WHERE graph_edges.created_by = EXCLUDED.created_by
            RETURNING (xmax = 0) AS was_insert
            """,
            source,
            target,
            "related",
            weight,
            created_by,
        )

    async def test_first_write_is_an_insert_and_the_second_is_not(self, conn):
        await _fresh_table(conn)
        a, b = await _chunk(conn), await _chunk(conn)

        first = await self._upsert(conn, a, b, weight=1, created_by="kc2-geometry@v1")
        second = await self._upsert(conn, a, b, weight=2, created_by="kc2-geometry@v1")

        assert first["was_insert"] is True
        assert second["was_insert"] is False, "xmax = 0 must distinguish the DO UPDATE branch"
        assert await conn.fetchval("SELECT count(*) FROM graph_edges") == 1
        assert int(await conn.fetchval("SELECT weight FROM graph_edges")) == 2

    async def test_a_different_creator_is_refused_not_silently_resolved(self, conn):
        """Neither stealing the row nor leaving a stale owner on a new weight."""
        await _fresh_table(conn)
        a, b = await _chunk(conn), await _chunk(conn)
        await self._upsert(conn, a, b, weight=1, created_by="dreamer")

        refused = await self._upsert(conn, a, b, weight=99, created_by="kc2-geometry@v1")

        assert refused is None, "the conflicting upsert must return no row"
        row = await conn.fetchrow("SELECT weight, created_by FROM graph_edges")
        assert row["created_by"] == "dreamer", "ownership must not be silently transferred"
        assert int(row["weight"]) == 1, "the other creator's weight must not be applied"

    async def test_retirement_by_creator_still_reaches_its_own_rows(self, conn):
        """The reason the refusal matters: DELETE ... WHERE created_by is how edges retire."""
        await _fresh_table(conn)
        a, b = await _chunk(conn), await _chunk(conn)
        await self._upsert(conn, a, b, weight=1, created_by="kc2-geometry@v1")

        result = await conn.execute("DELETE FROM graph_edges WHERE created_by = $1", "kc2-geometry@v1")
        assert int(result.split()[-1]) == 1
        assert await conn.fetchval("SELECT count(*) FROM graph_edges") == 0
