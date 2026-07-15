from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scrutator.auth.models import TenantContext
from scrutator.config import settings
from scrutator.ltm.models import IngestRequest
from scrutator.ltm.router import ingest

CTX = TenantContext(
    principal_id="muneral-writer",
    principal_type="service",
    allowed_namespace_ids=frozenset({7}),
    allowed_namespace_names=frozenset({"muneral"}),
)
GRAPH = {
    "schema_version": 1,
    "content_hash": "a" * 64,
    "entities": [
        {"name": "MUN:1", "entity_type": "task", "properties": {"state": "open"}},
        {"name": "MUN:2", "entity_type": "task", "properties": {}},
    ],
    "edges": [{"source": "MUN:1", "target": "MUN:2", "relation": "depends_on"}],
}


@asynccontextmanager
async def _writer_settings():
    old = settings.ltm_writer_token, settings.ltm_writer_namespaces
    settings.ltm_writer_token = "ltm-secret"
    settings.ltm_writer_namespaces = "muneral"
    try:
        yield
    finally:
        settings.ltm_writer_token, settings.ltm_writer_namespaces = old


@pytest.mark.asyncio
async def test_first_structured_ingest_indexes_and_persists_without_llm():
    req = IngestRequest(
        content="MUN:1 depends on MUN:2",
        source_path="muneral://tasks/1",
        namespace="muneral",
        structured_graph=GRAPH,
    )
    async with _writer_settings():
        with (
            patch("scrutator.ltm.router.repository.get_namespace_id", new_callable=AsyncMock, return_value=None),
            patch("scrutator.ltm.router.repository.upsert_namespace", new_callable=AsyncMock, return_value=7),
            patch("scrutator.ltm.router.repository.create_ltm_job", new_callable=AsyncMock, return_value="job-1"),
            patch("scrutator.ltm.router.repository.update_ltm_job", new_callable=AsyncMock),
            patch(
                "scrutator.ltm.router.repository.get_source_graph_provenance",
                new_callable=AsyncMock,
                return_value={"entity_ids": [], "edge_ids": []},
            ),
            patch(
                "scrutator.search.indexer.index_document",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(chunks_indexed=1),
            ) as index_document,
            patch(
                "scrutator.ltm.router.repository.get_chunk_ids_by_source",
                new_callable=AsyncMock,
                return_value=["chunk-new"],
            ) as chunks,
            patch(
                "scrutator.ltm.router.repository.apply_structured_graph",
                new_callable=AsyncMock,
                return_value={"entities_upserted": 2, "edges_upserted": 1, "idempotent_noop": False},
            ) as apply_graph,
            patch("scrutator.ltm.router._create_llm_client") as llm_factory,
        ):
            response = await ingest(req, CTX, "ltm-secret")

    assert response.entities_upserted == 2
    assert response.edges_upserted == 1
    assert response.idempotent_noop is False
    index_document.assert_awaited_once()
    chunks.assert_awaited_once_with("muneral://tasks/1", 7)
    assert [entity["name"] for entity in apply_graph.await_args.kwargs["entities"]] == ["MUN:1", "MUN:2"]
    assert apply_graph.await_args.kwargs["edges"] == [
        {"source": "MUN:1", "target": "MUN:2", "relation": "depends_on", "weight": 1.0}
    ]
    assert apply_graph.await_args.kwargs["source_chunk_id"] == "chunk-new"
    llm_factory.assert_not_called()


@pytest.mark.asyncio
async def test_matching_structured_hash_is_noop_before_job_and_index_dml():
    req = IngestRequest(
        content="same",
        source_path="muneral://tasks/1",
        namespace="muneral",
        structured_graph=GRAPH,
    )
    async with _writer_settings():
        with (
            patch("scrutator.ltm.router.repository.get_namespace_id", new_callable=AsyncMock, return_value=7),
            patch("scrutator.ltm.router.repository.upsert_namespace", new_callable=AsyncMock) as upsert_namespace,
            patch(
                "scrutator.ltm.router.repository.get_structured_graph_hash",
                new_callable=AsyncMock,
                return_value="a" * 64,
            ),
            patch("scrutator.ltm.router.repository.create_ltm_job", new_callable=AsyncMock) as create_job,
            patch("scrutator.search.indexer.index_document", new_callable=AsyncMock) as index_document,
            patch("scrutator.ltm.router.repository.apply_structured_graph", new_callable=AsyncMock) as apply_graph,
            patch("scrutator.ltm.router._create_llm_client") as llm_factory,
        ):
            response = await ingest(req, CTX, "ltm-secret")

    assert response.idempotent_noop is True
    upsert_namespace.assert_not_awaited()
    create_job.assert_not_awaited()
    index_document.assert_not_awaited()
    apply_graph.assert_not_awaited()
    llm_factory.assert_not_called()


def _pool_with_connection(conn):
    pool = MagicMock()
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire
    return pool


@pytest.mark.asyncio
async def test_repository_converges_source_associations_and_deletes_only_orphan_edges():
    from scrutator.db.repository import apply_structured_graph

    conn = AsyncMock()
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)
    conn.fetchval.return_value = None
    conn.fetchrow.side_effect = [{"id": "entity-1"}, {"id": "entity-2"}, {"id": 41}]
    conn.fetch.return_value = [{"edge_id": 40}]
    pool = _pool_with_connection(conn)

    with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=pool):
        result = await apply_structured_graph(
            namespace_id=7,
            source_path="muneral://tasks/1",
            content_hash="b" * 64,
            entities=GRAPH["entities"],
            edges=GRAPH["edges"],
            source_chunk_id="chunk-new",
            prior_entity_ids=["legacy-entity"],
            prior_edge_ids=[40],
        )

    assert result == {"entities_upserted": 2, "edges_upserted": 1, "idempotent_noop": False}
    sql = "\n".join(str(call.args[0]) for call in [*conn.execute.await_args_list, *conn.fetch.await_args_list])
    assert "pg_advisory_xact_lock" in sql
    assert "INSERT INTO entity_sources" in sql
    assert "INSERT INTO entity_edge_sources" in sql
    assert "DELETE FROM entity_sources" in sql and "NOT (entity_id = ANY" in sql
    assert "DELETE FROM entity_edge_sources" in sql and "NOT (edge_id = ANY" in sql
    assert "DELETE FROM entity_edges" in sql and "NOT EXISTS" in sql
    edge_delete_sql = next(
        call.args[0] for call in conn.execute.await_args_list if "DELETE FROM entity_edges ee" in call.args[0]
    )
    assert "ee.id = ANY" in edge_delete_sql, "only edges detached from this source may be garbage-collected"
    assert "DELETE FROM entities" not in sql
    # Hash publication is the final statement in the transaction.
    assert "INSERT INTO structured_graph_sources" in conn.execute.await_args_list[-1].args[0]


@pytest.mark.asyncio
async def test_repository_hash_short_circuit_has_no_graph_dml():
    from scrutator.db.repository import apply_structured_graph

    conn = AsyncMock()
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)
    conn.fetchval.return_value = "a" * 64
    pool = _pool_with_connection(conn)

    with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=pool):
        result = await apply_structured_graph(
            namespace_id=7,
            source_path="muneral://tasks/1",
            content_hash="a" * 64,
            entities=GRAPH["entities"],
            edges=GRAPH["edges"],
        )

    assert result["idempotent_noop"] is True
    sql = "\n".join(str(call.args[0]) for call in conn.execute.await_args_list)
    assert "pg_advisory_xact_lock" in sql
    assert "INSERT INTO entities" not in sql
    assert "INSERT INTO entity_edges" not in sql


@pytest.mark.asyncio
async def test_hash_is_not_advanced_when_graph_dml_raises():
    from scrutator.db.repository import apply_structured_graph

    conn = AsyncMock()
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)
    conn.fetchval.return_value = "a" * 64
    conn.fetchrow.side_effect = RuntimeError("entity write failed")
    pool = _pool_with_connection(conn)

    with (
        patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=pool),
        pytest.raises(RuntimeError, match="entity write failed"),
    ):
        await apply_structured_graph(
            namespace_id=7,
            source_path="muneral://tasks/1",
            content_hash="b" * 64,
            entities=GRAPH["entities"],
            edges=GRAPH["edges"],
        )

    sql = "\n".join(str(call.args[0]) for call in conn.execute.await_args_list)
    assert "INSERT INTO structured_graph_sources" not in sql
    tx.__aexit__.assert_awaited()


@pytest.mark.asyncio
async def test_source_provenance_lookup_is_namespace_scoped():
    from scrutator.db.repository import get_source_graph_provenance

    conn = AsyncMock()
    conn.fetchrow.return_value = {"entity_ids": ["e-1"], "edge_ids": [1]}
    pool = _pool_with_connection(conn)
    with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=pool):
        await get_source_graph_provenance(7, "same.md")

    sql, namespace_id, source_path = conn.fetchrow.await_args.args
    assert "c.namespace_id = $1" in sql
    assert namespace_id == 7
    assert source_path == "same.md"
