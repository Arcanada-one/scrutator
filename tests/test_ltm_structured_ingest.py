from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
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


def test_provenance_schema_keys_are_not_nullable():
    schema = (Path(__file__).parents[1] / "src/scrutator/db/schema.sql").read_text()
    for column in (
        "entity_id UUID REFERENCES entities(id) ON DELETE CASCADE NOT NULL",
        "edge_id INT REFERENCES entity_edges(id) ON DELETE CASCADE NOT NULL",
        "namespace_id INT REFERENCES namespaces(id) ON DELETE CASCADE NOT NULL",
        "source_path TEXT NOT NULL",
        "content_hash TEXT NOT NULL",
    ):
        assert column in schema


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
    lock_sql = conn.execute.await_args_list[0].args[0]
    assert "hashtextextended($1::int::text || ':' || $2, 0)" in lock_sql
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


class _StateTransaction:
    def __init__(self, conn):
        self.conn = conn
        self.snapshot = None

    async def __aenter__(self):
        self.snapshot = deepcopy(self.conn.state)

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.conn.state = self.snapshot
        return False


class _StateConnection:
    """Small asyncpg state fake for behavioral convergence/rollback tests."""

    def __init__(self):
        self.state = {
            "hashes": {},
            "entities": {},
            "edges": {},
            "entity_sources": set(),
            "edge_sources": set(),
            "next_entity": 1,
            "next_edge": 1,
        }
        self.fail_on = None

    def transaction(self):
        return _StateTransaction(self)

    def _fail(self, sql):
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("injected graph write failure")

    async def fetchval(self, sql, namespace_id, source_path):
        return self.state["hashes"].get((namespace_id, source_path))

    async def fetchrow(self, sql, *args):
        self._fail(sql)
        if "INSERT INTO entities" in sql:
            namespace_id, name, entity_type = args[:3]
            key = (namespace_id, name, entity_type)
            entity_id = self.state["entities"].get(key)
            if entity_id is None:
                entity_id = f"00000000-0000-0000-0000-{self.state['next_entity']:012d}"
                self.state["next_entity"] += 1
                self.state["entities"][key] = entity_id
            return {"id": entity_id}
        if "INSERT INTO entity_edges" in sql:
            source_id, target_id, relation = args[:3]
            key = (source_id, target_id, relation)
            edge_id = self.state["edges"].get(key)
            if edge_id is None:
                edge_id = self.state["next_edge"]
                self.state["next_edge"] += 1
                self.state["edges"][key] = edge_id
            return {"id": edge_id}
        raise AssertionError(f"unexpected fetchrow SQL: {sql}")

    async def execute(self, sql, *args):
        self._fail(sql)
        if "pg_advisory_xact_lock" in sql:
            return "SELECT 1"
        if "INSERT INTO entity_sources" in sql and "SELECT id" not in sql:
            entity_id, namespace_id, source_path = args[:3]
            self.state["entity_sources"].add((entity_id, namespace_id, source_path))
        elif "INSERT INTO entity_edge_sources" in sql and "SELECT ee.id" not in sql:
            edge_id, namespace_id, source_path = args[:3]
            self.state["edge_sources"].add((edge_id, namespace_id, source_path))
        elif "DELETE FROM entity_sources" in sql:
            namespace_id, source_path, current_ids = args
            self.state["entity_sources"] = {
                assoc
                for assoc in self.state["entity_sources"]
                if assoc[1:3] != (namespace_id, source_path) or assoc[0] in current_ids
            }
        elif "DELETE FROM entity_edges ee" in sql:
            removed_ids = set(args[0])
            shared_ids = {assoc[0] for assoc in self.state["edge_sources"]}
            deleted_ids = removed_ids - shared_ids
            self.state["edges"] = {
                key: edge_id for key, edge_id in self.state["edges"].items() if edge_id not in deleted_ids
            }
        elif "INSERT INTO structured_graph_sources" in sql:
            namespace_id, source_path, content_hash = args
            self.state["hashes"][(namespace_id, source_path)] = content_hash
        return "OK"

    async def fetch(self, sql, namespace_id, source_path, current_ids):
        self._fail(sql)
        removed = [
            assoc
            for assoc in self.state["edge_sources"]
            if assoc[1:3] == (namespace_id, source_path) and assoc[0] not in current_ids
        ]
        self.state["edge_sources"].difference_update(removed)
        return [{"edge_id": assoc[0]} for assoc in removed]


def _state_pool(conn):
    pool = MagicMock()
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire
    return pool


@pytest.mark.asyncio
async def test_behavioral_shared_edge_convergence_retains_entities_and_isolates_namespace():
    from scrutator.db.repository import apply_structured_graph

    conn = _StateConnection()
    pool = _state_pool(conn)
    entities = GRAPH["entities"]
    edges = GRAPH["edges"]
    with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=pool):
        await apply_structured_graph(7, "shared.md", "a" * 64, entities, edges)
        await apply_structured_graph(7, "other.md", "b" * 64, entities, edges)
        await apply_structured_graph(8, "shared.md", "c" * 64, entities, edges)

        # Removing source A leaves the namespace-7 shared edge owned by source B.
        await apply_structured_graph(7, "shared.md", "d" * 64, entities, [])
        namespace_7_edge_ids = {
            edge_id
            for key, edge_id in conn.state["edges"].items()
            if key[0] in {entity_id for (ns, _, _), entity_id in conn.state["entities"].items() if ns == 7}
        }
        assert any(assoc[0] in namespace_7_edge_ids and assoc[2] == "other.md" for assoc in conn.state["edge_sources"])

        # Removing the final namespace-7 owner deletes only that edge.
        await apply_structured_graph(7, "other.md", "e" * 64, entities, [])

    assert not any(edge_id in namespace_7_edge_ids for edge_id in conn.state["edges"].values())
    namespace_8_entity_ids = {
        entity_id for (namespace_id, _, _), entity_id in conn.state["entities"].items() if namespace_id == 8
    }
    assert any(key[0] in namespace_8_entity_ids for key in conn.state["edges"]), (
        "same source_path in another namespace is untouched"
    )
    assert len(conn.state["entities"]) == 4, "entities are retained after all source edges are removed"
    assert conn.state["hashes"][(8, "shared.md")] == "c" * 64


@pytest.mark.asyncio
async def test_behavioral_transaction_rollback_restores_graph_and_prior_hash():
    from scrutator.db.repository import apply_structured_graph

    conn = _StateConnection()
    pool = _state_pool(conn)
    with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=pool):
        await apply_structured_graph(7, "rollback.md", "a" * 64, GRAPH["entities"], GRAPH["edges"])
        committed = deepcopy(conn.state)
        conn.fail_on = "INSERT INTO entity_edge_sources"
        with pytest.raises(RuntimeError, match="injected graph write failure"):
            await apply_structured_graph(
                7,
                "rollback.md",
                "b" * 64,
                GRAPH["entities"],
                [{"source": "MUN:2", "target": "MUN:1", "relation": "blocks"}],
            )

    assert conn.state == committed
    assert conn.state["hashes"][(7, "rollback.md")] == "a" * 64
