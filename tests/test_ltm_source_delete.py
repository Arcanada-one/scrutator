import copy
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from scrutator.auth.models import TenantContext
from scrutator.config import settings
from scrutator.health import app
from tests.conftest import override_tenant_context

READER = TenantContext(
    principal_id="muneral-reader",
    principal_type="service",
    allowed_namespace_ids=frozenset({7}),
    allowed_namespace_names=frozenset({"muneral"}),
)
ANONYMOUS = TenantContext(
    principal_id="anonymous",
    principal_type="service",
    allowed_namespace_ids=frozenset(),
    allowed_namespace_names=frozenset(),
)
BODY = {"namespace": "muneral", "source_path": "muneral://task/task-2"}


def _delete(*, token=None, body=None, ctx=READER):
    headers = {"X-LTM-Writer-Token": token} if token is not None else {}
    with (
        override_tenant_context(app, ctx),
        patch("scrutator.ltm.router.repository.get_namespace_id", new_callable=AsyncMock, return_value=7) as get_ns,
        patch("scrutator.ltm.router.repository.delete_ltm_source", new_callable=AsyncMock) as delete_source,
        TestClient(app) as client,
    ):
        response = client.request("DELETE", "/v1/ltm/source", json=body or BODY, headers=headers)
    return response, get_ns, delete_source


@pytest.fixture
def source_delete_settings():
    original = (
        settings.ltm_writer_token,
        settings.ltm_writer_namespaces,
        settings.ltm_writer_source_prefixes,
    )
    settings.ltm_writer_token = "ltm-secret"
    settings.ltm_writer_namespaces = "muneral"
    settings.ltm_writer_source_prefixes = json.dumps({"muneral": ["muneral://task/"]})
    yield
    (
        settings.ltm_writer_token,
        settings.ltm_writer_namespaces,
        settings.ltm_writer_source_prefixes,
    ) = original


@pytest.mark.usefixtures("source_delete_settings")
@pytest.mark.parametrize("ctx,token", [(ANONYMOUS, None), (READER, "wrong-secret")])
def test_source_delete_rejects_anonymous_or_wrong_writer_before_dml(ctx, token):
    response, get_ns, delete_source = _delete(token=token, ctx=ctx)
    assert response.status_code == 401
    get_ns.assert_not_awaited()
    delete_source.assert_not_awaited()


@pytest.mark.usefixtures("source_delete_settings")
def test_reader_bearer_alone_cannot_delete_source():
    response, get_ns, delete_source = _delete()
    assert response.status_code == 401
    get_ns.assert_not_awaited()
    delete_source.assert_not_awaited()


@pytest.mark.usefixtures("source_delete_settings")
def test_source_delete_rejects_out_of_namespace_before_dml():
    response, get_ns, delete_source = _delete(
        token="ltm-secret", body={"namespace": "other", "source_path": "muneral://task/task-2"}
    )
    assert response.status_code == 403
    get_ns.assert_not_awaited()
    delete_source.assert_not_awaited()


@pytest.mark.usefixtures("source_delete_settings")
def test_source_delete_rejects_out_of_prefix_before_dml():
    response, get_ns, delete_source = _delete(
        token="ltm-secret", body={"namespace": "muneral", "source_path": "memory://muneral/task-2"}
    )
    assert response.status_code == 403
    get_ns.assert_not_awaited()
    delete_source.assert_not_awaited()


@pytest.mark.usefixtures("source_delete_settings")
@pytest.mark.parametrize("mapping", ["", "not-json", "[]", '{"muneral": []}', '{"muneral": [""]}'])
def test_source_delete_fails_closed_for_absent_or_invalid_prefix_config(mapping):
    settings.ltm_writer_source_prefixes = mapping
    response, get_ns, delete_source = _delete(token="ltm-secret")
    assert response.status_code == 503
    get_ns.assert_not_awaited()
    delete_source.assert_not_awaited()


@pytest.mark.usefixtures("source_delete_settings")
def test_source_delete_returns_exact_repository_counts():
    expected = {
        "chunks_deleted": 1,
        "entity_sources_deleted": 2,
        "edge_sources_deleted": 3,
        "edges_deleted": 1,
        "entities_deleted": 0,
        "idempotent_noop": False,
    }
    with (
        override_tenant_context(app, READER),
        patch("scrutator.ltm.router.repository.get_namespace_id", new_callable=AsyncMock, return_value=7),
        patch(
            "scrutator.ltm.router.repository.delete_ltm_source", new_callable=AsyncMock, return_value=expected
        ) as delete_source,
        TestClient(app) as client,
    ):
        response = client.request("DELETE", "/v1/ltm/source", json=BODY, headers={"X-LTM-Writer-Token": "ltm-secret"})
    assert response.status_code == 200
    assert response.json() == expected
    delete_source.assert_awaited_once_with(7, BODY["source_path"])


class _Transaction:
    def __init__(self, conn):
        self.conn = conn
        self.before = None

    async def __aenter__(self):
        self.before = copy.deepcopy(self.conn.state)

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.conn.state.clear()
            self.conn.state.update(self.before)


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


class _StatefulDeleteConnection:
    def __init__(self, state, *, fail_on=None):
        self.state = state
        self.fail_on = fail_on

    def transaction(self):
        return _Transaction(self)

    async def fetch(self, sql, *args):
        namespace_id, source_path = args
        if "FROM entity_sources" in sql:
            provenance = [
                {"entity_id": item[0]}
                for item in self.state["entity_sources"]
                if item[1:] == (namespace_id, source_path)
            ]
            legacy = []
            if "JOIN chunks" in sql:
                legacy = [
                    {"entity_id": entity_id}
                    for entity_id, entity in self.state["entities"].items()
                    if (entity["source_chunk_id"], namespace_id, source_path) in self.state["chunks"]
                ]
            return list({row["entity_id"]: row for row in (*provenance, *legacy)}.values())
        if "FROM entity_edge_sources" in sql:
            provenance = [
                {"edge_id": item[0]} for item in self.state["edge_sources"] if item[1:] == (namespace_id, source_path)
            ]
            legacy = []
            if "JOIN chunks" in sql:
                legacy = [
                    {"edge_id": edge_id}
                    for edge_id, chunk_id in self.state["edge_source_chunks"].items()
                    if (chunk_id, namespace_id, source_path) in self.state["chunks"]
                ]
            return list({row["edge_id"]: row for row in (*provenance, *legacy)}.values())
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        compact = " ".join(sql.split())
        if self.fail_on and self.fail_on in compact:
            raise RuntimeError("injected failure")
        if "pg_advisory_xact_lock(hashtextextended" in compact:
            return "SELECT 1"
        if compact.startswith("DELETE FROM entity_edge_sources"):
            namespace_id, source_path = args[:2]
            before = len(self.state["edge_sources"])
            self.state["edge_sources"] = {x for x in self.state["edge_sources"] if x[1:] != (namespace_id, source_path)}
            return f"DELETE {before - len(self.state['edge_sources'])}"
        if compact.startswith("DELETE FROM entity_sources"):
            namespace_id, source_path = args[:2]
            before = len(self.state["entity_sources"])
            self.state["entity_sources"] = {
                x for x in self.state["entity_sources"] if x[1:] != (namespace_id, source_path)
            }
            return f"DELETE {before - len(self.state['entity_sources'])}"
        if compact.startswith("DELETE FROM structured_graph_sources"):
            namespace_id, source_path = args[:2]
            existed = (namespace_id, source_path) in self.state["hashes"]
            self.state["hashes"].discard((namespace_id, source_path))
            return f"DELETE {int(existed)}"
        if compact.startswith("DELETE FROM chunks"):
            namespace_id, source_path = args[:2]
            doomed = {x for x in self.state["chunks"] if x[1:] == (namespace_id, source_path)}
            self.state["chunks"] -= doomed
            doomed_ids = {x[0] for x in doomed}
            for entity in self.state["entities"].values():
                if entity["source_chunk_id"] in doomed_ids:
                    entity["source_chunk_id"] = None
            for edge_id, chunk_id in self.state["edge_source_chunks"].items():
                if chunk_id in doomed_ids:
                    self.state["edge_source_chunks"][edge_id] = None
            return f"DELETE {len(doomed)}"
        if compact.startswith("DELETE FROM entity_edges"):
            candidate_ids = set(args[0])
            remaining_sources = {x[0] for x in self.state["edge_sources"]}
            doomed = {
                edge_id
                for edge_id in candidate_ids - remaining_sources
                if self.state["edge_source_chunks"].get(edge_id) is None
            }
            self.state["edges"] -= doomed
            for edge_id in doomed:
                self.state["edge_source_chunks"].pop(edge_id, None)
            return f"DELETE {len(doomed)}"
        if compact.startswith("DELETE FROM entities"):
            candidate_ids = set(args[0])
            sourced = {x[0] for x in self.state["entity_sources"]}
            incident = {
                endpoint
                for edge_id, endpoints in self.state["edge_endpoints"].items()
                if edge_id in self.state["edges"]
                for endpoint in endpoints
            }
            doomed = {
                entity_id
                for entity_id in candidate_ids
                if entity_id not in sourced
                and self.state["entities"][entity_id]["source_chunk_id"] is None
                and entity_id not in incident
            }
            for entity_id in doomed:
                del self.state["entities"][entity_id]
            return f"DELETE {len(doomed)}"
        raise AssertionError(sql)


def _state():
    return {
        "chunks": {("chunk-a", 7, "muneral://task/a")},
        "hashes": {(7, "muneral://task/a"), (8, "muneral://task/a")},
        "entities": {
            "task": {"source_chunk_id": "chunk-a"},
            "shared": {"source_chunk_id": "chunk-a"},
            "lonely": {"source_chunk_id": "chunk-a"},
            "legacy": {"source_chunk_id": "legacy-chunk"},
        },
        "entity_sources": {
            ("task", 7, "muneral://task/a"),
            ("shared", 7, "muneral://task/a"),
            ("shared", 7, "muneral://task/b"),
            ("lonely", 7, "muneral://task/a"),
            ("legacy", 7, "muneral://task/a"),
            ("legacy", 8, "muneral://task/a"),
        },
        "edges": {11},
        "edge_endpoints": {11: ("task", "shared")},
        "edge_source_chunks": {11: "chunk-a"},
        "edge_sources": {
            (11, 7, "muneral://task/a"),
            (11, 7, "muneral://task/b"),
        },
    }


@pytest.mark.asyncio
async def test_repository_delete_preserves_shared_provenance_and_is_idempotent():
    from scrutator.db.repository import delete_ltm_source

    state = _state()
    conn = _StatefulDeleteConnection(state)
    with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=_Pool(conn)):
        first = await delete_ltm_source(7, "muneral://task/a")
        second = await delete_ltm_source(7, "muneral://task/a")

    assert first == {
        "chunks_deleted": 1,
        "entity_sources_deleted": 4,
        "edge_sources_deleted": 1,
        "edges_deleted": 0,
        "entities_deleted": 1,
        "idempotent_noop": False,
    }
    assert second == {
        "chunks_deleted": 0,
        "entity_sources_deleted": 0,
        "edge_sources_deleted": 0,
        "edges_deleted": 0,
        "entities_deleted": 0,
        "idempotent_noop": True,
    }
    assert "shared" in state["entities"]
    assert "task" in state["entities"]
    assert "lonely" not in state["entities"]
    assert "legacy" in state["entities"]
    assert ("legacy", 8, "muneral://task/a") in state["entity_sources"]
    assert (8, "muneral://task/a") in state["hashes"]
    assert ("shared", 7, "muneral://task/b") in state["entity_sources"]
    assert (11, 7, "muneral://task/b") in state["edge_sources"]
    assert 11 in state["edges"]


@pytest.mark.asyncio
async def test_repository_delete_rolls_back_all_state_on_failure():
    from scrutator.db.repository import delete_ltm_source

    state = _state()
    before = copy.deepcopy(state)
    conn = _StatefulDeleteConnection(state, fail_on="DELETE FROM chunks")
    with (
        patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=_Pool(conn)),
        pytest.raises(RuntimeError, match="injected failure"),
    ):
        await delete_ltm_source(7, "muneral://task/a")
    assert state == before


@pytest.mark.asyncio
async def test_repository_deletes_last_edge_provenance_then_unreferenced_nodes():
    from scrutator.db.repository import delete_ltm_source

    state = _state()
    state["entity_sources"] = {item for item in state["entity_sources"] if item[1:] == (7, "muneral://task/a")}
    state["edge_sources"] = {(11, 7, "muneral://task/a")}
    state["entities"].pop("legacy")
    state["entity_sources"] = {item for item in state["entity_sources"] if item[0] != "legacy"}
    conn = _StatefulDeleteConnection(state)
    with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=_Pool(conn)):
        result = await delete_ltm_source(7, "muneral://task/a")

    assert result["edges_deleted"] == 1
    assert result["entities_deleted"] == 3
    assert state["edges"] == set()
    assert state["entities"] == {}


@pytest.mark.asyncio
async def test_repository_deletes_chunk_only_legacy_entity_and_edge():
    from scrutator.db.repository import delete_ltm_source

    state = {
        "chunks": {("legacy-chunk", 7, "muneral://task/legacy")},
        "hashes": {(7, "muneral://task/legacy")},
        "entities": {
            "legacy-source": {"source_chunk_id": "legacy-chunk"},
            "legacy-target": {"source_chunk_id": "legacy-chunk"},
        },
        "entity_sources": set(),
        "edges": {22},
        "edge_endpoints": {22: ("legacy-source", "legacy-target")},
        "edge_source_chunks": {22: "legacy-chunk"},
        "edge_sources": set(),
    }
    conn = _StatefulDeleteConnection(state)
    with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=_Pool(conn)):
        result = await delete_ltm_source(7, "muneral://task/legacy")

    assert result == {
        "chunks_deleted": 1,
        "entity_sources_deleted": 0,
        "edge_sources_deleted": 0,
        "edges_deleted": 1,
        "entities_deleted": 2,
        "idempotent_noop": False,
    }
    assert state["edges"] == set()
    assert state["entities"] == {}


@pytest.mark.asyncio
async def test_repository_preserves_association_edge_with_surviving_legacy_chunk_owner():
    from scrutator.db.repository import delete_ltm_source

    state = _state()
    state["edge_sources"] = {(11, 7, "muneral://task/a")}
    state["edge_source_chunks"][11] = "other-source-chunk"
    conn = _StatefulDeleteConnection(state)
    with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=_Pool(conn)):
        result = await delete_ltm_source(7, "muneral://task/a")

    assert result["edge_sources_deleted"] == 1
    assert result["edges_deleted"] == 0
    assert 11 in state["edges"]
    assert state["edge_source_chunks"][11] == "other-source-chunk"
