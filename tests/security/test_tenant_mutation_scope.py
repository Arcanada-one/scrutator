from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from scrutator.auth.models import TenantContext
from scrutator.health import app
from scrutator.memory.models import MemoryStats
from tests.conftest import override_tenant_context

TENANT_A = TenantContext(
    principal_id="tenant-a-reader",
    principal_type="service",
    allowed_namespace_ids=frozenset({7}),
    allowed_namespace_names=frozenset({"tenant-a"}),
)


def test_cross_tenant_memory_write_is_denied_before_embedding_or_storage():
    with (
        override_tenant_context(app, TENANT_A),
        patch("scrutator.health.index_memory", new_callable=AsyncMock) as index_memory,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/memories",
            json={"content": "secret", "actor": "agent", "namespace": "tenant-b"},
        )

    assert response.status_code == 403
    index_memory.assert_not_awaited()


def test_namespace_creation_cannot_escape_existing_grants():
    with (
        override_tenant_context(app, TENANT_A),
        patch("scrutator.health.upsert_namespace", new_callable=AsyncMock) as upsert,
        TestClient(app) as client,
    ):
        response = client.post("/v1/namespaces", json={"name": "tenant-b"})

    assert response.status_code == 403
    upsert.assert_not_awaited()


def test_edge_read_and_write_forward_the_tenant_scope_to_repository():
    edge = {
        "source_chunk_id": "12345678-1234-5678-1234-567812345678",
        "target_chunk_id": "87654321-4321-8765-4321-876543218765",
        "edge_type": "related",
    }
    with (
        override_tenant_context(app, TENANT_A),
        patch("scrutator.health.insert_edges", new_callable=AsyncMock, return_value=1) as insert,
        patch("scrutator.health.get_edges_for_chunk", new_callable=AsyncMock, return_value=[]) as get_edges,
        TestClient(app) as client,
    ):
        write_response = client.post("/v1/edges", json=[edge])
        read_response = client.get(f"/v1/edges/{edge['source_chunk_id']}")

    assert write_response.status_code == 200
    assert read_response.status_code == 200
    assert insert.await_args.args[1] == frozenset({7})
    assert get_edges.await_args.args[1] == frozenset({7})


def test_memory_stats_are_repository_scoped():
    empty = MemoryStats(total_memories=0)
    with (
        override_tenant_context(app, TENANT_A),
        patch("scrutator.health.get_memory_stats", new_callable=AsyncMock, return_value=empty) as stats,
        TestClient(app) as client,
    ):
        response = client.get("/v1/memories/stats")

    assert response.status_code == 200
    stats.assert_awaited_once_with(frozenset({7}))
