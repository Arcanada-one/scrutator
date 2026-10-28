"""Project-principal writes stay inside the dedicated, server-selected namespace."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from scrutator.auth.models import TenantContext
from scrutator.config import settings
from scrutator.db.models import IndexResponse
from scrutator.health import app
from tests.conftest import override_tenant_context

PROJECT = TenantContext(
    principal_id="project-system-key",
    principal_type="service",
    allowed_namespace_ids=frozenset({7}),
    allowed_namespace_names=frozenset({"knowledge-contract"}),
    scopes=frozenset({"kb:knowledge-contract.read", "kb:ltm.write"}),
)


def test_project_write_preflight_is_read_only_and_bound_to_principal():
    with (
        patch.object(settings, "project_write_namespace", "knowledge-contract"),
        override_tenant_context(app, PROJECT),
        patch(
            "scrutator.auth.dependency.resolve_namespace_selector", new_callable=AsyncMock, return_value=7
        ) as resolve,
        patch("scrutator.health.index_document", new_callable=AsyncMock) as index,
        patch("scrutator.health.delete_by_source", new_callable=AsyncMock) as delete,
        TestClient(app) as client,
    ):
        own = client.get("/v1/project/write-preflight", params={"namespace": "knowledge-contract"})
        wrong = client.get("/v1/project/write-preflight", params={"namespace": "arcanada"})

    assert own.status_code == 200
    assert own.json() == {"namespace": "knowledge-contract", "writable": True}
    assert wrong.status_code == 403
    resolve.assert_awaited_once_with(PROJECT, "knowledge-contract")
    index.assert_not_awaited()
    delete.assert_not_awaited()


def test_project_index_uses_server_namespace_and_wrong_namespace_never_reaches_indexer():
    indexed = IndexResponse(
        chunks_indexed=1, source_path="paper.md", namespace="knowledge-contract", strategy_used="markdown"
    )
    with (
        patch.object(settings, "project_write_namespace", "knowledge-contract"),
        override_tenant_context(app, PROJECT),
        patch("scrutator.auth.dependency.resolve_namespace_selector", new_callable=AsyncMock, return_value=7),
        patch("scrutator.health.index_document", new_callable=AsyncMock, return_value=indexed) as index,
        TestClient(app) as client,
    ):
        wrong = client.post(
            "/v1/project/index", json={"namespace": "arcanada", "content": "synthetic", "source_path": "paper.md"}
        )
        assert wrong.status_code == 403
        index.assert_not_awaited()
        own = client.post(
            "/v1/project/index",
            json={"namespace": "knowledge-contract", "content": "synthetic", "source_path": "paper.md"},
        )

    assert own.status_code == 200
    assert own.json()["namespace"] == "knowledge-contract"
    assert index.await_args.kwargs["namespace"] == "knowledge-contract"
    assert index.await_args.kwargs["project"] == "knowledge-contract"
    assert index.await_args.kwargs["authorized_namespace_id"] == 7


def test_project_index_rejects_mismatched_source_result():
    mismatched = IndexResponse(
        chunks_indexed=1, source_path="other.md", namespace="knowledge-contract", strategy_used="markdown"
    )
    with (
        patch.object(settings, "project_write_namespace", "knowledge-contract"),
        override_tenant_context(app, PROJECT),
        patch("scrutator.auth.dependency.resolve_namespace_selector", new_callable=AsyncMock, return_value=7),
        patch("scrutator.health.index_document", new_callable=AsyncMock, return_value=mismatched),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/project/index",
            json={"namespace": "knowledge-contract", "content": "synthetic", "source_path": "paper.md"},
        )
    assert response.status_code == 503
    assert response.json() == {"detail": "project index result is inconsistent"}


def test_project_delete_uses_granted_namespace_id_and_wrong_namespace_never_reaches_repository():
    with (
        patch.object(settings, "project_write_namespace", "knowledge-contract"),
        override_tenant_context(app, PROJECT),
        patch("scrutator.auth.dependency.resolve_namespace_selector", new_callable=AsyncMock, return_value=7),
        patch("scrutator.health.delete_by_source", new_callable=AsyncMock, return_value=1) as delete,
        TestClient(app) as client,
    ):
        wrong = client.request("DELETE", "/v1/project/index", json={"namespace": "arcanada", "source_path": "paper.md"})
        assert wrong.status_code == 403
        delete.assert_not_awaited()
        own = client.request(
            "DELETE", "/v1/project/index", json={"namespace": "knowledge-contract", "source_path": "paper.md"}
        )

    assert own.status_code == 200
    assert own.json() == {"namespace": "knowledge-contract", "source_path": "paper.md", "chunks_deleted": 1}
    delete.assert_awaited_once_with("paper.md", 7)


def test_project_write_is_off_on_shared_instance_even_with_project_key():
    with (
        patch.object(settings, "project_write_namespace", ""),
        override_tenant_context(app, PROJECT),
        patch("scrutator.health.index_document", new_callable=AsyncMock) as index,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/project/index",
            json={"namespace": "knowledge-contract", "content": "synthetic", "source_path": "paper.md"},
        )
    assert response.status_code == 503
    index.assert_not_awaited()


def test_project_write_requires_service_principal_and_write_scope():
    for context in (
        TenantContext(
            principal_id="reader",
            principal_type="service",
            allowed_namespace_ids=frozenset({7}),
            allowed_namespace_names=frozenset({"knowledge-contract"}),
            scopes=frozenset({"kb:knowledge-contract.read"}),
        ),
        TenantContext(
            principal_id="human",
            principal_type="user",
            allowed_namespace_ids=frozenset({7}),
            allowed_namespace_names=frozenset({"knowledge-contract"}),
            scopes=frozenset({"kb:ltm.write"}),
        ),
    ):
        with (
            patch.object(settings, "project_write_namespace", "knowledge-contract"),
            override_tenant_context(app, context),
            patch("scrutator.health.index_document", new_callable=AsyncMock) as index,
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/project/index",
                json={"namespace": "knowledge-contract", "content": "synthetic", "source_path": "paper.md"},
            )
        assert response.status_code == 403
        index.assert_not_awaited()


def test_project_write_requires_a_namespace_grant_before_mutation():
    ungranted = TenantContext(
        principal_id="project-system-key",
        principal_type="service",
        allowed_namespace_ids=frozenset(),
        allowed_namespace_names=frozenset(),
        scopes=frozenset({"kb:knowledge-contract.read", "kb:ltm.write"}),
    )
    with (
        patch.object(settings, "project_write_namespace", "knowledge-contract"),
        override_tenant_context(app, ungranted),
        patch("scrutator.health.index_document", new_callable=AsyncMock) as index,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/project/index",
            json={"namespace": "knowledge-contract", "content": "synthetic", "source_path": "paper.md"},
        )
    assert response.status_code == 403
    index.assert_not_awaited()


def test_payload_project_cannot_redirect_project_provenance():
    with (
        patch.object(settings, "project_write_namespace", "knowledge-contract"),
        override_tenant_context(app, PROJECT),
        patch("scrutator.auth.dependency.resolve_namespace_selector", new_callable=AsyncMock, return_value=7),
        patch("scrutator.health.index_document", new_callable=AsyncMock) as index,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/project/index",
            json={
                "namespace": "knowledge-contract",
                "project": "arcanada",
                "content": "synthetic",
                "source_path": "paper.md",
            },
        )
    assert response.status_code == 403
    index.assert_not_awaited()


def test_project_bearer_does_not_authorize_global_feeder_or_rollback_routes():
    with (
        patch.object(settings, "project_write_namespace", "knowledge-contract"),
        override_tenant_context(app, PROJECT),
        patch("scrutator.health.index_document", new_callable=AsyncMock) as index,
        patch("scrutator.health.delete_by_source", new_callable=AsyncMock) as delete,
        TestClient(app) as client,
    ):
        headers = {"Authorization": "Bearer arc_api_project_fixture"}
        payload = {"namespace": "knowledge-contract", "content": "synthetic", "source_path": "paper.md"}
        global_index = client.post("/v1/index", json=payload, headers=headers)
        global_delete = client.request(
            "DELETE", "/v1/index", json={"namespace": "knowledge-contract", "source_path": "paper.md"}, headers=headers
        )
    assert global_index.status_code == 401
    assert global_delete.status_code == 401
    index.assert_not_awaited()
    delete.assert_not_awaited()
