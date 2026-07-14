from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from scrutator.health import app
from tests.conftest import make_tenant_context, override_tenant_context


def test_delete_source_uses_granted_namespace_id():
    with (
        override_tenant_context(app, make_tenant_context(frozenset({7}), frozenset({"wiki"}))),
        patch("scrutator.health.resolve_namespace_selector", new_callable=AsyncMock, return_value=7),
        patch("scrutator.health.delete_by_source", new_callable=AsyncMock, return_value=3) as delete,
        TestClient(app) as client,
    ):
        response = client.request("DELETE", "/v1/index", json={"namespace": "wiki", "source_path": "wiki/a.md"})

    assert response.status_code == 200
    assert response.json()["chunks_deleted"] == 3
    delete.assert_awaited_once_with("wiki/a.md", 7)


def test_delete_source_rejects_namespace_outside_grant():
    ctx = make_tenant_context(frozenset({7}), frozenset({"wiki"}))
    with override_tenant_context(app, ctx), TestClient(app) as client:
        response = client.request("DELETE", "/v1/index", json={"namespace": "ecosystem-core", "source_path": "a.md"})
    assert response.status_code == 403
