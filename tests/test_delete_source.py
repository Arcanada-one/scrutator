from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from scrutator.config import settings
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


def test_delete_source_accepts_dedicated_rollback_token():
    anonymous = make_tenant_context(frozenset(), frozenset(), principal_id="anonymous")
    original = settings.rollback_token
    original_scope = settings.rollback_namespaces
    settings.rollback_token = "test-rollback-secret"
    settings.rollback_namespaces = "wiki"
    try:
        with (
            override_tenant_context(app, anonymous),
            patch("scrutator.health.get_pool", new_callable=AsyncMock) as get_pool,
            patch("scrutator.health.delete_by_source", new_callable=AsyncMock, return_value=2) as delete,
            TestClient(app) as client,
        ):
            conn = AsyncMock()
            conn.fetchval.return_value = 9
            pool = MagicMock()
            pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
            pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            get_pool.return_value = pool
            response = client.request(
                "DELETE",
                "/v1/index",
                json={"namespace": "wiki", "source_path": "wiki/new.md"},
                headers={"X-KB-Rollback-Token": "test-rollback-secret"},
            )
        assert response.status_code == 200
        delete.assert_awaited_once_with("wiki/new.md", 9)
    finally:
        settings.rollback_token = original
        settings.rollback_namespaces = original_scope


def test_scheduled_rollback_token_rejects_namespace_outside_scope():
    anonymous = make_tenant_context(frozenset(), frozenset(), principal_id="anonymous")
    original = (settings.rollback_token, settings.rollback_namespaces)
    settings.rollback_token = "test-rollback-secret"
    settings.rollback_namespaces = "wiki"
    try:
        with override_tenant_context(app, anonymous), TestClient(app) as client:
            response = client.request(
                "DELETE",
                "/v1/index",
                json={"namespace": "ecosystem-core", "source_path": "a.md"},
                headers={"X-KB-Rollback-Token": "test-rollback-secret"},
            )
        assert response.status_code == 403
    finally:
        settings.rollback_token, settings.rollback_namespaces = original


def test_operator_rollback_token_can_target_other_namespace():
    anonymous = make_tenant_context(frozenset(), frozenset(), principal_id="anonymous")
    original = settings.operator_rollback_token
    settings.operator_rollback_token = "operator-secret"
    try:
        with (
            override_tenant_context(app, anonymous),
            patch("scrutator.health.get_pool", new_callable=AsyncMock) as get_pool,
            patch("scrutator.health.delete_by_source", new_callable=AsyncMock, return_value=1),
            TestClient(app) as client,
        ):
            conn = AsyncMock()
            conn.fetchval.return_value = 8
            pool = MagicMock()
            pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
            pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            get_pool.return_value = pool
            response = client.request(
                "DELETE",
                "/v1/index",
                json={"namespace": "ecosystem-core", "source_path": "a.md"},
                headers={"X-KB-Rollback-Token": "operator-secret"},
            )
        assert response.status_code == 200
    finally:
        settings.operator_rollback_token = original
