from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from scrutator.auth.models import TenantContext
from scrutator.config import settings
from scrutator.health import app
from tests.conftest import override_tenant_context

ANON = TenantContext(
    principal_id="anonymous",
    principal_type="anonymous",
    allowed_namespace_ids=frozenset(),
    allowed_namespace_names=frozenset(),
)
BODY = {"content": "# safe", "source_path": "wiki/a.md", "namespace": "wiki"}


def test_index_rejects_anonymous_writer_without_feeder_token():
    with override_tenant_context(app, ANON), TestClient(app) as client:
        response = client.post("/v1/index", json=BODY)
    assert response.status_code == 401


def test_index_accepts_scoped_feeder_token():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "wiki,ecosystem-core"
    try:
        with (
            override_tenant_context(app, ANON),
            patch("scrutator.health.index_document", new_callable=AsyncMock) as index,
            TestClient(app) as client,
        ):
            index.return_value = {
                "chunks_indexed": 1,
                "namespace": "wiki",
                "project": None,
                "source_path": "wiki/a.md",
                "chunk_ids": [],
                "strategy_used": "markdown",
            }
            response = client.post(
                "/v1/index",
                json=BODY,
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
        assert response.status_code == 200
    finally:
        settings.feeder_token, settings.feeder_namespaces = original


def test_index_rejects_feeder_namespace_outside_scope():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "wiki"
    try:
        with override_tenant_context(app, ANON), TestClient(app) as client:
            response = client.post(
                "/v1/index",
                json={**BODY, "namespace": "ecosystem-core"},
                headers={"X-KB-Feeder-Token": "feeder-secret"},
            )
        assert response.status_code == 403
    finally:
        settings.feeder_token, settings.feeder_namespaces = original
