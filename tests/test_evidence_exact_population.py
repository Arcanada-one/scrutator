"""Supported raw-only population capability; HTTP source tests, not deployed receiving."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from scrutator.config import settings
from scrutator.health import app
from scrutator.search import indexer


@pytest.fixture
def population_config(monkeypatch):
    monkeypatch.setattr(settings, "feeder_token", "synthetic-fixture-writer")
    monkeypatch.setattr(settings, "feeder_namespaces", "kc2-store")
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", ["kc2-store"])


@pytest.mark.parametrize(
    "scope,flag,status",
    [(None, True, 409), ([], True, 409), (["kc2-store", "other"], True, 409), (["kc2-store"], False, 409)],
)
def test_unqualified_configuration_never_calls_store(population_config, monkeypatch, scope, flag, status):
    monkeypatch.setattr(settings, "evidence_exact_namespaces", scope)
    monkeypatch.setattr(settings, "evidence_exact_bytes", flag)
    with patch.object(indexer, "populate_exact_evidence_atomic", AsyncMock()) as store:
        response = TestClient(app).post(
            "/v1/index/evidence-exact",
            json={"content": "# Exact", "source_path": "x.md"},
            headers={"X-KB-Feeder-Token": "synthetic-fixture-writer"},
        )
    assert response.status_code == status
    store.assert_not_awaited()


def test_missing_writer_and_wrong_scope_never_call_store(population_config, monkeypatch):
    with patch.object(indexer, "populate_exact_evidence_atomic", AsyncMock()) as store:
        client = TestClient(app)
        body = {"content": "# Exact", "source_path": "x.md"}
        assert client.post("/v1/index/evidence-exact", json=body).status_code == 401
        monkeypatch.setattr(settings, "feeder_namespaces", "other")
        assert (
            client.post(
                "/v1/index/evidence-exact", json=body, headers={"X-KB-Feeder-Token": "synthetic-fixture-writer"}
            ).status_code
            == 403
        )
    store.assert_not_awaited()


def test_payload_namespace_cannot_redirect_server_derived_effect(population_config):
    body = {"content": "---\ntitle: Exact\n---\n# Original\n", "source_path": "x.md", "namespace": "other"}
    with (
        patch.object(indexer, "populate_exact_evidence_atomic", AsyncMock(return_value=True)) as store,
        patch.object(indexer, "embed_texts", AsyncMock()) as embed,
        patch.object(indexer, "replace_source_chunks_atomic", AsyncMock()) as replace,
    ):
        response = TestClient(app).post(
            "/v1/index/evidence-exact", json=body, headers={"X-KB-Feeder-Token": "synthetic-fixture-writer"}
        )
    assert response.status_code == 200
    assert response.json()["namespace"] == "kc2-store"
    assert response.json()["chunks_indexed"] == 0
    assert response.json()["strategy_used"] == "evidence_exact_created"
    assert store.await_args.args[0] == "kc2-store"
    assert store.await_args.args[1]["raw_content"] == body["content"]
    embed.assert_not_awaited()
    replace.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversize_population_refuses_before_database(population_config, monkeypatch):
    monkeypatch.setattr(settings, "evidence_population_max_bytes", 8)
    with (
        patch.object(indexer, "populate_exact_evidence_atomic", AsyncMock()) as store,
        pytest.raises(ValueError, match="byte bound"),
    ):
        await indexer.populate_exact_evidence("λ" * 5, "x.md", "kc2-store")
    store.assert_not_awaited()
