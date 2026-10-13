from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from scrutator.config import settings
from scrutator.db.models import IndexResponse
from scrutator.health import app

FEEDER_TOKEN = "isolated-test-feeder-token"
NAMESPACE = "kc2-store"


def _configure_exact_evidence(monkeypatch, *, feeder_namespaces: str = NAMESPACE) -> None:
    monkeypatch.setattr(settings, "feeder_token", FEEDER_TOKEN)
    monkeypatch.setattr(settings, "feeder_namespaces", feeder_namespaces)
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", [NAMESPACE])


def _request_body() -> dict[str, str]:
    return {
        "content": "candidate-local exact evidence",
        "source_path": "tests/fixtures/exact-evidence.md",
        # The handler must choose its server-pinned namespace instead of this value.
        "namespace": "caller-selected-other-namespace",
    }


def test_exact_evidence_rejects_absent_feeder_token_without_store_call(monkeypatch):
    _configure_exact_evidence(monkeypatch)
    store = AsyncMock()

    with patch("scrutator.health.populate_exact_evidence", store):
        response = TestClient(app).post("/v1/index/evidence-exact", json=_request_body())

    assert response.status_code == 401
    store.assert_not_awaited()


def test_exact_evidence_rejects_wrong_feeder_token_without_store_call(monkeypatch):
    _configure_exact_evidence(monkeypatch)
    store = AsyncMock()

    with patch("scrutator.health.populate_exact_evidence", store):
        response = TestClient(app).post(
            "/v1/index/evidence-exact",
            headers={"X-KB-Feeder-Token": "wrong-isolated-test-token"},
            json=_request_body(),
        )

    assert response.status_code == 401
    store.assert_not_awaited()


def test_exact_evidence_rejects_feeder_outside_server_namespace_without_store_call(monkeypatch):
    _configure_exact_evidence(monkeypatch, feeder_namespaces="talomnia")
    store = AsyncMock()

    with patch("scrutator.health.populate_exact_evidence", store):
        response = TestClient(app).post(
            "/v1/index/evidence-exact",
            headers={"X-KB-Feeder-Token": FEEDER_TOKEN},
            json=_request_body(),
        )

    assert response.status_code == 403
    store.assert_not_awaited()


def test_exact_evidence_allows_scoped_feeder_and_uses_server_namespace(monkeypatch):
    _configure_exact_evidence(monkeypatch)
    store = AsyncMock(
        return_value=IndexResponse(
            chunks_indexed=0,
            source_path="tests/fixtures/exact-evidence.md",
            namespace=NAMESPACE,
            strategy_used="evidence_exact_created",
        )
    )

    with patch("scrutator.health.populate_exact_evidence", store):
        response = TestClient(app).post(
            "/v1/index/evidence-exact",
            headers={"X-KB-Feeder-Token": FEEDER_TOKEN},
            json=_request_body(),
        )

    assert response.status_code == 200
    assert response.json()["namespace"] == NAMESPACE
    store.assert_awaited_once_with(
        "candidate-local exact evidence",
        "tests/fixtures/exact-evidence.md",
        NAMESPACE,
    )
