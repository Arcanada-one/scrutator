from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from scrutator.config import settings
from scrutator.db.models import IndexResponse
from scrutator.health import app
from scrutator.request_rate_limit import SlidingWindowRateLimiter

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


def test_exact_evidence_rate_limit_refuses_before_store_call(monkeypatch):
    import scrutator.health as health

    _configure_exact_evidence(monkeypatch)
    monkeypatch.setattr(health, "evidence_exact_rate_limiter", SlidingWindowRateLimiter(limit=2, window_seconds=60))
    store = AsyncMock(
        return_value=IndexResponse(
            chunks_indexed=0,
            source_path="tests/fixtures/exact-evidence.md",
            namespace=NAMESPACE,
            strategy_used="evidence_exact_created",
        )
    )

    with patch("scrutator.health.populate_exact_evidence", store):
        client = TestClient(app)
        headers = {"X-KB-Feeder-Token": FEEDER_TOKEN}
        first = client.post("/v1/index/evidence-exact", headers=headers, json=_request_body())
        second = client.post("/v1/index/evidence-exact", headers=headers, json=_request_body())
        limited = client.post("/v1/index/evidence-exact", headers=headers, json=_request_body())

    assert first.status_code == 200
    assert second.status_code == 200
    assert limited.status_code == 429
    assert limited.json() == {"detail": "exact evidence request rate limit exceeded"}
    assert 1 <= int(limited.headers["retry-after"]) <= 60
    assert store.await_count == 2


def test_exact_evidence_authorization_refusals_do_not_consume_rate_budget(monkeypatch):
    import scrutator.health as health

    _configure_exact_evidence(monkeypatch)
    monkeypatch.setattr(health, "evidence_exact_rate_limiter", SlidingWindowRateLimiter(limit=1, window_seconds=60))
    store = AsyncMock(
        return_value=IndexResponse(
            chunks_indexed=0,
            source_path="tests/fixtures/exact-evidence.md",
            namespace=NAMESPACE,
            strategy_used="evidence_exact_created",
        )
    )
    with patch("scrutator.health.populate_exact_evidence", store):
        client = TestClient(app)
        no_token = client.post("/v1/index/evidence-exact", json=_request_body())
        wrong_token = client.post(
            "/v1/index/evidence-exact",
            headers={"X-KB-Feeder-Token": "wrong-isolated-test-token"},
            json=_request_body(),
        )
        monkeypatch.setattr(settings, "feeder_namespaces", "talomnia")
        out_of_scope = client.post(
            "/v1/index/evidence-exact",
            headers={"X-KB-Feeder-Token": FEEDER_TOKEN},
            json=_request_body(),
        )
        monkeypatch.setattr(settings, "feeder_namespaces", NAMESPACE)
        allowed = client.post(
            "/v1/index/evidence-exact",
            headers={"X-KB-Feeder-Token": FEEDER_TOKEN},
            json=_request_body(),
        )

    assert [no_token.status_code, wrong_token.status_code, out_of_scope.status_code] == [401, 401, 403]
    assert allowed.status_code == 200
    store.assert_awaited_once_with(
        "candidate-local exact evidence",
        "tests/fixtures/exact-evidence.md",
        NAMESPACE,
    )


def test_exact_evidence_limit_does_not_throttle_other_routes(monkeypatch):
    import scrutator.health as health

    _configure_exact_evidence(monkeypatch)
    monkeypatch.setattr(health, "evidence_exact_rate_limiter", SlidingWindowRateLimiter(limit=1, window_seconds=60))
    store = AsyncMock(
        return_value=IndexResponse(
            chunks_indexed=0,
            source_path="tests/fixtures/exact-evidence.md",
            namespace=NAMESPACE,
            strategy_used="evidence_exact_created",
        )
    )
    with patch("scrutator.health.populate_exact_evidence", store):
        client = TestClient(app)
        headers = {"X-KB-Feeder-Token": FEEDER_TOKEN}
        first = client.post("/v1/index/evidence-exact", headers=headers, json=_request_body())
        health_response = client.get("/health")
        limited = client.post("/v1/index/evidence-exact", headers=headers, json=_request_body())

    assert first.status_code == 200
    assert health_response.status_code == 200
    assert limited.status_code == 429
    store.assert_awaited_once()
