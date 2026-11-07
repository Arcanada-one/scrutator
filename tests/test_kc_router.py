"""KB-CONS-01-N: private-generation HTTP surface (mock-based, runs in the ordinary CI without PostgreSQL).

With no real authority backend every dedicated route is a refusal; legacy routes keep their behaviour."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from scrutator.auth.dependency import require_private_tenant_context
from scrutator.db import kc_generations as kc
from scrutator.health import app
from tests.conftest import make_tenant_context

LEASE = "11111111-1111-4111-8111-111111111111"


def _client():
    ctx = make_tenant_context(namespace_ids=frozenset({5}), namespace_names=frozenset({"kc"}))
    app.dependency_overrides[require_private_tenant_context] = lambda: ctx
    return TestClient(app)


def _done():
    app.dependency_overrides.pop(require_private_tenant_context, None)


def test_default_authority_backend_refuses_every_use():
    assert isinstance(kc.get_authority_backend(), kc.UnavailableAuthority)


def test_fetch_route_maps_authority_unavailable_to_503_without_leaking_details():
    rc = MagicMock()
    with (
        patch("scrutator.kc_router.kc.load_context", AsyncMock(return_value=rc)),
        patch("scrutator.kc_router.get_pool", AsyncMock(return_value=MagicMock())),
        patch(
            "scrutator.kc_router.kc.read_member_exact",
            AsyncMock(side_effect=kc.KcGenerationError("CURRENT_AUTHORITY_UNAVAILABLE", "secret detail")),
        ),
    ):
        r = _client().post("/v1/kc/read/fetch", json={"lease_id": LEASE, "member_id": "m"})
    _done()
    assert r.status_code == 503 and r.json() == {"detail": "CURRENT_AUTHORITY_UNAVAILABLE"}


def test_unknown_or_foreign_lease_is_404_for_every_dedicated_route():
    with (
        patch("scrutator.kc_router.kc.load_context", AsyncMock(side_effect=kc.KcGenerationError("LEASE_NOT_FOUND"))),
        patch("scrutator.kc_router.get_pool", AsyncMock(return_value=MagicMock())),
    ):
        c = _client()
        codes = {
            path: c.post(path, json=body).status_code
            for path, body in {
                "/v1/kc/read/search": {"lease_id": LEASE, "query": "q"},
                "/v1/kc/read/fetch": {"lease_id": LEASE, "member_id": "m"},
                "/v1/kc/read/chunk": {"lease_id": LEASE, "chunk_id": LEASE},
                "/v1/kc/read/related": {"lease_id": LEASE, "chunk_id": LEASE},
            }.items()
        }
    _done()
    assert set(codes.values()) == {404}


def test_graph_and_ltm_views_answer_unsupported_not_a_body():
    r = _client().post("/v1/kc/read/graph", json={"lease_id": LEASE})
    _done()
    assert r.status_code == 501 and r.json() == {"detail": "UNSUPPORTED_PRIVATE_OPERATION"}


def test_search_route_binds_the_captured_generation_and_never_a_request_project():
    rc = MagicMock(namespace_id=5)
    rc.eligibility = kc.GenerationEligibility("s", "g", 5, 9)
    captured = {}

    async def fake_search(**kw):
        captured.update(kw)
        from scrutator.db.models import SearchResponse

        return SearchResponse(results=[], total=0, query=kw["query"], search_time_ms=0.0)

    with (
        patch("scrutator.kc_router.kc.load_context", AsyncMock(return_value=rc)),
        patch("scrutator.kc_router.kc.authorize_use", AsyncMock()),
        patch("scrutator.kc_router.get_pool", AsyncMock(return_value=MagicMock())),
        patch("scrutator.kc_router.search", fake_search),
    ):
        r = _client().post(
            "/v1/kc/read/search", json={"lease_id": LEASE, "query": "q", "project": "other", "namespace": "x"}
        )
    _done()
    assert r.status_code == 200
    assert captured["generation"] == rc.eligibility and callable(captured["pre_disclosure_check"])
    assert "project" not in captured  # the request's project field is not an input of the private search
