"""SRCH-0023 V-AC-5 — auth is enforced and fails closed.

Unauthenticated /v1/* and /v1/ltm/* return 401 once SCRUTATOR_AUTH_ENFORCE=True; invalid/
expired token returns 401; JWKS-unreachable denies (never fails open). `/health` stays
unauthenticated regardless of the flag.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from scrutator.auth.verifier import Unauthenticated


def _client():
    from scrutator.health import app

    return TestClient(app, raise_server_exceptions=False)


class TestAuthEnforceTrueDeniesUnauthenticated:
    def test_missing_auth_denies_401_on_search(self):
        with patch("scrutator.auth.dependency.settings") as mock_settings:
            mock_settings.auth_enforce = True
            resp = _client().post("/v1/search", json={"query": "q"})
        assert resp.status_code == 401

    def test_missing_auth_denies_401_on_ltm_recall(self):
        with patch("scrutator.auth.dependency.settings") as mock_settings:
            mock_settings.auth_enforce = True
            resp = _client().post("/v1/ltm/recall", json={"query": "q"})
        assert resp.status_code == 401

    def test_missing_auth_denies_401_on_navigation_before_resource_lookup(self):
        with patch("scrutator.auth.dependency.settings") as mock_settings:
            mock_settings.auth_enforce = True
            outline = _client().get(
                "/v1/navigate/outline",
                params={"namespace": "wiki", "source_path": "missing.md"},
            )
            section = _client().get(
                "/v1/navigate/section",
                params={"chunk_id": "00000000-0000-0000-0000-000000000000"},
            )
        assert outline.status_code == 401
        assert section.status_code == 401

    def test_invalid_token_denies_401(self):
        with (
            patch("scrutator.auth.dependency.settings") as mock_settings,
            patch(
                "scrutator.auth.dependency.verify_bearer_token",
                new_callable=AsyncMock,
                side_effect=Unauthenticated("bad signature"),
            ),
        ):
            mock_settings.auth_enforce = True
            resp = _client().post("/v1/search", json={"query": "q"}, headers={"Authorization": "Bearer garbage"})
        assert resp.status_code == 401

    def test_jwks_unreachable_denies_401_not_fail_open(self):
        """JWKS host down MUST deny — never silently grant an unverified principal."""
        with (
            patch("scrutator.auth.dependency.settings") as mock_settings,
            patch(
                "scrutator.auth.dependency.verify_bearer_token",
                new_callable=AsyncMock,
                side_effect=Unauthenticated("JWKS lookup failed: ConnectionError"),
            ),
        ):
            mock_settings.auth_enforce = True
            resp = _client().post(
                "/v1/search",
                json={"query": "q"},
                headers={"Authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.fake.token"},
            )
        assert resp.status_code == 401


class TestHealthStaysUnauthenticated:
    def test_health_never_requires_auth_enforce_true(self):
        with patch("scrutator.auth.dependency.settings") as mock_settings:
            mock_settings.auth_enforce = True
            resp = _client().get("/health")
        assert resp.status_code == 200

    def test_health_never_requires_auth_enforce_false(self):
        with patch("scrutator.auth.dependency.settings") as mock_settings:
            mock_settings.auth_enforce = False
            resp = _client().get("/health")
        assert resp.status_code == 200


class TestGraceWindowLogsWouldDenyButDoesNotReject:
    def test_missing_auth_grace_window_not_401(self):
        """SCRUTATOR_AUTH_ENFORCE=False: request is not hard-rejected — it is granted an
        empty-context TenantContext instead (see test_tenant_isolation.py for proof this
        still can't read cross-tenant data)."""
        with patch("scrutator.auth.dependency.settings") as mock_settings:
            mock_settings.auth_enforce = False
            resp = _client().post("/v1/search", json={"query": "q"})
        assert resp.status_code != 401
