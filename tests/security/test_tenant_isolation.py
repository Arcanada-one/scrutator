"""SRCH-0023 negative cross-tenant suite — the crux (V-AC-1, V-AC-2, V-AC-3, V-AC-11).

Proves the confirmed live leak (`POST /v1/search {"query": "..."}` with no namespace
returning the WHOLE corpus across every tenant) is closed:
- no-namespace request never returns cross-tenant rows (V-AC-1).
- a spoofed / out-of-set namespace is rejected 403, never silently re-scoped (V-AC-2).

TDD: this file is written RED against the pre-fix code (the `($2::int IS NULL OR ...)`
escape hatch + unauthenticated routes) and GREEN once repository.py/health.py/searcher.py
land the fix. Confirmed both states — see task report.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from .conftest import mock_authenticated_principal


def _client():
    from scrutator.health import app

    return TestClient(app, raise_server_exceptions=False)


class TestNoNamespaceNeverLeaksFullCorpus:
    """V-AC-1: omitting `namespace` must never return rows across every tenant."""

    def test_single_tenant_principal_scopes_to_own_namespace_only(self):
        """A principal authorized for exactly one namespace omitting `namespace` gets that
        tenant's own namespace_id resolved server-side — never None, never all-tenants."""
        with (
            mock_authenticated_principal("svc-1", frozenset({1})),
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock, return_value=[0.1] * 8),
            patch("scrutator.search.searcher.embed_sparse", new_callable=AsyncMock, return_value=[{}]),
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock, return_value=[]) as mock_hs,
        ):
            resp = _client().post(
                "/v1/search",
                json={"query": "find the secret"},
                headers={"Authorization": "Bearer arc_api_test"},
            )

        assert resp.status_code == 200
        mock_hs.assert_awaited_once()
        # namespace_id resolved server-side to the principal's own tenant — never None.
        assert mock_hs.call_args.kwargs["namespace_id"] == 1

    def test_multi_tenant_principal_omitting_namespace_is_ambiguous_not_all_rows(self):
        """A principal authorized for >1 namespace who omits `namespace` gets 400
        (must disambiguate) — the old code's `NULL -> all rows` behaviour is gone."""
        with (
            mock_authenticated_principal("svc-multi", frozenset({1, 2})),
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_hs,
        ):
            resp = _client().post(
                "/v1/search",
                json={"query": "find the secret"},
                headers={"Authorization": "Bearer arc_api_test"},
            )

        assert resp.status_code == 400
        mock_hs.assert_not_called()

    def test_unauthenticated_grace_window_never_falls_back_to_all_rows(self):
        """SCRUTATOR_AUTH_ENFORCE=False (dual-auth grace): an unauthenticated caller is not
        hard-rejected with 401, but it MUST NOT get the old NULL-namespace full-corpus read —
        it resolves to an empty allowed-set and is denied at the namespace-selector stage."""
        with (
            patch("scrutator.auth.dependency.settings") as mock_settings,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_hs,
        ):
            mock_settings.auth_enforce = False
            resp = _client().post("/v1/search", json={"query": "find the secret"})

        assert resp.status_code == 403
        mock_hs.assert_not_called()


class TestSpoofedNamespaceRejected:
    """V-AC-2: a principal scoped to tenant A requesting tenant B's namespace gets 403,
    zero B rows — never silently re-scoped to A, never widened."""

    def test_out_of_set_namespace_denied_403(self):
        with (
            mock_authenticated_principal("svc-1", frozenset({1})),
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_hs,
        ):
            resp = _client().post(
                "/v1/search",
                json={"query": "find the secret", "namespace": "secret-tenant"},
                headers={"Authorization": "Bearer arc_api_test"},
            )

        assert resp.status_code == 403
        mock_hs.assert_not_called()

    def test_in_set_namespace_resolves_correctly(self):
        """Sanity check: requesting a namespace that IS in the allowed-set still works."""
        with (
            mock_authenticated_principal("svc-both", frozenset({1, 2})),
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock, return_value=[0.1] * 8),
            patch("scrutator.search.searcher.embed_sparse", new_callable=AsyncMock, return_value=[{}]),
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock, return_value=[]) as mock_hs,
        ):
            resp = _client().post(
                "/v1/search",
                json={"query": "find the secret", "namespace": "secret-tenant"},
                headers={"Authorization": "Bearer arc_api_test"},
            )

        assert resp.status_code == 200
        assert mock_hs.call_args.kwargs["namespace_id"] == 2


class TestUnscopedReadEscapeHatchGone:
    """V-AC-3 at the API layer: the endpoint never calls hybrid_search with namespace_id=None,
    regardless of what the caller sends."""

    @pytest.mark.parametrize("body", [{"query": "q"}, {"query": "q", "namespace": "arcanada"}])
    def test_hybrid_search_never_called_with_none_namespace_id(self, body):
        with (
            mock_authenticated_principal("svc-1", frozenset({1})),
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock, return_value=[0.1] * 8),
            patch("scrutator.search.searcher.embed_sparse", new_callable=AsyncMock, return_value=[{}]),
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock, return_value=[]) as mock_hs,
        ):
            resp = _client().post("/v1/search", json=body, headers={"Authorization": "Bearer arc_api_test"})

        assert resp.status_code == 200
        assert mock_hs.call_args.kwargs["namespace_id"] is not None
