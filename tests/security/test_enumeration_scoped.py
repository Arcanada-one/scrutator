"""SRCH-0023 V-AC-6 — enumeration endpoints are scoped to the caller's allowed-set.

`GET /v1/namespaces` and `GET /v1/stats` for a principal scoped to tenant A must never
list tenant B — closes the tenant-label enumeration oracle (CWE-200).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from .conftest import mock_authenticated_principal


def _client():
    from scrutator.health import app

    return TestClient(app, raise_server_exceptions=False)


class TestNamespacesEndpointScoped:
    def test_lists_only_allowed_namespace(self):
        with (
            mock_authenticated_principal("svc-1", frozenset({1})),
            patch(
                "scrutator.health.get_namespaces",
                new_callable=AsyncMock,
                return_value=[
                    {"id": 1, "name": "arcanada", "description": None, "chunk_count": 3},
                ],
            ) as mock_get_ns,
        ):
            resp = _client().get("/v1/namespaces", headers={"Authorization": "Bearer arc_api_test"})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "arcanada"
        mock_get_ns.assert_awaited_once_with(namespace_ids=frozenset({1}))

    def test_zero_grant_principal_sees_no_namespaces(self):
        with (
            mock_authenticated_principal("svc-orphan", frozenset()),
            patch("scrutator.health.get_namespaces", new_callable=AsyncMock, return_value=[]) as mock_get_ns,
        ):
            resp = _client().get("/v1/namespaces", headers={"Authorization": "Bearer arc_api_test"})

        assert resp.status_code == 200
        assert resp.json() == []
        mock_get_ns.assert_awaited_once_with(namespace_ids=frozenset())


class TestStatsEndpointScoped:
    def test_stats_scoped_to_allowed_namespaces(self):
        with (
            mock_authenticated_principal("svc-1", frozenset({1})),
            patch(
                "scrutator.health.get_stats",
                new_callable=AsyncMock,
                return_value={
                    "total_chunks": 3,
                    "total_namespaces": 1,
                    "total_projects": 1,
                    "namespaces": [],
                },
            ) as mock_get_stats,
        ):
            resp = _client().get("/v1/stats", headers={"Authorization": "Bearer arc_api_test"})

        assert resp.status_code == 200
        assert resp.json()["total_namespaces"] == 1
        mock_get_stats.assert_awaited_once_with(namespace_ids=frozenset({1}))
