"""SRCH-0038 S2 / V-AC-5 — per-caller namespace authorization on fetch (IDOR defense).

A caller scoped to namespace A fetching a doc/chunk id in namespace B is denied (404 within
allowed scope — no existence oracle). Fetch reuses `require_tenant_context` and every fetch
query filters `namespace_id = ANY(ctx.allowed_namespace_ids)`.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from scrutator.db import repository

from .conftest import mock_authenticated_principal

# Fixture doc lives in namespace id 2 ("secret-tenant"); the caller below is scoped to id 1.
_DOC = "# Secret\n\nConfidential body. " + ("word " * 120)


def _client():
    from scrutator.health import app

    return TestClient(app, raise_server_exceptions=False)


def _scoped_fake_fetch(owner_namespace_id: int, rows: list[dict]):
    """Simulate the DB scoping: return rows only when the owning namespace is in the allowed set,
    exactly as `WHERE namespace_id = ANY($allowed)` would."""

    async def _fake(_id: str, allowed_namespace_ids):
        return rows if owner_namespace_id in allowed_namespace_ids else []

    return _fake


class TestCrossNamespaceDenied:
    def test_cross_namespace_fetch_denied(self):
        from ..conftest import build_indexed_doc

        doc_id, _h, rows = build_indexed_doc(_DOC, namespace="secret-tenant")
        with (
            mock_authenticated_principal("svc-A", frozenset({1})),  # authorized for ns 1 only
            patch(
                "scrutator.search.fetcher.fetch_chunks_by_doc_id",
                new=_scoped_fake_fetch(owner_namespace_id=2, rows=rows),
            ),
        ):
            resp = _client().post(
                "/v1/fetch",
                json={"by": "source_id", "id": doc_id, "range": "full"},
                headers={"Authorization": "Bearer arc_api_test"},
            )
        assert resp.status_code == 404  # denied within allowed scope, no existence oracle

    def test_in_scope_fetch_allowed(self):
        from ..conftest import build_indexed_doc

        doc_id, content_hash, rows = build_indexed_doc(_DOC, namespace="arcanada")
        with (
            mock_authenticated_principal("svc-A", frozenset({1})),  # ns 1 == "arcanada"
            patch(
                "scrutator.search.fetcher.fetch_chunks_by_doc_id",
                new=_scoped_fake_fetch(owner_namespace_id=1, rows=rows),
            ),
        ):
            resp = _client().post(
                "/v1/fetch",
                json={"by": "source_id", "id": doc_id, "range": "full"},
                headers={"Authorization": "Bearer arc_api_test"},
            )
        assert resp.status_code == 200
        assert resp.json()["content_hash"] == content_hash


class TestFailClosedAndScopedQuery:
    @pytest.mark.asyncio
    async def test_empty_allowed_set_fetches_nothing(self):
        """Grace-window / zero-grant principal (empty allowed set) fetches nothing — fail-closed,
        no DB round-trip even attempted."""
        assert await repository.fetch_chunks_by_doc_id("0123456789abcdef", frozenset()) == []
        assert await repository.fetch_chunks_by_chunk_id("11111111-2222-3333-4444-555555555555", frozenset()) == []

    def test_fetch_queries_are_namespace_scoped(self):
        """Static S2 guard: both fetch queries carry the `namespace_id = ANY(...)` filter and use
        parameterized equality on the opaque doc_id — never an interpolated path join (S3)."""
        src = inspect.getsource(repository.fetch_chunks_by_doc_id)
        assert "namespace_id = ANY($2::int[])" in src
        assert "metadata->'section'->>'doc_id' = $1" in src

        src_chunk = inspect.getsource(repository.fetch_chunks_by_chunk_id)
        assert "namespace_id = ANY($2::int[])" in src_chunk
        assert "id = $1::uuid" in src_chunk
