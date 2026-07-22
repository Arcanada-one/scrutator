"""SRCH-0038 — POST /v1/fetch endpoint, SearchHit augmentation, selector resolution, roundtrip.

Mock-based (the whole Scrutator suite is): repository fetch functions and search are patched;
`tests.conftest.build_indexed_doc` reproduces the real ingest hash stamp so the ingest→read
integrity path is genuinely exercised with Postgres bypassed.
"""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scrutator.db.models import (
    ChunkManifestEntry,
    FetchRequest,
    FetchResponse,
    SearchResult,
    doc_fields_from_metadata,
)

from .conftest import build_indexed_doc, override_tenant_context

_DOC = (
    "# Architecture\n\nScrutator is the retrieval engine. "
    + ("word " * 120)
    + "\n\n## Details\n\nMore. "
    + ("token " * 120)
)


def _client():
    from scrutator.health import app

    return TestClient(app, raise_server_exceptions=False)


# ── Step 1: request/response model shape ─────────────────────────────


class TestFetchModelShape:
    def test_fetch_request_response_shape(self):
        req = FetchRequest(by="source_id", id="0123456789abcdef")
        assert req.range == "full"
        assert req.include == ["content", "provenance"]

        resp = FetchResponse(
            source_id="0123456789abcdef",
            path="concepts/architecture.md",
            content="hello",
            content_len_tokens=1,
            content_hash="sha256:abc",
            index_snapshot_id="deadbeefdeadbeef",
            indexed_at="2026-07-22T10:00:00+00:00",
            embedding_model_id="bge-m3",
            namespace="arcanada",
            trust_class="evidence",
            chunk_manifest=[ChunkManifestEntry(chunk_id="c1", offset_start=0, offset_end=5)],
            stale=False,
        )
        assert resp.trust_class == "evidence"
        assert resp.chunk_manifest[0].offset_end == 5

    def test_trust_class_is_closed_literal(self):
        with pytest.raises(ValidationError):
            FetchResponse(
                source_id="x",
                path="p",
                content="c",
                content_len_tokens=0,
                content_hash="sha256:x",
                index_snapshot_id="s",
                indexed_at="t",
                embedding_model_id="m",
                namespace="n",
                trust_class="root",  # not in {skill, evidence}
            )


# ── Step 2: SearchHit augmentation (V-AC-2) ──────────────────────────


class TestSearchHitAugmentation:
    def test_search_result_defaults_are_empty_non_breaking(self):
        # Additive with defaults → the frozen search-baseline structural contract still holds.
        r = SearchResult(
            chunk_id="c1", source_path="p.md", source_type="md", chunk_index=0, score=0.1, namespace="arcanada"
        )
        assert r.content_hash == ""
        assert r.source_id == ""

    def test_doc_fields_from_metadata_projection(self):
        meta = {"section": {"doc_id": "0123456789abcdef", "doc_content_hash": "sha256:abc"}}
        source_id, content_hash = doc_fields_from_metadata(meta)
        assert source_id == "0123456789abcdef"
        assert content_hash == "sha256:abc"
        # Legacy / un-backfilled rows degrade to "" (never a recomputed value).
        assert doc_fields_from_metadata({}) == ("", "")
        assert doc_fields_from_metadata(None) == ("", "")
        assert doc_fields_from_metadata({"section": None}) == ("", "")

    @pytest.mark.asyncio
    async def test_search_hit_has_content_hash_and_source_id(self):
        """The searcher projects source_id + content_hash from the already-selected metadata —
        no new join. Exercised through the filtered (search_with_filters) branch."""
        doc_id, content_hash, rows = build_indexed_doc(_DOC, namespace="arcanada")
        raw = [
            {
                "chunk_id": rows[0]["chunk_id"],
                "content": rows[0]["content"],
                "source_path": rows[0]["source_path"],
                "source_type": rows[0]["source_type"],
                "chunk_index": 0,
                "score": 0.1,
                "namespace": "arcanada",
                "project": None,
                "metadata": rows[0]["metadata"],
            }
        ]
        from scrutator.search import searcher

        with patch.object(searcher, "search_with_filters", new_callable=AsyncMock, return_value=raw):
            resp = await searcher.search(query="q", namespace_id=1, source_type="md")

        hit = resp.results[0]
        assert hit.source_id == doc_id
        assert hit.content_hash == content_hash


# ── Step 4/5: selector resolution (V-AC-3) ───────────────────────────


class TestSelectorResolution:
    @pytest.mark.asyncio
    async def test_fetch_by_each_selector_resolves_same_doc(self):
        doc_id, content_hash, rows = build_indexed_doc(_DOC, namespace="arcanada")
        from scrutator.search import fetcher

        with (
            patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
            patch.object(fetcher, "fetch_chunks_by_chunk_id", new_callable=AsyncMock, return_value=rows),
        ):
            by_doc = await fetcher.fetch(FetchRequest(by="document_id", id=doc_id), frozenset({1}))
            by_src = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id), frozenset({1}))
            by_chunk = await fetcher.fetch(FetchRequest(by="chunk_id", id=rows[0]["chunk_id"]), frozenset({1}))

        assert by_doc.source_id == by_src.source_id == by_chunk.source_id == doc_id
        assert by_doc.content == by_src.content == by_chunk.content
        assert by_doc.content_hash == by_src.content_hash == by_chunk.content_hash == content_hash


# ── Step 6: route + roundtrip (V-AC-1, V-AC-12) ──────────────────────


class TestFetchRoute:
    def test_fetch_full_returns_typed_response(self):
        doc_id, content_hash, rows = build_indexed_doc(_DOC, namespace="arcanada")
        from scrutator.health import app
        from scrutator.search import fetcher

        with (
            override_tenant_context(app),
            patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        ):
            resp = _client().post("/v1/fetch", json={"by": "source_id", "id": doc_id, "range": "full"})

        assert resp.status_code == 200
        body = resp.json()
        for field in (
            "source_id",
            "path",
            "content",
            "content_len_tokens",
            "content_hash",
            "index_snapshot_id",
            "indexed_at",
            "embedding_model_id",
            "namespace",
            "trust_class",
            "chunk_manifest",
            "stale",
        ):
            assert field in body, f"missing spec field: {field}"
        assert body["source_id"] == doc_id
        assert body["content_hash"] == content_hash
        assert body["embedding_model_id"] == "bge-m3"
        assert body["stale"] is False
        assert body["trust_class"] == "evidence"

    def test_fetch_unknown_id_returns_404_no_oracle(self):
        from scrutator.health import app
        from scrutator.search import fetcher

        with (
            override_tenant_context(app),
            patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=[]),
        ):
            resp = _client().post("/v1/fetch", json={"by": "source_id", "id": "ffffffffffffffff", "range": "full"})
        assert resp.status_code == 404

    def test_path_like_id_rejected_422_at_route(self):
        from scrutator.health import app

        with override_tenant_context(app):
            resp = _client().post("/v1/fetch", json={"by": "source_id", "id": "../../etc/passwd"})
        assert resp.status_code == 422

    def test_index_search_fetch_hash_roundtrip(self):
        """V-AC-12: index → /v1/search hit carries content_hash+source_id → /v1/fetch by
        source_id returns the whole doc with a byte-equal content_hash."""
        doc_id, content_hash, rows = build_indexed_doc(_DOC, namespace="arcanada")

        # The search hit as the real repository projection would build it from the same metadata.
        search_hit = SearchResult(
            chunk_id=rows[0]["chunk_id"],
            content=rows[0]["content"],
            source_path=rows[0]["source_path"],
            source_type=rows[0]["source_type"],
            chunk_index=0,
            score=0.1,
            namespace="arcanada",
            metadata=rows[0]["metadata"],
            source_id=doc_fields_from_metadata(rows[0]["metadata"])[0],
            content_hash=doc_fields_from_metadata(rows[0]["metadata"])[1],
        )
        assert search_hit.source_id == doc_id
        assert search_hit.content_hash == content_hash

        from scrutator.health import app
        from scrutator.search import fetcher

        with (
            override_tenant_context(app),
            patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        ):
            resp = _client().post("/v1/fetch", json={"by": "source_id", "id": search_hit.source_id, "range": "full"})

        assert resp.status_code == 200
        body = resp.json()
        # Byte-equal roundtrip: both search hit and fetch READ the same ingest-bound stamp.
        assert body["content_hash"] == search_hit.content_hash == content_hash
        # Whole doc returned (reassembly of every chunk).
        assert body["content"] == "".join(r["content"] for r in rows)
        assert len(body["chunk_manifest"]) == len(rows)


# ── Step 8: additive-only schema guard (V-AC-13) ─────────────────────


def test_no_destructive_schema_change():
    """schema.sql change is JSONB-only (zero-DDL, D3): no ALTER/DROP against the branch base."""
    diff = subprocess.run(
        ["git", "diff", "origin/main", "--", "src/scrutator/db/schema.sql"],
        cwd=subprocess.os.path.dirname(subprocess.os.path.dirname(subprocess.os.path.abspath(__file__))),
        capture_output=True,
        text=True,
    ).stdout
    added = [ln for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    lowered = "\n".join(added).lower()
    assert "drop " not in lowered, f"destructive DROP in schema.sql delta:\n{diff}"
    assert "alter " not in lowered, f"ALTER in schema.sql delta (expected zero-DDL):\n{diff}"
