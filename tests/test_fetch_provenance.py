"""SRCH-0038 — ingest hash stamp (S1 write side, V-AC-4) + derived provenance (V-AC-9/10/11)."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, patch

import pytest

from scrutator.config import settings
from scrutator.db.models import FetchRequest

from .conftest import build_indexed_doc

_DOC = "# Topic\n\nFirst paragraph. " + ("gamma " * 120) + "\n\n## More\n\nSecond. " + ("delta " * 120)


# ── Step 3: ingest stamps doc_content_hash (write side, S1) ──────────


class TestIngestHashStamp:
    def test_ingest_stamps_doc_content_hash(self):
        from scrutator.search.indexer import _chunk_dicts, chunk_document, compute_doc_content_hash

        content = "# Doc\n\n" + ("lorem " * 200)
        result = chunk_document(content, "d.md", max_tokens=64, overlap_tokens=8)
        chunk_dicts = _chunk_dicts(result, "arcanada", "d.md", content)

        expected = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
        assert compute_doc_content_hash(content) == expected
        # Every chunk of the document carries the identical whole-doc hash.
        for cd in chunk_dicts:
            assert cd["metadata"]["section"]["doc_content_hash"] == expected

    def test_hash_is_over_full_content_not_chunk(self):
        from scrutator.search.indexer import _chunk_dicts, chunk_document

        content = "# Doc\n\n" + ("lorem " * 300)
        result = chunk_document(content, "d.md", max_tokens=64, overlap_tokens=8)
        chunk_dicts = _chunk_dicts(result, "arcanada", "d.md", content)
        # The doc-level hash differs from the per-chunk content_hash of any single chunk.
        doc_hash = chunk_dicts[0]["metadata"]["section"]["doc_content_hash"]
        assert doc_hash != "sha256:" + chunk_dicts[0]["content_hash"]


# ── Step 5: derived provenance (V-AC-9/10/11) ────────────────────────


class TestDerivedProvenance:
    @pytest.mark.asyncio
    async def test_trust_class_derived_from_namespace(self):
        from scrutator.search import fetcher

        # A doc in the skills namespace → "skill".
        skills_ns = settings.skills_namespace
        doc_id, _h, rows = build_indexed_doc(_DOC, namespace=skills_ns)
        with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows):
            resp = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id), frozenset({1}))
        assert resp.trust_class == "skill"

        # A doc elsewhere → "evidence".
        doc_id2, _h2, rows2 = build_indexed_doc(_DOC, namespace="arcanada")
        with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows2):
            resp2 = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id2), frozenset({1}))
        assert resp2.trust_class == "evidence"

    @pytest.mark.asyncio
    async def test_stale_defaults_false(self):
        from scrutator.search import fetcher

        doc_id, _h, rows = build_indexed_doc(_DOC)
        with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows):
            resp = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id), frozenset({1}))
        assert resp.stale is False

    @pytest.mark.asyncio
    async def test_embedding_model_id_from_config(self):
        from scrutator.search import fetcher

        doc_id, _h, rows = build_indexed_doc(_DOC)
        with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows):
            resp = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id), frozenset({1}))
        assert resp.embedding_model_id == settings.embedding_model_id

    @pytest.mark.asyncio
    async def test_index_snapshot_id_stable_and_reindex_changes(self):
        from scrutator.search import fetcher

        doc_id, _h, rows = build_indexed_doc(_DOC, indexed_at="2026-07-22T10:00:00+00:00")
        with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows):
            r1 = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id), frozenset({1}))
            r2 = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id), frozenset({1}))
        # Deterministic for identical index state.
        assert r1.index_snapshot_id == r2.index_snapshot_id
        assert r1.indexed_at == "2026-07-22T10:00:00+00:00"

        # Re-index (max indexed_at moves) → snapshot id changes.
        _did, _h2, rows_reindexed = build_indexed_doc(_DOC, indexed_at="2026-07-23T11:00:00+00:00")
        with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows_reindexed):
            r3 = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id), frozenset({1}))
        assert r3.index_snapshot_id != r1.index_snapshot_id
        assert r3.indexed_at == "2026-07-23T11:00:00+00:00"
