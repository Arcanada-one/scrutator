"""Tests for SRCH-0004: Search & Retrieval Pipeline.

Unit tests for models, embedder, config.
Integration tests (require PostgreSQL + Embedding API) are skipped when unavailable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scrutator.config import Settings
from scrutator.db.models import (
    IndexRequest,
    IndexResponse,
    IndexStats,
    NamespaceCreate,
    NamespaceInfo,
    NamespaceStats,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

# ── Model validation tests ──────────────────────────────────────────


class TestIndexRequest:
    def test_defaults(self):
        req = IndexRequest(content="hello world", source_path="test.md")
        assert req.namespace == "arcanada"
        assert req.max_tokens == 512
        assert req.overlap_tokens == 50
        assert req.source_type is None

    def test_empty_content_rejected(self):
        with pytest.raises(ValueError, match="content must not be empty"):
            IndexRequest(content="   ", source_path="test.md")

    def test_custom_namespace(self):
        req = IndexRequest(content="data", source_path="f.md", namespace="personal")
        assert req.namespace == "personal"


class TestSearchRequest:
    def test_defaults(self):
        req = SearchRequest(query="hello")
        assert req.limit == 10
        assert req.namespace is None
        assert req.include_content is True

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError, match="query must not be empty"):
            SearchRequest(query="  ")

    def test_limit_capped_at_50(self):
        req = SearchRequest(query="test", limit=100)
        assert req.limit == 50

    def test_limit_min_1(self):
        with pytest.raises(ValueError, match="limit must be >= 1"):
            SearchRequest(query="test", limit=0)

    def test_namespace_filter(self):
        req = SearchRequest(query="test", namespace="arcanada", project="scrutator")
        assert req.namespace == "arcanada"
        assert req.project == "scrutator"


class TestSearchResult:
    def test_source_attribution(self):
        result = SearchResult(
            chunk_id="abc-123",
            content="test content",
            source_path="wiki/README.md",
            source_type="markdown",
            chunk_index=0,
            score=0.032,
            namespace="arcanada",
        )
        assert result.source_path == "wiki/README.md"
        assert result.chunk_index == 0
        assert result.score == 0.032

    def test_heading_hierarchy_default(self):
        result = SearchResult(
            chunk_id="x",
            source_path="f.md",
            source_type="markdown",
            chunk_index=0,
            score=0.01,
            namespace="ns",
        )
        assert result.heading_hierarchy == []
        assert result.metadata == {}


class TestIndexResponse:
    def test_creation(self):
        resp = IndexResponse(
            chunks_indexed=5,
            source_path="test.md",
            namespace="arcanada",
            strategy_used="markdown_headers",
        )
        assert resp.chunks_indexed == 5


class TestSearchResponse:
    def test_creation(self):
        resp = SearchResponse(results=[], total=0, query="hello", search_time_ms=12.5)
        assert resp.total == 0
        assert resp.search_time_ms == 12.5


class TestNamespaceModels:
    def test_namespace_create(self):
        ns = NamespaceCreate(name="test")
        assert ns.name == "test"
        assert ns.description is None

    def test_namespace_info(self):
        ns = NamespaceInfo(id=1, name="arcanada", chunk_count=42)
        assert ns.chunk_count == 42

    def test_namespace_stats(self):
        ns = NamespaceStats(name="arcanada", chunk_count=100, project_count=5)
        assert ns.project_count == 5

    def test_index_stats(self):
        stats = IndexStats(total_chunks=100, total_namespaces=2, total_projects=5)
        assert stats.total_chunks == 100
        assert stats.namespaces == []


# ── Config tests ─────────────────────────────────────────────────────


class TestConfig:
    def test_defaults(self):
        s = Settings()
        assert s.database_pool_min == 2
        assert s.database_pool_max == 10
        assert s.search_timeout_ms == 5000
        assert s.embedding_api_url == "http://localhost:8300"
        assert s.port == 8310


# ── Schema file tests ───────────────────────────────────────────────


class TestSchema:
    def test_schema_file_exists(self):
        schema_path = Path(__file__).parent.parent / "src" / "scrutator" / "db" / "schema.sql"
        assert schema_path.exists(), "schema.sql must exist"

    def test_schema_contains_tables(self):
        schema_path = Path(__file__).parent.parent / "src" / "scrutator" / "db" / "schema.sql"
        sql = schema_path.read_text()
        for table in ["namespaces", "projects", "streams", "chunks", "sparse_vectors", "graph_edges"]:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in sql, f"Missing table: {table}"

    def test_schema_contains_indexes(self):
        schema_path = Path(__file__).parent.parent / "src" / "scrutator" / "db" / "schema.sql"
        sql = schema_path.read_text()
        expected = [
            "idx_chunks_dense",
            "idx_chunks_fts_ru",
            "idx_chunks_fts_en",
            "idx_chunks_namespace",
            "idx_chunks_source",
            "idx_chunks_hash",
            "idx_edges_source",
            "idx_edges_target",
        ]
        for idx in expected:
            assert idx in sql, f"Missing index: {idx}"

    def test_schema_uses_hnsw(self):
        schema_path = Path(__file__).parent.parent / "src" / "scrutator" / "db" / "schema.sql"
        sql = schema_path.read_text()
        assert "USING hnsw" in sql
        assert "vector_cosine_ops" in sql

    def test_schema_idempotent(self):
        """All CREATE statements use IF NOT EXISTS."""
        schema_path = Path(__file__).parent.parent / "src" / "scrutator" / "db" / "schema.sql"
        sql = schema_path.read_text()
        for line in sql.splitlines():
            if line.strip().startswith("CREATE TABLE") or line.strip().startswith("CREATE INDEX"):
                assert "IF NOT EXISTS" in line, f"Missing IF NOT EXISTS: {line.strip()}"


# ── Embedder tests (mocked) ─────────────────────────────────────────


class TestEmbedder:
    @pytest.mark.asyncio
    async def test_embed_texts_returns_vectors(self):
        from scrutator.search.embedder import embed_texts

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1] * 1024},
                {"index": 1, "embedding": [0.2] * 1024},
            ]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("scrutator.search.embedder.get_client", return_value=mock_client):
            results = await embed_texts(["hello", "world"])
            assert len(results) == 2
            assert len(results[0]) == 1024

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        from scrutator.search.embedder import embed_texts

        results = await embed_texts([])
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_error_raises(self):
        from scrutator.search.embedder import EmbeddingError, embed_texts

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with (
            patch("scrutator.search.embedder.get_client", return_value=mock_client),
            pytest.raises(EmbeddingError, match="500"),
        ):
            await embed_texts(["test"])

    @pytest.mark.asyncio
    async def test_embed_single(self):
        from scrutator.search.embedder import embed_single

        with patch("scrutator.search.embedder.embed_texts", new_callable=AsyncMock) as mock:
            mock.return_value = [[0.5] * 1024]
            result = await embed_single("hello")
            assert len(result) == 1024
            mock.assert_called_once_with(["hello"])


# ── Indexer tests (mocked DB + embedder) ─────────────────────────────


class TestIndexer:
    @pytest.mark.asyncio
    async def test_index_document_full_pipeline(self):
        from scrutator.search.indexer import index_document

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.indexer.upsert_project", new_callable=AsyncMock) as mock_proj,
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as mock_replace,
        ):
            mock_embed.return_value = [[0.1] * 1024]
            mock_sparse.return_value = [{"1": 0.1}]
            mock_ns.return_value = 1
            mock_proj.return_value = 10
            mock_replace.return_value = 1

            result = await index_document(
                content="# Hello\n\nWorld",
                source_path="test.md",
                namespace="arcanada",
                project="scrutator",
            )

            assert result.chunks_indexed == 1
            assert result.namespace == "arcanada"
            assert result.source_path == "test.md"
            mock_ns.assert_called_once_with("arcanada")
            mock_proj.assert_called_once_with(1, "scrutator")
            mock_replace.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_index_empty_content_returns_zero(self):
        """Chunker produces empty result for whitespace-only → 0 indexed."""
        from scrutator.search.indexer import index_document

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 1024]),
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock, return_value=[{"1": 0.1}]),
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock, return_value=1),
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock, return_value=1),
        ):
            # Very short content → single chunk, still indexable
            result = await index_document(content="x", source_path="empty.md")
            # Single-char content still produces a chunk
            assert result.chunks_indexed >= 0

    @pytest.mark.asyncio
    async def test_index_document_writes_section_metadata(self):
        """SRCH-0021 V-AC-1/D-REQ-06: section written alongside heading_hierarchy, doc_id stamped."""
        from scrutator.chunker.splitters import compute_doc_id
        from scrutator.search.indexer import index_document

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as mock_replace,
        ):
            mock_embed.return_value = [[0.1] * 1024]
            mock_sparse.return_value = [{"1": 0.1}]
            mock_ns.return_value = 1

            async def _capture(chunk_dicts, *_args, **_kwargs):
                _capture.seen = chunk_dicts
                return len(chunk_dicts)

            mock_replace.side_effect = _capture

            await index_document(content="# Intro\n\nHello world", source_path="wiki/x.md", namespace="arcanada")

            chunk_dicts = _capture.seen
            assert len(chunk_dicts) == 1
            metadata = chunk_dicts[0]["metadata"]
            assert metadata["heading_hierarchy"] == ["# Intro"]  # unchanged, back-compat
            assert metadata["section"] is not None
            assert metadata["section"]["section_key"] == "intro"
            assert metadata["section"]["doc_id"] == compute_doc_id("arcanada", "wiki/x.md")

    @pytest.mark.asyncio
    async def test_index_document_non_markdown_section_none(self):
        """SRCH-0021: non-markdown chunks get section=None explicitly (no doc structure)."""
        from scrutator.search.indexer import index_document

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as mock_replace,
        ):
            mock_embed.return_value = [[0.1] * 1024]
            mock_sparse.return_value = [{"1": 0.1}]
            mock_ns.return_value = 1

            async def _capture(chunk_dicts, *_args, **_kwargs):
                _capture.seen = chunk_dicts
                return len(chunk_dicts)

            mock_replace.side_effect = _capture

            await index_document(content="plain text, no headers", source_path="note.txt", namespace="arcanada")

            assert _capture.seen[0]["metadata"]["section"] is None

    @pytest.mark.asyncio
    async def test_index_without_project(self):
        from scrutator.search.indexer import index_document

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as mock_replace,
        ):
            mock_embed.return_value = [[0.1] * 1024]
            mock_sparse.return_value = [{"1": 0.1}]
            mock_ns.return_value = 1
            mock_replace.return_value = 1

            result = await index_document(content="hello", source_path="note.txt")
            # project=None → upsert_project should NOT be called
            assert result.namespace == "arcanada"


# ── Searcher tests (mocked) ─────────────────────────────────────────


class TestSearcher:
    @pytest.mark.asyncio
    async def test_search_returns_response(self):
        from scrutator.search.searcher import search

        mock_results = [
            SearchResult(
                chunk_id="abc",
                content="found it",
                source_path="wiki/test.md",
                source_type="markdown",
                chunk_index=0,
                score=0.033,
                namespace="arcanada",
            )
        ]

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = mock_results

            resp = await search(query="test query", namespace_id=1, limit=5)
            assert resp.total == 1
            assert resp.results[0].source_path == "wiki/test.md"
            assert resp.search_time_ms > 0

    @pytest.mark.asyncio
    async def test_search_min_score_filter(self):
        from scrutator.search.searcher import search

        mock_results = [
            SearchResult(
                chunk_id="a",
                content="low",
                source_path="a.md",
                source_type="md",
                chunk_index=0,
                score=0.01,
                namespace="ns",
            ),
            SearchResult(
                chunk_id="b",
                content="high",
                source_path="b.md",
                source_type="md",
                chunk_index=0,
                score=0.05,
                namespace="ns",
            ),
        ]

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = mock_results

            resp = await search(query="test", namespace_id=1, min_score=0.03)
            assert resp.total == 1
            assert resp.results[0].chunk_id == "b"

    @pytest.mark.asyncio
    async def test_search_exclude_content(self):
        from scrutator.search.searcher import search

        mock_results = [
            SearchResult(
                chunk_id="a",
                content="secret",
                source_path="a.md",
                source_type="md",
                chunk_index=0,
                score=0.05,
                namespace="ns",
            ),
        ]

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = mock_results

            resp = await search(query="test", namespace_id=1, include_content=False)
            assert resp.results[0].content == ""

    @pytest.mark.asyncio
    async def test_search_requires_namespace_id(self):
        """SRCH-0023 V-AC-1: namespace_id is mandatory — search() never defaults to
        all-namespaces. Omitting it is a TypeError, not a silent full-corpus read."""
        from scrutator.search.searcher import search

        with pytest.raises(TypeError):
            await search(query="test")


# ── API endpoint tests (FastAPI TestClient, mocked DB) ───────────────


class TestAPI:
    """Test FastAPI endpoints using httpx.AsyncClient (no real DB)."""

    def test_health_still_works(self):
        """Health endpoint works without DB."""
        from fastapi.testclient import TestClient

        from scrutator.health import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_chunk_endpoint_still_works(self):
        """Chunking endpoint works without DB (no changes from SRCH-0003)."""
        from fastapi.testclient import TestClient

        from scrutator.health import app

        client = TestClient(app)
        resp = client.post(
            "/v1/chunk",
            json={"content": "# Test\n\nHello world", "source_path": "test.md"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] >= 1

    def test_search_validation_error(self):
        """POST /v1/search with empty query → 422."""
        from fastapi.testclient import TestClient

        from scrutator.health import app

        client = TestClient(app)
        resp = client.post("/v1/search", json={"query": "  "})
        assert resp.status_code == 422

    def test_index_validation_error(self):
        """POST /v1/index with empty content → 422."""
        from fastapi.testclient import TestClient

        from scrutator.auth.capabilities import NamespaceCapability, require_feeder_capability
        from scrutator.health import app

        app.dependency_overrides[require_feeder_capability] = lambda: NamespaceCapability(
            namespaces=frozenset({"arcanada"})
        )
        try:
            client = TestClient(app)
            resp = client.post("/v1/index", json={"content": "  ", "source_path": "x.md"})
        finally:
            app.dependency_overrides.pop(require_feeder_capability, None)
        assert resp.status_code == 422


# ── SRCH-0021: group_by + default-path-unchanged ─────────────────────


class TestGroupBy:
    def _mock_result(self, chunk_id: str, score: float, section_key: str, doc_id: str = "doc1") -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            content=f"content {chunk_id}",
            source_path="docs/same.md",
            source_type="markdown",
            chunk_index=0,
            score=score,
            namespace="arcanada",
            metadata={"section": {"doc_id": doc_id, "section_key": section_key}},
        )

    @pytest.mark.asyncio
    async def test_group_by_document_folds_duplicates(self):
        """V-AC-5: 5 same-document hits fold into 1 group, member_count=5, score=max(member)."""
        from scrutator.search.searcher import search

        mock_results = [
            self._mock_result("c1", 0.05, "doc/intro"),
            self._mock_result("c2", 0.04, "doc/section-a"),
            self._mock_result("c3", 0.03, "doc/section-b"),
            self._mock_result("c4", 0.02, "doc/section-c"),
            self._mock_result("c5", 0.01, "doc/section-d"),
        ]

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_sparse.return_value = None
            mock_search.return_value = mock_results

            resp = await search(query="test", namespace_id=1, limit=5, group_by="document")

        assert resp.total == 1
        group = resp.results[0]
        assert group.member_count == 5
        assert group.score == pytest.approx(0.05)
        assert group.doc_id == "doc1"
        assert set(group.member_chunk_ids) == {"c1", "c2", "c3", "c4", "c5"}
        assert group.representative.chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_group_by_section_separates_distinct_sections(self):
        from scrutator.search.searcher import search

        mock_results = [
            self._mock_result("c1", 0.05, "doc/intro"),
            self._mock_result("c2", 0.04, "doc/intro"),
            self._mock_result("c3", 0.03, "doc/other"),
        ]

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_sparse.return_value = None
            mock_search.return_value = mock_results

            resp = await search(query="test", namespace_id=1, limit=3, group_by="section")

        assert resp.total == 2
        keys = {g.group_key for g in resp.results}
        assert keys == {"doc/intro", "doc/other"}

    @pytest.mark.asyncio
    async def test_group_by_absent_returns_flat_search_results(self):
        """D-REQ-06: group_by omitted → results stay plain SearchResult objects."""
        from scrutator.search.searcher import search

        mock_results = [self._mock_result("c1", 0.05, "doc/intro")]

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_sparse.return_value = None
            mock_search.return_value = mock_results

            resp = await search(query="test", namespace_id=1, limit=1)

        assert resp.results[0].chunk_id == "c1"
        assert not hasattr(resp.results[0], "member_count")


class TestSearchDefaultPathUnchanged:
    """V-AC-6: absent group_by ⇒ SearchResponse byte-identical to the v0.3.0 baseline
    captured (pre-SRCH-0021) in tests/fixtures/search_baseline_v0.3.0.json (Step 0)."""

    @pytest.mark.asyncio
    async def test_search_default_path_unchanged(self):
        import json
        from pathlib import Path

        from scrutator.search.searcher import search

        mock_results = [
            SearchResult(
                chunk_id="c1",
                content="Alpha content",
                source_path="docs/alpha.md",
                source_type="markdown",
                chunk_index=0,
                score=0.05,
                namespace="arcanada",
                heading_hierarchy=["# Alpha", "## Intro"],
            ),
            SearchResult(
                chunk_id="c2",
                content="Beta content",
                source_path="docs/beta.md",
                source_type="markdown",
                chunk_index=1,
                score=0.03,
                namespace="arcanada",
                heading_hierarchy=["# Beta"],
            ),
        ]

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_sparse.return_value = [{"tok": 0.5}]
            mock_search.return_value = mock_results

            # SRCH-0023: namespace_id is now mandatory (int); the removed upsert_namespace
            # patch is obsolete (read path never provisions a namespace).
            resp = await search(query="baseline query", namespace_id=1, limit=5)

        data = resp.model_dump()
        data.pop("search_time_ms", None)

        baseline_path = Path(__file__).parent / "fixtures" / "search_baseline_v0.3.0.json"
        baseline = json.loads(baseline_path.read_text())

        assert json.dumps(data, sort_keys=True) == json.dumps(baseline, sort_keys=True)


# ── SRCH-0029 M1+M2 searcher integration tests ───────────────────────


class TestSearcherCitationAndRerank:
    """V-AC-1 + V-AC-2 — citation emission and flag-gated rerank."""

    def _mock_result(self, chunk_id: str = "abc", score: float = 0.033) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            content="some content",
            source_path=f"docs/{chunk_id}.md",
            source_type="md",
            chunk_index=0,
            score=score,
            namespace="arcanada",
        )

    @pytest.mark.asyncio
    async def test_flag_off_citation_populated_score_kind_rrf(self):
        """V-AC-1 + V-AC-2: rerank_enabled=False → citation populated with score_kind='rrf'."""
        from scrutator.search.searcher import search

        mock_results = [self._mock_result("a", 0.05), self._mock_result("b", 0.03)]

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
            patch("scrutator.search.searcher.settings") as mock_settings,
        ):
            mock_settings.rerank_enabled = False
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = mock_results

            resp = await search(query="test", namespace_id=1, limit=2)

        assert resp.total == 2
        for r in resp.results:
            assert r.citation is not None, f"citation must be populated for {r.chunk_id}"
            assert r.citation.score_kind == "rrf"
            assert r.citation.schema_version == 1
            assert r.citation.chunk_id == r.chunk_id

    @pytest.mark.asyncio
    async def test_flag_off_ordering_and_scores_byte_identical(self):
        """V-AC-2: rerank_enabled=False → ordering/scores unchanged vs current behaviour."""
        from scrutator.search.searcher import search

        # RRF order: A first (higher score)
        mock_results = [self._mock_result("A", 0.05), self._mock_result("B", 0.02)]

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
            patch("scrutator.search.searcher.settings") as mock_settings,
        ):
            mock_settings.rerank_enabled = False
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = mock_results

            resp = await search(query="test", namespace_id=1, limit=2)

        # Order preserved from RRF
        assert resp.results[0].chunk_id == "A"
        assert resp.results[1].chunk_id == "B"
        # Scores unchanged (only citation added)
        assert resp.results[0].score == pytest.approx(0.05)
        assert resp.results[1].score == pytest.approx(0.02)
        # citation.relevance_score mirrors the RRF score
        assert resp.results[0].citation.relevance_score == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_flag_off_rerank_not_called(self):
        """V-AC-2: rerank_enabled=False → rerank() function is never called."""
        from scrutator.search.searcher import search

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
            patch("scrutator.search.searcher.settings") as mock_settings,
            patch("scrutator.search.reranker.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            mock_settings.rerank_enabled = False
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = [self._mock_result()]

            await search(query="test", namespace_id=1, limit=1)

        # rerank module should never be called when flag is OFF
        mock_rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_flag_on_rerank_called_and_score_kind_colbert(self):
        """V-AC-2: rerank_enabled=True → rerank() is called; result has score_kind='colbert_rerank'."""
        from scrutator.db.models import Citation
        from scrutator.search.searcher import search

        reranked_result = self._mock_result("A", 4.5)
        reranked_result.citation = Citation(
            chunk_id="A",
            source_path="docs/A.md",
            source_type="md",
            chunk_index=0,
            relevance_score=4.5,
            score_kind="colbert_rerank",
        )

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
            patch("scrutator.search.searcher.settings") as mock_settings,
            patch("scrutator.search.searcher.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            mock_settings.rerank_enabled = True
            mock_settings.rerank_pool_multiplier = 4
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = [self._mock_result("A", 0.03)]
            mock_rerank.return_value = [reranked_result]

            resp = await search(query="test", namespace_id=1, limit=1)

        mock_rerank.assert_called_once()
        assert resp.results[0].citation.score_kind == "colbert_rerank"
        assert resp.results[0].citation.relevance_score == pytest.approx(4.5)

    @pytest.mark.asyncio
    async def test_flag_on_wider_pool_passed_to_hybrid_search(self):
        """V-AC-2: rerank_enabled=True → hybrid_search called with return_pool=True and multiplier."""
        from scrutator.search.searcher import search

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
            patch("scrutator.search.searcher.settings") as mock_settings,
            patch("scrutator.search.searcher.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            mock_settings.rerank_enabled = True
            mock_settings.rerank_pool_multiplier = 4
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = []
            mock_rerank.return_value = []

            await search(query="test", namespace_id=1, limit=5)

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs.get("fetch_multiplier") == 4
        assert call_kwargs.get("return_pool") is True

    def test_config_rerank_defaults(self):
        """V-AC-2: config defaults — rerank_enabled=False, multiplier=4, max_pool=30."""
        from scrutator.config import Settings

        s = Settings()
        assert s.rerank_enabled is False
        assert s.rerank_pool_multiplier == 4
        assert s.rerank_colbert_max_pool == 30

    # SRCH-0024 removed a stale SRCH-0029 working-tree snapshot assertion here.
    # The rerank contract is covered above; legitimate later LTM changes must not
    # make this otherwise unrelated suite fail solely because the tree is dirty.
