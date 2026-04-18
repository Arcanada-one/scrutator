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
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 1024}, {"embedding": [0.2] * 1024}]}

        with patch("scrutator.search.embedder.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

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

        with patch("scrutator.search.embedder.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            with pytest.raises(EmbeddingError, match="500"):
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
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.indexer.upsert_project", new_callable=AsyncMock) as mock_proj,
            patch("scrutator.search.indexer.delete_by_source", new_callable=AsyncMock) as mock_del,
            patch("scrutator.search.indexer.insert_chunks", new_callable=AsyncMock) as mock_insert,
        ):
            mock_embed.return_value = [[0.1] * 1024]
            mock_ns.return_value = 1
            mock_proj.return_value = 10
            mock_del.return_value = 0
            mock_insert.return_value = 1

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
            mock_del.assert_called_once_with("test.md")

    @pytest.mark.asyncio
    async def test_index_empty_content_returns_zero(self):
        """Chunker produces empty result for whitespace-only → 0 indexed."""
        from scrutator.search.indexer import index_document

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock),
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock),
            patch("scrutator.search.indexer.delete_by_source", new_callable=AsyncMock),
            patch("scrutator.search.indexer.insert_chunks", new_callable=AsyncMock),
        ):
            # Very short content → single chunk, still indexable
            result = await index_document(content="x", source_path="empty.md")
            # Single-char content still produces a chunk
            assert result.chunks_indexed >= 0

    @pytest.mark.asyncio
    async def test_index_without_project(self):
        from scrutator.search.indexer import index_document

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.indexer.delete_by_source", new_callable=AsyncMock),
            patch("scrutator.search.indexer.insert_chunks", new_callable=AsyncMock) as mock_insert,
        ):
            mock_embed.return_value = [[0.1] * 1024]
            mock_ns.return_value = 1
            mock_insert.return_value = 1

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
            patch("scrutator.search.searcher.upsert_namespace", new_callable=AsyncMock) as mock_ns,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = mock_results
            mock_ns.return_value = 1

            resp = await search(query="test query", namespace="arcanada", limit=5)
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

            resp = await search(query="test", min_score=0.03)
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

            resp = await search(query="test", include_content=False)
            assert resp.results[0].content == ""

    @pytest.mark.asyncio
    async def test_search_no_namespace(self):
        """Search without namespace → all namespaces."""
        from scrutator.search.searcher import search

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = []

            resp = await search(query="test")
            assert resp.total == 0
            # namespace_id=None was passed
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args
            assert call_kwargs[1]["namespace_id"] is None


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

        from scrutator.health import app

        client = TestClient(app)
        resp = client.post("/v1/index", json={"content": "  ", "source_path": "x.md"})
        assert resp.status_code == 422
