"""Tests for chunk lookup by source_path — repository + API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scrutator.db.models import ChunkLookupResult

# ── Model tests ───────────────────────────────────────────────────


class TestChunkLookupResult:
    def test_defaults(self):
        result = ChunkLookupResult(
            chunk_id="abc-123",
            chunk_index=0,
            source_path="wiki/AI/ML.md",
            source_type="markdown",
        )
        assert result.content_preview == ""
        assert result.metadata == {}

    def test_full(self):
        result = ChunkLookupResult(
            chunk_id="abc-123",
            chunk_index=2,
            source_path="wiki/AI/ML.md",
            source_type="markdown",
            content_preview="Some content...",
            metadata={"heading": "Introduction"},
        )
        assert result.chunk_index == 2
        assert result.content_preview == "Some content..."
        assert result.metadata["heading"] == "Introduction"


# ── Repository tests ──────────────────────────────────────────────


def _mock_pool():
    """Create a mock asyncpg pool with context manager support."""
    pool = MagicMock()
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


class TestGetChunksBySourcePath:
    @pytest.mark.asyncio
    async def test_returns_chunks(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [
            {
                "chunk_id": "uuid-1",
                "chunk_index": 0,
                "source_type": "markdown",
                "source_path": "wiki/AI/ML.md",
                "content_preview": "Machine learning is...",
                "metadata": "{}",
            },
            {
                "chunk_id": "uuid-2",
                "chunk_index": 1,
                "source_type": "markdown",
                "source_path": "wiki/AI/ML.md",
                "content_preview": "Supervised learning...",
                "metadata": '{"heading": "Types"}',
            },
        ]

        with patch("scrutator.db.repository.get_pool", return_value=pool):
            from scrutator.db.repository import get_chunks_by_source_path

            results = await get_chunks_by_source_path("wiki/AI/ML.md")
            assert len(results) == 2
            assert results[0].chunk_id == "uuid-1"
            assert results[0].chunk_index == 0
            assert results[1].chunk_index == 1
            assert results[1].metadata == {"heading": "Types"}

    @pytest.mark.asyncio
    async def test_no_results(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []

        with patch("scrutator.db.repository.get_pool", return_value=pool):
            from scrutator.db.repository import get_chunks_by_source_path

            results = await get_chunks_by_source_path("nonexistent/path.md")
            assert results == []

    @pytest.mark.asyncio
    async def test_with_namespace(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [
            {
                "chunk_id": "uuid-1",
                "chunk_index": 0,
                "source_type": "markdown",
                "source_path": "wiki/AI/ML.md",
                "content_preview": "Content",
                "metadata": "{}",
            },
        ]

        with patch("scrutator.db.repository.get_pool", return_value=pool):
            from scrutator.db.repository import get_chunks_by_source_path

            results = await get_chunks_by_source_path("wiki/AI/ML.md", namespace_id=1)
            assert len(results) == 1
            # Verify the namespace_id branch was used (2 params in query)
            call_args = conn.fetch.call_args
            assert call_args[0][1] == "wiki/AI/ML.md"
            assert call_args[0][2] == 1

    @pytest.mark.asyncio
    async def test_ordered_by_chunk_index(self):
        pool, conn = _mock_pool()
        # Return in correct order (SQL ORDER BY chunk_index)
        conn.fetch.return_value = [
            {
                "chunk_id": "uuid-0",
                "chunk_index": 0,
                "source_type": "markdown",
                "source_path": "wiki/AI/ML.md",
                "content_preview": "First",
                "metadata": "{}",
            },
            {
                "chunk_id": "uuid-1",
                "chunk_index": 1,
                "source_type": "markdown",
                "source_path": "wiki/AI/ML.md",
                "content_preview": "Second",
                "metadata": "{}",
            },
            {
                "chunk_id": "uuid-2",
                "chunk_index": 2,
                "source_type": "markdown",
                "source_path": "wiki/AI/ML.md",
                "content_preview": "Third",
                "metadata": "{}",
            },
        ]

        with patch("scrutator.db.repository.get_pool", return_value=pool):
            from scrutator.db.repository import get_chunks_by_source_path

            results = await get_chunks_by_source_path("wiki/AI/ML.md")
            assert len(results) == 3
            assert [r.chunk_index for r in results] == [0, 1, 2]


# ── API tests ─────────────────────────────────────────────────────


class TestChunkLookupAPI:
    def test_get_chunks_endpoint(self):
        mock_results = [
            ChunkLookupResult(
                chunk_id="uuid-1",
                chunk_index=0,
                source_path="wiki/AI/ML.md",
                source_type="markdown",
                content_preview="Content preview",
            ),
        ]

        with (
            patch("scrutator.health.get_chunks_by_source_path", new_callable=AsyncMock, return_value=mock_results),
            patch("scrutator.health.get_namespaces", new_callable=AsyncMock, return_value=[]),
        ):
            from scrutator.health import app

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/v1/chunks", params={"source_path": "wiki/AI/ML.md"})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["chunk_id"] == "uuid-1"
            assert data[0]["chunk_index"] == 0

    def test_get_chunks_endpoint_empty(self):
        with (
            patch("scrutator.health.get_chunks_by_source_path", new_callable=AsyncMock, return_value=[]),
            patch("scrutator.health.get_namespaces", new_callable=AsyncMock, return_value=[]),
        ):
            from scrutator.health import app

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/v1/chunks", params={"source_path": "nonexistent.md"})
            assert resp.status_code == 200
            assert resp.json() == []
