"""Tests for memory module — models, service, repository, API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scrutator.memory.models import (
    MemoryBulkRequest,
    MemoryDeleteRequest,
    MemoryRecallRequest,
    MemoryRecord,
    MemoryStats,
)

# ── Model validation tests ─────────────────────────────────────────


class TestMemoryRecord:
    def test_defaults(self):
        r = MemoryRecord(content="test fact", actor="dreamer")
        assert r.memory_type == "fact"
        assert r.namespace == "arcanada"
        assert r.importance == 0.5
        assert r.tags == []
        assert r.project is None

    def test_empty_content_rejected(self):
        with pytest.raises(ValueError, match="content must not be empty"):
            MemoryRecord(content="   ", actor="dreamer")

    def test_long_content_rejected(self):
        with pytest.raises(ValueError, match="exceeds 10000"):
            MemoryRecord(content="x" * 10_001, actor="dreamer")

    def test_empty_actor_rejected(self):
        with pytest.raises(ValueError, match="actor must not be empty"):
            MemoryRecord(content="test", actor="  ")

    def test_invalid_memory_type(self):
        with pytest.raises(ValueError, match="memory_type must be one of"):
            MemoryRecord(content="test", actor="dreamer", memory_type="invalid")

    def test_valid_memory_types(self):
        for mt in ("fact", "preference", "decision", "event", "observation"):
            r = MemoryRecord(content="test", actor="dreamer", memory_type=mt)
            assert r.memory_type == mt

    def test_importance_out_of_range(self):
        with pytest.raises(ValueError, match="importance must be between"):
            MemoryRecord(content="test", actor="dreamer", importance=1.5)
        with pytest.raises(ValueError, match="importance must be between"):
            MemoryRecord(content="test", actor="dreamer", importance=-0.1)

    def test_full_record(self):
        r = MemoryRecord(
            content="User prefers Russian",
            actor="datarim",
            memory_type="preference",
            namespace="personal",
            project="notes",
            tags=["lang", "ui"],
            importance=0.9,
            valid_from="2026-04-18T00:00:00Z",
            source_ref="SRCH-0006",
        )
        assert r.namespace == "personal"
        assert r.tags == ["lang", "ui"]
        assert r.importance == 0.9


class TestMemoryBulkRequest:
    def test_empty_list_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            MemoryBulkRequest(memories=[])

    def test_oversized_list_rejected(self):
        memories = [MemoryRecord(content=f"fact {i}", actor="test") for i in range(101)]
        with pytest.raises(ValueError, match="exceeds 100"):
            MemoryBulkRequest(memories=memories)

    def test_valid_bulk(self):
        memories = [MemoryRecord(content="fact 1", actor="test")]
        req = MemoryBulkRequest(memories=memories)
        assert len(req.memories) == 1


class TestMemoryRecallRequest:
    def test_defaults(self):
        r = MemoryRecallRequest(query="test")
        assert r.limit == 10
        assert r.importance_boost is True
        assert r.include_expired is False

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError, match="query must not be empty"):
            MemoryRecallRequest(query="   ")

    def test_limit_capped(self):
        r = MemoryRecallRequest(query="test", limit=999)
        assert r.limit == 50

    def test_invalid_memory_type_filter(self):
        with pytest.raises(ValueError, match="memory_type must be one of"):
            MemoryRecallRequest(query="test", memory_type="bogus")

    def test_none_memory_type_allowed(self):
        r = MemoryRecallRequest(query="test", memory_type=None)
        assert r.memory_type is None


class TestMemoryDeleteRequest:
    def test_empty_actor_rejected(self):
        with pytest.raises(ValueError, match="actor must not be empty"):
            MemoryDeleteRequest(actor="  ")

    def test_valid(self):
        r = MemoryDeleteRequest(actor="dreamer", namespace="arcanada")
        assert r.actor == "dreamer"


class TestMemoryStats:
    def test_defaults(self):
        s = MemoryStats(total_memories=0)
        assert s.by_namespace == {}
        assert s.by_actor == {}
        assert s.by_type == {}

    def test_populated(self):
        s = MemoryStats(
            total_memories=10,
            by_namespace={"arcanada": 8, "personal": 2},
            by_actor={"dreamer": 6, "user": 4},
            by_type={"fact": 5, "decision": 3, "event": 2},
        )
        assert s.total_memories == 10
        assert s.by_actor["dreamer"] == 6


# ── Service tests ──────────────────────────────────────────────────


def _make_pool_mock(mock_conn):
    """Create asyncpg pool mock with acquire() context manager."""
    mock_pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx
    return mock_pool


class TestServiceIndexMemory:
    @pytest.mark.asyncio
    async def test_index_creates_chunk(self):
        record = MemoryRecord(content="test fact", actor="dreamer")

        with (
            patch("scrutator.memory.service.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.memory.service.repository") as mock_repo,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_repo.upsert_namespace = AsyncMock(return_value=1)
            mock_repo.upsert_project = AsyncMock(return_value=1)
            mock_repo.insert_chunks = AsyncMock(return_value=1)

            from scrutator.memory.service import index_memory

            result = await index_memory(record)

        assert result.namespace == "arcanada"
        assert result.memory_id  # non-empty UUID
        mock_repo.insert_chunks.assert_called_once()
        args = mock_repo.insert_chunks.call_args
        chunk = args[0][0][0]
        assert chunk["source_type"] == "memory"
        assert chunk["source_path"].startswith("memory://arcanada/_/dreamer/")
        assert args[0][0][0]["metadata"]["actor"] == "dreamer"

    @pytest.mark.asyncio
    async def test_index_with_project(self):
        record = MemoryRecord(content="project fact", actor="user", project="scrutator")

        with (
            patch("scrutator.memory.service.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.memory.service.repository") as mock_repo,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_repo.upsert_namespace = AsyncMock(return_value=1)
            mock_repo.upsert_project = AsyncMock(return_value=2)
            mock_repo.insert_chunks = AsyncMock(return_value=1)

            from scrutator.memory.service import index_memory

            await index_memory(record)

        mock_repo.upsert_project.assert_called_once_with(1, "scrutator")
        args = mock_repo.insert_chunks.call_args
        chunk = args[0][0][0]
        assert "scrutator" in chunk["source_path"]


class TestServiceBulkIndex:
    @pytest.mark.asyncio
    async def test_bulk_indexes_all(self):
        records = [
            MemoryRecord(content="fact 1", actor="dreamer"),
            MemoryRecord(content="fact 2", actor="dreamer"),
        ]

        with (
            patch("scrutator.memory.service.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.memory.service.repository") as mock_repo,
        ):
            mock_embed.return_value = [0.1] * 1024
            mock_repo.upsert_namespace = AsyncMock(return_value=1)
            mock_repo.upsert_project = AsyncMock(return_value=1)
            mock_repo.insert_chunks = AsyncMock(return_value=1)

            from scrutator.memory.service import bulk_index

            result = await bulk_index(records)

        assert result.indexed == 2
        assert len(result.memory_ids) == 2


class TestServiceRecall:
    @pytest.mark.asyncio
    async def test_recall_returns_results(self):
        request = MemoryRecallRequest(query="test query", namespace="arcanada")

        with patch("scrutator.memory.service.repository") as mock_repo:
            mock_repo.upsert_namespace = AsyncMock(return_value=1)
            mock_repo.search_with_filters = AsyncMock(
                return_value=[
                    {
                        "chunk_id": "abc-123",
                        "content": "test fact",
                        "source_path": "memory://arcanada/_/dreamer/abc-123",
                        "source_type": "memory",
                        "chunk_index": 0,
                        "score": 0.85,
                        "namespace": "arcanada",
                        "project": None,
                        "metadata": {
                            "memory_id": "abc-123",
                            "actor": "dreamer",
                            "memory_type": "fact",
                            "importance": 0.8,
                            "tags": ["search"],
                        },
                        "created_at": "2026-04-18T12:00:00+00:00",
                    }
                ]
            )

            from scrutator.memory.service import recall

            result = await recall(request)

        assert result.total == 1
        assert result.results[0].actor == "dreamer"
        assert result.results[0].importance == 0.8
        assert result.results[0].tags == ["search"]

    @pytest.mark.asyncio
    async def test_recall_filters_by_min_score(self):
        request = MemoryRecallRequest(query="test", min_score=0.9)

        with patch("scrutator.memory.service.repository") as mock_repo:
            mock_repo.upsert_namespace = AsyncMock(return_value=1)
            mock_repo.search_with_filters = AsyncMock(
                return_value=[
                    {
                        "chunk_id": "abc",
                        "content": "low score",
                        "source_path": "memory://arcanada/_/dreamer/abc",
                        "source_type": "memory",
                        "chunk_index": 0,
                        "score": 0.5,
                        "namespace": "arcanada",
                        "metadata": {"actor": "dreamer", "memory_type": "fact", "importance": 0.5},
                    }
                ]
            )

            from scrutator.memory.service import recall

            result = await recall(request)

        assert result.total == 0

    @pytest.mark.asyncio
    async def test_recall_empty_results(self):
        request = MemoryRecallRequest(query="nothing here")

        with patch("scrutator.memory.service.repository") as mock_repo:
            mock_repo.upsert_namespace = AsyncMock(return_value=1)
            mock_repo.search_with_filters = AsyncMock(return_value=[])

            from scrutator.memory.service import recall

            result = await recall(request)

        assert result.total == 0
        assert result.results == []


class TestServiceGetStats:
    @pytest.mark.asyncio
    async def test_get_stats(self):
        with patch("scrutator.memory.service.repository") as mock_repo:
            mock_repo.memory_stats = AsyncMock(
                return_value=MemoryStats(
                    total_memories=5,
                    by_namespace={"arcanada": 5},
                    by_actor={"dreamer": 3, "user": 2},
                    by_type={"fact": 4, "decision": 1},
                )
            )

            from scrutator.memory.service import get_memory_stats

            result = await get_memory_stats()

        assert result.total_memories == 5
        assert result.by_actor["dreamer"] == 3


# ── API endpoint tests ─────────────────────────────────────────────


class TestMemoryAPI:
    def test_create_memory_validation_error(self):
        from scrutator.health import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/memories", json={"content": "", "actor": "dreamer"})
        assert resp.status_code == 422

    def test_create_memory_invalid_type(self):
        from scrutator.health import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/memories",
            json={"content": "test", "actor": "dreamer", "memory_type": "bogus"},
        )
        assert resp.status_code == 422

    def test_bulk_empty_rejected(self):
        from scrutator.health import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/memories/bulk", json={"memories": []})
        assert resp.status_code == 422

    def test_recall_validation_error(self):
        from scrutator.health import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/memories/recall", json={"query": ""})
        assert resp.status_code == 422

    def test_recall_invalid_type_filter(self):
        from scrutator.health import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/memories/recall",
            json={"query": "test", "memory_type": "invalid"},
        )
        assert resp.status_code == 422

    def test_search_with_source_type_field(self):
        """SearchRequest now accepts source_type field."""
        from scrutator.db.models import SearchRequest

        r = SearchRequest(query="test", source_type="memory")
        assert r.source_type == "memory"

        r2 = SearchRequest(query="test")
        assert r2.source_type is None


# ── Config tests ───────────────────────────────────────────────────


class TestMemoryConfig:
    def test_memory_config_defaults(self):
        from scrutator.config import Settings

        s = Settings()
        assert s.memory_default_importance == 0.5
        assert s.memory_decay_days == 180
        assert s.memory_max_bulk_size == 100
