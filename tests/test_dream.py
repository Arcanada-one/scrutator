"""Tests for dream module — models, repository, analyzer, API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scrutator.dream.models import (
    BoostScore,
    CrossReference,
    DreamAnalysisRequest,
    DreamAnalysisResult,
    DuplicatePair,
    EdgeCreate,
    EdgeInfo,
    OrphanChunk,
    StaleChunk,
)

# ── Model validation tests ──────────────────────────────────────────


class TestDreamAnalysisRequest:
    def test_valid_request(self):
        req = DreamAnalysisRequest(namespace="arcanada")
        assert req.namespace == "arcanada"
        assert req.min_similarity == 0.7
        assert req.dedup_threshold == 0.92
        assert req.max_results_per_type == 50
        assert req.stale_days == 90
        assert req.include_boost is True

    def test_namespace_stripped(self):
        req = DreamAnalysisRequest(namespace="  arcanada  ")
        assert req.namespace == "arcanada"

    def test_namespace_empty_fails(self):
        with pytest.raises(ValueError, match="namespace must not be empty"):
            DreamAnalysisRequest(namespace="   ")

    def test_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            DreamAnalysisRequest(namespace="test", min_similarity=0.0)
        with pytest.raises(ValueError, match="threshold must be in"):
            DreamAnalysisRequest(namespace="test", dedup_threshold=1.5)

    def test_max_results_capped(self):
        req = DreamAnalysisRequest(namespace="test", max_results_per_type=500)
        assert req.max_results_per_type == 200

    def test_max_results_negative_fails(self):
        with pytest.raises(ValueError, match="max_results_per_type must be >= 1"):
            DreamAnalysisRequest(namespace="test", max_results_per_type=0)


class TestDuplicatePair:
    def test_creation(self):
        pair = DuplicatePair(
            chunk_id_a="a1",
            chunk_id_b="b2",
            similarity=0.95,
            source_path_a="/doc/a.md",
            source_path_b="/doc/b.md",
            content_preview_a="Hello world",
            content_preview_b="Hello world!",
        )
        assert pair.similarity == 0.95
        assert pair.chunk_id_a == "a1"


class TestCrossReference:
    def test_creation_with_defaults(self):
        ref = CrossReference(
            chunk_id_a="a1",
            chunk_id_b="b2",
            similarity=0.78,
            source_path_a="/a.md",
            source_path_b="/b.md",
        )
        assert ref.suggested_edge_type == "related"

    def test_custom_edge_type(self):
        ref = CrossReference(
            chunk_id_a="a1",
            chunk_id_b="b2",
            similarity=0.78,
            source_path_a="/a.md",
            source_path_b="/b.md",
            suggested_edge_type="refines",
        )
        assert ref.suggested_edge_type == "refines"


class TestOrphanChunk:
    def test_creation(self):
        orphan = OrphanChunk(chunk_id="c1", source_path="/x.md", edge_count=0, created_at="2026-01-01T00:00:00Z")
        assert orphan.edge_count == 0


class TestStaleChunk:
    def test_creation(self):
        stale = StaleChunk(chunk_id="c1", source_path="/old.md", days_since_update=120, edge_count=2)
        assert stale.days_since_update == 120


class TestBoostScore:
    def test_creation(self):
        boost = BoostScore(chunk_id="c1", source_path="/a.md", edge_count=5, avg_edge_weight=0.8, boost_score=0.75)
        assert boost.boost_score == 0.75


class TestEdgeCreate:
    def test_valid_edge(self):
        edge = EdgeCreate(source_chunk_id="a1", target_chunk_id="b2", edge_type="related")
        assert edge.weight == 1.0
        assert edge.created_by == "dreamer"

    def test_edge_type_empty_fails(self):
        with pytest.raises(ValueError, match="edge_type must not be empty"):
            EdgeCreate(source_chunk_id="a1", target_chunk_id="b2", edge_type="  ")


class TestEdgeInfo:
    def test_creation(self):
        info = EdgeInfo(
            id=1,
            source_chunk_id="a1",
            target_chunk_id="b2",
            edge_type="related",
            weight=0.9,
            created_by="dreamer",
            created_at="2026-01-01T00:00:00Z",
        )
        assert info.id == 1


class TestDreamAnalysisResult:
    def test_empty_result(self):
        result = DreamAnalysisResult(
            namespace="test",
            duplicates=[],
            cross_references=[],
            orphans=[],
            stale=[],
            boosts=[],
            stats={"total_chunks": 0, "total_edges": 0, "analysis_time_ms": 10},
        )
        assert result.namespace == "test"
        assert len(result.duplicates) == 0
        assert result.stats["analysis_time_ms"] == 10


# ── Repository tests (graph edges CRUD + analysis queries) ──────────


class TestDreamConfig:
    def test_dream_defaults(self):
        from scrutator.config import Settings

        s = Settings()
        assert s.dream_dedup_threshold == 0.92
        assert s.dream_crossref_threshold == 0.7
        assert s.dream_stale_days == 90
        assert s.dream_max_results == 50
        assert s.dream_analysis_timeout_ms == 30000


# ── Repository mock helper ──────────────────────────────────────────


def _make_pool_mock(mock_conn):
    """Create a properly configured asyncpg pool mock with async context manager."""
    mock_pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx
    return mock_pool


class TestInsertEdges:
    @pytest.mark.asyncio
    async def test_insert_edges_batch(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        mock_pool = _make_pool_mock(mock_conn)

        async def fake_get_pool():
            return mock_pool

        with patch("scrutator.db.repository.get_pool", side_effect=fake_get_pool):
            from scrutator.db.repository import insert_edges

            count = await insert_edges(
                [
                    {
                        "source_chunk_id": "a1",
                        "target_chunk_id": "b2",
                        "edge_type": "related",
                        "weight": 1.0,
                        "created_by": "dreamer",
                    },
                    {
                        "source_chunk_id": "c3",
                        "target_chunk_id": "d4",
                        "edge_type": "contradicts",
                        "weight": 0.8,
                        "created_by": "dreamer",
                    },
                ]
            )
            assert count == 2
            assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_insert_edges_empty(self):
        mock_get_pool = AsyncMock()

        with patch("scrutator.db.repository.get_pool", mock_get_pool):
            from scrutator.db.repository import insert_edges

            count = await insert_edges([])
            assert count == 0
            mock_get_pool.assert_not_called()


class TestGetEdgesForChunk:
    @pytest.mark.asyncio
    async def test_get_edges_returns_list(self):
        mock_rows = [
            {
                "id": 1,
                "source_chunk_id": "a1",
                "target_chunk_id": "b2",
                "edge_type": "related",
                "weight": 1.0,
                "created_by": "dreamer",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": 2,
                "source_chunk_id": "c3",
                "target_chunk_id": "a1",
                "edge_type": "refines",
                "weight": 0.9,
                "created_by": "dreamer",
                "created_at": "2026-01-02T00:00:00+00:00",
            },
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = _make_pool_mock(mock_conn)

        async def fake_get_pool():
            return mock_pool

        with patch("scrutator.db.repository.get_pool", side_effect=fake_get_pool):
            from scrutator.db.repository import get_edges_for_chunk

            edges = await get_edges_for_chunk("a1")
            assert len(edges) == 2
            assert edges[0]["edge_type"] == "related"


class TestDeleteEdgesByCreator:
    @pytest.mark.asyncio
    async def test_delete_by_creator(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 5")
        mock_pool = _make_pool_mock(mock_conn)

        async def fake_get_pool():
            return mock_pool

        with patch("scrutator.db.repository.get_pool", side_effect=fake_get_pool):
            from scrutator.db.repository import delete_edges_by_creator

            count = await delete_edges_by_creator("dreamer")
            assert count == 5

    @pytest.mark.asyncio
    async def test_delete_with_namespace(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 3")
        mock_pool = _make_pool_mock(mock_conn)

        async def fake_get_pool():
            return mock_pool

        with patch("scrutator.db.repository.get_pool", side_effect=fake_get_pool):
            from scrutator.db.repository import delete_edges_by_creator

            count = await delete_edges_by_creator("dreamer", namespace_id=1)
            assert count == 3
            call_args = mock_conn.execute.call_args
            assert "namespace_id" in call_args[0][0] or len(call_args[0]) > 2


# ── Repository analysis queries ─────────────────────────────────────


class TestFindSimilarPairs:
    @pytest.mark.asyncio
    async def test_returns_pairs(self):
        from uuid import uuid4

        id_a, id_b = str(uuid4()), str(uuid4())
        mock_rows = [
            {
                "chunk_id_a": id_a,
                "chunk_id_b": id_b,
                "similarity": 0.95,
                "source_path_a": "/a.md",
                "source_path_b": "/b.md",
                "content_a": "Hello world content",
                "content_b": "Hello world content!",
            }
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = _make_pool_mock(mock_conn)

        async def fake_get_pool():
            return mock_pool

        with patch("scrutator.db.repository.get_pool", side_effect=fake_get_pool):
            from scrutator.db.repository import find_similar_pairs

            pairs = await find_similar_pairs(namespace_id=1, threshold=0.92)
            assert len(pairs) == 1
            assert pairs[0]["similarity"] == 0.95


class TestGetOrphanChunks:
    @pytest.mark.asyncio
    async def test_returns_orphans(self):
        mock_rows = [
            {"chunk_id": "c1", "source_path": "/lonely.md", "edge_count": 0, "created_at": "2026-01-01T00:00:00+00:00"},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = _make_pool_mock(mock_conn)

        async def fake_get_pool():
            return mock_pool

        with patch("scrutator.db.repository.get_pool", side_effect=fake_get_pool):
            from scrutator.db.repository import get_orphan_chunks

            orphans = await get_orphan_chunks(namespace_id=1)
            assert len(orphans) == 1
            assert orphans[0]["edge_count"] == 0


class TestFindStaleChunks:
    @pytest.mark.asyncio
    async def test_returns_stale(self):
        mock_rows = [
            {"chunk_id": "c1", "source_path": "/old.md", "days_since_update": 120, "edge_count": 2},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = _make_pool_mock(mock_conn)

        async def fake_get_pool():
            return mock_pool

        with patch("scrutator.db.repository.get_pool", side_effect=fake_get_pool):
            from scrutator.db.repository import find_stale_chunks

            stale = await find_stale_chunks(namespace_id=1, stale_days=90)
            assert len(stale) == 1
            assert stale[0]["days_since_update"] == 120


class TestGetEdgeStats:
    @pytest.mark.asyncio
    async def test_returns_stats(self):
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=42)
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"edge_type": "related", "count": 30, "avg_weight": 0.85},
                {"edge_type": "contradicts", "count": 12, "avg_weight": 0.6},
            ]
        )
        mock_pool = _make_pool_mock(mock_conn)

        async def fake_get_pool():
            return mock_pool

        with patch("scrutator.db.repository.get_pool", side_effect=fake_get_pool):
            from scrutator.db.repository import get_edge_stats

            stats = await get_edge_stats()
            assert stats["total_edges"] == 42


# ── Analyzer tests ──────────────────────────────────────────────────


class TestAnalyzerResolveNamespace:
    @pytest.mark.asyncio
    async def test_unknown_namespace_returns_empty(self):
        from scrutator.db.models import NamespaceInfo
        from scrutator.dream.analyzer import analyze

        with patch("scrutator.dream.analyzer.repository") as mock_repo:
            mock_repo.get_namespaces = AsyncMock(
                return_value=[
                    NamespaceInfo(id=1, name="other", description=None, chunk_count=10),
                ]
            )
            result = await analyze(DreamAnalysisRequest(namespace="nonexistent"))
            assert result.namespace == "nonexistent"
            assert result.stats.get("error") == "namespace_not_found"
            assert len(result.duplicates) == 0


class TestAnalyzerFindDuplicates:
    @pytest.mark.asyncio
    async def test_returns_duplicates(self):
        from scrutator.dream.analyzer import find_semantic_duplicates

        with patch("scrutator.dream.analyzer.repository") as mock_repo:
            mock_repo.find_similar_pairs = AsyncMock(
                return_value=[
                    {
                        "chunk_id_a": "a1",
                        "chunk_id_b": "b2",
                        "similarity": 0.95,
                        "source_path_a": "/a.md",
                        "source_path_b": "/b.md",
                        "content_a": "Hello",
                        "content_b": "Hello!",
                    }
                ]
            )
            dups = await find_semantic_duplicates(namespace_id=1, threshold=0.92, limit=50)
            assert len(dups) == 1
            assert dups[0].similarity == 0.95


class TestAnalyzerCrossReferences:
    @pytest.mark.asyncio
    async def test_excludes_duplicates(self):
        from scrutator.dream.analyzer import find_cross_references

        with patch("scrutator.dream.analyzer.repository") as mock_repo:
            mock_repo.find_similar_pairs = AsyncMock(
                return_value=[
                    {
                        "chunk_id_a": "a1",
                        "chunk_id_b": "b2",
                        "similarity": 0.95,
                        "source_path_a": "/a.md",
                        "source_path_b": "/b.md",
                        "content_a": "X",
                        "content_b": "Y",
                    },
                    {
                        "chunk_id_a": "c3",
                        "chunk_id_b": "d4",
                        "similarity": 0.78,
                        "source_path_a": "/c.md",
                        "source_path_b": "/d.md",
                        "content_a": "A",
                        "content_b": "B",
                    },
                ]
            )
            refs = await find_cross_references(namespace_id=1, min_similarity=0.7, dedup_threshold=0.92, limit=50)
            assert len(refs) == 1
            assert refs[0].similarity == 0.78


class TestAnalyzerOrphanChunks:
    @pytest.mark.asyncio
    async def test_returns_orphans(self):
        from scrutator.dream.analyzer import find_orphan_chunks

        with patch("scrutator.dream.analyzer.repository") as mock_repo:
            mock_repo.get_orphan_chunks = AsyncMock(
                return_value=[
                    {"chunk_id": "c1", "source_path": "/x.md", "edge_count": 0, "created_at": "2026-01-01"},
                ]
            )
            orphans = await find_orphan_chunks(namespace_id=1, limit=50)
            assert len(orphans) == 1
            assert orphans[0].edge_count == 0


class TestAnalyzerFullAnalysis:
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self):
        from scrutator.db.models import NamespaceInfo
        from scrutator.dream.analyzer import analyze

        with patch("scrutator.dream.analyzer.repository") as mock_repo:
            mock_repo.get_namespaces = AsyncMock(
                return_value=[
                    NamespaceInfo(id=1, name="arcanada", description=None, chunk_count=100),
                ]
            )
            mock_repo.find_similar_pairs = AsyncMock(return_value=[])
            mock_repo.get_orphan_chunks = AsyncMock(return_value=[])
            mock_repo.find_stale_chunks = AsyncMock(return_value=[])
            mock_repo.get_edge_stats = AsyncMock(return_value={"total_edges": 0, "by_type": []})
            mock_repo.get_stats = AsyncMock(
                return_value={
                    "total_chunks": 100,
                    "total_namespaces": 1,
                    "total_projects": 2,
                    "namespaces": [],
                }
            )

            result = await analyze(DreamAnalysisRequest(namespace="arcanada"))
            assert result.namespace == "arcanada"
            assert result.stats["total_chunks"] == 100
            assert result.stats["analysis_time_ms"] >= 0
            assert len(result.duplicates) == 0
            assert len(result.cross_references) == 0

    @pytest.mark.asyncio
    async def test_analysis_with_boost(self):
        from scrutator.db.models import NamespaceInfo
        from scrutator.dream.analyzer import analyze

        with patch("scrutator.dream.analyzer.repository") as mock_repo:
            mock_repo.get_namespaces = AsyncMock(
                return_value=[
                    NamespaceInfo(id=1, name="test", description=None, chunk_count=50),
                ]
            )
            mock_repo.find_similar_pairs = AsyncMock(return_value=[])
            mock_repo.get_orphan_chunks = AsyncMock(return_value=[])
            mock_repo.find_stale_chunks = AsyncMock(return_value=[])
            mock_repo.get_edge_stats = AsyncMock(
                return_value={
                    "total_edges": 10,
                    "by_type": [
                        {"edge_type": "related", "count": 7, "avg_weight": 0.9},
                        {"edge_type": "contradicts", "count": 3, "avg_weight": 0.5},
                    ],
                }
            )
            mock_repo.get_stats = AsyncMock(
                return_value={
                    "total_chunks": 50,
                    "total_namespaces": 1,
                    "total_projects": 1,
                    "namespaces": [],
                }
            )

            result = await analyze(DreamAnalysisRequest(namespace="test", include_boost=True))
            assert len(result.boosts) == 2
            assert result.boosts[0].edge_count == 7


# ── API endpoint tests ──────────────────────────────────────────────


class TestDreamAPI:
    def test_dream_analyze_validation_error(self):
        from scrutator.health import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/dream/analyze", json={"namespace": ""})
        assert resp.status_code == 422

    def test_dream_analyze_returns_result(self):
        from scrutator.health import app

        mock_result = DreamAnalysisResult(
            namespace="test",
            duplicates=[],
            cross_references=[],
            orphans=[],
            stale=[],
            boosts=[],
            stats={"total_chunks": 0, "total_edges": 0, "analysis_time_ms": 5},
        )

        with patch("scrutator.health.dream_analyze", new_callable=AsyncMock, return_value=mock_result):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/dream/analyze", json={"namespace": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["namespace"] == "test"
            assert data["stats"]["total_chunks"] == 0

    def test_create_edges_endpoint(self):
        from scrutator.health import app

        with patch("scrutator.health.insert_edges", new_callable=AsyncMock, return_value=2):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/edges",
                json=[
                    {"source_chunk_id": "a1", "target_chunk_id": "b2", "edge_type": "related"},
                    {"source_chunk_id": "c3", "target_chunk_id": "d4", "edge_type": "contradicts"},
                ],
            )
            assert resp.status_code == 200
            assert resp.json()["created"] == 2

    def test_create_edges_validation_error(self):
        from scrutator.health import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/edges",
            json=[
                {"source_chunk_id": "a1", "target_chunk_id": "b2", "edge_type": ""},
            ],
        )
        assert resp.status_code == 422

    def test_get_edges_endpoint(self):
        from scrutator.health import app

        mock_edges = [
            {
                "id": 1,
                "source_chunk_id": "a1",
                "target_chunk_id": "b2",
                "edge_type": "related",
                "weight": 1.0,
                "created_by": "dreamer",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]

        with patch("scrutator.health.get_edges_for_chunk", new_callable=AsyncMock, return_value=mock_edges):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/v1/edges/a1")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["edge_type"] == "related"

    def test_delete_edges_endpoint(self):
        from scrutator.health import app

        with patch("scrutator.health.delete_edges_by_creator", new_callable=AsyncMock, return_value=3):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.delete("/v1/edges?created_by=dreamer")
            assert resp.status_code == 200
            assert resp.json()["deleted"] == 3
