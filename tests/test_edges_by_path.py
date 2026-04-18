"""Tests for edge creation by source_path — models, service, API."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from scrutator.db.models import ChunkLookupResult
from scrutator.dream.models import EdgeCreateByPath, EdgeCreateByPathResponse

# ── Model validation tests ────────────────────────────────────────


class TestEdgeCreateByPath:
    def test_defaults(self):
        edge = EdgeCreateByPath(
            source_path="wiki/AI/ML.md",
            target_path="wiki/AI/DL.md",
            edge_type="related",
        )
        assert edge.weight == 1.0
        assert edge.created_by == "dreamer"
        assert edge.source_chunk_index == 0
        assert edge.target_chunk_index == 0

    def test_empty_edge_type_fails(self):
        with pytest.raises(ValueError, match="edge_type must not be empty"):
            EdgeCreateByPath(
                source_path="wiki/AI/ML.md",
                target_path="wiki/AI/DL.md",
                edge_type="  ",
            )

    def test_empty_source_path_fails(self):
        with pytest.raises(ValueError, match="path must not be empty"):
            EdgeCreateByPath(
                source_path="",
                target_path="wiki/AI/DL.md",
                edge_type="related",
            )

    def test_empty_target_path_fails(self):
        with pytest.raises(ValueError, match="path must not be empty"):
            EdgeCreateByPath(
                source_path="wiki/AI/ML.md",
                target_path="  ",
                edge_type="related",
            )


# ── Service tests ─────────────────────────────────────────────────


def _chunk(chunk_id: str, chunk_index: int = 0, source_path: str = "test.md") -> ChunkLookupResult:
    return ChunkLookupResult(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        source_path=source_path,
        source_type="markdown",
    )


class TestCreateEdgesByPath:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        edges = [
            EdgeCreateByPath(
                source_path="wiki/AI/ML.md",
                target_path="wiki/AI/DL.md",
                edge_type="related",
            ),
        ]

        with (
            patch(
                "scrutator.dream.edges.get_chunks_by_source_path",
                new_callable=AsyncMock,
                side_effect=lambda path, **kw: {
                    "wiki/AI/ML.md": [_chunk("uuid-s", 0, "wiki/AI/ML.md")],
                    "wiki/AI/DL.md": [_chunk("uuid-t", 0, "wiki/AI/DL.md")],
                }.get(path, []),
            ),
            patch(
                "scrutator.dream.edges.insert_edges",
                new_callable=AsyncMock,
                return_value=1,
            ) as mock_insert,
        ):
            from scrutator.dream.edges import create_edges_by_path

            result = await create_edges_by_path(edges)
            assert result.created == 1
            assert result.not_found == []
            # Verify resolved UUIDs passed to insert_edges
            inserted = mock_insert.call_args[0][0]
            assert inserted[0]["source_chunk_id"] == "uuid-s"
            assert inserted[0]["target_chunk_id"] == "uuid-t"

    @pytest.mark.asyncio
    async def test_not_found(self):
        edges = [
            EdgeCreateByPath(
                source_path="wiki/missing.md",
                target_path="wiki/also-missing.md",
                edge_type="related",
            ),
        ]

        with patch(
            "scrutator.dream.edges.get_chunks_by_source_path",
            new_callable=AsyncMock,
            return_value=[],
        ):
            from scrutator.dream.edges import create_edges_by_path

            result = await create_edges_by_path(edges)
            assert result.created == 0
            assert "wiki/missing.md" in result.not_found

    @pytest.mark.asyncio
    async def test_mixed_found_and_not_found(self):
        edges = [
            EdgeCreateByPath(source_path="wiki/found.md", target_path="wiki/also-found.md", edge_type="related"),
            EdgeCreateByPath(source_path="wiki/found.md", target_path="wiki/missing.md", edge_type="related"),
        ]

        with (
            patch(
                "scrutator.dream.edges.get_chunks_by_source_path",
                new_callable=AsyncMock,
                side_effect=lambda path, **kw: {
                    "wiki/found.md": [_chunk("uuid-a", 0, "wiki/found.md")],
                    "wiki/also-found.md": [_chunk("uuid-b", 0, "wiki/also-found.md")],
                }.get(path, []),
            ),
            patch(
                "scrutator.dream.edges.insert_edges",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            from scrutator.dream.edges import create_edges_by_path

            result = await create_edges_by_path(edges)
            assert result.created == 1
            assert "wiki/missing.md" in result.not_found

    @pytest.mark.asyncio
    async def test_custom_chunk_index(self):
        edges = [
            EdgeCreateByPath(
                source_path="wiki/AI/ML.md",
                target_path="wiki/AI/DL.md",
                edge_type="related",
                source_chunk_index=2,
                target_chunk_index=1,
            ),
        ]

        with (
            patch(
                "scrutator.dream.edges.get_chunks_by_source_path",
                new_callable=AsyncMock,
                side_effect=lambda path, **kw: {
                    "wiki/AI/ML.md": [
                        _chunk("uuid-s0", 0, "wiki/AI/ML.md"),
                        _chunk("uuid-s1", 1, "wiki/AI/ML.md"),
                        _chunk("uuid-s2", 2, "wiki/AI/ML.md"),
                    ],
                    "wiki/AI/DL.md": [
                        _chunk("uuid-t0", 0, "wiki/AI/DL.md"),
                        _chunk("uuid-t1", 1, "wiki/AI/DL.md"),
                    ],
                }.get(path, []),
            ),
            patch(
                "scrutator.dream.edges.insert_edges",
                new_callable=AsyncMock,
                return_value=1,
            ) as mock_insert,
        ):
            from scrutator.dream.edges import create_edges_by_path

            result = await create_edges_by_path(edges)
            assert result.created == 1
            inserted = mock_insert.call_args[0][0]
            assert inserted[0]["source_chunk_id"] == "uuid-s2"
            assert inserted[0]["target_chunk_id"] == "uuid-t1"


# ── API tests ─────────────────────────────────────────────────────


class TestEdgesByPathAPI:
    def test_edges_by_path_endpoint(self):
        mock_response = EdgeCreateByPathResponse(created=2, not_found=[])

        with patch(
            "scrutator.health.create_edges_by_path",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            from scrutator.health import app

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/edges/by-path",
                json=[
                    {
                        "source_path": "wiki/AI/ML.md",
                        "target_path": "wiki/AI/DL.md",
                        "edge_type": "related",
                    },
                ],
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["created"] == 2
            assert data["not_found"] == []

    def test_edges_by_path_validation_empty_edge_type(self):
        from scrutator.health import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/edges/by-path",
            json=[
                {
                    "source_path": "wiki/AI/ML.md",
                    "target_path": "wiki/AI/DL.md",
                    "edge_type": "  ",
                },
            ],
        )
        assert resp.status_code == 422
