"""Tests for SRCH-0021: hierarchical navigation (outline + section-context)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scrutator.db.models import ChunkLookupResult, NamespaceInfo


def _mock_pool():
    """Mock asyncpg pool with context-manager support (matches test_chunk_lookup.py pattern)."""
    pool = MagicMock()
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


def _row(chunk_id: str, chunk_index: int, source_path: str, section: dict | None) -> ChunkLookupResult:
    return ChunkLookupResult(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        source_path=source_path,
        source_type="markdown",
        content_preview="preview",
        metadata={"section": section} if section is not None else {},
    )


def _section(heading_path: list[str], anchor_path: list[str], doc_id: str = "docid1234567890a") -> dict:
    return {
        "doc_id": doc_id,
        "heading_path": heading_path,
        "depth": len(heading_path),
        "anchor": anchor_path[-1],
        "anchor_path": anchor_path,
        "section_key": "/".join(anchor_path),
        "schema_version": 1,
    }


# ── build_outline ─────────────────────────────────────────────────────


class TestBuildOutline:
    @pytest.mark.asyncio
    async def test_outline_tree_shape(self):
        from scrutator.search.navigator import build_outline

        rows = [
            _row("c1", 0, "notes/doc.md", _section(["Doc"], ["doc"])),
            _row("c2", 1, "notes/doc.md", _section(["Doc", "Section"], ["doc", "section"])),
            _row("c3", 2, "notes/doc.md", _section(["Doc", "Section", "Sub"], ["doc", "section", "sub"])),
        ]

        with (
            patch("scrutator.search.navigator.get_namespaces", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.navigator.get_chunks_by_source_path", new_callable=AsyncMock) as mock_get,
        ):
            mock_ns.return_value = [NamespaceInfo(id=1, name="arcanada", chunk_count=3)]
            mock_get.return_value = rows

            result = await build_outline(namespace="arcanada", source_path="notes/doc.md")

        assert result.total_chunks == 3
        assert len(result.outline) == 1
        root = result.outline[0]
        assert root.title == "Doc"
        assert root.section_key == "doc"
        assert root.chunk_ids == ["c1"]
        assert len(root.children) == 1
        section_node = root.children[0]
        assert section_node.title == "Section"
        assert section_node.section_key == "doc/section"
        assert len(section_node.children) == 1
        sub_node = section_node.children[0]
        assert sub_node.title == "Sub"
        assert sub_node.section_key == "doc/section/sub"
        assert sub_node.chunk_ids == ["c3"]

    @pytest.mark.asyncio
    async def test_outline_fallback_for_unbackfilled_doc(self):
        """Un-backfilled chunks (no `section` key) degrade to a single flat root — HARD-GATE safety net."""
        from scrutator.search.navigator import build_outline

        rows = [
            _row("c1", 0, "notes/legacy.md", None),
            _row("c2", 1, "notes/legacy.md", None),
        ]

        with (
            patch("scrutator.search.navigator.get_namespaces", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.navigator.get_chunks_by_source_path", new_callable=AsyncMock) as mock_get,
        ):
            mock_ns.return_value = [NamespaceInfo(id=1, name="arcanada", chunk_count=2)]
            mock_get.return_value = rows

            result = await build_outline(namespace="arcanada", source_path="notes/legacy.md")

        assert len(result.outline) == 1
        assert result.outline[0].section_key == "root"
        assert result.outline[0].chunk_ids == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_outline_unknown_source_404(self):
        from fastapi import HTTPException

        from scrutator.search.navigator import build_outline

        with (
            patch("scrutator.search.navigator.get_namespaces", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.navigator.get_chunks_by_source_path", new_callable=AsyncMock) as mock_get,
        ):
            mock_ns.return_value = [NamespaceInfo(id=1, name="arcanada", chunk_count=0)]
            mock_get.return_value = []

            with pytest.raises(HTTPException) as exc_info:
                await build_outline(namespace="arcanada", source_path="notes/missing.md")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_outline_unknown_namespace_404(self):
        from fastapi import HTTPException

        from scrutator.search.navigator import build_outline

        with patch("scrutator.search.navigator.get_namespaces", new_callable=AsyncMock) as mock_ns:
            mock_ns.return_value = []

            with pytest.raises(HTTPException) as exc_info:
                await build_outline(namespace="does-not-exist", source_path="notes/doc.md")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_outline_max_nodes_422(self):
        """Fork 3: row-count checked BEFORE tree assembly, typed 422 (not silent truncation)."""
        from fastapi import HTTPException

        from scrutator.search.navigator import build_outline

        rows = [_row(f"c{i}", i, "notes/huge.md", _section([f"Sec{i}"], [f"sec{i}"])) for i in range(5)]

        with (
            patch("scrutator.search.navigator.get_namespaces", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.navigator.get_chunks_by_source_path", new_callable=AsyncMock) as mock_get,
        ):
            mock_ns.return_value = [NamespaceInfo(id=1, name="arcanada", chunk_count=5)]
            mock_get.return_value = rows

            with pytest.raises(HTTPException) as exc_info:
                await build_outline(namespace="arcanada", source_path="notes/huge.md", max_nodes=3)

        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_outline_max_nodes_hard_ceiling(self):
        """Fork 3: caller-supplied max_nodes is capped server-side at 10000."""
        from scrutator.search import navigator

        with (
            patch("scrutator.search.navigator.get_namespaces", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.navigator.get_chunks_by_source_path", new_callable=AsyncMock) as mock_get,
        ):
            mock_ns.return_value = [NamespaceInfo(id=1, name="arcanada", chunk_count=1)]
            mock_get.return_value = [_row("c1", 0, "notes/doc.md", _section(["Doc"], ["doc"]))]

            await navigator.build_outline(namespace="arcanada", source_path="notes/doc.md", max_nodes=999_999_999)
            # No assertion error raised — the (small) fixture never exceeds any cap;
            # this test only exercises that an absurd max_nodes doesn't crash the clamp.


# ── build_section_context ────────────────────────────────────────────


class TestBuildSectionContext:
    @pytest.mark.asyncio
    async def test_section_context_ancestors_siblings(self):
        from scrutator.search.navigator import build_section_context

        doc_rows = [
            _row("c1", 0, "notes/doc.md", _section(["Doc"], ["doc"])),
            _row("c2", 1, "notes/doc.md", _section(["Doc", "Section"], ["doc", "section"])),
            _row("c3", 2, "notes/doc.md", _section(["Doc", "Section", "Sub"], ["doc", "section", "sub"])),
            _row("c4", 3, "notes/doc.md", _section(["Doc", "Section", "Sub2"], ["doc", "section", "sub2"])),
        ]

        chunk_id = "12345678-1234-5678-1234-567812345678"
        doc_rows[2] = _row(chunk_id, 2, "notes/doc.md", _section(["Doc", "Section", "Sub"], ["doc", "section", "sub"]))

        with patch("scrutator.search.navigator.get_section_siblings_children", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"doc_rows": doc_rows}

            result = await build_section_context(chunk_id)

        assert result.chunk_id == chunk_id
        assert result.section_key == "doc/section/sub"
        assert [a.section_key for a in result.ancestors] == ["doc", "doc/section"]
        assert result.self_.title == "Sub"
        assert result.self_.chunk_ids == [chunk_id]
        sibling_keys = {s.section_key for s in result.siblings}
        assert sibling_keys == {"doc/section/sub2"}
        assert result.children == []

    @pytest.mark.asyncio
    async def test_invalid_chunk_id_422(self):
        from fastapi import HTTPException

        from scrutator.search.navigator import build_section_context

        with pytest.raises(HTTPException) as exc_info:
            await build_section_context("not-a-uuid")

        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_unknown_chunk_id_404(self):
        from fastapi import HTTPException

        from scrutator.search.navigator import build_section_context

        with patch("scrutator.search.navigator.get_section_siblings_children", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await build_section_context("12345678-1234-5678-1234-567812345678")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_section_context_unbackfilled_fallback(self):
        """Un-backfilled chunk (no `section` key) still resolves — flat root fallback."""
        from scrutator.search.navigator import build_section_context

        chunk_id = "12345678-1234-5678-1234-567812345678"
        doc_rows = [_row(chunk_id, 0, "notes/legacy.md", None)]

        with patch("scrutator.search.navigator.get_section_siblings_children", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"doc_rows": doc_rows}

            result = await build_section_context(chunk_id)

        assert result.section_key == "root"
        assert result.ancestors == []
        assert result.self_.chunk_ids == [chunk_id]


# ── repository: get_section_siblings_children (Step 6, V-AC-4) ──────
# Mocked-pool pattern (matches tests/test_chunk_lookup.py) — no real DB needed.


class TestGetSectionSiblingsChildrenRepository:
    @pytest.mark.asyncio
    async def test_chunk_not_found_returns_none(self):
        pool, conn = _mock_pool()
        conn.fetchrow.return_value = None

        with patch("scrutator.db.repository.get_pool", return_value=pool):
            from scrutator.db.repository import get_section_siblings_children

            result = await get_section_siblings_children("12345678-1234-5678-1234-567812345678")
            assert result is None

    @pytest.mark.asyncio
    async def test_backfilled_chunk_filters_by_doc_id(self):
        pool, conn = _mock_pool()
        conn.fetchrow.return_value = {
            "chunk_id": "c2",
            "chunk_index": 1,
            "source_type": "markdown",
            "source_path": "notes/doc.md",
            "namespace_id": 1,
            "content_preview": "preview",
            "metadata": '{"section": {"doc_id": "abc123", "section_key": "doc/section"}}',
        }
        conn.fetch.return_value = [
            {
                "chunk_id": "c1",
                "chunk_index": 0,
                "source_type": "markdown",
                "source_path": "notes/doc.md",
                "content_preview": "p1",
                "metadata": '{"section": {"doc_id": "abc123", "section_key": "doc"}}',
            },
            {
                "chunk_id": "c2",
                "chunk_index": 1,
                "source_type": "markdown",
                "source_path": "notes/doc.md",
                "content_preview": "p2",
                "metadata": '{"section": {"doc_id": "abc123", "section_key": "doc/section"}}',
            },
        ]

        with patch("scrutator.db.repository.get_pool", return_value=pool):
            from scrutator.db.repository import get_section_siblings_children

            result = await get_section_siblings_children("c2")

        assert result is not None
        assert [r.chunk_id for r in result["doc_rows"]] == ["c1", "c2"]
        # doc_id-filtered query bound namespace_id + doc_id, parameterized (no f-string SQL)
        fetch_call = conn.fetch.call_args
        assert fetch_call[0][1] == 1
        assert fetch_call[0][2] == "abc123"

    @pytest.mark.asyncio
    async def test_unbackfilled_chunk_falls_back_to_source_path(self):
        pool, conn = _mock_pool()
        conn.fetchrow.return_value = {
            "chunk_id": "c1",
            "chunk_index": 0,
            "source_type": "markdown",
            "source_path": "notes/legacy.md",
            "namespace_id": 1,
            "content_preview": "preview",
            "metadata": "{}",
        }
        conn.fetch.return_value = [
            {
                "chunk_id": "c1",
                "chunk_index": 0,
                "source_type": "markdown",
                "source_path": "notes/legacy.md",
                "content_preview": "p1",
                "metadata": "{}",
            },
        ]

        with patch("scrutator.db.repository.get_pool", return_value=pool):
            from scrutator.db.repository import get_section_siblings_children

            result = await get_section_siblings_children("c1")

        assert result is not None
        fetch_call = conn.fetch.call_args
        assert fetch_call[0][1] == 1
        assert fetch_call[0][2] == "notes/legacy.md"


# ── API endpoints (Step 7, V-AC-3/4/7) ───────────────────────────────


class TestNavigateOutlineEndpoint:
    def test_happy_path(self):
        from fastapi.testclient import TestClient

        from scrutator.db.models import OutlineNode, OutlineResponse
        from scrutator.health import app

        mock_response = OutlineResponse(
            source_path="notes/doc.md",
            namespace="arcanada",
            doc_id="abc123",
            total_chunks=1,
            outline=[OutlineNode(title="Doc", anchor="doc", depth=1, section_key="doc", chunk_ids=["c1"])],
        )

        with patch("scrutator.health.build_outline", new_callable=AsyncMock, return_value=mock_response):
            client = TestClient(app)
            resp = client.get("/v1/navigate/outline", params={"namespace": "arcanada", "source_path": "notes/doc.md"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "abc123"
        assert data["outline"][0]["title"] == "Doc"

    def test_unknown_source_404(self):
        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        from scrutator.health import app

        with patch(
            "scrutator.health.build_outline",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404, detail="unknown source_path: x.md"),
        ):
            client = TestClient(app)
            resp = client.get("/v1/navigate/outline", params={"namespace": "arcanada", "source_path": "x.md"})

        assert resp.status_code == 404

    def test_max_nodes_422(self):
        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        from scrutator.health import app

        with patch(
            "scrutator.health.build_outline",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=422, detail="document exceeds max_nodes"),
        ):
            client = TestClient(app)
            resp = client.get(
                "/v1/navigate/outline",
                params={"namespace": "arcanada", "source_path": "huge.md", "max_nodes": 1},
            )

        assert resp.status_code == 422


class TestNavigateSectionEndpoint:
    def test_happy_path(self):
        from fastapi.testclient import TestClient

        from scrutator.db.models import SectionContext, SectionSelf
        from scrutator.health import app

        chunk_id = "12345678-1234-5678-1234-567812345678"
        mock_response = SectionContext(
            chunk_id=chunk_id,
            doc_id="abc123",
            section_key="doc/section",
            self_=SectionSelf(title="Section", section_key="doc/section", depth=2, chunk_ids=[chunk_id]),
        )

        with patch("scrutator.health.build_section_context", new_callable=AsyncMock, return_value=mock_response):
            client = TestClient(app)
            resp = client.get("/v1/navigate/section", params={"chunk_id": chunk_id})

        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_id"] == chunk_id
        assert data["self"]["title"] == "Section"

    def test_invalid_chunk_id_422(self):
        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        from scrutator.health import app

        with patch(
            "scrutator.health.build_section_context",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=422, detail="chunk_id is not a valid UUID"),
        ):
            client = TestClient(app)
            resp = client.get("/v1/navigate/section", params={"chunk_id": "not-a-uuid"})

        assert resp.status_code == 422

    def test_unknown_chunk_404(self):
        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        from scrutator.health import app

        chunk_id = "12345678-1234-5678-1234-567812345678"
        with patch(
            "scrutator.health.build_section_context",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404, detail="chunk not found"),
        ):
            client = TestClient(app)
            resp = client.get("/v1/navigate/section", params={"chunk_id": chunk_id})

        assert resp.status_code == 404
