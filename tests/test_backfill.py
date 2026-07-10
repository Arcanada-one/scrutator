"""Tests for tools/backfill_sections.py (SRCH-0021, D-REQ-02, V-AC-2).

HARD-GATE: every test here uses a fully mocked asyncpg pool (matching the
tests/test_chunk_lookup.py pattern) — no real database connection is ever
opened, and no namespace (dry-run or live) is ever backfilled by this suite.
"""

from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import backfill_sections


def _mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


def _stale_row(chunk_id: str, source_path: str, chunk_index: int, heading_hierarchy: list[str]) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_path": source_path,
        "chunk_index": chunk_index,
        "metadata": json.dumps({"heading_hierarchy": heading_hierarchy}),
    }


class TestComputeSectionForRow:
    def test_derives_from_heading_hierarchy(self):
        section = backfill_sections.compute_section_for_row(
            "arcanada", "notes/doc.md", {"heading_hierarchy": ["# Doc", "## Sub"]}
        )
        assert section["heading_path"] == ["Doc", "Sub"]
        assert section["section_key"] == "doc/sub"
        assert section["doc_id"]

    def test_fallback_root_when_no_hierarchy(self):
        section = backfill_sections.compute_section_for_row("arcanada", "notes/flat.md", {})
        assert section["section_key"] == "root"
        assert section["depth"] == 1


class TestRunBackfillDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_default_makes_no_writes(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [
            _stale_row("c1", "notes/a.md", 0, ["# A"]),
            _stale_row("c2", "notes/b.md", 0, ["# B"]),
        ]

        with patch("tools.backfill_sections.get_pool", new_callable=AsyncMock, return_value=pool):
            result = await backfill_sections.run_backfill(namespace="arcanada")

        assert result["dry_run"] is True
        assert result["candidates"] == 2
        assert result["updated"] == 0
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_explicit_true_makes_no_writes(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [_stale_row("c1", "notes/a.md", 0, ["# A"])]

        with patch("tools.backfill_sections.get_pool", new_callable=AsyncMock, return_value=pool):
            result = await backfill_sections.run_backfill(namespace="arcanada", dry_run=True)

        assert result["updated"] == 0
        conn.execute.assert_not_called()


class TestRunBackfillLive:
    @pytest.mark.asyncio
    async def test_live_run_updates_matching_count(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [
            _stale_row("c1", "notes/a.md", 0, ["# A"]),
            _stale_row("c2", "notes/b.md", 0, []),  # empty hierarchy → fallback root
        ]

        with patch("tools.backfill_sections.get_pool", new_callable=AsyncMock, return_value=pool):
            result = await backfill_sections.run_backfill(namespace="arcanada", dry_run=False)

        assert result["updated"] == 2
        assert conn.execute.call_count == 2
        # parameterized UPDATE — chunk_id bound as the 2nd positional param, no f-string SQL
        first_call = conn.execute.call_args_list[0]
        assert first_call[0][2] == "c1"

    @pytest.mark.asyncio
    async def test_backfill_idempotent_second_run_zero_updates(self):
        """Fork 7: once schema_version matches, the WHERE clause excludes the row —
        simulated here by the second call returning 0 stale rows."""
        pool, conn = _mock_pool()

        with patch("tools.backfill_sections.get_pool", new_callable=AsyncMock, return_value=pool):
            conn.fetch.return_value = [_stale_row("c1", "notes/a.md", 0, ["# A"])]
            first = await backfill_sections.run_backfill(namespace="arcanada", dry_run=False)
            assert first["updated"] == 1

            conn.fetch.return_value = []  # now excluded by the schema_version WHERE clause
            second = await backfill_sections.run_backfill(namespace="arcanada", dry_run=False)

        assert second["candidates"] == 0
        assert second["updated"] == 0

    @pytest.mark.asyncio
    async def test_live_run_zero_embedding_calls(self):
        """V-AC-2: backfill must never import or call the embedding client."""
        pool, conn = _mock_pool()
        conn.fetch.return_value = [_stale_row("c1", "notes/a.md", 0, ["# A"])]

        with (
            patch("tools.backfill_sections.get_pool", new_callable=AsyncMock, return_value=pool),
            patch("scrutator.search.embedder.embed_texts", new_callable=AsyncMock) as mock_embed,
        ):
            await backfill_sections.run_backfill(namespace="arcanada", dry_run=False)

        mock_embed.assert_not_called()

    def test_module_has_no_embedding_client_import(self):
        """V-AC-2 (structural): zero embedding-client imports anywhere in the module source."""
        source = inspect.getsource(backfill_sections)
        assert "embedder" not in source
        assert "embed_texts" not in source
        assert "embed_single" not in source


class TestParseArgs:
    def test_namespace_required(self):
        with pytest.raises(SystemExit):
            backfill_sections._parse_args([])

    def test_default_is_dry_run(self):
        args = backfill_sections._parse_args(["--namespace", "arcanada"])
        assert args.live is False

    def test_live_flag(self):
        args = backfill_sections._parse_args(["--namespace", "arcanada", "--live"])
        assert args.live is True
