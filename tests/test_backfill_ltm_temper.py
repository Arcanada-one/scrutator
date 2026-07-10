"""Tests for tools/backfill_ltm_temper.py (LTM-0014).

HARD-GATE: every test here uses a fully mocked asyncpg pool and a mocked
LtmLlmClient / IngestPipeline — no real database connection is ever opened,
no LLM call is ever made, and no namespace (dry-run or live) is ever
backfilled by this suite.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import backfill_ltm_temper


def _mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


def _orphan_row(chunk_id: str, source_path: str, content: str = "some chunk text") -> dict:
    return {"chunk_id": chunk_id, "source_path": source_path, "content": content}


class TestRunBackfillDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_default_makes_no_writes_no_llm_calls(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [
            _orphan_row("c1", "notes/a.md"),
            _orphan_row("c2", "notes/b.md"),
        ]

        with (
            patch("tools.backfill_ltm_temper.get_pool", new_callable=AsyncMock, return_value=pool),
            patch("tools.backfill_ltm_temper._create_llm_client") as mock_llm_factory,
        ):
            result = await backfill_ltm_temper.run_backfill(namespace="arcanada")

        assert result["dry_run"] is True
        assert result["candidates"] == 2
        assert result["processed"] == 0
        assert result["failed"] == 0
        mock_llm_factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_explicit_true_makes_no_writes(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [_orphan_row("c1", "notes/a.md")]

        with patch("tools.backfill_ltm_temper.get_pool", new_callable=AsyncMock, return_value=pool):
            result = await backfill_ltm_temper.run_backfill(namespace="arcanada", dry_run=True)

        assert result["processed"] == 0
        conn.fetchrow.assert_not_called()

    @pytest.mark.asyncio
    async def test_candidate_query_scoped_to_orphan_chunks(self):
        """The fetch query must exclude chunks that already have a linked entity."""
        pool, conn = _mock_pool()
        conn.fetch.return_value = []

        with patch("tools.backfill_ltm_temper.get_pool", new_callable=AsyncMock, return_value=pool):
            await backfill_ltm_temper.run_backfill(namespace="arcanada")

        query = conn.fetch.call_args[0][0]
        assert "LEFT JOIN entities e ON e.source_chunk_id = c.id" in query
        assert "e.id IS NULL" in query


class TestRunBackfillLive:
    @pytest.mark.asyncio
    async def test_live_run_processes_each_candidate_chunk(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [
            _orphan_row("c1", "notes/a.md"),
            _orphan_row("c2", "notes/b.md"),
        ]
        conn.fetchrow.return_value = {"id": 7}

        mock_pipeline = AsyncMock()
        mock_pipeline.process_chunk = AsyncMock(return_value=([], []))

        with (
            patch("tools.backfill_ltm_temper.get_pool", new_callable=AsyncMock, return_value=pool),
            patch("tools.backfill_ltm_temper._create_llm_client", return_value=MagicMock()),
            patch("tools.backfill_ltm_temper.IngestPipeline", return_value=mock_pipeline) as mock_pipeline_cls,
        ):
            result = await backfill_ltm_temper.run_backfill(namespace="arcanada", dry_run=False)

        assert result["processed"] == 2
        assert result["failed"] == 0
        assert mock_pipeline.process_chunk.call_count == 2
        mock_pipeline.process_chunk.assert_any_call("c1", "some chunk text")
        mock_pipeline.process_chunk.assert_any_call("c2", "some chunk text")
        # namespace_id resolved read-only, not auto-provisioned
        _, kwargs = mock_pipeline_cls.call_args
        assert kwargs["namespace_id"] == 7

    @pytest.mark.asyncio
    async def test_live_run_continues_past_per_chunk_failure(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [
            _orphan_row("c1", "notes/a.md"),
            _orphan_row("c2", "notes/b.md"),
        ]
        conn.fetchrow.return_value = {"id": 7}

        mock_pipeline = AsyncMock()
        mock_pipeline.process_chunk = AsyncMock(side_effect=[RuntimeError("LLM 500"), ([], [])])

        with (
            patch("tools.backfill_ltm_temper.get_pool", new_callable=AsyncMock, return_value=pool),
            patch("tools.backfill_ltm_temper._create_llm_client", return_value=MagicMock()),
            patch("tools.backfill_ltm_temper.IngestPipeline", return_value=mock_pipeline),
        ):
            result = await backfill_ltm_temper.run_backfill(namespace="arcanada", dry_run=False)

        assert result["processed"] == 1
        assert result["failed"] == 1
        assert mock_pipeline.process_chunk.call_count == 2

    @pytest.mark.asyncio
    async def test_live_run_refuses_unknown_namespace_no_auto_provision(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [_orphan_row("c1", "notes/a.md")]
        conn.fetchrow.return_value = None  # namespace not found

        with (
            patch("tools.backfill_ltm_temper.get_pool", new_callable=AsyncMock, return_value=pool),
            patch("tools.backfill_ltm_temper._create_llm_client") as mock_llm_factory,
        ):
            result = await backfill_ltm_temper.run_backfill(namespace="does-not-exist", dry_run=False)

        assert result["processed"] == 0
        assert "error" in result
        mock_llm_factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_run_no_candidates_is_noop(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []

        with (
            patch("tools.backfill_ltm_temper.get_pool", new_callable=AsyncMock, return_value=pool),
            patch("tools.backfill_ltm_temper._create_llm_client") as mock_llm_factory,
        ):
            result = await backfill_ltm_temper.run_backfill(namespace="arcanada", dry_run=False)

        assert result["candidates"] == 0
        assert result["processed"] == 0
        mock_llm_factory.assert_not_called()


class TestLimitParam:
    @pytest.mark.asyncio
    async def test_limit_appends_sql_limit_clause(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []

        with patch("tools.backfill_ltm_temper.get_pool", new_callable=AsyncMock, return_value=pool):
            await backfill_ltm_temper.run_backfill(namespace="arcanada", limit=50)

        query, ns_arg, limit_arg = conn.fetch.call_args[0]
        assert "LIMIT $2" in query
        assert ns_arg == "arcanada"
        assert limit_arg == 50

    @pytest.mark.asyncio
    async def test_no_limit_omits_sql_limit_clause(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []

        with patch("tools.backfill_ltm_temper.get_pool", new_callable=AsyncMock, return_value=pool):
            await backfill_ltm_temper.run_backfill(namespace="arcanada")

        query, ns_arg = conn.fetch.call_args[0]
        assert "LIMIT" not in query
        assert ns_arg == "arcanada"


class TestCliArgs:
    def test_dry_run_is_default(self):
        args = backfill_ltm_temper._parse_args(["--namespace", "arcanada"])
        assert args.live is False

    def test_live_flag_required_to_opt_in(self):
        args = backfill_ltm_temper._parse_args(["--namespace", "arcanada", "--live"])
        assert args.live is True
