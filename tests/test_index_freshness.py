"""Tests for scrutator.tools.index_freshness (SRCH-0036).

Covers: stale detection, missing detection, empty/clean case, report shape,
corpus scanning, manifest loading, the dry-run re-index plan, and the CLI
entrypoint end-to-end with a mocked DB read. No test touches a live database
or a live Scrutator index — the DB read path is mocked out.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from scrutator.tools.index_freshness import (
    IndexedSource,
    build_reindex_plan,
    detect_freshness,
    fetch_indexed_sources,
    load_manifest_paths,
    main,
    probe_health,
    run_detection,
    scan_corpus_paths,
)

GENERATED_AT = "2026-07-10T00:00:00+00:00"


class TestDetectFreshness:
    def test_stale_detection(self):
        indexed = [IndexedSource("a.md", 3), IndexedSource("gone.md", 5)]
        corpus = {"a.md"}
        report = detect_freshness(indexed, corpus, "arcanada", GENERATED_AT)
        assert report.stale == [{"source_path": "gone.md", "chunk_count": 5}]
        assert report.missing == []
        assert report.stale_count == 1
        assert not report.is_clean

    def test_missing_detection(self):
        indexed = [IndexedSource("a.md", 3)]
        corpus = {"a.md", "new.md"}
        report = detect_freshness(indexed, corpus, "arcanada", GENERATED_AT)
        assert report.missing == [{"source_path": "new.md"}]
        assert report.stale == []
        assert report.missing_count == 1
        assert not report.is_clean

    def test_stale_and_missing_together(self):
        indexed = [IndexedSource("a.md", 1), IndexedSource("gone.md", 2)]
        corpus = {"a.md", "new.md"}
        report = detect_freshness(indexed, corpus, "arcanada", GENERATED_AT)
        assert [e["source_path"] for e in report.stale] == ["gone.md"]
        assert [e["source_path"] for e in report.missing] == ["new.md"]

    def test_empty_clean_case(self):
        indexed = [IndexedSource("a.md", 1), IndexedSource("b.md", 2)]
        corpus = {"a.md", "b.md"}
        report = detect_freshness(indexed, corpus, "arcanada", GENERATED_AT)
        assert report.is_clean
        assert report.stale == []
        assert report.missing == []
        assert report.indexed_count == 2
        assert report.corpus_count == 2

    def test_both_empty(self):
        report = detect_freshness([], set(), "arcanada", GENERATED_AT)
        assert report.is_clean
        assert report.indexed_count == 0
        assert report.corpus_count == 0

    def test_results_sorted_deterministically(self):
        indexed = [IndexedSource("z-gone.md", 1), IndexedSource("a-gone.md", 1)]
        report = detect_freshness(indexed, set(), "arcanada", GENERATED_AT)
        assert [e["source_path"] for e in report.stale] == ["a-gone.md", "z-gone.md"]


class TestReportShape:
    def test_to_dict_shape(self):
        indexed = [IndexedSource("gone.md", 5)]
        report = detect_freshness(indexed, {"new.md"}, "arcanada", GENERATED_AT)
        d = report.to_dict()
        assert d == {
            "namespace": "arcanada",
            "generated_at": GENERATED_AT,
            "indexed_count": 1,
            "corpus_count": 1,
            "stale_count": 1,
            "missing_count": 1,
            "clean": False,
            "stale": [{"source_path": "gone.md", "chunk_count": 5}],
            "missing": [{"source_path": "new.md"}],
        }

    def test_to_dict_is_json_serializable(self):
        report = detect_freshness([IndexedSource("gone.md", 5)], set(), "arcanada", GENERATED_AT)
        json.dumps(report.to_dict())  # must not raise

    def test_human_summary_clean(self):
        report = detect_freshness([IndexedSource("a.md", 1)], {"a.md"}, "arcanada", GENERATED_AT)
        summary = report.human_summary()
        assert "clean" in summary
        assert "STALE" not in summary
        assert "MISSING" not in summary

    def test_human_summary_reports_counts(self):
        report = detect_freshness([IndexedSource("gone.md", 5)], {"new.md"}, "arcanada", GENERATED_AT)
        summary = report.human_summary()
        assert "STALE (1)" in summary
        assert "MISSING (1)" in summary
        assert "gone.md" in summary
        assert "new.md" in summary

    def test_human_summary_truncates_long_lists(self):
        indexed = [IndexedSource(f"gone-{i}.md", 1) for i in range(25)]
        report = detect_freshness(indexed, set(), "arcanada", GENERATED_AT)
        summary = report.human_summary()
        assert "... and 5 more" in summary


class TestScanCorpusPaths:
    def test_scans_matching_extensions(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("x")
        (tmp_path / "b.pdf").write_text("x")
        (tmp_path / "ignore.txt").write_text("x")
        sub = tmp_path / "nested"
        sub.mkdir()
        (sub / "c.md").write_text("x")

        paths = scan_corpus_paths(tmp_path)
        assert paths == {"a.md", "b.pdf", "nested/c.md"}

    def test_custom_extensions(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.md").write_text("x")
        paths = scan_corpus_paths(tmp_path, extensions=(".py",))
        assert paths == {"a.py"}

    def test_missing_root_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            scan_corpus_paths(tmp_path / "does-not-exist")

    def test_empty_dir(self, tmp_path: Path):
        assert scan_corpus_paths(tmp_path) == set()


class TestLoadManifestPaths:
    def test_bare_array(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(["a.md", "b.md"]))
        assert load_manifest_paths(manifest) == {"a.md", "b.md"}

    def test_paths_object(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"paths": ["a.md", "b.md"], "generated_at": "x"}))
        assert load_manifest_paths(manifest) == {"a.md", "b.md"}

    def test_object_without_paths_key_is_empty(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"generated_at": "x"}))
        assert load_manifest_paths(manifest) == set()

    def test_invalid_type_raises(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps("just a string"))
        with pytest.raises(ValueError):
            load_manifest_paths(manifest)


class TestBuildReindexPlan:
    def test_plan_is_never_executed(self):
        report = detect_freshness([IndexedSource("gone.md", 5)], {"new.md"}, "arcanada", GENERATED_AT)
        plan = build_reindex_plan(report)
        assert plan["executed"] is False

    def test_plan_actions(self):
        report = detect_freshness([IndexedSource("gone.md", 5)], {"new.md"}, "arcanada", GENERATED_AT)
        plan = build_reindex_plan(report)
        assert plan["action_count"] == 2
        assert {"action": "delete", "source_path": "gone.md", "reason": "stale-indexed-but-gone"} in plan["actions"]
        assert {"action": "reingest", "source_path": "new.md", "reason": "on-disk-but-unindexed"} in plan["actions"]

    def test_plan_empty_when_clean(self):
        report = detect_freshness([IndexedSource("a.md", 1)], {"a.md"}, "arcanada", GENERATED_AT)
        plan = build_reindex_plan(report)
        assert plan["action_count"] == 0
        assert plan["actions"] == []


class TestFetchIndexedSources:
    async def test_queries_and_closes_connection(self):
        fake_conn = AsyncMock()
        fake_conn.fetch.return_value = [
            {"source_path": "a.md", "chunk_count": 3},
            {"source_path": "b.md", "chunk_count": 1},
        ]
        with patch("asyncpg.connect", new=AsyncMock(return_value=fake_conn)) as connect_mock:
            result = await fetch_indexed_sources("postgresql://x", "arcanada")

        connect_mock.assert_awaited_once_with(dsn="postgresql://x")
        fake_conn.fetch.assert_awaited_once()
        query, namespace_arg = fake_conn.fetch.await_args.args
        assert "SELECT" in query
        assert "GROUP BY c.source_path" in query
        assert namespace_arg == "arcanada"
        fake_conn.close.assert_awaited_once()
        assert result == [IndexedSource("a.md", 3), IndexedSource("b.md", 1)]

    async def test_closes_connection_even_on_error(self):
        fake_conn = AsyncMock()
        fake_conn.fetch.side_effect = RuntimeError("boom")
        with patch("asyncpg.connect", new=AsyncMock(return_value=fake_conn)), pytest.raises(RuntimeError):
            await fetch_indexed_sources("postgresql://x", "arcanada")
        fake_conn.close.assert_awaited_once()


class TestProbeHealth:
    async def test_ok(self):
        fake_response = AsyncMock()
        fake_response.status_code = 200
        with patch("httpx.AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.get = AsyncMock(return_value=fake_response)
            reachable, reason = await probe_health("http://example.invalid")
        assert reachable is True
        assert reason == "ok"

    async def test_non_200(self):
        fake_response = AsyncMock()
        fake_response.status_code = 503
        with patch("httpx.AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.get = AsyncMock(return_value=fake_response)
            reachable, reason = await probe_health("http://example.invalid")
        assert reachable is False
        assert "503" in reason

    async def test_unreachable(self):
        import httpx

        with patch("httpx.AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            reachable, reason = await probe_health("http://example.invalid")
        assert reachable is False
        assert "unreachable" in reason


class TestRunDetection:
    async def test_requires_corpus_source(self):
        with pytest.raises(ValueError, match="corpus_root or manifest_path"):
            await run_detection(
                database_url="postgresql://x",
                namespace="arcanada",
                corpus_root=None,
                manifest_path=None,
            )

    async def test_uses_manifest_when_given(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(["a.md"]))
        with patch(
            "scrutator.tools.index_freshness.fetch_indexed_sources",
            new=AsyncMock(return_value=[IndexedSource("a.md", 1), IndexedSource("gone.md", 2)]),
        ):
            report = await run_detection(
                database_url="postgresql://x",
                namespace="arcanada",
                corpus_root=None,
                manifest_path=manifest,
            )
        assert report.stale == [{"source_path": "gone.md", "chunk_count": 2}]
        assert report.missing == []

    async def test_uses_corpus_root_when_given(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("x")
        with patch(
            "scrutator.tools.index_freshness.fetch_indexed_sources",
            new=AsyncMock(return_value=[]),
        ):
            report = await run_detection(
                database_url="postgresql://x",
                namespace="arcanada",
                corpus_root=tmp_path,
                manifest_path=None,
            )
        assert report.missing == [{"source_path": "a.md"}]


class TestCliMain:
    def test_requires_corpus_source(self, capsys):
        exit_code = main(["--namespace", "arcanada"])
        assert exit_code == 2
        assert "corpus-root or --manifest" in capsys.readouterr().err

    def test_end_to_end_report_and_output(self, tmp_path: Path, capsys):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(["a.md"]))
        output_path = tmp_path / "report.json"

        with patch(
            "scrutator.tools.index_freshness.fetch_indexed_sources",
            new=AsyncMock(return_value=[IndexedSource("a.md", 1), IndexedSource("gone.md", 5)]),
        ):
            exit_code = main(
                [
                    "--namespace",
                    "arcanada",
                    "--manifest",
                    str(manifest),
                    "--database-url",
                    "postgresql://unused",
                    "--output",
                    str(output_path),
                ]
            )

        assert exit_code == 0  # --fail-on-stale not passed
        written = json.loads(output_path.read_text())
        assert written["report"]["stale_count"] == 1
        assert "plan" not in written
        assert "gone.md" in capsys.readouterr().out

    def test_plan_flag_adds_plan_without_executing(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(["a.md"]))
        output_path = tmp_path / "report.json"

        with patch(
            "scrutator.tools.index_freshness.fetch_indexed_sources",
            new=AsyncMock(return_value=[IndexedSource("gone.md", 5)]),
        ):
            main(
                [
                    "--namespace",
                    "arcanada",
                    "--manifest",
                    str(manifest),
                    "--database-url",
                    "postgresql://unused",
                    "--output",
                    str(output_path),
                    "--plan",
                ]
            )

        written = json.loads(output_path.read_text())
        assert written["plan"]["executed"] is False
        assert written["plan"]["action_count"] == 2  # delete gone.md + reingest a.md

    def test_fail_on_stale_exit_code(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([]))
        with patch(
            "scrutator.tools.index_freshness.fetch_indexed_sources",
            new=AsyncMock(return_value=[IndexedSource("gone.md", 1)]),
        ):
            exit_code = main(
                [
                    "--manifest",
                    str(manifest),
                    "--database-url",
                    "postgresql://unused",
                    "--fail-on-stale",
                ]
            )
        assert exit_code == 1

    def test_fail_on_stale_clean_exits_zero(self, tmp_path: Path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(["a.md"]))
        with patch(
            "scrutator.tools.index_freshness.fetch_indexed_sources",
            new=AsyncMock(return_value=[IndexedSource("a.md", 1)]),
        ):
            exit_code = main(
                [
                    "--manifest",
                    str(manifest),
                    "--database-url",
                    "postgresql://unused",
                    "--fail-on-stale",
                ]
            )
        assert exit_code == 0
