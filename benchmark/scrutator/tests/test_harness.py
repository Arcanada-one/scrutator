"""Unit tests for benchmark/scrutator/harness.py (SRCH-0015).

Covers (Test Plan step 1 + step 2):
- golden-row loading + validation
- corpus_pinned_at parsing
- liveness pre-flight fixture (mix of live + deliberately-missing paths) — V-AC-03
- recall@{1,5,10} / MRR / nDCG@5 math against hand-computed fixtures
- infra-fail vs. threshold-fail exit-code branch (D-REQ-05) via a mocked ModelClient — V-AC-02a
- consumer-side verdict smoke: decide_exit_code() against synthetic summary JSON — V-AC-02a/02b
- CLI --dry-run-infra-fail / --dry-run-threshold-fail smoke surface
"""

import json

import harness
import pytest

# --------------------------------------------------------------------------- golden-row loading


class TestLoadGoldenRows:
    def test_loads_valid_rows(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        path.write_text(
            '{"id": "F1", "class": "factual", "query": "q1", "gold_source_paths": ["a.md"], '
            '"corpus_pinned_at": "2026-07-10"}\n'
            '{"id": "F2", "class": "factual", "query": "q2", "gold_source_paths": ["b.md"]}\n'
        )
        rows = harness.load_golden_rows(path)
        assert len(rows) == 2
        assert rows[0].id == "F1"
        assert rows[0].cls == "factual"
        assert rows[0].corpus_pinned_at == "2026-07-10"
        # corpus_pinned_at is optional at the schema level (older/candidate rows) — parses to None
        assert rows[1].corpus_pinned_at is None

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        path.write_text('{"id": "F1", "class": "factual", "query": "q", "gold_source_paths": ["a.md"]}\n\n\n')
        rows = harness.load_golden_rows(path)
        assert len(rows) == 1

    def test_missing_required_field_raises(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        path.write_text('{"id": "F1", "class": "factual", "query": "q"}\n')  # missing gold_source_paths
        with pytest.raises(ValueError, match="gold_source_paths"):
            harness.load_golden_rows(path)

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        path.write_text("not json\n")
        with pytest.raises(ValueError, match="invalid JSON"):
            harness.load_golden_rows(path)


# --------------------------------------------------------------------------- liveness pre-flight


class TestLiveness:
    def _row(self, paths):
        return harness.GoldenRow(id="X1", cls="factual", query="q", gold_source_paths=paths)

    def test_row_live_when_path_exists(self, tmp_path):
        (tmp_path / "live.md").write_text("content")
        row = self._row(["live.md"])
        assert harness.is_row_live(row, tmp_path) is True

    def test_row_stale_when_path_missing(self, tmp_path):
        row = self._row(["dead.md"])
        assert harness.is_row_live(row, tmp_path) is False

    def test_multi_hop_row_live_if_any_path_survives(self, tmp_path):
        (tmp_path / "live.md").write_text("content")
        row = self._row(["dead.md", "live.md"])
        assert harness.is_row_live(row, tmp_path) is True

    def test_partition_stale_skipped_count(self, tmp_path):
        """V-AC-03 fixture: one live + one deliberately-deleted path → stale_skipped == 1,
        and the live count equals total - stale_skipped."""
        (tmp_path / "live.md").write_text("content")
        rows = [self._row(["live.md"]), self._row(["dead.md"])]
        live, stale = harness.partition_by_liveness(rows, tmp_path)
        assert len(stale) == 1
        assert len(live) == len(rows) - len(stale)


# --------------------------------------------------------------------------- scoring math


class TestScoringMath:
    def test_tail_hit_exact_and_suffix_match(self):
        assert harness.tail_hit(["docs/a.md"], ["docs/a.md"]) is True
        assert harness.tail_hit(["a.md"], ["docs/a.md"]) is True
        assert harness.tail_hit(["docs/a.md"], ["a.md"]) is True
        assert harness.tail_hit(["docs/a.md"], ["docs/b.md"]) is False

    def test_reciprocal_rank_first_hit(self):
        assert harness.reciprocal_rank(["a.md"], ["x.md", "a.md", "y.md"]) == pytest.approx(0.5)

    def test_reciprocal_rank_no_hit_is_zero(self):
        assert harness.reciprocal_rank(["a.md"], ["x.md", "y.md"]) == 0.0

    def test_reciprocal_rank_top1_hit(self):
        assert harness.reciprocal_rank(["a.md"], ["a.md", "x.md"]) == 1.0

    def test_ndcg_perfect_ranking_is_one(self):
        # single gold path, hit at rank 1 → nDCG@5 == 1.0
        assert harness.ndcg_at_k(["a.md"], ["a.md", "x.md", "y.md"], k=5) == pytest.approx(1.0)

    def test_ndcg_no_hit_is_zero(self):
        assert harness.ndcg_at_k(["a.md"], ["x.md", "y.md"], k=5) == 0.0

    def test_ndcg_hit_at_rank_2_below_perfect(self):
        # relevant doc at rank 2 instead of rank 1 → 0 < nDCG < 1
        score = harness.ndcg_at_k(["a.md"], ["x.md", "a.md"], k=5)
        assert 0.0 < score < 1.0
        # hand-computed: DCG = 1/log2(3), IDCG = 1/log2(2) = 1 → nDCG = 1/log2(3)
        import math

        assert score == pytest.approx(1.0 / math.log2(3))


class TestAggregate:
    def test_recall_and_mrr_over_fixture_rows(self):
        results = [
            harness.RowResult(
                row_id="F1",
                cls="factual",
                hit_at_1=True,
                hit_at_5=True,
                hit_at_10=True,
                rr=1.0,
                ndcg_at_5=1.0,
                latency_ms=100.0,
            ),
            harness.RowResult(
                row_id="F2",
                cls="factual",
                hit_at_1=False,
                hit_at_5=True,
                hit_at_10=True,
                rr=0.5,
                ndcg_at_5=0.6,
                latency_ms=200.0,
            ),
        ]
        agg = harness.aggregate(results, cost_usd=0.0)
        overall = agg["overall"]
        assert overall["n"] == 2
        assert overall["recall@1"] == pytest.approx(0.5)
        assert overall["recall@5"] == pytest.approx(1.0)
        assert overall["mrr"] == pytest.approx(0.75)
        assert overall["latency_p50_ms"] == pytest.approx(150.0)
        assert "factual" in agg

    def test_aggregate_empty_class_is_zeroed_not_crashing(self):
        agg = harness.aggregate([], cost_usd=0.0)
        assert agg["overall"]["n"] == 0
        assert agg["overall"]["recall@5"] == 0.0


# --------------------------------------------------------------------------- infra-fail vs threshold-fail (V-AC-02a)


class _FakeInfraFailClient:
    name = "fake-infra-fail"

    def retrieve(self, query, limit):
        raise harness.InfraError("simulated mesh connection refused")

    def cost_usd(self):
        return 0.0


class _FakeLowRecallClient:
    """Always misses — a successful (non-infra) run that scores below any real threshold."""

    name = "fake-low-recall"

    def retrieve(self, query, limit):
        return ["nonexistent/path.md"], 42.0

    def cost_usd(self):
        return 0.0


class _FakeHighRecallClient:
    def __init__(self, rows_by_query):
        self._rows_by_query = rows_by_query

    name = "fake-high-recall"

    def retrieve(self, query, limit):
        gold = self._rows_by_query[query]
        return gold[:limit], 10.0

    def cost_usd(self):
        return 0.0


class TestInfraVsThresholdFail:
    def test_run_model_propagates_infra_error(self, monkeypatch):
        rows = [harness.GoldenRow(id="F1", cls="factual", query="q", gold_source_paths=["a.md"])]
        monkeypatch.setattr(harness, "build_client", lambda name, **kw: _FakeInfraFailClient())
        with pytest.raises(harness.InfraError):
            harness.run_model("fake-infra-fail", rows, endpoint="x", namespace="ns")

    def test_run_model_low_recall_is_successful_not_infra(self, monkeypatch):
        rows = [harness.GoldenRow(id="F1", cls="factual", query="q", gold_source_paths=["a.md"])]
        monkeypatch.setattr(harness, "build_client", lambda name, **kw: _FakeLowRecallClient())
        summary, row_results = harness.run_model("fake-low-recall", rows, endpoint="x", namespace="ns")
        assert summary["overall"]["recall@5"] == 0.0
        assert len(row_results) == 1

    def test_main_infra_fail_exit_code_distinct_from_threshold_fail(self, monkeypatch, tmp_path):
        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": "F1", "class": "factual", "query": "q", "gold_source_paths": ["a.md"]}\n')
        (tmp_path / "a.md").write_text("content")

        monkeypatch.setattr(harness, "build_client", lambda name, **kw: _FakeInfraFailClient())
        rc_infra = harness.main(
            [
                "--golden",
                str(golden),
                "--corpus-root",
                str(tmp_path),
                "--out-summary",
                str(tmp_path / "summary.json"),
            ]
        )
        assert rc_infra == harness.INFRA_FAIL_CODE

        monkeypatch.setattr(harness, "build_client", lambda name, **kw: _FakeLowRecallClient())
        rc_threshold = harness.main(
            [
                "--golden",
                str(golden),
                "--corpus-root",
                str(tmp_path),
                "--out-summary",
                str(tmp_path / "summary2.json"),
                "--out-detail",
                str(tmp_path / "detail2.json"),
            ]
        )
        assert rc_threshold == harness.THRESHOLD_FAIL_CODE
        assert rc_infra != rc_threshold

    def test_main_success_exit_code_zero(self, monkeypatch, tmp_path):
        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": "F1", "class": "factual", "query": "q", "gold_source_paths": ["a.md"]}\n')
        (tmp_path / "a.md").write_text("content")

        client = _FakeHighRecallClient({"q": ["a.md"]})
        monkeypatch.setattr(harness, "build_client", lambda name, **kw: client)
        rc = harness.main(
            [
                "--golden",
                str(golden),
                "--corpus-root",
                str(tmp_path),
                "--out-summary",
                str(tmp_path / "summary.json"),
                "--out-detail",
                str(tmp_path / "detail.json"),
                "--recall-threshold",
                "0.5",
            ]
        )
        assert rc == harness.SUCCESS_CODE
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["exit_reason"] == "ok"
        assert summary["stale_skipped"] == 0
        assert "bge-m3" in summary["models"]
        detail = json.loads((tmp_path / "detail.json").read_text())
        assert len(detail["bge-m3"]) == 1

    def test_main_stale_row_excluded_from_denominator(self, monkeypatch, tmp_path):
        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            '{"id": "F1", "class": "factual", "query": "q1", "gold_source_paths": ["a.md"]}\n'
            '{"id": "F2", "class": "factual", "query": "q2", "gold_source_paths": ["dead.md"]}\n'
        )
        (tmp_path / "a.md").write_text("content")

        client = _FakeHighRecallClient({"q1": ["a.md"]})
        monkeypatch.setattr(harness, "build_client", lambda name, **kw: client)
        rc = harness.main(
            [
                "--golden",
                str(golden),
                "--corpus-root",
                str(tmp_path),
                "--out-summary",
                str(tmp_path / "summary.json"),
                "--out-detail",
                str(tmp_path / "detail.json"),
                "--recall-threshold",
                "0.5",
            ]
        )
        assert rc == harness.SUCCESS_CODE
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["stale_skipped"] == 1
        assert summary["models"]["bge-m3"]["overall"]["n"] == 1  # total(2) - stale_skipped(1)


# --------------------------------------------------------------------------- consumer-side verdict smoke


class TestDecideExitCode:
    def test_passing_synthetic_summary(self):
        summaries = {"bge-m3": {"overall": {"recall@5": 0.92}}}
        code, reason = harness.decide_exit_code(summaries, threshold=0.879)
        assert code == harness.SUCCESS_CODE
        assert reason == "ok"

    def test_failing_synthetic_summary(self):
        summaries = {"bge-m3": {"overall": {"recall@5": 0.5}}}
        code, reason = harness.decide_exit_code(summaries, threshold=0.879)
        assert code == harness.THRESHOLD_FAIL_CODE
        assert reason == "threshold_fail"

    def test_multi_model_any_failure_fails_the_run(self):
        summaries = {
            "bge-m3": {"overall": {"recall@5": 0.95}},
            "bge-reranker": {"overall": {"recall@5": 0.2}},
        }
        code, _ = harness.decide_exit_code(summaries, threshold=0.879)
        assert code == harness.THRESHOLD_FAIL_CODE


# --------------------------------------------------------------------------- CLI dry-run smoke


class TestDryRunCli:
    def test_dry_run_infra_fail_no_golden_needed(self):
        rc = harness.main(["--dry-run-infra-fail"])
        assert rc == harness.INFRA_FAIL_CODE

    def test_dry_run_threshold_fail_no_golden_needed(self):
        rc = harness.main(["--dry-run-threshold-fail", "0.5"])
        assert rc == harness.THRESHOLD_FAIL_CODE

    def test_missing_golden_without_dry_run_errors(self):
        with pytest.raises(SystemExit):
            harness.main([])


# --------------------------------------------------------------------------- not-wired models


class TestNotWiredClient:
    def test_bge_reranker_raises_not_implemented(self):
        client = harness.build_client("bge-reranker", endpoint="x", namespace="ns")
        with pytest.raises(NotImplementedError):
            client.retrieve("q", 5)

    def test_llm_alias_raises_not_implemented(self):
        client = harness.build_client("llm:gpt-4o-mini", endpoint="x", namespace="ns")
        with pytest.raises(NotImplementedError):
            client.retrieve("q", 5)

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError):
            harness.build_client("not-a-real-model", endpoint="x", namespace="ns")
