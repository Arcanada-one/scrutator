"""Unit tests for recall_gate.py.

Tests cover:
- Green report (all classes at/above baseline) -> exit 0, all PASS
- Regressed report (temporal below threshold) -> exit 1, temporal FAIL (failing-build proof)
- Per-class independence: factual-only regression fails factual, temporal PASS
- Multi-hop-only regression fails multi-hop, others PASS
- Transport error (all-zero + num_retrieved==0) -> exit 2, NOT exit 1
- Baseline-load: missing file -> SystemExit with informative message
"""

import hashlib
import subprocess
import sys
from pathlib import Path

# Resolve paths relative to this file
GATE_DIR = Path(__file__).parent.parent
GATE_SCRIPT = GATE_DIR / "recall_gate.py"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
BASELINE = GATE_DIR / "baseline.json"
THRESHOLDS = GATE_DIR / "thresholds.json"
VENDOR_DIR = GATE_DIR / "vendor"


def run_gate(*args) -> subprocess.CompletedProcess:
    """Run recall_gate.py as a subprocess and return the result."""
    cmd = [
        sys.executable,
        str(GATE_SCRIPT),
        "--baseline",
        str(BASELINE),
        "--thresholds",
        str(THRESHOLDS),
        *args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_vendored_harness_snapshot_is_complete_and_pinned():
    expected = {
        VENDOR_DIR / "ltm-bench-query.py": "6a31f688301a7fab8d2412500d1ecf22daa638fb22a291605d4a0c2cda1a7b81",
        VENDOR_DIR / "queries/factual.jsonl": "66ebbec22459763f6337d87503bbd35a913d10c2c1481d36b242fab13fc20767",
        VENDOR_DIR / "queries/multi-hop.jsonl": "136af39e509a18658380473350929a313019003afdf717d67b1dec078f5f595f",
        VENDOR_DIR / "queries/temporal.jsonl": "14206a5707bb12afd30aee16d2b42ecd8e98ea7a4e28a701e24df2848535a032",
    }
    for path, digest in expected.items():
        assert path.is_file(), f"missing vendored recall input: {path}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    workflow = (GATE_DIR.parent.parent / ".github/workflows/recall-regression.yml").read_text()
    assert "HARNESS_PATH: benchmark/recall-gate/vendor/ltm-bench-query.py" in workflow
    assert "/home/ci-runner/arcanada/" not in workflow


class TestGreenReport:
    """Gate passes on a report at or above baseline."""

    def test_exit_0_on_green_report(self):
        result = run_gate("--report", str(FIXTURES_DIR / "report_green.json"))
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}.\n{result.stdout}\n{result.stderr}"

    def test_all_classes_pass_in_output(self):
        result = run_gate("--report", str(FIXTURES_DIR / "report_green.json"))
        assert "PASS" in result.stdout
        assert "FAIL" not in result.stdout


class TestRegressedReport:
    """Gate fails (exit 1) on a temporal regression — this is the failing-build proof."""

    def test_exit_1_on_temporal_regression(self):
        result = run_gate("--report", str(FIXTURES_DIR / "report_regressed.json"))
        assert result.returncode == 1, (
            f"Failing-build proof FAILED: expected exit 1 (recall regression), got {result.returncode}.\n"
            f"{result.stdout}\n{result.stderr}"
        )

    def test_temporal_class_marked_fail(self):
        result = run_gate("--report", str(FIXTURES_DIR / "report_regressed.json"))
        assert "temporal" in result.stdout.lower()
        assert "FAIL" in result.stdout

    def test_output_shows_regression_details(self):
        result = run_gate("--report", str(FIXTURES_DIR / "report_regressed.json"))
        # Must print a summary table with class, baseline, observed, delta
        assert "baseline" in result.stdout.lower() or "delta" in result.stdout.lower()


class TestPerClassIndependence:
    """Regression of one class fails only that class; others remain PASS."""

    def test_factual_only_regression_fails_factual(self):
        result = run_gate("--report", str(FIXTURES_DIR / "report_factual_regressed.json"))
        assert result.returncode == 1, "Expected exit 1 on factual regression"
        # factual must be FAIL
        lines = result.stdout
        assert "factual" in lines.lower()
        assert "FAIL" in lines

    def test_factual_only_regression_temporal_still_passes(self):
        result = run_gate("--report", str(FIXTURES_DIR / "report_factual_regressed.json"))
        # In a per-line scan: find "temporal" row and confirm it says PASS
        lines = result.stdout.splitlines()
        temporal_lines = [line for line in lines if "temporal" in line.lower()]
        assert any("PASS" in line for line in temporal_lines), (
            f"Expected temporal PASS in factual-regression report, got: {temporal_lines}"
        )

    def test_multi_hop_only_regression(self):
        result = run_gate("--report", str(FIXTURES_DIR / "report_multihop_regressed.json"))
        assert result.returncode == 1, "Expected exit 1 on multi-hop regression"
        lines = result.stdout.splitlines()
        multihop_lines = [line for line in lines if "multi" in line.lower()]
        assert any("FAIL" in line for line in multihop_lines), f"Expected multi-hop FAIL, got: {multihop_lines}"


class TestTransportError:
    """Transport error (all-zero + num_retrieved==0) must exit 2, not 1."""

    def test_transport_error_exits_2_not_1(self):
        result = run_gate("--report", str(FIXTURES_DIR / "report_transport_error.json"))
        assert result.returncode == 2, (
            f"Transport error must exit 2 (not 1 = recall regression, not 0 = pass). "
            f"Got: {result.returncode}.\n{result.stdout}\n{result.stderr}"
        )

    def test_transport_error_message_mentions_infra(self):
        result = run_gate("--report", str(FIXTURES_DIR / "report_transport_error.json"))
        combined = (result.stdout + result.stderr).lower()
        assert any(word in combined for word in ["transport", "infra", "error", "network"]), (
            f"Expected transport/infra error message, got:\n{result.stdout}\n{result.stderr}"
        )


class TestBaselineLoad:
    """Baseline file missing or malformed -> exit with informative error."""

    def test_missing_baseline_exits_nonzero(self, tmp_path):
        missing = tmp_path / "no_such_baseline.json"
        result = run_gate(
            "--report",
            str(FIXTURES_DIR / "report_green.json"),
            "--baseline",
            str(missing),
        )
        assert result.returncode != 0

    def test_missing_thresholds_exits_nonzero(self, tmp_path):
        missing = tmp_path / "no_such_thresholds.json"
        result = run_gate(
            "--report",
            str(FIXTURES_DIR / "report_green.json"),
            "--thresholds",
            str(missing),
        )
        assert result.returncode != 0
