#!/usr/bin/env python3
"""recall_gate.py — Scrutator recall@k regression CI gate.

Wraps the existing LTM-0009 harness (ltm-bench-query.py) to add:
  - baseline load (from baseline.json)
  - per-class delta comparison (factual / multi-hop / temporal independently)
  - non-zero exit on regression

This script does NOT recompute recall — it reads aggregate_by_class.*.mean_recall_at_5
from the harness JSON output and compares against the committed baseline.

Exit codes:
  0 — all classes at or above baseline (within threshold), gate passes
  1 — at least one class regressed beyond its per-class threshold, gate fails (recall regression)
  2 — transport / infrastructure error: all queries returned num_retrieved==0, which indicates
      a network or service failure rather than a real recall drop. A network flake MUST NOT
      be reported as a recall regression.

Usage:
  # Gate against an existing report JSON:
  python recall_gate.py --report <path-to-report.json>

  # Run the harness first, then gate the result:
  python recall_gate.py --run --harness /path/to/ltm-bench-query.py

  # Refresh the baseline with current per-class numbers (manual, requires review):
  python recall_gate.py --report <path> --update-baseline
"""

import json
import subprocess
import sys
from pathlib import Path

# Defaults relative to this script's directory (benchmark/recall-gate/)
_GATE_DIR = Path(__file__).parent
DEFAULT_BASELINE = _GATE_DIR / "baseline.json"
DEFAULT_THRESHOLDS = _GATE_DIR / "thresholds.json"

# Default harness path on the arcana-db runner (synced ~/arcanada tree)
DEFAULT_HARNESS = Path.home() / "arcanada/Projects/Long Term Memory/benchmark/scripts/ltm-bench-query.py"

CLASSES = ["factual", "multi-hop", "temporal"]


def load_json(path: Path, label: str) -> dict:
    """Load and parse a JSON file; exit with informative message on failure."""
    if not path.exists():
        print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
        sys.exit(3)
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"ERROR: {label} file is not valid JSON: {path}\n  {exc}", file=sys.stderr)
        sys.exit(3)


def is_transport_error(report: dict) -> bool:
    """Return True if all per-query rows show num_retrieved==0 — harness ERROR pattern."""
    per_query = report.get("per_query", [])
    if not per_query:
        return True
    return all(row.get("num_retrieved", 0) == 0 for row in per_query)


def extract_per_class_recall(report: dict) -> dict[str, float]:
    """Extract mean_recall_at_5 per class from aggregate_by_class."""
    by_class = report.get("aggregate_by_class", {})
    result: dict[str, float] = {}
    for cls in CLASSES:
        cls_data = by_class.get(cls, {})
        result[cls] = float(cls_data.get("mean_recall_at_5", 0.0))
    return result


def run_harness(harness_path: Path) -> dict:
    """Invoke the harness with --expand-entities and parse the JSON report it writes.

    Gate mode: with-entities (expand_entities=True).
    Rationale: the production /v1/ltm/recall endpoint defaults to expand_entities=True
    (RecallRequest model default).  The gate must defend the production retrieval path.
    Callers who explicitly pass expand_entities=false use a non-default mode not guarded here.
    """
    if not harness_path.exists():
        print(f"ERROR: harness not found at: {harness_path}", file=sys.stderr)
        print("  Set --harness <path> or ensure the synced ~/arcanada tree is accessible.", file=sys.stderr)
        sys.exit(3)

    print(f"Running harness: {harness_path} --expand-entities")
    try:
        subprocess.run(
            [sys.executable, str(harness_path), "--expand-entities"],
            capture_output=False,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("ERROR: harness timed out after 600 seconds — transport error assumed.", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"ERROR: failed to run harness: {exc}", file=sys.stderr)
        sys.exit(2)

    # The harness writes its report to a dated file under reports/v4/scrutator/.
    # With --expand-entities the filename suffix is .with-entities.json.
    harness_reports_dir = harness_path.parent.parent / "reports" / "v4" / "scrutator"
    if not harness_reports_dir.exists():
        print(f"ERROR: harness reports dir not found: {harness_reports_dir}", file=sys.stderr)
        sys.exit(2)

    import time

    today = time.strftime("%Y-%m-%d")
    # Find today's with-entities report (gate mode)
    candidates = sorted(harness_reports_dir.glob(f"{today}.datarim-kb.with-entities.json"))
    if not candidates:
        # Fall back to most recently modified JSON
        all_reports = sorted(harness_reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not all_reports:
            print(f"ERROR: no harness report found in {harness_reports_dir}", file=sys.stderr)
            sys.exit(2)
        report_path = all_reports[0]
    else:
        report_path = candidates[-1]

    print(f"Reading report: {report_path}")
    return load_json(report_path, "harness report")


def print_table(results: list[dict]) -> None:
    """Print per-class comparison table."""
    col_w = [10, 10, 10, 10, 10, 8]
    header = (
        f"{'class':<{col_w[0]}} {'baseline':>{col_w[1]}} {'observed':>{col_w[2]}}"
        f" {'delta':>{col_w[3]}} {'threshold':>{col_w[4]}} {'status':>{col_w[5]}}"
    )
    sep = "-" * len(header)
    print()
    print("Recall@5 per-class regression check:")
    print(sep)
    print(header)
    print(sep)
    for row in results:
        status = "FAIL" if row["fail"] else "PASS"
        delta_str = f"{row['delta']:+.4f}"
        print(
            f"{row['cls']:<{col_w[0]}} "
            f"{row['baseline']:>{col_w[1]}.4f} "
            f"{row['observed']:>{col_w[2]}.4f} "
            f"{delta_str:>{col_w[3]}} "
            f"{row['threshold']:>{col_w[4]}.4f} "
            f"{status:>{col_w[5]}}"
        )
    print(sep)
    print()


def gate(report: dict, baseline: dict, thresholds: dict) -> int:
    """Run the per-class regression check. Returns the exit code."""
    # Transport-error guard: if all queries show num_retrieved==0, this is an infra failure
    if is_transport_error(report):
        print(
            "TRANSPORT ERROR: all per-query rows show num_retrieved=0.\n"
            "  This indicates a network or service failure, NOT a recall regression.\n"
            "  Scrutator may be down or the namespace may be unreachable.\n"
            "  Exiting with code 2 (infrastructure error).",
            file=sys.stderr,
        )
        return 2

    observed = extract_per_class_recall(report)
    baseline_by_class: dict[str, float] = baseline.get("by_class", {})

    results = []
    any_fail = False
    for cls in CLASSES:
        baseline_val = float(baseline_by_class.get(cls, 0.0))
        observed_val = observed.get(cls, 0.0)
        threshold = float(thresholds.get(cls, 0.05))
        # Regression = how much observed fell below baseline
        regression = baseline_val - observed_val
        fail = regression > threshold
        if fail:
            any_fail = True
        results.append(
            {
                "cls": cls,
                "baseline": baseline_val,
                "observed": observed_val,
                "delta": observed_val - baseline_val,  # positive = improvement, negative = regression
                "threshold": threshold,
                "regression": regression,
                "fail": fail,
            }
        )

    print_table(results)

    if any_fail:
        failed = [r["cls"] for r in results if r["fail"]]
        print(f"RECALL REGRESSION DETECTED: {', '.join(failed)} exceeded regression threshold.")
        print("  Gate FAILS (exit 1). Fix the regression or update the baseline after review.")
        return 1
    else:
        print("All classes within threshold. Gate PASSES (exit 0).")
        return 0


def update_baseline(report: dict, baseline_path: Path) -> None:
    """Write current per-class recall numbers to baseline.json."""
    import time

    observed = extract_per_class_recall(report)
    existing = load_json(baseline_path, "baseline") if baseline_path.exists() else {}
    updated = {
        **existing,
        "captured_at": time.strftime("%Y-%m-%d"),
        "by_class": observed,
    }
    with open(baseline_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"Baseline updated: {baseline_path}")
    for cls, val in observed.items():
        print(f"  {cls}: {val:.4f}")


def parse_args(argv: list[str]) -> dict:
    """Minimal CLI argument parser (no external deps)."""
    args: dict = {
        "report": None,
        "baseline": DEFAULT_BASELINE,
        "thresholds": DEFAULT_THRESHOLDS,
        "run": False,
        "harness": DEFAULT_HARNESS,
        "update_baseline": False,
    }
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--report" and i + 1 < len(argv):
            args["report"] = Path(argv[i + 1])
            i += 2
        elif arg == "--baseline" and i + 1 < len(argv):
            args["baseline"] = Path(argv[i + 1])
            i += 2
        elif arg == "--thresholds" and i + 1 < len(argv):
            args["thresholds"] = Path(argv[i + 1])
            i += 2
        elif arg == "--harness" and i + 1 < len(argv):
            args["harness"] = Path(argv[i + 1])
            i += 2
        elif arg == "--run":
            args["run"] = True
            i += 1
        elif arg == "--update-baseline":
            args["update_baseline"] = True
            i += 1
        else:
            print(f"WARNING: unknown argument: {arg}", file=sys.stderr)
            i += 1
    return args


def main() -> None:
    args = parse_args(sys.argv)

    baseline = load_json(args["baseline"], "baseline")
    thresholds = load_json(args["thresholds"], "thresholds")

    if args["run"]:
        report = run_harness(args["harness"])
    elif args["report"] is not None:
        report = load_json(args["report"], "report")
    else:
        print("ERROR: provide --report <path> or --run (to invoke the harness)", file=sys.stderr)
        sys.exit(3)

    if args["update_baseline"]:
        update_baseline(report, args["baseline"])
        sys.exit(0)

    exit_code = gate(report, baseline, thresholds)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
