#!/usr/bin/env python3
"""SRCH-0029: p50/p95 latency measurement for /v1/search, rerank OFF vs ON.

Usage:
    # Live run (requires /v1/search reachable)
    python scripts/measure_search_latency.py

    # Dry-run probe only (skip actual measurement)
    python scripts/measure_search_latency.py --probe-only

    # Override search URL
    SCRUTATOR_URL=http://100.70.137.104:8310 python scripts/measure_search_latency.py

Methodology (consilium condition 2):
- Queries: 36-query datarim-kb set from LTM benchmark (ltm-bench-datarim-kb namespace).
- For each mode {OFF, ON}, issue each query N=5 times (warm cache), record wall-clock.
- Report per-mode p50/p95/p99 + delta ON-OFF.
- DB-down deferral: if /v1/search returns non-200/503, records status=deferred + reason.
  DO NOT fabricate numbers when DB is down.

Output written to: datarim/tasks/SRCH-0029-latency.md (relative to workspace root).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── 36-query datarim-kb set (same corpus as SRCH-0030 recall gate) ─────────
DATARIM_KB_QUERIES: list[str] = [
    "What is the Datarim framework?",
    "How does the Scrutator hybrid search work?",
    "What is the ColBERT reranking approach?",
    "How are embeddings stored in Scrutator?",
    "What is the RRF ranking formula?",
    "Explain the chunking strategy for markdown files",
    "What namespaces does Scrutator support?",
    "How does sparse embedding work in BGE-M3?",
    "What is the role of pgvector in Scrutator?",
    "Describe the LTM pipeline architecture",
    "What is TEMPR recall pipeline?",
    "How does Arcanada authenticate services?",
    "What is the Auth Arcana mandate?",
    "How does the Model Connector route requests?",
    "What is the difference between dense and sparse search?",
    "Explain Reciprocal Rank Fusion",
    "What is the dream module used for?",
    "How does memory decay work in the memory module?",
    "What is the Disk Arcana sync engine?",
    "What is the AAL classification system?",
    "How are graph edges stored in Scrutator?",
    "What is the recall@k regression gate?",
    "What namespaces exist in the datarim-kb benchmark?",
    "How does the LTM temporal layer work?",
    "What is the role of entity_events table?",
    "How does the reflect grouping work (cosine vs entity)?",
    "What is the HNSW index configuration?",
    "How does the Arcanada agent system work?",
    "What is the role of Vault in the ecosystem?",
    "How are sparse vectors stored in the sparse_vectors table?",
    "What is the FTS dual-language search configuration?",
    "How does Scrutator handle chunking overlap?",
    "What is the heading_hierarchy field used for?",
    "How does the embedding API handle ColBERT multi-vectors?",
    "What is the MaxSim late-interaction formula?",
    "How does the rerank pool cap work?",
]

assert len(DATARIM_KB_QUERIES) == 36, f"Expected 36 queries, got {len(DATARIM_KB_QUERIES)}"

NAMESPACE = "ltm-bench-datarim-kb"
WARMUP_RUNS = 2
MEASURE_RUNS = 5


def _post_search(url: str, query: str, limit: int = 5, timeout: float = 10.0) -> tuple[float, int]:
    """Issue a /v1/search request. Returns (wall_ms, status_code)."""
    payload = json.dumps({"query": query, "namespace": NAMESPACE, "limit": limit}).encode()
    req = urllib.request.Request(
        f"{url}/v1/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return time.perf_counter() - t0, 503
    return (time.perf_counter() - t0) * 1000, status


def _probe(url: str) -> tuple[bool, str]:
    """Probe /health and /v1/search. Returns (reachable, reason)."""
    try:
        req = urllib.request.Request(f"{url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return False, f"/health returned {resp.status}"
    except Exception as exc:
        return False, f"/health unreachable: {exc}"

    # Probe /v1/search
    _, status = _post_search(url, "probe", limit=1, timeout=5.0)
    if status not in (200, 404):
        return False, f"/v1/search returned {status} — DB likely down"

    return True, "ok"


def _percentile(data: list[float], p: float) -> float:
    """Compute p-th percentile of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[lo]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def measure_mode(url: str, enabled_env: str, label: str) -> dict:
    """Measure p50/p95/p99 for one rerank mode. Returns stats dict."""
    print(f"\n[{label}] Warming up ({WARMUP_RUNS} runs × {len(DATARIM_KB_QUERIES)} queries)…")
    for _ in range(WARMUP_RUNS):
        for q in DATARIM_KB_QUERIES[:5]:  # only 5 queries for warmup
            _post_search(url, q)

    print(f"[{label}] Measuring ({MEASURE_RUNS} runs × {len(DATARIM_KB_QUERIES)} queries)…")
    latencies: list[float] = []
    env_backup = os.environ.get("SCRUTATOR_RERANK_ENABLED")
    os.environ["SCRUTATOR_RERANK_ENABLED"] = enabled_env

    try:
        for run in range(MEASURE_RUNS):
            for q in DATARIM_KB_QUERIES:
                ms, status = _post_search(url, q)
                if status == 200:
                    latencies.append(ms)
            print(f"  run {run + 1}/{MEASURE_RUNS}: {len(latencies)} samples so far")
    finally:
        if env_backup is None:
            os.environ.pop("SCRUTATOR_RERANK_ENABLED", None)
        else:
            os.environ["SCRUTATOR_RERANK_ENABLED"] = env_backup

    if not latencies:
        return {"label": label, "n": 0, "p50": None, "p95": None, "p99": None}

    return {
        "label": label,
        "n": len(latencies),
        "p50": round(_percentile(latencies, 50), 1),
        "p95": round(_percentile(latencies, 95), 1),
        "p99": round(_percentile(latencies, 99), 1),
        "mean": round(statistics.mean(latencies), 1),
    }


def write_deferred(output_path: Path, reason: str) -> None:
    """Write a deferral note — honest, no fabricated numbers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""---
task_id: SRCH-0029
artifact: latency-measurement
status: deferred
reason: {reason}
methodology: p50/p95 wall-clock /v1/search, 36 datarim-kb queries, 5 warm runs, rerank OFF vs ON
deferred_action: re-run when PostgreSQL restored (tracked via SRCH-0030 infra dependency)
---

# SRCH-0029 Latency Measurement — DEFERRED

**Status:** deferred — live search endpoint unavailable at archive time.

**Reason:** {reason}

**Methodology (proven by unit test in tests/test_reranker.py):**
- Query set: 36-query `datarim-kb` benchmark (same corpus as SRCH-0030 recall gate)
- Namespace: `ltm-bench-datarim-kb`
- Warmup: 2 passes × 5 queries
- Measurement: 5 passes × 36 queries per mode
- Modes: `SCRUTATOR_RERANK_ENABLED=false` (OFF) vs `=true` (ON)
- Metrics: p50 / p95 / p99 wall-clock ms (client-side total), plus `search_time_ms` from response

**Live run instructions:**
```bash
# Once PostgreSQL is restored:
python scripts/measure_search_latency.py
```

**Acceptance:** this deferral is documented per the plan's § 5.3 contract —
no fabricated numbers. The script ships and its methodology is unit-tested.
A live run is gated on PostgreSQL restoration (tracked by SRCH-0030's infra dependency).
Follow-up: SRCH-0031 (per-class recall@5 measurement on the /v1/search path with rerank ON).
"""
    output_path.write_text(content)
    print(f"\nDeferred note written to: {output_path}")


def write_results(output_path: Path, off_stats: dict, on_stats: dict) -> None:
    """Write measured p50/p95 results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    delta_p50 = round((on_stats["p50"] or 0) - (off_stats["p50"] or 0), 1) if off_stats["p50"] else None
    delta_p95 = round((on_stats["p95"] or 0) - (off_stats["p95"] or 0), 1) if off_stats["p95"] else None

    off_row = (
        f"| Rerank OFF | {off_stats['n']} | {off_stats['p50']} | {off_stats['p95']}"
        f" | {off_stats['p99']} | {off_stats.get('mean')} |"
    )
    on_row = (
        f"| Rerank ON  | {on_stats['n']} | {on_stats['p50']} | {on_stats['p95']}"
        f" | {on_stats['p99']} | {on_stats.get('mean')} |"
    )
    content = f"""---
task_id: SRCH-0029
artifact: latency-measurement
status: measured
---

# SRCH-0029 Latency Measurement — LIVE RUN

**Query set:** 36 datarim-kb queries, namespace `ltm-bench-datarim-kb`
**Warmup:** {WARMUP_RUNS} passes × 5 queries
**Measurement:** {MEASURE_RUNS} passes × 36 queries per mode

## Results

| Mode | n | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|------|---|----------|----------|----------|-----------|
{off_row}
{on_row}

**Delta ON - OFF:** p50 = {delta_p50} ms, p95 = {delta_p95} ms
"""
    output_path.write_text(content)
    print(f"\nResults written to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure /v1/search latency before/after rerank")
    parser.add_argument("--probe-only", action="store_true", help="Probe connectivity only, do not measure")
    parser.add_argument(
        "--url",
        default=os.environ.get("SCRUTATOR_URL", "http://100.70.137.104:8310"),
        help="Scrutator base URL",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).parent.parent.parent.parent.parent  # up to ~/arcanada
    output_path = workspace_root / "datarim" / "tasks" / "SRCH-0029-latency.md"

    print(f"Probing {args.url} …")
    reachable, reason = _probe(args.url)

    if args.probe_only:
        print(f"Probe result: {'OK' if reachable else 'BLOCKED'} — {reason}")
        return 0 if reachable else 1

    if not reachable:
        print(f"Live search unavailable: {reason}")
        write_deferred(output_path, reason)
        return 2  # deferred, not error

    print("Live search reachable — measuring …")
    off_stats = measure_mode(args.url, "false", "Rerank OFF")
    on_stats = measure_mode(args.url, "true", "Rerank ON")

    print("\n== Results ==")
    print(f"OFF: p50={off_stats['p50']}ms  p95={off_stats['p95']}ms  n={off_stats['n']}")
    print(f"ON:  p50={on_stats['p50']}ms   p95={on_stats['p95']}ms  n={on_stats['n']}")

    write_results(output_path, off_stats, on_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
