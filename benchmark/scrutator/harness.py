#!/usr/bin/env python3
"""benchmark/scrutator/harness.py — SRCH-0015 multi-model `/v1/search` recall harness.

Extends `measure.py`'s request/scoring shape (ported verbatim as the starting point) with:
  - a pluggable model-dispatch interface (`ModelClient`),
  - a `corpus_pinned_at` + runtime liveness pre-flight (D-REQ-02/D-REQ-03),
  - recall@{1,5,10}, MRR, nDCG@5, latency p50/p95, cost per model,
  - infra-fail vs. threshold-fail exit-code separation (D-REQ-05).

Scope decision (SRCH-0015 /dr-do, recorded in CONSUMERS.md and the task description):
only `bge-m3` is wired to a live network call in this task. `bge-reranker` (no confirmed
live cross-encoder endpoint in this repo) and any `llm:*` baseline (real, non-trivial API
spend at volume — the operator's HARD-GATE) are registered in `build_client()` but raise
`NotImplementedError` — tests inject a fake `ModelClient` to prove the multi-model dispatch
loop and exit-code logic (V-AC-02a) without making those calls live. Also per PRD § Context
Analysis: `rerank` is a no-op on `/v1/search` today (SRCH-0029 not shipped) — this harness
does not report "rerank on" vs. "rerank off" as two distinct measured conditions.

Stdlib-only by design (matches `measure.py`): this script must run on a bare self-hosted
runner or the operator's Mac without installing Scrutator's own dependency set.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

INFRA_FAIL_CODE = 2
THRESHOLD_FAIL_CODE = 1
SUCCESS_CODE = 0

DEFAULT_ENDPOINT = "http://100.70.137.104:8310/v1/search"
DEFAULT_NAMESPACE = "arcanada"
DEFAULT_TIMEOUT_S = 30.0
# Lower bound of SRCH-0031's baseline recall@5 (0.909) ± 0.03, per V-AC-06.
DEFAULT_RECALL_THRESHOLD = 0.879


class InfraError(Exception):
    """Raised by a ModelClient when a request cannot complete due to infra failure
    (connection refused, mesh timeout, provider error) — distinct from a low-but-successful
    recall result. See D-REQ-05."""


# --------------------------------------------------------------------------- golden rows


@dataclass
class GoldenRow:
    id: str
    cls: str
    query: str
    gold_source_paths: list[str]
    corpus_pinned_at: str | None = None

    @classmethod
    def from_json(cls_, obj: dict) -> GoldenRow:
        missing = [k for k in ("id", "class", "query", "gold_source_paths") if k not in obj]
        if missing:
            raise ValueError(f"golden row missing required field(s): {missing} — row: {obj}")
        return cls_(
            id=obj["id"],
            cls=obj["class"],
            query=obj["query"],
            gold_source_paths=obj["gold_source_paths"],
            corpus_pinned_at=obj.get("corpus_pinned_at"),
        )


def load_golden_rows(path: str | Path) -> list[GoldenRow]:
    """Load a golden-arcanada-v{N}.jsonl file into GoldenRow objects."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc
            rows.append(GoldenRow.from_json(obj))
    return rows


# --------------------------------------------------------------------------- liveness pre-flight


def is_row_live(row: GoldenRow, corpus_root: Path) -> bool:
    """A row is live if at least one of its gold_source_paths still exists under corpus_root.

    Design note: multi-hop rows carry >1 gold path. A row is only reported SKIPPED (stale)
    when *every* one of its gold paths is dead — a single surviving path is still enough to
    score tail_hit() against. This is stricter than "any path dead ⇒ skip" (which would
    silently drop multi-hop rows far more often than the corpus staleness actually warrants).
    """
    return any((corpus_root / p).exists() for p in row.gold_source_paths)


def partition_by_liveness(rows: list[GoldenRow], corpus_root: Path) -> tuple[list[GoldenRow], list[GoldenRow]]:
    """Split rows into (live, stale) per the liveness pre-flight (D-REQ-03)."""
    live, stale = [], []
    for row in rows:
        (live if is_row_live(row, corpus_root) else stale).append(row)
    return live, stale


# --------------------------------------------------------------------------- scoring primitives


def tail_hit(gold_paths: list[str], returned_paths: list[str]) -> bool:
    """Path-matching convention ported verbatim from measure.py's tail_hit()."""
    for g in gold_paths:
        for r in returned_paths:
            if r == g or r.endswith("/" + g) or g.endswith("/" + r):
                return True
    return False


def reciprocal_rank(gold_paths: list[str], returned_paths: list[str]) -> float:
    """1/rank of the first returned path that tail-hits a gold path; 0.0 if none hit."""
    for i, r in enumerate(returned_paths, start=1):
        if tail_hit(gold_paths, [r]):
            return 1.0 / i
    return 0.0


def ndcg_at_k(gold_paths: list[str], returned_paths: list[str], k: int) -> float:
    """Binary-relevance nDCG@k: each returned path is relevant (1) if it tail-hits any gold
    path, else 0. Ideal ranking assumes min(len(gold_paths), k) relevant docs at the top."""
    rel = [1.0 if tail_hit(gold_paths, [r]) else 0.0 for r in returned_paths[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal_count = min(len(gold_paths), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


# --------------------------------------------------------------------------- model dispatch


class ModelClient(Protocol):
    name: str

    def retrieve(self, query: str, limit: int) -> tuple[list[str], float]:
        """Return (ordered source_paths, latency_ms). Raise InfraError on infra failure."""
        ...

    def cost_usd(self) -> float:
        """Cost of the calls made so far, in USD. 0.0 for calls with no per-request billing."""
        ...


class BGEM3Client:
    """Live dispatch to Scrutator's hybrid `/v1/search` (dense+sparse RRF). `rerank` is
    intentionally never sent — SRCH-0029's ColBERT stage is a no-op today (see module
    docstring); sending `rerank=true` here would silently duplicate this same measurement
    under a second label."""

    name = "bge-m3"

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        namespace: str = DEFAULT_NAMESPACE,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        self.endpoint = endpoint
        self.namespace = namespace
        self.timeout = timeout

    def retrieve(self, query: str, limit: int) -> tuple[list[str], float]:
        body = {
            "query": query,
            "namespace": self.namespace,
            "limit": limit,
            "min_score": 0.0,
            "include_content": False,
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        t0 = time.time()
        try:
            # nosec B310 — endpoint is a fixed mesh-internal http:// URL (config/CLI-supplied
            # by the operator, same pattern as measure.py), never attacker-controlled input.
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                out = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise InfraError(f"bge-m3 request failed: {exc}") from exc
        dt_ms = (time.time() - t0) * 1000
        paths = [r["source_path"] for r in out.get("results", [])]
        return paths, dt_ms

    def cost_usd(self) -> float:
        return 0.0  # mesh-internal embedding call, no per-request billing


class NotWiredClient:
    """Placeholder for a model with no live dispatch path in this task's scope.

    See module docstring's "Scope decision" note. `build_client()` returns this for
    `bge-reranker` and any `llm:*` alias; `retrieve()` always raises `NotImplementedError`.
    """

    def __init__(self, name: str):
        self.name = name

    def retrieve(self, query: str, limit: int) -> tuple[list[str], float]:
        raise NotImplementedError(
            f"model '{self.name}' has no live dispatch path yet — operator-gated per "
            "SRCH-0015 scope, see benchmark/scrutator/README.md"
        )

    def cost_usd(self) -> float:
        return 0.0


def build_client(model_name: str, *, endpoint: str, namespace: str) -> ModelClient:
    if model_name == "bge-m3":
        return BGEM3Client(endpoint=endpoint, namespace=namespace)
    if model_name == "bge-reranker" or model_name.startswith("llm:"):
        return NotWiredClient(model_name)
    raise ValueError(f"unknown model: {model_name}")


# --------------------------------------------------------------------------- row/model scoring


@dataclass
class RowResult:
    row_id: str
    cls: str
    hit_at_1: bool
    hit_at_5: bool
    hit_at_10: bool
    rr: float
    ndcg_at_5: float
    latency_ms: float


def score_row(client: ModelClient, row: GoldenRow) -> RowResult:
    """Score one (already-liveness-filtered) golden row against a model client.

    May raise InfraError, propagated by the caller as a run-level infra failure.
    """
    paths, latency_ms = client.retrieve(row.query, 10)
    return RowResult(
        row_id=row.id,
        cls=row.cls,
        hit_at_1=tail_hit(row.gold_source_paths, paths[:1]),
        hit_at_5=tail_hit(row.gold_source_paths, paths[:5]),
        hit_at_10=tail_hit(row.gold_source_paths, paths[:10]),
        rr=reciprocal_rank(row.gold_source_paths, paths),
        ndcg_at_5=ndcg_at_k(row.gold_source_paths, paths, k=5),
        latency_ms=latency_ms,
    )


def _aggregate_class(rows: list[RowResult], cost_usd: float) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "cost_usd": 0.0,
        }
    lat_sorted = sorted(r.latency_ms for r in rows)
    return {
        "n": n,
        "recall@1": sum(1 for r in rows if r.hit_at_1) / n,
        "recall@5": sum(1 for r in rows if r.hit_at_5) / n,
        "recall@10": sum(1 for r in rows if r.hit_at_10) / n,
        "mrr": sum(r.rr for r in rows) / n,
        "ndcg@5": sum(r.ndcg_at_5 for r in rows) / n,
        "latency_p50_ms": round(statistics.median(lat_sorted), 1),
        "latency_p95_ms": round(_percentile(lat_sorted, 0.95), 1),
        "cost_usd": cost_usd,
    }


def aggregate(results: list[RowResult], cost_usd: float = 0.0) -> dict:
    """Per-class + overall aggregation. Does not include stale_skipped (a run-level,
    not per-model, quantity — see main())."""
    by_class = {
        cls: _aggregate_class([r for r in results if r.cls == cls], cost_usd=0.0)
        for cls in sorted({r.cls for r in results})
    }
    by_class["overall"] = _aggregate_class(results, cost_usd=cost_usd)
    return by_class


def run_model(
    model_name: str, live_rows: list[GoldenRow], *, endpoint: str, namespace: str
) -> tuple[dict, list[RowResult]]:
    """Run one model over all live rows. Raises InfraError on the first infra failure.

    Returns (per-class + overall summary dict, per-row detail results).
    """
    client = build_client(model_name, endpoint=endpoint, namespace=namespace)
    results = [score_row(client, row) for row in live_rows]
    return aggregate(results, cost_usd=client.cost_usd()), results


# --------------------------------------------------------------------------- exit-code verdict


def decide_exit_code(summaries: dict[str, dict], threshold: float, metric: str = "recall@5") -> tuple[int, str]:
    """V-AC-02a/02b verdict: any model's overall metric below threshold ⇒ THRESHOLD_FAIL_CODE."""
    for summary in summaries.values():
        if summary.get("overall", {}).get(metric, 0.0) < threshold:
            return THRESHOLD_FAIL_CODE, "threshold_fail"
    return SUCCESS_CODE, "ok"


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SRCH-0015 Scrutator /v1/search recall harness")
    p.add_argument("--golden", default=None, help="path to golden-arcanada-v{N}.jsonl (required unless --dry-run-*)")
    p.add_argument(
        "--models",
        default="bge-m3",
        help="comma-separated model names, e.g. bge-m3,bge-reranker,llm:gpt-4o-mini",
    )
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--corpus-root", default=".", help="repo root gold_source_paths are relative to")
    p.add_argument("--out-summary", default="results-summary.json")
    p.add_argument("--out-detail", default="results-detail.json")
    p.add_argument(
        "--recall-threshold",
        type=float,
        default=DEFAULT_RECALL_THRESHOLD,
        help="minimum overall recall@5 required for exit 0 (default: SRCH-0031 baseline lower bound)",
    )
    p.add_argument(
        "--dry-run-infra-fail",
        action="store_true",
        help="simulate a mocked Scrutator connection error; exits INFRA_FAIL_CODE without any network call",
    )
    p.add_argument(
        "--dry-run-threshold-fail",
        type=float,
        default=None,
        metavar="RECALL",
        help="simulate a low-recall-but-successful run at the given recall@5; exits THRESHOLD_FAIL_CODE",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.dry_run_infra_fail:
        print("DRY-RUN: simulating infra failure (mocked Scrutator connection error)", file=sys.stderr)
        return INFRA_FAIL_CODE

    if args.dry_run_threshold_fail is not None:
        print(f"DRY-RUN: simulating threshold failure at recall@5={args.dry_run_threshold_fail}", file=sys.stderr)
        return THRESHOLD_FAIL_CODE

    if not args.golden:
        build_arg_parser().error(
            "--golden is required unless --dry-run-infra-fail or --dry-run-threshold-fail is passed"
        )

    rows = load_golden_rows(args.golden)
    corpus_root = Path(args.corpus_root)
    live_rows, stale_rows = partition_by_liveness(rows, corpus_root)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    summaries: dict[str, dict] = {}
    details: dict[str, list[dict]] = {}
    try:
        for model_name in model_names:
            summary, row_results = run_model(model_name, live_rows, endpoint=args.endpoint, namespace=args.namespace)
            summaries[model_name] = summary
            details[model_name] = [asdict(r) for r in row_results]
    except InfraError as exc:
        print(f"INFRA FAILURE: {exc}", file=sys.stderr)
        return INFRA_FAIL_CODE

    exit_code, exit_reason = decide_exit_code(summaries, args.recall_threshold)

    out = {
        "stale_skipped": len(stale_rows),
        "exit_reason": exit_reason,
        "models": summaries,
    }
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(args.out_detail, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
