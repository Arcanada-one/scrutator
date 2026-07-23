#!/usr/bin/env python3
"""Paired `/v1/search` recall gate for SRCH-0031.

The runner compares two already-running endpoints that differ only in the
process-global `SCRUTATOR_RERANK_ENABLED` setting.  It fails closed when the
ColBERT treatment is not proven by the response citation contract.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SUCCESS_CODE = 0
QUALITY_FAIL_CODE = 1
INVALID_EVIDENCE_CODE = 2

EXPECTED_CLASS_COUNTS = {"factual": 15, "multi-hop": 8, "temporal": 10}
HISTORICAL_CLASS_FLOORS = {"factual": 14, "multi-hop": 8, "temporal": 8}
DEFAULT_LATENCY_P95_BUDGET_MS = 5000.0


class InvalidEvidence(RuntimeError):
    """The experiment did not exercise the declared treatment cleanly."""


def _path_matches_gold(gold_path: str, returned_path: str) -> bool:
    """Match a repo-relative gold path to an exact or absolute-rooted result."""
    gold = gold_path.replace("\\", "/").lstrip("./")
    returned = returned_path.replace("\\", "/").lstrip("./")
    return returned == gold or returned.endswith(f"/{gold}")


def gold_hit(gold_paths: list[str], returned_paths: list[str]) -> bool:
    return any(_path_matches_gold(gold, returned) for gold in gold_paths for returned in returned_paths)


def all_gold_hit(gold_paths: list[str], returned_paths: list[str]) -> bool:
    return all(any(_path_matches_gold(gold, returned) for returned in returned_paths) for gold in gold_paths)


def validate_response(payload: dict[str, Any], *, expected_score_kind: str) -> list[dict[str, Any]]:
    """Validate that one response proves the expected ranking implementation."""
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 5:
        raise InvalidEvidence("response must contain exactly 5 results")

    chunk_ids: list[str] = []
    scores: list[float] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise InvalidEvidence(f"result {index} is not an object")
        chunk_id = result.get("chunk_id")
        source_path = result.get("source_path")
        score = result.get("score")
        citation = result.get("citation")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise InvalidEvidence(f"result {index} has no chunk_id")
        if not isinstance(source_path, str) or not source_path:
            raise InvalidEvidence(f"result {index} has no source_path")
        if not isinstance(score, int | float) or not math.isfinite(float(score)):
            raise InvalidEvidence(f"result {index} has invalid score")
        if not isinstance(citation, dict):
            raise InvalidEvidence(f"result {index} has no citation")
        if citation.get("score_kind") != expected_score_kind:
            raise InvalidEvidence(
                f"result {index} score_kind={citation.get('score_kind')!r}, expected {expected_score_kind!r}"
            )
        if citation.get("chunk_id") != chunk_id or citation.get("source_path") != source_path:
            raise InvalidEvidence(f"result {index} citation identity mismatch")
        chunk_ids.append(chunk_id)
        scores.append(float(score))

    if len(set(chunk_ids)) != len(chunk_ids):
        raise InvalidEvidence("response contains duplicate chunk_id values")
    if any(left == right for left, right in zip(scores, scores[1:], strict=False)):
        raise InvalidEvidence("observed score tie makes rank ordering ambiguous")
    return results


def _row_hit(row: dict[str, Any], mode: str) -> bool:
    explicit = row.get(f"{mode}_hit")
    if isinstance(explicit, bool):
        return explicit
    return gold_hit(row["gold_source_paths"], row[f"{mode}_paths"])


def _row_all_gold(row: dict[str, Any], mode: str) -> bool:
    explicit = row.get(f"{mode}_all_gold")
    if isinstance(explicit, bool):
        return explicit
    return all_gold_hit(row["gold_source_paths"], row[f"{mode}_paths"])


def summarize_transitions(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        cls = row["class"]
        bucket = summary.setdefault(
            cls,
            {
                "n": 0,
                "off_hits": 0,
                "on_hits": 0,
                "gains": 0,
                "losses": 0,
                "off_all_gold": 0,
                "on_all_gold": 0,
            },
        )
        off_hit = _row_hit(row, "off")
        on_hit = _row_hit(row, "on")
        off_all = _row_all_gold(row, "off")
        on_all = _row_all_gold(row, "on")
        bucket["n"] += 1
        bucket["off_hits"] += int(off_hit)
        bucket["on_hits"] += int(on_hit)
        bucket["gains"] += int(not off_hit and on_hit)
        bucket["losses"] += int(off_hit and not on_hit)
        bucket["off_all_gold"] += int(off_all)
        bucket["on_all_gold"] += int(on_all)
    return summary


def decide_verdict(
    transitions: dict[str, dict[str, int]],
    *,
    on_latency_p95_ms: float,
    latency_budget_ms: float = DEFAULT_LATENCY_P95_BUDGET_MS,
) -> dict[str, Any]:
    reasons: list[str] = []
    total_gains = 0
    for cls, expected_n in EXPECTED_CLASS_COUNTS.items():
        bucket = transitions.get(cls)
        if bucket is None:
            reasons.append(f"{cls}: class missing")
            continue
        if bucket["n"] != expected_n:
            reasons.append(f"{cls}: N={bucket['n']} expected {expected_n}")
        if bucket["losses"] > 0:
            reasons.append(f"{cls}: {bucket['losses']} paired loss(es)")
        if bucket["on_hits"] < HISTORICAL_CLASS_FLOORS[cls]:
            reasons.append(f"{cls}: ON hits {bucket['on_hits']} below historical floor {HISTORICAL_CLASS_FLOORS[cls]}")
        if cls == "multi-hop" and bucket["on_all_gold"] < bucket["off_all_gold"]:
            reasons.append(f"{cls}: all-gold retrieval regressed {bucket['off_all_gold']}->{bucket['on_all_gold']}")
        total_gains += bucket["gains"]

    if total_gains == 0:
        reasons.append("no paired gain")
    if on_latency_p95_ms > latency_budget_ms:
        reasons.append(f"ON p95 {on_latency_p95_ms:.1f} ms exceeds {latency_budget_ms:.1f} ms budget")

    return {"status": "KEEP_OFF" if reasons else "ELIGIBLE_TO_FLIP", "reasons": reasons}


def require_repeat_stability(repetitions: list[list[dict[str, Any]]]) -> None:
    if not repetitions:
        raise InvalidEvidence("no repetitions recorded")
    baseline = {
        row["id"]: (row["off_paths"], row["on_paths"], _row_hit(row, "off"), _row_hit(row, "on"))
        for row in repetitions[0]
    }
    for repeat_index, rows in enumerate(repetitions[1:], start=2):
        current = {
            row["id"]: (row["off_paths"], row["on_paths"], _row_hit(row, "off"), _row_hit(row, "on")) for row in rows
        }
        if current != baseline:
            raise InvalidEvidence(f"repeat {repeat_index} is unstable")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise InvalidEvidence("no latency observations")
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _load_golden(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InvalidEvidence(f"{path}:{line_number}: invalid JSON") from exc
        missing = {"id", "class", "query", "gold_source_paths"} - row.keys()
        if missing:
            raise InvalidEvidence(f"{path}:{line_number}: missing {sorted(missing)}")
        rows.append(row)

    counts = {cls: sum(row["class"] == cls for row in rows) for cls in EXPECTED_CLASS_COUNTS}
    if counts != EXPECTED_CLASS_COUNTS:
        raise InvalidEvidence(f"golden class counts {counts}, expected {EXPECTED_CLASS_COUNTS}")
    if len({row["id"] for row in rows}) != len(rows):
        raise InvalidEvidence("golden contains duplicate ids")
    return rows


def _request_json(url: str, *, body: dict[str, Any] | None, timeout_s: float) -> tuple[dict[str, Any], float]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    parsed = urlsplit(url)
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.scheme != "http" or parsed.port is None:
        raise InvalidEvidence(f"request URL must be loopback HTTP with an explicit port: {url}")
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
    started = time.monotonic()
    try:
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        connection.request("POST" if body is not None else "GET", path, body=data, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise InvalidEvidence(f"request returned HTTP {response.status} for {url}: {raw[:500]!r}")
        payload = json.loads(raw)
    except (TimeoutError, OSError, json.JSONDecodeError, http.client.HTTPException) as exc:
        raise InvalidEvidence(f"request failed for {url}: {exc}") from exc
    finally:
        connection.close()
    return payload, (time.monotonic() - started) * 1000


def _validate_endpoint(url: str) -> str:
    allowed_prefixes = ("http://127.0.0.1:", "http://localhost:")
    if not url.startswith(allowed_prefixes):
        raise InvalidEvidence(f"benchmark endpoint must be loopback-only: {url}")
    return url.rstrip("/")


def _measure_one(
    *,
    endpoint: str,
    row: dict[str, Any],
    expected_score_kind: str,
    timeout_s: float,
) -> dict[str, Any]:
    payload, client_latency_ms = _request_json(
        f"{endpoint}/v1/search",
        body={
            "query": row["query"],
            "namespace": "arcanada",
            "limit": 5,
            "min_score": 0.0,
            "include_content": False,
        },
        timeout_s=timeout_s,
    )
    results = validate_response(payload, expected_score_kind=expected_score_kind)
    paths = [result["source_path"] for result in results]
    return {
        "paths": paths,
        "chunk_ids": [result["chunk_id"] for result in results],
        "scores": [result["score"] for result in results],
        "score_kinds": [result["citation"]["score_kind"] for result in results],
        "hit": gold_hit(row["gold_source_paths"], paths),
        "all_gold": all_gold_hit(row["gold_source_paths"], paths),
        "server_latency_ms": payload.get("search_time_ms"),
        "client_latency_ms": round(client_latency_ms, 3),
    }


def _fingerprint(endpoint: str, timeout_s: float) -> dict[str, Any]:
    payload, _ = _request_json(f"{endpoint}/__benchmark/fingerprint", body=None, timeout_s=timeout_s)
    required = {"namespace", "namespace_id", "chunk_count", "sha256"}
    if required - payload.keys():
        raise InvalidEvidence(f"fingerprint missing {sorted(required - payload.keys())}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_experiment(args: argparse.Namespace) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    off_endpoint = _validate_endpoint(args.off_endpoint)
    on_endpoint = _validate_endpoint(args.on_endpoint)
    golden_path = Path(args.golden)
    golden = _load_golden(golden_path)
    repetitions: list[list[dict[str, Any]]] = []
    fingerprints: list[dict[str, Any]] = []
    on_latencies: list[float] = []

    for repeat_index in range(args.repeats):
        before_off = _fingerprint(off_endpoint, args.timeout)
        before_on = _fingerprint(on_endpoint, args.timeout)
        if before_off != before_on:
            raise InvalidEvidence(f"repeat {repeat_index + 1}: OFF/ON fingerprints differ before run")

        measured_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(golden):
            order = ("off", "on") if (repeat_index + row_index) % 2 == 0 else ("on", "off")
            observations: dict[str, dict[str, Any]] = {}
            for mode in order:
                endpoint = off_endpoint if mode == "off" else on_endpoint
                expected_kind = "rrf" if mode == "off" else "colbert_rerank"
                observations[mode] = _measure_one(
                    endpoint=endpoint,
                    row=row,
                    expected_score_kind=expected_kind,
                    timeout_s=args.timeout,
                )
                if mode == "on":
                    on_latencies.append(observations[mode]["client_latency_ms"])

            measured_rows.append(
                {
                    "id": row["id"],
                    "class": row["class"],
                    "query": row["query"],
                    "gold_source_paths": row["gold_source_paths"],
                    "request_order": list(order),
                    **{f"off_{key}": value for key, value in observations["off"].items()},
                    **{f"on_{key}": value for key, value in observations["on"].items()},
                }
            )

        after_off = _fingerprint(off_endpoint, args.timeout)
        after_on = _fingerprint(on_endpoint, args.timeout)
        if before_off != after_off or before_on != after_on or after_off != after_on:
            raise InvalidEvidence(f"repeat {repeat_index + 1}: corpus fingerprint changed")
        fingerprints.append(before_off)
        repetitions.append(measured_rows)
        _write_json(Path(args.out_dir) / f"repeat-{repeat_index + 1}.json", measured_rows)

    require_repeat_stability(repetitions)
    transitions = summarize_transitions(repetitions[0])
    on_p95 = round(_percentile(on_latencies, 0.95), 3)
    verdict = decide_verdict(
        transitions,
        on_latency_p95_ms=on_p95,
        latency_budget_ms=args.latency_p95_budget_ms,
    )
    summary = {
        "schema": "scrutator-search-rerank-gate/1",
        "golden_path": str(golden_path),
        "golden_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "fingerprints": fingerprints,
        "transitions": transitions,
        "on_latency_p50_ms": round(statistics.median(on_latencies), 3),
        "on_latency_p95_ms": on_p95,
        "latency_p95_budget_ms": args.latency_p95_budget_ms,
        "verdict": verdict,
    }
    return summary, repetitions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SRCH-0031 paired /v1/search rerank gate")
    parser.add_argument("--off-endpoint", required=True)
    parser.add_argument("--on-endpoint", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--latency-p95-budget-ms", type=float, default=DEFAULT_LATENCY_P95_BUDGET_MS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.repeats < 2:
            raise InvalidEvidence("at least two repetitions are required")
        summary, _ = run_experiment(args)
        _write_json(Path(args.out_dir) / "summary.json", summary)
    except InvalidEvidence as exc:
        print(f"INVALID EVIDENCE: {exc}", file=sys.stderr)
        return INVALID_EVIDENCE_CODE

    print(json.dumps(summary["verdict"], sort_keys=True))
    return SUCCESS_CODE if summary["verdict"]["status"] == "ELIGIBLE_TO_FLIP" else QUALITY_FAIL_CODE


if __name__ == "__main__":
    sys.exit(main())
