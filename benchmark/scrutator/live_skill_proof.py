#!/usr/bin/env python3
"""Live proof contract for the skills-kb-discovery-probe artifact (INFRA-0365 AC-2).

Validates the exact 695-byte promoted skill plan against the committed promotion
registry.  Produces redacted evidence only — no content, no tokens, no paths beyond
the fixed registry identity.  Fail-closed: any mismatch, missing grant, or unexpected
response shape is invalid evidence, never a silent pass.

Stdlib-only by design — runs on a bare self-hosted runner without Scrutator's
dependency set (same pattern as ``rerank_gate.py`` and ``harness.py``).

Exit codes:
  0 — proof valid (every contract check passed)
  1 — contract mismatch (wrong hash, wrong byte count, wrong namespace/maturity)
  2 — invalid evidence / infrastructure (self-consistency failure, non-loopback
      endpoint, transport error, unexpected response shape)
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

# ── Pinned from src/scrutator/search/skill_promotions.json (repository-tracked) ──

PROBE_SOURCE_PATH = "skills/skills-kb-discovery-probe.json"
PROBE_CONTENT_HASH = (
    "sha256:a568ac9631c86ff90fc4bf5b893f70a7113cbf851e0991c59acb16d087165633"
)
PROBE_BYTE_COUNT = 695
PROMOTED_MATURITY = "production"
REQUIRED_NAMESPACE = "skills"
PROBE_SEARCH_QUERY = PROBE_SOURCE_PATH

# Derived: doc_id = first 16 hex chars of sha256("skills|skills/skills-kb-discovery-probe.json")
_PROBE_DOC_ID = hashlib.sha256(
    f"{REQUIRED_NAMESPACE}|{PROBE_SOURCE_PATH}".encode()
).hexdigest()[:16]

SUCCESS_CODE = 0
CONTRACT_MISMATCH_CODE = 1
INVALID_EVIDENCE_CODE = 2

# ───────────────────────────────────────────────────────────────────────────────────
# BLOCKED — INFRA-0365 AC-2 production-mutation verification
#
# Requires the inherited explicit proof checkpoint + a dedicated feeder credential.
# To verify: POST /v1/index with a mutated plan (e.g. altered maturity or injected
# marker in a semantic field) → expect 422 rejection with "Unsafe skill semantic
# content" or a SkillPlanContractError reason.  This remains checkpoint-gated and
# is never callable by this read-only runner.
#
# def _verify_production_write_rejection(endpoint: str, timeout_s: float) -> None:
#     '''POST a deliberately broken skill plan and confirm 422 rejection.'''
#     raise NotImplementedError(
#         "BLOCKED — requires the inherited explicit proof checkpoint + feeder credential"
#     )
# ───────────────────────────────────────────────────────────────────────────────────


class InvalidEvidence(RuntimeError):
    """The live proof did not establish the required contract."""


class ContractMismatch(InvalidEvidence):
    """The service responded, but a pinned proof field did not match."""


# ── HTTP helpers (mirrors rerank_gate.py) ────────────────────────────────────────


def _validate_endpoint(url: str) -> str:
    """Reject non-loopback endpoints before any request is made."""
    allowed_prefixes = ("http://127.0.0.1:", "http://localhost:")
    if not url.startswith(allowed_prefixes):
        raise InvalidEvidence(
            f"benchmark endpoint must be loopback-only: {url}"
        )
    return url.rstrip("/")


def _request_json(
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> tuple[Any, float]:
    """Make an HTTP request; return (parsed JSON body, client latency ms).

    Only GET (body=None) and POST (body=dict) are supported — this proof never
    writes to the endpoint (no PUT / PATCH / DELETE).
    """
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    parsed = urlsplit(url)
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.scheme != "http":
        raise InvalidEvidence(
            f"request URL must be loopback HTTP: {url}"
        )
    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port, timeout=timeout_s
    )
    started = time.monotonic()
    try:
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        method = "POST" if body is not None else "GET"
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise InvalidEvidence(f"request returned HTTP {response.status} for {url}")
        payload = json.loads(raw)
    except (
        TimeoutError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        http.client.HTTPException,
    ) as exc:
        raise InvalidEvidence(f"request failed for {url}: {exc}") from exc
    finally:
        connection.close()
    return payload, (time.monotonic() - started) * 1000


# ── Self-consistency: verify the tracked registry matches pinned constants ────────


def _registry_path() -> Path:
    """Resolve the promotion registry relative to this script."""
    script_dir = Path(__file__).resolve().parent
    return script_dir / ".." / ".." / "src" / "scrutator" / "search" / "skill_promotions.json"


def _check_registry_self_consistency() -> None:
    """Load the committed promotion registry and verify it matches pinned constants.

    Any mismatch means this script is stale / corrupted — the pinned constants
    and the committed registry must be identical.
    """
    path = _registry_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidEvidence(
            f"promotion registry not found at {path}: {exc}"
        ) from exc

    try:
        registry = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidEvidence(
            f"promotion registry is not valid JSON: {exc}"
        ) from exc

    if registry.get("schema_version") != 1:
        raise InvalidEvidence(
            "promotion registry schema_version is not 1"
        )

    promotions = registry.get("promotions")
    if not isinstance(promotions, list) or len(promotions) != 1:
        raise InvalidEvidence(
            "promotion registry must contain exactly one entry"
        )

    entry = promotions[0]
    if not isinstance(entry, dict):
        raise InvalidEvidence("promotion registry entry is not an object")

    actual = {
        "source_path": entry.get("source_path"),
        "content_hash": entry.get("content_hash"),
        "maturity": entry.get("maturity"),
    }
    expected = {
        "source_path": PROBE_SOURCE_PATH,
        "content_hash": PROBE_CONTENT_HASH,
        "maturity": PROMOTED_MATURITY,
    }
    if actual != expected:
        raise InvalidEvidence(
            f"promotion registry entry does not match pinned constants: "
            f"actual={actual} expected={expected}"
        )


# ── Contract checks against the live endpoint ────────────────────────────────────


def _check_health(endpoint: str, timeout_s: float) -> dict[str, Any]:
    """Confirm the service is Scrutator and record its version."""
    payload, _ = _request_json(f"{endpoint}/health", timeout_s=timeout_s)
    if not isinstance(payload, dict):
        raise InvalidEvidence("/health did not return an object")
    service = payload.get("service", "")
    version = payload.get("version", "")
    if service != "Scrutator":
        raise InvalidEvidence(f"health endpoint reports service={service!r}, expected Scrutator")
    return {"scrutator_version": version, "service": service}


def _check_namespace_grants(endpoint: str, timeout_s: float) -> list[str]:
    """Retrieve the authorized namespace list and confirm it includes 'skills'.

    A missing 'skills' namespace means the benchmark principal lacks the grant
    required to fetch the probe — fail-closed, never silently skip.
    """
    payload, _ = _request_json(f"{endpoint}/v1/namespaces", timeout_s=timeout_s)
    if not isinstance(payload, list):
        raise InvalidEvidence("/v1/namespaces did not return a list")
    names = [ns.get("name") if isinstance(ns, dict) else str(ns) for ns in payload]
    if any(not isinstance(name, str) or not name for name in names):
        raise InvalidEvidence("/v1/namespaces contained an invalid namespace name")
    if REQUIRED_NAMESPACE not in names:
        raise InvalidEvidence(
            f"'skills' namespace not in authorized list: {sorted(names)}"
        )
    return names


def _check_probe_search(endpoint: str, timeout_s: float) -> dict[str, Any]:
    """Prove the pinned artifact is visible through the read-only search index."""
    payload, latency_ms = _request_json(
        f"{endpoint}/v1/search",
        body={
            "query": PROBE_SEARCH_QUERY,
            "namespace": REQUIRED_NAMESPACE,
            "limit": 10,
            "min_score": 0.0,
            "include_content": False,
            "maturity": PROMOTED_MATURITY,
        },
        timeout_s=timeout_s,
    )
    if not isinstance(payload, dict):
        raise InvalidEvidence("/v1/search did not return an object")
    results = payload.get("results")
    if not isinstance(results, list):
        raise InvalidEvidence("/v1/search did not return a results list")

    matching = [
        result
        for result in results
        if isinstance(result, dict) and result.get("source_path") == PROBE_SOURCE_PATH
    ]
    if not matching:
        raise ContractMismatch("search did not return the pinned probe path")

    result = matching[0]
    if result.get("source_id") != _PROBE_DOC_ID:
        raise ContractMismatch("search returned an unexpected probe document id")
    if result.get("namespace") != REQUIRED_NAMESPACE:
        raise ContractMismatch("search returned an unexpected probe namespace")
    if result.get("content_hash") != PROBE_CONTENT_HASH:
        raise ContractMismatch("search returned an unexpected probe content hash")

    return {
        "search_verified": True,
        "search_result_count": len(results),
        "search_latency_ms": round(latency_ms, 3),
    }


def _check_probe_discovery(endpoint: str, timeout_s: float) -> dict[str, Any]:
    """Prove the pinned artifact is discoverable through read-only navigation."""
    query = urlencode(
        {
            "namespace": REQUIRED_NAMESPACE,
            "source_path": PROBE_SOURCE_PATH,
            "max_nodes": 2000,
        }
    )
    payload, latency_ms = _request_json(
        f"{endpoint}/v1/navigate/outline?{query}", timeout_s=timeout_s
    )
    if not isinstance(payload, dict):
        raise InvalidEvidence("/v1/navigate/outline did not return an object")
    if payload.get("source_path") != PROBE_SOURCE_PATH:
        raise ContractMismatch("navigation returned an unexpected probe path")
    if payload.get("namespace") != REQUIRED_NAMESPACE:
        raise ContractMismatch("navigation returned an unexpected probe namespace")
    if payload.get("doc_id") != _PROBE_DOC_ID:
        raise ContractMismatch("navigation returned an unexpected probe document id")
    total_chunks = payload.get("total_chunks")
    if not isinstance(total_chunks, int) or total_chunks < 1:
        raise ContractMismatch("navigation did not expose an indexed probe document")
    outline = payload.get("outline")
    if not isinstance(outline, list) or not outline:
        raise ContractMismatch("navigation returned no probe outline")

    return {
        "discovery_verified": True,
        "discovery_total_chunks": total_chunks,
        "discovery_latency_ms": round(latency_ms, 3),
    }


def _check_probe_fetch(endpoint: str, timeout_s: float) -> dict[str, Any]:
    """Fetch the probe by its opaque doc_id and validate every contract field."""
    body = {
        "by": "source_id",
        "id": _PROBE_DOC_ID,
        "range": "full",
        "include": ["content", "provenance"],
    }
    payload, latency_ms = _request_json(
        f"{endpoint}/v1/fetch", body=body, timeout_s=timeout_s
    )

    # Validate response shape — every required field must be present and correct
    errors: list[str] = []

    actual_hash = payload.get("content_hash")
    if actual_hash != PROBE_CONTENT_HASH:
        errors.append(
            f"content_hash mismatch: got {actual_hash!r}, "
            f"expected {PROBE_CONTENT_HASH!r}"
        )

    content = payload.get("content")
    if not isinstance(content, str):
        errors.append("content is not a string")
    else:
        actual_bytes = len(content.encode("utf-8"))
        if actual_bytes != PROBE_BYTE_COUNT:
            errors.append(
                f"byte count mismatch: got {actual_bytes}, "
                f"expected {PROBE_BYTE_COUNT}"
            )

    actual_ns = payload.get("namespace")
    if actual_ns != REQUIRED_NAMESPACE:
        errors.append(
            f"namespace mismatch: got {actual_ns!r}, "
            f"expected {REQUIRED_NAMESPACE!r}"
        )

    actual_trust_class = payload.get("trust_class")
    if actual_trust_class != "skill":
        errors.append(
            f"trust_class mismatch: got {actual_trust_class!r}, expected 'skill'"
        )

    actual_exact = payload.get("content_exact")
    if actual_exact is not True:
        errors.append(
            f"content_exact is {actual_exact!r}, expected True"
        )

    actual_path = payload.get("path")
    if actual_path != PROBE_SOURCE_PATH:
        errors.append(
            f"path mismatch: got {actual_path!r}, "
            f"expected {PROBE_SOURCE_PATH!r}"
        )

    if errors:
        raise ContractMismatch("probe fetch contract violations: " + "; ".join(errors))

    return {
        "content_hash_verified": True,
        "byte_count_verified": PROBE_BYTE_COUNT,
        "namespace_verified": REQUIRED_NAMESPACE,
        "trust_class_verified": "skill",
        "content_exact_verified": True,
        "path_verified": PROBE_SOURCE_PATH,
        "doc_id_resolved": _PROBE_DOC_ID,
        "fetch_latency_ms": round(latency_ms, 3),
    }


# ── Evidence output ──────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: Any) -> None:
    """Atomic JSON write: tmp file + rename (mirrors rerank_gate.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _build_redacted_evidence(
    endpoint: str,
    health: dict[str, Any],
    namespace_names: list[str],
    search_result: dict[str, Any],
    discovery_result: dict[str, Any],
    fetch_result: dict[str, Any],
) -> dict[str, Any]:
    """Assemble redacted evidence — no content, no tokens, no raw paths."""
    return {
        "schema": "scrutator-skill-proof/1",
        "probe_source_path": PROBE_SOURCE_PATH,
        "probe_content_hash_verified": fetch_result["content_hash_verified"],
        "probe_byte_count_verified": fetch_result["byte_count_verified"],
        "promotion_maturity": PROMOTED_MATURITY,
        "namespace_verified": fetch_result["namespace_verified"],
        "trust_class_verified": fetch_result["trust_class_verified"],
        "content_exact_verified": fetch_result["content_exact_verified"],
        "doc_id_resolved": fetch_result["doc_id_resolved"],
        "registry_self_consistent": True,
        "scrutator_version": health["scrutator_version"],
        "authorized_namespaces": sorted(namespace_names),
        "search_verified": search_result["search_verified"],
        "search_result_count": search_result["search_result_count"],
        "search_latency_ms": search_result["search_latency_ms"],
        "discovery_verified": discovery_result["discovery_verified"],
        "discovery_total_chunks": discovery_result["discovery_total_chunks"],
        "discovery_latency_ms": discovery_result["discovery_latency_ms"],
        "fetch_latency_ms": fetch_result["fetch_latency_ms"],
        "endpoint": "loopback",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": {"status": "PROOF_VALID", "reasons": []},
    }


# ── CLI ──────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live proof contract for the skills-kb-discovery-probe artifact"
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="loopback Scrutator endpoint, e.g. http://127.0.0.1:8310",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="directory for redacted evidence output",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate self-consistency and exit-code taxonomy only; no network calls",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Self-consistency always runs first — a stale script must never reach the network.
    try:
        _check_registry_self_consistency()
    except InvalidEvidence as exc:
        print(f"INVALID EVIDENCE: {exc}", file=sys.stderr)
        return INVALID_EVIDENCE_CODE

    if args.dry_run:
        print("DRY-RUN: registry self-consistency passed; skipping network calls.")
        return SUCCESS_CODE

    out_dir = Path(args.out_dir)
    timeout_s = args.timeout

    try:
        endpoint = _validate_endpoint(args.endpoint)
        health = _check_health(endpoint, timeout_s)
        namespace_names = _check_namespace_grants(endpoint, timeout_s)
        search_result = _check_probe_search(endpoint, timeout_s)
        discovery_result = _check_probe_discovery(endpoint, timeout_s)
        fetch_result = _check_probe_fetch(endpoint, timeout_s)
    except ContractMismatch as exc:
        print(f"CONTRACT MISMATCH: {exc}", file=sys.stderr)
        return CONTRACT_MISMATCH_CODE
    except InvalidEvidence as exc:
        print(f"INVALID EVIDENCE: {exc}", file=sys.stderr)
        return INVALID_EVIDENCE_CODE

    evidence = _build_redacted_evidence(
        endpoint,
        health,
        namespace_names,
        search_result,
        discovery_result,
        fetch_result,
    )

    # Contract-mismatch verdict: any individual check failure is already raised
    # as InvalidEvidence above.  If we reach here, every check passed.
    _write_json(out_dir / "summary.json", evidence)
    print(json.dumps(evidence["verdict"], sort_keys=True))
    return SUCCESS_CODE


if __name__ == "__main__":
    sys.exit(main())
