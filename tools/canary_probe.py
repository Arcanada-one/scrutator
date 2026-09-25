#!/usr/bin/env python3
"""CanaryPlan/v1 -> CanaryResult/v1 against a live contour (AUP-GRAPH-008 rule C1..C5).

WHY THIS FILE EXISTS RATHER THAN A CALL TO THE PROGRAM'S OWN TOOL. `tools/graph/deploy_gate.py
canary` is the producer named by the mandate, and it is unreachable from here: the program
repository is private and the vendored GraphAdmissionBundle carries only the tools the GATE needs
(`ci_gate.py:49 BUNDLE_FILES` — admit_change, verify, contract_diff, build_graph, impact,
canary_evidence, process_observation, schema_check, sshsig; no deploy_gate). What travels is the
DOCUMENT contract, and that is what is honoured here: every result this writes is handed to the
bundled `canary_evidence.consume` — the gate's own reader — in `tests/test_canary_probe.py`, so the
shape is checked by the real consumer and not by a fixture of ours (A2-287).

One deliberate superset of the program's producer: a probe may carry a `body`. The negative probe
this repository needs is an UNAUTHENTICATED `POST /v1/edges` with `[]`, which must answer 403 once
A2-308's per-route scope enforcement is live; `deploy_gate.http_probe` sends no request body at
all. Reported as a gap, implemented here rather than worked around.

    tools/canary_probe.py --plan deploy/canary/<plan>.json --phase post --base-url https://…
                          --repo . --out receipts/canary/<name>.json

Rules this implements, each one falsifiable in tests/test_canary_probe.py:
  C3  a probe that could not run is `not_measured`, never `verified` — an unreachable contour, a
      refused credential and a skipped probe are all absence of measurement;
  C4  a probe whose method is not GET/HEAD/OPTIONS needs `mutating: true` AND a plan owner, or the
      run refuses. The 403 probe does not write — that is the point of it — but if the service ever
      accepted it, it WOULD write, so it is declared mutating rather than described as safe;
  DEC-AUP-0040  the result is bound to an immutable Git subject (GitCanarySubject/v1 over a
      MeasuredGitSource/v1 claim), because evidence about one tree must never vouch for another's.

Route presence (A2-324) is NOT read off the status code alone. The first production run went red
on `GET /v1/ltm/jobs/{job_id}`: the probe's placeholder id named no job, the HANDLER answered 404
"Job not found", and "404 => the route is gone" called a served route missing. A 404 is ambiguous —
the router saying no route matched, or a handler saying no such resource — and so is a 405, which
says the PATH matched and the VERB did not (a rewritten verb, the other half of the mutant class).
Two observations decide instead, neither of which the handler's own answer can supply:
  - the resident version's OpenAPI document must declare this path template with this verb. It is
    generated from the routing table the process is serving, so a route removed from the
    application is absent from it; a route it does not declare is `failed` whatever the status;
  - a 404 must be distinguishable from the router's own no-match answer, fingerprinted live on the
    same contour by a GET to an unrouted sibling path (`<path>/__canary_unrouted__`). A 404 whose
    body is byte-identical to that fingerprint IS the no-match answer: `failed`. A different body
    is a handler answering "absent", and with the declaration that is a served route.
When neither decides (OpenAPI unreadable and no usable fingerprint) the verdict is `not_measured`.

Secrets: a token is read from the environment, sent in a header, and never written to the result,
the log or the evidence file. Response bodies are recorded only as a length and a sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
RANK = {"verified": 0, "not_measured": 1, "failed": 2}
TOOL = "tools/canary_probe.py"
VERSION = "1.0.0"


class Refusal(Exception):
    pass


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


UNROUTED_SUFFIX = "/__canary_unrouted__"


def fetch(url: str, timeout: float) -> tuple[int | None, bytes]:
    """An unauthenticated GET that treats a 4xx/5xx as an observation. (None, b"") if it could not run."""
    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", "aup-orchestrator/1.0")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read(65536)
    except Exception:  # noqa: BLE001 — absence of an observation, handled by the caller
        return None, b""


class Resident:
    """What the resident version says about its own routing table, read once per run.

    `declares(method, template)` is True/False when the OpenAPI document was read, None when it was
    not — an unreadable document is absence of measurement, never evidence either way."""

    def __init__(self, base_url: str, openapi_path: str, timeout: float):
        self.url = base_url.rstrip("/") + openapi_path
        self.timeout = timeout
        self._paths: dict | None = None
        self.observation: dict | None = None

    def paths(self) -> dict | None:
        if self.observation is None:
            status, payload = fetch(self.url, self.timeout)
            self.observation = {"url": self.url, "status": status, "sha256": sha_bytes(payload)}
            try:
                document = json.loads(payload) if status == 200 else None
                paths = document.get("paths") if isinstance(document, dict) else None
                self._paths = paths if isinstance(paths, dict) else None
            except ValueError:
                self._paths = None
            self.observation["paths"] = None if self._paths is None else len(self._paths)
        return self._paths

    def declares(self, method: str, template: str) -> bool | None:
        paths = self.paths()
        if paths is None:
            return None
        operations = paths.get(template)
        return isinstance(operations, dict) and method.lower() in operations


def route_presence(
    base_url: str, spec: dict, method: str, status: int, payload: bytes, resident: Resident | None, timeout: float
) -> tuple[str, str, dict]:
    """(outcome, reason, extra result fields) for `expect.route_present` — see the module docstring."""
    template = spec.get("route", spec["path"])
    declared = resident.declares(method, template) if resident is not None else None
    extra: dict = {"route": template, "declared_by_resident_openapi": declared}
    if declared is False:
        return "failed", f"the resident OpenAPI does not declare {method} {template}: the route is NOT served", extra
    if 500 <= status < 600:
        # A 5xx says the route is reachable, and says nothing trustworthy about it: the observation
        # is of a fault. Rule C3's third verdict is what that is — never a pass, never a regression.
        return "not_measured", f"status {status}: the contour answered with a fault, not with the route", extra
    if status == 405:
        return "failed", f"405: the path matches but {method} is NOT served on it", extra
    if status != 404:
        suffix = "; declared in the resident OpenAPI" if declared else ""
        return "verified", f"status {status} != 404: the route is served{suffix}", extra
    unrouted_status, unrouted_body = fetch(base_url.rstrip("/") + spec["path"].rstrip("/") + UNROUTED_SUFFIX, timeout)
    fingerprint = sha_bytes(unrouted_body) if unrouted_status == 404 else None
    extra["unrouted_404_sha256"] = fingerprint
    if fingerprint is not None and fingerprint == sha_bytes(payload):
        return "failed", "404 identical to the router's own no-match answer: the route is NOT served", extra
    if declared and fingerprint is not None:
        return (
            "verified",
            "404 from the handler (body differs from the router's no-match answer) on a route the resident "
            "OpenAPI declares: the resource is absent, the route is served",
            extra,
        )
    return (
        "not_measured",
        f"404 not attributable: resident OpenAPI {'unreadable' if declared is None else 'declares it'}, "
        f"no-match fingerprint {'unavailable' if fingerprint is None else 'differs'}",
        extra,
    )


def probe(base_url: str, spec: dict, token: str | None, timeout: float, resident: Resident | None = None) -> dict:
    url = base_url.rstrip("/") + spec["path"]
    method = spec.get("method", "GET").upper()
    body = json.dumps(spec["body"]).encode() if "body" in spec else None
    request = urllib.request.Request(url, method=method, data=body)
    request.add_header("User-Agent", "aup-orchestrator/1.0")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    if spec.get("auth") == "bearer":
        if not token:
            # Rule C3 names this case explicitly: a refused credential is absence of measurement,
            # not a failed measurement. A contour whose canary secret is unset must not turn the
            # admission red — it must leave the entity `not_measured`, which is never a pass.
            return {
                "id": spec["id"],
                "kind": "http",
                "method": method,
                "url": url,
                "status": None,
                "outcome": "not_measured",
                "executed": False,
                "reason": "the canary credential is not configured on this host (rule C3)",
                "elapsed_ms": 0,
            }
        request.add_header("Authorization", "Bearer " + token)
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, payload = response.status, response.read(65536)
    except urllib.error.HTTPError as error:
        # A 4xx IS an observation. The whole point of the negative probe is to read a 403.
        status, payload = error.code, error.read(65536)
    except Exception as error:  # noqa: BLE001 — an unreachable contour is absence of measurement (C3)
        return {
            "id": spec["id"],
            "kind": "http",
            "method": method,
            "url": url,
            "status": None,
            "outcome": "not_measured",
            "executed": True,
            "reason": f"probe could not run: {type(error).__name__}",
            "elapsed_ms": int((time.time() - started) * 1000),
        }
    result = {
        "id": spec["id"],
        "kind": "http",
        "method": method,
        "url": url,
        "status": status,
        "executed": True,
        "elapsed_ms": int((time.time() - started) * 1000),
        "body_sha256": sha_bytes(payload),
        "body_bytes": len(payload),
    }
    expect, reasons, outcome = spec.get("expect", {}), [], "verified"
    if "status_in" in expect:
        if status in expect["status_in"]:
            reasons.append(f"status {status} in {expect['status_in']}")
        else:
            outcome = "failed"
            reasons.append(f"status {status} not in {expect['status_in']}")
    if expect.get("route_present"):
        presence, reason, extra = route_presence(base_url, spec, method, status, payload, resident, timeout)
        result.update(extra)
        if RANK[presence] > RANK[outcome]:
            outcome = presence
        reasons.append(reason)
    text = payload.decode("utf-8", "replace")
    for key in expect.get("json_has", []):
        try:
            document = json.loads(text)
        except ValueError:
            outcome, _ = "not_measured", reasons.append("response is not JSON: shape not measured")
            break
        first = document[0] if isinstance(document, list) and document else document
        if not isinstance(first, dict) or key not in first:
            outcome, _ = "failed", reasons.append(f"response has no key `{key}`")
        else:
            reasons.append(f"key `{key}` present")
    if "version_key" in expect:
        try:
            result["observed_version"] = json.loads(text).get(expect["version_key"])
            reasons.append(f"resident version {result['observed_version']}")
        except ValueError:
            outcome, _ = "not_measured", reasons.append("version could not be read from the response")
    result["outcome"] = outcome
    result["reason"] = "; ".join(reasons)
    return result


def subject_of(repo: Path, evidence_path: Path) -> tuple[dict, dict]:
    """The immutable Git subject of this measurement, and the source claim it rests on.

    A dirty tree has no immutable subject at all (`canary_evidence.source_errors` refuses it), so
    it refuses here too rather than producing evidence that would be rejected later with a vaguer
    reason."""
    if git(repo, "status", "--porcelain"):
        raise Refusal("the working tree is dirty: a canary measures a commit, not a working copy")
    commit = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    claim = {
        "schema": "MeasuredGitSource/v1",
        "commit": commit,
        "tree": tree,
        "dirty": False,
        "producer": {"tool": TOOL, "version": VERSION},
        "captured_at_utc": now_iso(),
    }
    raw = json.dumps(claim, indent=1, sort_keys=True).encode() + b"\n"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(raw)
    subject = {
        "schema": "GitCanarySubject/v1",
        "commit": commit,
        "tree": tree,
        # A2-263: a sibling NAME, never this host's absolute path — the gate resolves references
        # beside the result document in git objects.
        "evidence": {"path": evidence_path.name, "sha256": sha_bytes(raw), "source_field": "commit"},
    }
    return subject, claim


def run(plan: dict, base_url: str, phase: str, repo: Path, out: Path, timeout: float, offline: bool = False) -> dict:
    if plan.get("schema") != "CanaryPlan/v1":
        raise Refusal("plan is not a CanaryPlan/v1")
    owner = plan.get("owner")
    for spec in plan["probes"]:
        if spec.get("method", "GET").upper() not in SAFE_METHODS and not (spec.get("mutating") and owner):
            raise Refusal(
                f"MUTATING_PROBE_UNDECLARED: probe {spec['id']} is {spec.get('method')} but the plan "
                f"declares neither mutating:true nor an owner (rule C4)"
            )
    token = os.environ.get(plan["api_key_env"]) if plan.get("api_key_env") else None
    subject, _ = subject_of(repo, out.with_name(out.stem + ".source-evidence.json"))

    results, resident = [], None
    openapi = Resident(base_url, plan.get("openapi_path", "/openapi.json"), timeout)
    for spec in plan["probes"]:
        if offline:
            results.append(
                {
                    "id": spec["id"],
                    "kind": "http",
                    "method": spec.get("method", "GET").upper(),
                    "url": base_url.rstrip("/") + spec["path"],
                    "status": None,
                    "executed": False,
                    "outcome": "not_measured",
                    "reason": "--offline: the probe was not executed",
                    "elapsed_ms": 0,
                }
            )
            continue
        result = probe(base_url, spec, token, timeout, openapi)
        resident = result.get("observed_version") or resident
        results.append(result)

    by_id = {r["id"]: r for r in results}
    rows: dict[str, dict] = {}
    for spec in plan["probes"]:
        result = by_id[spec["id"]]
        for entity in spec.get("entities", []):
            row = rows.setdefault(
                entity,
                {
                    "entity": entity,
                    "verdict": result["outcome"],
                    "reason": f"{spec['id']}: {result['reason']}"[:400],
                    "probe_ids": [],
                },
            )
            row["probe_ids"].append(spec["id"])
            if RANK[result["outcome"]] > RANK[row["verdict"]]:
                row["verdict"] = result["outcome"]
                row["reason"] = f"{spec['id']}: {result['reason']}"[:400]

    read_only = all(s.get("method", "GET").upper() in SAFE_METHODS for s in plan["probes"])
    document = {
        "schema": "CanaryResult/v1",
        "card": "A2-311",
        "captured_at_utc": now_iso(),
        "producer": {"tool": TOOL, "version": VERSION},
        "model": "none: deterministic HTTP producer",
        "plan": {
            "id": plan["id"],
            "path": Path(plan["_path"]).name if plan.get("_path") else plan["id"],
            "digest": plan.get("_digest"),
        },
        "environment": plan["environment"],
        "base_url": base_url,
        "phase": phase,
        "read_only": read_only,
        "resident_version": resident,
        "resident_openapi": openapi.observation,
        "subject": subject,
        "probes": results,
        "entity_verdicts": sorted(rows.values(), key=lambda r: r["entity"]),
        "counters": {k: sum(1 for r in results if r["outcome"] == k) for k in RANK},
        "rules": [
            "C3 a probe that could not run is not_measured, never verified",
            "C4 probes are read-only unless the plan declares mutating with an owner",
        ],
    }
    if not read_only:
        document["mutating_owner"] = owner
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--offline", action="store_true", help="record every probe as not_measured without touching the contour"
    )
    args = parser.parse_args(argv)

    raw = Path(args.plan).read_bytes()
    plan = json.loads(raw)
    plan["_path"], plan["_digest"] = args.plan, "sha256:" + sha_bytes(raw)
    try:
        document = run(
            plan,
            args.base_url or plan["base_url"],
            args.phase,
            Path(args.repo),
            Path(args.out),
            args.timeout,
            offline=args.offline,
        )
    except Refusal as refusal:
        print(f"canary refused: {refusal}", file=sys.stderr)
        return 2
    worst = max((RANK[r["verdict"]] for r in document["entity_verdicts"]), default=0)
    for row in document["entity_verdicts"]:
        print(f"{row['verdict']:12} {row['entity']}  {row['reason'][:120]}")
    print(f"{args.out}: {document['counters']}")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
