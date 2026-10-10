"""Exact field/span classification; disabled candidates never clear findings.

The caller supplies a policy digest ONLY after native admission. This module does
not perform admission, fetch proof, rewrite fields, or enable a candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

from .secretscan import SEV_INFO, Finding, ScanError, ScanResult, _set_verdict, scan_serialized

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_OID = re.compile(r"[0-9a-f]{40}\Z")


def policy_digest(policy: Mapping) -> str:
    return hashlib.sha256(json.dumps(dict(policy), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _proven(entry: Mapping) -> bool:
    proof = entry.get("provenance")
    common = (
        isinstance(proof, dict)
        and isinstance(entry.get("task_id"), str)
        and bool(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", entry["task_id"]))
        and entry.get("field") in {"title", "description"}
        and type(entry.get("line")) is int
        and entry["line"] > 0
        and bool(_HEX.fullmatch(str(entry.get("current_field_sha256", ""))))
        and bool(_HEX.fullmatch(str(entry.get("span_sha256", ""))))
        and proof.get("span_sha256") == entry.get("span_sha256")
    )
    if not common:
        return False
    if proof.get("kind") == "historical_orca_accounting":
        records = proof.get("records")
        return (
            entry.get("metadata_key") == "agentTerminalHandle"
            and proof.get("source_field") == "agentTerminalHandle"
            and bool(_HEX.fullmatch(str(proof.get("artifact_sha256", ""))))
            and bool(_HEX.fullmatch(str(proof.get("report_sha256", ""))))
            and isinstance(records, list)
            and bool(records)
            and all(
                isinstance(r, dict)
                and r.get("source_field") == "agentTerminalHandle"
                and r.get("span_sha256") == entry["span_sha256"]
                and bool(re.fullmatch(r"run_[a-z0-9]+", str(r.get("runId", ""))))
                and bool(re.fullmatch(r"task_[a-z0-9]+", str(r.get("taskId", ""))))
                and bool(re.fullmatch(r"ctx_[a-z0-9]+", str(r.get("dispatchId", ""))))
                for r in records
            )
        )
    return (
        proof.get("kind") in {None, "immutable_git"}
        and bool(re.fullmatch(r"Arcanada-one/[A-Za-z0-9_.-]+", str(proof.get("repo", ""))))
        and bool(_OID.fullmatch(str(proof.get("commit", ""))))
        and bool(_OID.fullmatch(str(proof.get("blob", ""))))
        and bool(_HEX.fullmatch(str(proof.get("file_sha256", ""))))
        and isinstance(proof.get("path"), str)
        and bool(proof["path"])
        and not proof["path"].startswith("/")
        and ".." not in proof["path"].split("/")
        and isinstance(proof.get("pointers"), list)
        and bool(proof["pointers"])
        and all(isinstance(p, str) and p.startswith("$") for p in proof["pointers"])
    )


def scan_task_field(
    text: str,
    *,
    task_id: str,
    field: str,
    policy: Mapping | None = None,
    admitted_policy_sha256: str | None = None,
) -> ScanResult:
    """Scan every rule first; classify only admitted exact generic-entropy hits.

    Named critical and gitleaks findings are immutable blockers. No info regex,
    field-wide exemption or policy-provided rule selector is accepted. Policy
    provenance is an immutable reference, not an assertion that admission ran.
    """
    result = scan_serialized(text)
    if policy is None or policy.get("enabled") is not True:
        return result
    if (
        policy.get("schema") != "ExactSpanClassificationPolicy/v1"
        or policy.get("status") != "admitted"
        or not isinstance(admitted_policy_sha256, str)
        or not _HEX.fullmatch(admitted_policy_sha256)
        or policy_digest(policy) != admitted_policy_sha256
    ):
        raise ScanError("exact-span policy lacks caller-verified native admission binding")
    entries = policy.get("entries")
    if not isinstance(entries, list) or not entries or any(not isinstance(e, dict) or not _proven(e) for e in entries):
        raise ScanError("exact-span policy provenance invalid")
    identities = [
        (e.get("task_id"), e.get("field"), e.get("line"), e.get("current_field_sha256"), e.get("span_sha256"))
        for e in entries
    ]
    if len(set(identities)) != len(identities):
        raise ScanError("exact-span policy contains duplicate identities")
    whole_hash = hashlib.sha256(text.encode()).hexdigest()
    allowed = {(e[2], e[4]) for e in identities if e[0] == task_id and e[1] == field and e[3] == whole_hash}
    result.findings = [
        Finding(f.rule, SEV_INFO, f.line, f.span_hash)
        if f.rule == "generic-entropy" and (f.line, f.span_hash) in allowed
        else f
        for f in result.findings
    ]
    _set_verdict(result)
    return result
