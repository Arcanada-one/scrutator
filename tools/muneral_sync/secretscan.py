"""SRCH-0041-compatible pure-Python secret scan for exact outbound payloads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

SEV_CRITICAL = "CRITICAL"
SEV_INFO = "INFO"
VERDICT_CLEAN = "clean"
VERDICT_INFO = "info"
VERDICT_CRITICAL = "critical"

_CRITICAL_RULES = [
    ("vault-token-hvs", re.compile(r"hvs\.[A-Za-z0-9]{20,}")),
    ("vault-token-legacy", re.compile(r"\bs\.[A-Za-z0-9]{24,}")),
    (
        "approle-secret-id",
        re.compile(
            r"secret_id[\"'\s:=]+[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            re.IGNORECASE,
        ),
    ),
    ("pgpassword", re.compile(r"PGPASSWORD\s*=\s*[\"']?([^\s\"']+)")),
    ("cloudflare-origin-token", re.compile(r"v1\.0-[A-Za-z0-9-]{100,}")),
    ("pem-private-key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
]
_ENTROPY_ASSIGN = re.compile(r"(?P<key>[\w.\-]{2,})[\"']?\s*[:=]\s*[\"'](?P<val>[A-Za-z0-9+/=_\-]{20,})[\"']")
_ENTROPY_THRESHOLD = 4.0

# Muneral c5b5df36: the entropy rule was written for `key = "value"` assignments but runs over the serialized
# graph JSON, where it also reads the entity names (`muneral_id`, `display_name`, `description`). A long
# human-readable kebab slug such as `word-word-word-word-2026-11-12` has Shannon entropy above 4.0 simply
# because it is long and mixes many letters, so every such task title blocked its own sync and made the unit
# fail on every run (13 of the first 200 tasks). A slug of real words is not a credential; a random token
# (mixed case, hex, base64, a letters-and-digits part) is not a slug and is still reported.
_SLUG_SHAPE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+){2,}")
_SLUG_TIMESTAMP = re.compile(r"\d{8}t\d{4,6}z")
ALLOWLIST_PATH = Path(__file__).with_name("secret_allowlist.json")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def is_identifier_slug(value: str) -> bool:
    """True for a lowercase kebab/snake slug of real words (at most one leading capital).

    Every word part (3+ letters) must look like a word: a vowel and no run of five consonants, so a
    random lowercase license-key shape such as ``xqzmk-dpvrw-tbnhj`` is not a slug. At least three
    word parts and 60% of the non-numeric parts must be words; numeric parts and a ``20261029t0518z``
    timestamp are allowed; a part that mixes letters and digits is not (that is what a token looks like).
    This is a heuristic without a dictionary: a random lowercase key whose every group happens to hold
    a vowel would pass here, and is still caught by gitleaks and the named rules.
    """
    candidate = value[:1].lower() + value[1:] if value[:1].isupper() else value
    if not _SLUG_SHAPE.fullmatch(candidate):
        return False
    parts = re.split(r"[-_]", candidate)
    if any(len(part) > 20 for part in parts):
        return False
    words = 0
    non_numeric = 0
    for part in parts:
        if part.isdigit() or _SLUG_TIMESTAMP.fullmatch(part):
            continue
        non_numeric += 1
        if re.fullmatch(r"[a-z]{1,2}", part):
            continue
        if not re.fullmatch(r"[a-z]{3,}", part):
            return False
        if not re.search(r"[aeiouy]", part) or re.search(r"[^aeiouy]{5}", part):
            return False
        words += 1
    return words >= 3 and words * 5 >= non_numeric * 3


def load_allowlist(path: Path | None = None) -> dict[tuple[str, str], str]:
    """Reviewed false positives, keyed by (rule, sha256 of the matched span); the value is the recorded reason.

    A missing file means no entries. A malformed file fails CLOSED (ScanError): the allowlist can only
    relax the scan, so an unreadable one must never be read as an empty or a complete one.
    """
    target = path or ALLOWLIST_PATH
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        raise ScanError("secret allowlist unreadable") from None
    try:
        entries = json.loads(raw)["entries"]
        allowed: dict[tuple[str, str], str] = {}
        for entry in entries:
            rule, span_hash, reason = entry["rule"], entry["span_hash"], entry["reason"]
            if not (
                isinstance(rule, str)
                and rule
                and _SHA256_HEX.fullmatch(str(span_hash))
                and isinstance(reason, str)
                and reason.strip()
            ):
                raise ValueError("bad entry")
            allowed[(rule, span_hash)] = reason
    except (ValueError, KeyError, TypeError):
        raise ScanError("secret allowlist malformed") from None
    return allowed


def apply_allowlist(findings: list[Finding], allowed: dict[tuple[str, str], str]) -> list[Finding]:
    """Downgrade a CRITICAL finding to INFO when its (rule, span hash) is a recorded false positive."""
    return [
        replace(finding, severity=SEV_INFO)
        if finding.severity == SEV_CRITICAL and (finding.rule, finding.span_hash) in allowed
        else finding
        for finding in findings
    ]


class ScanError(RuntimeError):
    """Outbound payload failed or could not complete the secret-scan gate.

    ``scan`` is ``ScanResult.as_dict()`` (rule, severity, line, span_hash — never cleartext)
    when the payload was blocked by a critical finding, and ``None`` when the scanner itself
    could not complete: a blocked payload may be skipped by the caller, a failed scanner never.
    """

    def __init__(self, message: str, *, scan: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.scan = scan

    @property
    def blocked(self) -> bool:
        return self.scan is not None

    @property
    def findings(self) -> list[dict[str, object]]:
        if not self.scan:
            return []
        return [dict(finding) for finding in self.scan.get("findings", []) if isinstance(finding, dict)]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def shannon_entropy(data: str) -> float:
    if not data:
        return 0.0
    counts = {character: data.count(character) for character in set(data)}
    return -sum((count / len(data)) * math.log2(count / len(data)) for count in counts.values())


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    line: int
    span_hash: str

    def as_dict(self) -> dict[str, str | int]:
        return {"rule": self.rule, "severity": self.severity, "line": self.line, "span_hash": self.span_hash}


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    verdict: str = VERDICT_CLEAN

    @property
    def is_critical(self) -> bool:
        return self.verdict == VERDICT_CRITICAL

    def as_dict(self) -> dict[str, object]:
        return {"verdict": self.verdict, "findings": [finding.as_dict() for finding in self.findings]}


def _set_verdict(result: ScanResult) -> None:
    result.verdict = (
        VERDICT_CRITICAL
        if any(finding.severity == SEV_CRITICAL for finding in result.findings)
        else VERDICT_INFO
        if result.findings
        else VERDICT_CLEAN
    )


def scan_text(text: str, *, info_patterns: list[str] | None = None) -> ScanResult:
    """Scan text without retaining or returning any matched cleartext."""
    compiled_info = []
    for pattern in info_patterns or []:
        try:
            compiled_info.append(re.compile(pattern))
        except re.error:
            continue
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for rule_id, pattern in _CRITICAL_RULES:
            for match in pattern.finditer(line):
                span = match.group(0)
                severity = SEV_INFO if any(pattern.search(span) for pattern in compiled_info) else SEV_CRITICAL
                findings.append(Finding(rule_id, severity, line_number, _sha256(span)))
        for match in _ENTROPY_ASSIGN.finditer(line):
            value = match.group("val")
            if (
                shannon_entropy(value) > _ENTROPY_THRESHOLD
                and not is_identifier_slug(value)
                and not any(pattern.search(value) for pattern in compiled_info)
            ):
                findings.append(Finding("generic-entropy", SEV_CRITICAL, line_number, _sha256(value)))
        for pattern in compiled_info:
            for match in pattern.finditer(line):
                findings.append(Finding("info-pattern", SEV_INFO, line_number, _sha256(match.group(0))))
    result = ScanResult(findings=findings)
    _set_verdict(result)
    return result


def _run_gitleaks(text: str) -> list[Finding]:
    """Run optional gitleaks over the exact body and fail closed on operational errors."""
    gitleaks_path = shutil.which("gitleaks")
    if gitleaks_path is None:
        return []
    try:
        with tempfile.TemporaryDirectory(prefix="muneral-ltm-scan-") as scan_dir:
            payload_path = os.path.join(scan_dir, "payload.json")
            # gitleaks >= 8.30 refuses to write its report to /dev/stdout
            # ("Report path is not writable: open /dev/stdout: permission
            # denied") and exits 1 with no report at all. Read back through the
            # exit code alone, that is indistinguishable from "leaks found",
            # so every scan raised ScanError and the caller failed closed on a
            # healthy payload. Write the report to a real file instead.
            report_path = os.path.join(scan_dir, "report.json")
            with open(payload_path, "w", encoding="utf-8") as handle:
                handle.write(text)
            process = subprocess.run(  # noqa: S603 - fixed executable and arguments, no shell
                [
                    gitleaks_path,
                    "detect",
                    "--no-git",
                    "--source",
                    payload_path,
                    "--report-format",
                    "json",
                    "--report-path",
                    report_path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            try:
                with open(report_path, encoding="utf-8") as report_handle:
                    raw_report = report_handle.read().strip()
            except FileNotFoundError:
                # gitleaks writes no report when it finds nothing.
                raw_report = ""
    except (OSError, subprocess.SubprocessError):
        raise ScanError("gitleaks scan failed closed") from None
    try:
        data = json.loads(raw_report or "[]")
    except json.JSONDecodeError:
        raise ScanError("gitleaks scan failed closed") from None
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ScanError("gitleaks scan failed closed")
    if not ((process.returncode == 0 and not data) or (process.returncode == 1 and data)):
        raise ScanError("gitleaks scan failed closed")
    return [
        Finding(
            rule=f"gitleaks:{item.get('RuleID') or item.get('Description') or 'gitleaks'}",
            severity=SEV_CRITICAL,
            line=int(item.get("StartLine") or item.get("Line") or 0),
            span_hash=_sha256(str(item.get("Secret") or item.get("Match") or "")),
        )
        for item in data
    ]


def scan_serialized(text: str, *, info_patterns: list[str] | None = None) -> ScanResult:
    """Apply the always-on Python scan plus optional gitleaks to exact wire text."""
    result = scan_text(text, info_patterns=info_patterns)
    result.findings.extend(_run_gitleaks(text))
    result.findings = apply_allowlist(result.findings, load_allowlist())
    _set_verdict(result)
    return result
