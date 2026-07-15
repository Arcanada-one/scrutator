"""SRCH-0041-compatible pure-Python secret scan for exact outbound payloads."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

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


class ScanError(RuntimeError):
    """Outbound payload failed or could not complete the secret-scan gate."""


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
            if shannon_entropy(value) > _ENTROPY_THRESHOLD and not any(
                pattern.search(value) for pattern in compiled_info
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
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="muneral-ltm-payload-", suffix=".json", encoding="utf-8"
        ) as handle:
            handle.write(text)
            handle.flush()
            process = subprocess.run(  # noqa: S603 - fixed executable and arguments, no shell
                [
                    gitleaks_path,
                    "detect",
                    "--no-git",
                    "--source",
                    handle.name,
                    "--report-format",
                    "json",
                    "--report-path",
                    "/dev/stdout",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
    except (OSError, subprocess.SubprocessError):
        raise ScanError("gitleaks scan failed closed") from None
    try:
        data = json.loads((process.stdout or "[]").strip() or "[]")
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
    _set_verdict(result)
    return result
