"""Ingest-side prompt-injection scanning + source trust-tiering (ARAS-0055).

Defense-in-depth companion to ARAS-0049's read-side nonce-fenced envelope and SRCH-0038's
namespace-derived ``trust_class``. This module hardens the KB at INGEST time:

- :func:`scan_injection` — a fast regex/set-based scan (deliberately NO LLM call in the ingest
  hot path) that LABELS a document with a non-blocking injection signal. Ingestion is NEVER
  blocked by this signal — the KB must stay complete; the label is an observability layer that
  feeds downstream trust decisions.
- :func:`source_trust_tier` — a provenance-derived tier. Unreviewed raw dumps (``wiki/_raw_/``)
  get a LOWER tier (``raw``) than curated sources (``curated``). The tier is a WEIGHTING hint
  that composes with ``trust_class``; it NEVER promotes evidence to skill/exec (no
  cross-promotion — trust_class stays namespace-derived, this axis is orthogonal).

Both signals are computed server-side (the scanner runs over document *content*; the tier over
the *source_path*), so a document whose body merely quotes an injection string can label itself
flagged but can never forge its own ``trust_class`` or promote its own tier.
"""

from __future__ import annotations

import re
from typing import Literal

TrustTier = Literal["raw", "curated"]

# Path segment marking unreviewed raw dumps (mirrors the KB's `wiki/_raw_/` convention). Any
# source whose path contains this segment is tiered `raw` (lower trust) than curated content.
_RAW_SEGMENT = "_raw_"

# Score at/above which a document is flagged. Tuned so a single strong marker (role marker,
# override imperative, or exfiltration directive — each weight 3) trips the flag, while a lone
# tool mention (weight 1, e.g. ordinary `curl https://…` docs) does NOT — keeping the labeling
# layer from drowning normal technical prose in false positives.
INJECTION_FLAG_THRESHOLD = 3

# Chat-template / role markers used to smuggle a fake system/user turn into indexed content.
# Matched as literal substrings (case-insensitive) — fast, no regex backtracking.
_ROLE_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "[inst]",
    "[/inst]",
    "<<sys>>",
    "<</sys>>",
)

# "ignore previous instructions"-class imperatives that try to override the operator's prompt.
_OVERRIDE_RE = re.compile(
    r"ignore\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|earlier)\s+instructions"
    r"|disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+(?:instructions|context|text)"
    r"|forget\s+(?:everything|all\s+previous|your\s+instructions)"
    r"|override\s+(?:your\s+)?(?:instructions|system\s+prompt|guidelines)"
    r"|you\s+are\s+now\s+(?:a|an|the)\b"
    r"|new\s+instructions\s*:",
    re.IGNORECASE,
)

# Prompt/credential exfiltration + reveal directives.
_EXFIL_RE = re.compile(
    r"reveal\s+(?:your\s+)?(?:system\s+)?prompt"
    r"|print\s+(?:out\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions)"
    r"|repeat\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)\s+verbatim"
    r"|exfiltrat"
    r"|leak\s+(?:the\s+)?(?:api[_\s-]?key|secret|credentials|token)",
    re.IGNORECASE,
)

# Tool/network directives — weaker on their own (documentation legitimately contains these), so
# weight 1: they raise the score but do not by themselves flag.
_TOOL_RE = re.compile(
    r"\bcurl\s+https?://"
    r"|\bwget\s+https?://"
    r"|fetch\(\s*['\"]https?://"
    r"|POST\s+\S*\s*https?://"
    r"|api[_\s-]?key\s*[:=]",
    re.IGNORECASE,
)

# (category name, weight). Order is irrelevant — `patterns` is returned sorted.
_CATEGORY_WEIGHTS: dict[str, int] = {
    "role_marker": 3,
    "override_imperative": 3,
    "exfiltration": 3,
    "tool_directive": 1,
}


def scan_injection(content: str) -> dict:
    """Scan ``content`` for prompt-injection / instruction-override markers.

    Returns a small, JSONB-safe signal dict ``{"flag": bool, "risk_score": int,
    "patterns": [category, ...]}`` — ``patterns`` is deduplicated and sorted (at most one entry
    per category, so bounded by ``len(_CATEGORY_WEIGHTS)``). This is a NON-BLOCKING label: the
    caller indexes the document regardless of the result.
    """
    matched: list[str] = []
    lowered = content.lower()

    if any(marker in lowered for marker in _ROLE_MARKERS):
        matched.append("role_marker")
    if _OVERRIDE_RE.search(content):
        matched.append("override_imperative")
    if _EXFIL_RE.search(content):
        matched.append("exfiltration")
    if _TOOL_RE.search(content):
        matched.append("tool_directive")

    risk_score = sum(_CATEGORY_WEIGHTS[name] for name in matched)
    return {
        "flag": risk_score >= INJECTION_FLAG_THRESHOLD,
        "risk_score": risk_score,
        "patterns": sorted(matched),
    }


def source_trust_tier(source_path: str) -> TrustTier:
    """Derive a provenance trust tier from ``source_path`` (server-side, never from doc body).

    Unreviewed raw dumps under a ``_raw_`` path segment (the KB's ``wiki/_raw_/`` convention) are
    lower-trust (``raw``); everything else is ``curated``. This is a WEIGHTING hint only — it
    composes with ``trust_class`` and NEVER promotes a document across the skill|evidence boundary.
    """
    segments = source_path.replace("\\", "/").lower().split("/")
    if _RAW_SEGMENT in segments:
        return "raw"
    return "curated"
