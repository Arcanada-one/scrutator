"""LTM-0012 temporal layer — hybrid date extraction (regex Layer 1 + LLM Layer 2).

Regex captures explicit ISO/labelled dates without LLM cost. LLM is only invoked when
the chunk has time-cue keywords (`after`, `before`, `released`, etc.) AND regex found
nothing — this keeps event extraction sub-second on the typical datarim corpus.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import ValidationError

from scrutator.ltm.models import EntityEvent

if TYPE_CHECKING:
    from scrutator.ltm.llm import LtmLlmClient

log = logging.getLogger("scrutator.ltm.temporal")

# Labelled markdown fields → canonical event_type
_LABEL_TO_TYPE: dict[str, str] = {
    "archived": "archived",
    "completed": "completed",
    "started": "started",
    "released": "released",
    "deployed": "deployed",
    "created": "created",
    "updated": "updated",
    "deprecated": "deprecated",
    "last updated": "updated",
    "last update": "updated",
    "generated": "created",
}

# Compiled regex patterns
_RE_LABELED = re.compile(
    r"\*\*(Archived|Completed|Started|Released|Deployed|Created|Updated|Deprecated|Last Update[d]?|Generated):\*\*\s*"
    r"([0-9T:Z\-+. ]{8,40})",
    re.IGNORECASE,
)
_RE_ISO_TS = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?\b")
_RE_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RE_DOT_DATE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")
_RE_TASK_ID = re.compile(r"\b[A-Z]+-\d{4}\b")

# Time-cue heuristic for triggering LLM Layer 2
_RE_TIME_CUE = re.compile(
    r"\b(after|before|previously|previously|released|deprecated|superseded|"
    r"last\s+(?:week|month|quarter|year)|next\s+(?:week|month|quarter|year)|"
    r"после|до|раньше|сейчас|выпущен|выпустил|опубликован|устарел)\b",
    re.IGNORECASE,
)


def HAS_TIME_CUE(text: str) -> bool:  # noqa: N802 (sentinel-style helper)
    """Return True if `text` contains any temporal-cue word — gates LLM Layer 2."""
    return bool(_RE_TIME_CUE.search(text or ""))


def _parse_iso(value: str) -> datetime | None:
    """Parse ISO-8601 string with permissive trailing Z. Returns None on failure."""
    if not value:
        return None
    s = value.strip().rstrip(".").rstrip(",")
    # Date-only forms first
    try:
        if len(s) == 10:  # YYYY-MM-DD
            return datetime.fromisoformat(s).replace(tzinfo=UTC)
        # ISO timestamp — fromisoformat handles most cases on py3.11+
        s_norm = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s_norm)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None


def _parse_dot_date(value: str) -> datetime | None:
    """Parse DD.MM.YYYY format."""
    m = _RE_DOT_DATE.search(value or "")
    if not m:
        return None
    day, month, year = m.groups()
    try:
        return datetime(int(year), int(month), int(day), tzinfo=UTC)
    except ValueError:
        return None


def _resolve_entity(text: str, candidates: list[str]) -> str | None:
    """Find which known entity is mentioned in text. Returns None if none match.

    Priority (LTM-0015):
      1. Task-id pattern `[A-Z]+-\\d{4}` (e.g., TUNE-0003) — most specific identifier.
      2. Longer generic match (length DESC) — fallback for prose entities.
    """
    if not candidates:
        return None
    candidate_set = {c for c in candidates if c}
    for m in _RE_TASK_ID.finditer(text or ""):
        tid = m.group(0)
        if tid in candidate_set:
            return tid
    for name in sorted(candidates, key=len, reverse=True):
        if name and name in text:
            return name
    return None


def extract_regex_events(content: str, entity_names: list[str]) -> list[EntityEvent]:
    """Layer 1 — extract events from explicit ISO / labelled dates. Pure regex."""
    if not content:
        return []
    events: list[EntityEvent] = []
    seen: set[tuple[str, str, datetime]] = set()

    # Pass 1 — labelled fields (have explicit event_type)
    for m in _RE_LABELED.finditer(content):
        label = m.group(1).lower().strip()
        event_type = _LABEL_TO_TYPE.get(label, "updated")
        date_value = m.group(2).strip()
        when_t = _parse_iso(date_value) or _parse_dot_date(date_value)
        if when_t is None:
            continue
        # Pick entity that lives in same chunk — fall back to first known entity if any
        entity = _resolve_entity(content, entity_names) or (entity_names[0] if entity_names else "")
        if not entity:
            continue
        key = (entity, event_type, when_t)
        if key in seen:
            continue
        try:
            events.append(
                EntityEvent(
                    entity_name=entity,
                    event_type=event_type,
                    when_t=when_t,
                    valid_from=when_t,
                )
            )
            seen.add(key)
        except ValidationError as exc:
            log.warning("regex event dropped: %s", exc)

    # Pass 2 — bare ISO / dot dates (no explicit event_type → default 'updated')
    for m in _RE_ISO_DATE.finditer(content):
        date_str = m.group(0)
        # Skip if part of a labelled match already processed
        ctx_start = max(0, m.start() - 30)
        if _RE_LABELED.search(content[ctx_start : m.end()]):
            continue
        when_t = _parse_iso(date_str)
        if when_t is None:
            continue
        entity = _resolve_entity(content, entity_names) or (entity_names[0] if entity_names else "")
        if not entity:
            continue
        key = (entity, "updated", when_t)
        if key in seen:
            continue
        try:
            events.append(
                EntityEvent(
                    entity_name=entity,
                    event_type="updated",
                    when_t=when_t,
                    valid_from=when_t,
                )
            )
            seen.add(key)
        except ValidationError as exc:
            log.warning("regex event dropped: %s", exc)

    for m in _RE_DOT_DATE.finditer(content):
        when_t = _parse_dot_date(m.group(0))
        if when_t is None:
            continue
        entity = _resolve_entity(content, entity_names) or (entity_names[0] if entity_names else "")
        if not entity:
            continue
        key = (entity, "updated", when_t)
        if key in seen:
            continue
        try:
            events.append(
                EntityEvent(
                    entity_name=entity,
                    event_type="updated",
                    when_t=when_t,
                    valid_from=when_t,
                )
            )
            seen.add(key)
        except ValidationError as exc:
            log.warning("regex event dropped: %s", exc)

    return events


class DateExtractor:
    """Hybrid extractor — Layer 1 regex, Layer 2 LLM fallback."""

    def __init__(self, llm: LtmLlmClient, max_events: int = 10):
        self.llm = llm
        self.max_events = max_events

    async def extract(self, content: str, entity_names: list[str]) -> list[EntityEvent]:
        """Run Layer 1 first; only call Layer 2 if regex empty AND time-cue present."""
        regex_events = extract_regex_events(content, entity_names)
        if regex_events:
            return regex_events[: self.max_events]

        if not HAS_TIME_CUE(content):
            return []

        # Layer 2 — LLM fallback
        from scrutator.ltm.prompts import format_event_extraction

        system, user = format_event_extraction(content, entity_names)
        try:
            raw = await self.llm.extract_json(user, system=system)
        except Exception:
            log.exception("LLM event extraction failed")
            return []

        if not isinstance(raw, list):
            log.warning("event extraction returned non-list: %s", type(raw).__name__)
            return []

        known = set(entity_names)
        out: list[EntityEvent] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            ent = (item.get("entity_name") or "").strip()
            etype = (item.get("event_type") or "").strip()
            if ent not in known or not etype:
                continue
            when_t = _parse_iso(item.get("when") or item.get("when_t") or "")
            valid_from = _parse_iso(item.get("valid_from") or "")
            valid_to = _parse_iso(item.get("valid_to") or "")
            if when_t is None and valid_from is None:
                continue
            try:
                out.append(
                    EntityEvent(
                        entity_name=ent,
                        event_type=etype,
                        when_t=when_t,
                        valid_from=valid_from or when_t,
                        valid_to=valid_to,
                        description=(item.get("description") or "").strip() or None,
                    )
                )
            except ValidationError as exc:
                log.warning("LLM event dropped: %s", exc)
            if len(out) >= self.max_events:
                break

        return out


def merge_overlapping_events(events: list[EntityEvent]) -> list[EntityEvent]:
    """Apply Graphiti-style supersede: for each (entity, event_type) cluster, the
    older event's `valid_to` is set to the newer event's `valid_from − 1µs` when
    intervals overlap (both have `valid_to=None`)."""
    if not events:
        return events

    # Group by (entity_name, event_type)
    by_key: dict[tuple[str, str], list[EntityEvent]] = {}
    for e in events:
        by_key.setdefault((e.entity_name, e.event_type), []).append(e)

    closed: list[EntityEvent] = []
    delta = timedelta(microseconds=1)
    for cluster in by_key.values():
        # Sort by valid_from ascending; missing → push to end (treat as oldest)
        cluster.sort(key=lambda e: e.valid_from or datetime.min.replace(tzinfo=UTC))
        for i, ev in enumerate(cluster):
            if i + 1 < len(cluster) and ev.valid_to is None:
                next_from = cluster[i + 1].valid_from
                if next_from is not None and ev.valid_from is not None and ev.valid_from < next_from:
                    closed.append(ev.model_copy(update={"valid_to": next_from - delta}))
                    continue
            closed.append(ev)
    return closed
