"""Tests for LTM-0012 temporal layer (regex + LLM hybrid date extraction)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from scrutator.ltm.models import EntityEvent
from scrutator.ltm.temporal import (
    HAS_TIME_CUE,
    DateExtractor,
    extract_regex_events,
    merge_overlapping_events,
)

# ---------- regex layer ----------------------------------------------------


class TestRegexExtraction:
    def test_iso_date_match(self):
        events = extract_regex_events("**Archived:** 2026-04-16", entity_names=["TUNE-0003"])
        assert len(events) >= 1
        e = events[0]
        assert e.event_type == "archived"
        assert e.when_t == datetime(2026, 4, 16, tzinfo=UTC)

    def test_dot_date_format(self):
        events = extract_regex_events("Migration finished 16.04.2026", entity_names=["X"])
        # Generic date — no explicit event_type label, so we get a 'updated' fallback
        assert any(e.when_t == datetime(2026, 4, 16, tzinfo=UTC) for e in events)

    def test_iso_timestamp_with_tz(self):
        events = extract_regex_events("**Generated:** 2026-04-14T08:20:43.557Z", entity_names=["dream-report"])
        # Loose match: at least the date part is captured
        dates = [e.when_t for e in events if e.when_t is not None]
        assert any(d.year == 2026 and d.month == 4 and d.day == 14 for d in dates)

    def test_labeled_field_archived(self):
        events = extract_regex_events("**Archived:** 2026-04-16", entity_names=["TUNE-0003"])
        e = next(x for x in events if x.event_type == "archived")
        assert e.entity_name == "TUNE-0003"

    def test_labeled_field_completed(self):
        events = extract_regex_events("**Completed:** 2026-04-15", entity_names=["INFRA-0009"])
        e = next(x for x in events if x.event_type == "completed")
        assert e.when_t == datetime(2026, 4, 15, tzinfo=UTC)

    def test_no_match_returns_empty(self):
        events = extract_regex_events("This is plain prose with no dates.", entity_names=["X"])
        assert events == []

    def test_no_known_entity_drops_event(self):
        # If no entity_names provided, regex events have entity_name="" — should be dropped
        events = extract_regex_events("**Archived:** 2026-04-16", entity_names=[])
        assert events == []


# ---------- time-cue heuristic ---------------------------------------------


class TestHasTimeCue:
    @pytest.mark.parametrize(
        "text",
        [
            "after the migration",
            "released last week",
            "до этого момента",
            "После запуска",
            "previously deprecated",
        ],
    )
    def test_positive_cues(self, text):
        assert HAS_TIME_CUE(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Datarim is a knowledge framework.",
            "Production server is at 65.108.236.39",
            "Just plain factual content.",
        ],
    )
    def test_negative_cues(self, text):
        assert HAS_TIME_CUE(text) is False


# ---------- LLM fallback ---------------------------------------------------


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.extract_json = AsyncMock()
    return llm


@pytest.fixture
def extractor(mock_llm):
    return DateExtractor(llm=mock_llm, max_events=10)


class TestLlmFallback:
    @pytest.mark.asyncio
    async def test_skip_llm_when_regex_finds(self, extractor, mock_llm):
        # Regex finds → LLM not called
        events = await extractor.extract(
            content="**Archived:** 2026-04-16",
            entity_names=["TUNE-0003"],
        )
        assert len(events) >= 1
        mock_llm.extract_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_llm_when_no_time_cue(self, extractor, mock_llm):
        # No regex match AND no time-cue → LLM not called
        events = await extractor.extract(
            content="Datarim runtime-first knowledge framework.",
            entity_names=["Datarim"],
        )
        assert events == []
        mock_llm.extract_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_llm_on_prose_with_time_cue(self, extractor, mock_llm):
        mock_llm.extract_json.return_value = [
            {
                "entity_name": "Mem0",
                "event_type": "released",
                "when": "2026-04-14",
                "valid_from": "2026-04-14",
                "valid_to": None,
                "description": "v2.0.0 released",
            }
        ]
        events = await extractor.extract(
            content="Mem0 v2.0.0 was released after the rewrite",
            entity_names=["Mem0"],
        )
        mock_llm.extract_json.assert_called_once()
        assert len(events) == 1
        assert events[0].event_type == "released"
        assert events[0].when_t == datetime(2026, 4, 14, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_llm_returns_garbage_returns_empty(self, extractor, mock_llm):
        mock_llm.extract_json.return_value = {"raw": "no JSON here"}
        events = await extractor.extract(
            content="Released after the merge",
            entity_names=["X"],
        )
        assert events == []

    @pytest.mark.asyncio
    async def test_llm_invalid_iso_dropped(self, extractor, mock_llm):
        mock_llm.extract_json.return_value = [
            {"entity_name": "X", "event_type": "released", "when": "next monday"},
            {"entity_name": "X", "event_type": "archived", "when": "2026-04-16"},
        ]
        events = await extractor.extract(
            content="X was released after Y",
            entity_names=["X"],
        )
        # Only the second one should survive
        assert len(events) == 1
        assert events[0].event_type == "archived"

    @pytest.mark.asyncio
    async def test_llm_unknown_entity_dropped(self, extractor, mock_llm):
        mock_llm.extract_json.return_value = [
            {"entity_name": "Stranger", "event_type": "archived", "when": "2026-04-16"},
            {"entity_name": "X", "event_type": "archived", "when": "2026-04-17"},
        ]
        events = await extractor.extract(
            content="X was archived after Stranger",
            entity_names=["X"],
        )
        # Stranger dropped (unknown entity), X kept
        assert len(events) == 1
        assert events[0].entity_name == "X"

    @pytest.mark.asyncio
    async def test_llm_max_events_enforced(self, extractor, mock_llm):
        extractor.max_events = 2
        mock_llm.extract_json.return_value = [
            {"entity_name": "X", "event_type": f"e{i}", "when": "2026-04-16"} for i in range(5)
        ]
        events = await extractor.extract(content="X after Y", entity_names=["X"])
        assert len(events) == 2


# ---------- merge_overlapping_events ---------------------------------------


class TestMergeOverlapping:
    def _ev(self, name: str, when: datetime, vf: datetime | None = None, vt: datetime | None = None):
        return EntityEvent(
            entity_name=name,
            event_type="archived",
            when_t=when,
            valid_from=vf or when,
            valid_to=vt,
        )

    def test_already_closed_period_not_touched(self):
        # Older event already has valid_to → no further closure needed
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 1, 31, tzinfo=UTC)
        t3 = datetime(2026, 2, 1, tzinfo=UTC)
        old = self._ev("X", t1, vf=t1, vt=t2)  # already closed
        new = self._ev("X", t3)
        out = merge_overlapping_events([old, new])
        older = next(e for e in out if e.valid_from == t1)
        newer = next(e for e in out if e.valid_from == t3)
        assert older.valid_to == t2  # unchanged
        assert newer.valid_to is None

    def test_supersede_older_when_overlap(self):
        # Two events for same entity+type, both with valid_to=None — newer must close older
        old = self._ev("X", datetime(2026, 1, 1, tzinfo=UTC))
        new = self._ev("X", datetime(2026, 2, 1, tzinfo=UTC))
        out = merge_overlapping_events([old, new])
        # older has valid_to set, newer doesn't
        older = next(e for e in out if e.valid_from == datetime(2026, 1, 1, tzinfo=UTC))
        newer = next(e for e in out if e.valid_from == datetime(2026, 2, 1, tzinfo=UTC))
        assert older.valid_to is not None
        assert older.valid_to <= newer.valid_from
        assert newer.valid_to is None

    def test_different_entities_not_merged(self):
        a = self._ev("X", datetime(2026, 1, 1, tzinfo=UTC))
        b = self._ev("Y", datetime(2026, 2, 1, tzinfo=UTC))
        out = merge_overlapping_events([a, b])
        assert all(e.valid_to is None for e in out)
