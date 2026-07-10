"""LTM-0016 — regression guard for the LTM-0015 entity-resolution priority fix.

LTM-0015 fixed `_resolve_entity()` in `src/scrutator/ltm/temporal.py` so that a
task-id-shaped candidate (pattern `[A-Z]+-\\d{4}`, e.g. `TUNE-0003`) always wins
over a longer generic candidate. `tests/test_ltm_temporal.py::TestResolveEntityPriority`
already unit-tests `_resolve_entity()` directly and would catch a regression in
the function's own body.

This file adds a guard one layer up: it exercises `extract_regex_events()`, the
public function that actually calls `_resolve_entity()` (three times, once per
regex pass — see archive-LTM-0015.md § Path discrepancy) and that is itself the
function `DateExtractor.extract()` calls in production. Testing at this layer
means a future change that re-wires *how* `extract_regex_events()` invokes the
resolver (e.g. swaps it for an inline re-implementation, or passes the wrong
candidate list) is caught even if `_resolve_entity()`'s own unit tests stay
green.

Layer: pure-function integration test. `extract_regex_events()` has no DB/IO —
there is no live/test Postgres in this project (see test_ltm_temporal.py), and
none is needed here either, so this is a plain call/assert test, not a
mock-pool test.
"""

from datetime import UTC, datetime

from scrutator.ltm.temporal import extract_regex_events


class TestExtractRegexEventsEntityPriority:
    """Regression guard: extract_regex_events() must resolve task-id entities
    correctly even when a longer, generic candidate name is also present in
    the same chunk — the exact LTM-0012 root-cause shape."""

    def test_task_id_wins_over_longer_generic_candidate(self):
        content = "**Archived:** 2026-04-16 — TUNE-0003 closed by Datarim Framework Team"
        entity_names = ["Datarim Framework Team", "TUNE-0003"]

        events = extract_regex_events(content, entity_names)

        assert len(events) >= 1
        archived = next(e for e in events if e.event_type == "archived")
        assert archived.entity_name == "TUNE-0003"
        assert archived.when_t == datetime(2026, 4, 16, tzinfo=UTC)
