"""LTM-0017 — Audit harness unit tests (RED→GREEN cycle).

Tests pure canonicalisation logic in benchmark/audit_entity_resolver.py.
No DB access; no asyncpg; no Scrutator imports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_AUDIT_PATH = Path(__file__).resolve().parent.parent / "benchmark" / "audit_entity_resolver.py"
_spec = importlib.util.spec_from_file_location("audit_entity_resolver", _AUDIT_PATH)
audit = importlib.util.module_from_spec(_spec)
sys.modules["audit_entity_resolver"] = audit
_spec.loader.exec_module(audit)


def test_casefold_canonicalisation_invariance() -> None:
    """[A, a, A ] under mode='all' → 1 canonical group (case + trailing space normalised)."""
    entities = [
        {"name": "A", "source_chunk_id": "c1"},
        {"name": "a", "source_chunk_id": "c2"},
        {"name": "A ", "source_chunk_id": "c3"},
    ]
    groups = audit.group_by_canonical(entities, mode="all")
    assert len(groups) == 1
    [(canon, members)] = groups.items()
    assert canon == "a"
    assert {m["source_chunk_id"] for m in members} == {"c1", "c2", "c3"}


def test_task_id_namespace_guard() -> None:
    """[TUNE-0003, TUNE-0004, tune-0003] → 3 separate canonicals; no task-ids merged.

    Task-id pattern bypasses casefold, so TUNE-0003 ≠ tune-0003 ≠ TUNE-0004.
    The lower-case 'tune-0003' does NOT match the pattern → falls through to
    casefold path → canonicalises to 'tune-0003'. Crucially: NEVER merged
    with TUNE-0003 (which keeps its uppercase identity).
    """
    entities = [
        {"name": "TUNE-0003", "source_chunk_id": "c1"},
        {"name": "TUNE-0004", "source_chunk_id": "c2"},
        {"name": "tune-0003", "source_chunk_id": "c3"},
    ]
    groups = audit.group_by_canonical(entities, mode="all")
    assert "TUNE-0003" in groups
    assert "TUNE-0004" in groups
    assert "tune-0003" in groups
    assert len(groups) == 3
    assert audit.detect_task_id_violations(groups) == []


def test_alias_skip_when_table_absent() -> None:
    """alias_map=None → alias step is identity, no exception raised."""
    entities = [
        {"name": "Cloudflare", "source_chunk_id": "c1"},
        {"name": "CloudFlare", "source_chunk_id": "c2"},
    ]
    groups = audit.group_by_canonical(entities, mode="all", alias_map=None)
    assert len(groups) == 1
    assert "cloudflare" in groups


def test_idempotent_recanonicalisation() -> None:
    """canonicalise(canonicalise(x)) == canonicalise(x) for all modes."""
    names = ["A", "a", "TUNE-0003", "Cloudflare ", " utilities.md", "  X  Y  "]
    for mode in ("casefold", "whitespace", "all"):
        once = [audit.canonicalise(n, mode) for n in names]
        twice = [audit.canonicalise(n, mode) for n in once]
        assert once == twice, f"mode={mode}: {once!r} != {twice!r}"
