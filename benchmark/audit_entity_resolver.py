#!/usr/bin/env python3
"""LTM-0017 — Entity resolver coverage audit.

Read-only diagnostic that quantifies the canonicalisation delta on a flat
list of entities. Operates on stdin JSON / file input, never touches DB
during testing. Live audit fetches entities via SSH/psql to a side file,
then feeds it through this same code path.

Output: aggregate JSON report with merge_ratio_pct, multi-chunk group
counts (raw vs canonicalised), task-id-namespace violations (must be
empty), and a decision_recommendation per LTM-0017 § Step 4 decision tree.

Scope: read-only; no DB writes; no production traffic.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

TASK_ID_RE = re.compile(r"^[A-Z]+-\d{4}$")
SECRET_PAT = re.compile(r"(api[_-]?key|password|secret|token|bearer)", re.IGNORECASE)


def task_id_canonical(name: str) -> str | None:
    """Return name unchanged when it matches `[A-Z]+-\\d{4}`; None otherwise.

    Task-ids are their own canonical entity — never merged across namespaces.
    """
    return name if TASK_ID_RE.match(name) else None


def canonicalise(name: str, mode: str, alias_map: dict[str, str] | None = None) -> str:
    """Apply canonicalisation; preserves task-id namespaces unconditionally."""
    if (tid := task_id_canonical(name)) is not None:
        return tid
    out = name
    if mode in ("whitespace", "all"):
        out = re.sub(r"\s+", " ", out).strip()
    if mode in ("casefold", "all"):
        out = out.casefold()
    if mode in ("alias", "all") and alias_map:
        out = alias_map.get(out, out)
    if mode not in ("casefold", "whitespace", "alias", "all"):
        raise ValueError(f"unknown mode: {mode}")
    return out


def group_by_canonical(
    entities: list[dict],
    mode: str,
    alias_map: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """Group entity rows by canonical name."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        canon = canonicalise(e["name"], mode, alias_map)
        groups[canon].append(e)
    return dict(groups)


def raw_groups(entities: list[dict]) -> dict[str, list[dict]]:
    """Identity grouping by raw `name` field (no canonicalisation)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        groups[e["name"]].append(e)
    return dict(groups)


def _multi_chunk_groups(groups: dict[str, list[dict]]) -> int:
    """Count groups whose distinct non-null source_chunk_id set has size ≥ 2."""
    count = 0
    for members in groups.values():
        chunks = {m.get("source_chunk_id") for m in members if m.get("source_chunk_id")}
        if len(chunks) >= 2:
            count += 1
    return count


def compute_stats(
    entities: list[dict],
    raw: dict[str, list[dict]],
    canon: dict[str, list[dict]],
) -> dict:
    """Build the audit report numerics."""
    raw_count = len(raw)
    canon_count = len(canon)
    merge_ratio_pct = 0.0
    if raw_count:
        merge_ratio_pct = round(100.0 * (raw_count - canon_count) / raw_count, 2)
    linked = sum(1 for e in entities if e.get("source_chunk_id"))
    return {
        "total_entities_raw": len(entities),
        "linked_entities": linked,
        "unlinked_entities": len(entities) - linked,
        "entity_groups_raw_count": raw_count,
        "entity_groups_with_2plus_raw_count": _multi_chunk_groups(raw),
        "entity_groups_canonical_count": canon_count,
        "entity_groups_with_2plus_canonical_count": _multi_chunk_groups(canon),
        "delta_multi_chunk_groups": _multi_chunk_groups(canon) - _multi_chunk_groups(raw),
        "merge_ratio_pct": merge_ratio_pct,
    }


def detect_task_id_violations(canon: dict[str, list[dict]]) -> list[dict]:
    """Identify any canonical group merging ≥2 distinct task-ids. MUST be empty."""
    violations = []
    for canonical, members in canon.items():
        task_ids = {m["name"] for m in members if TASK_ID_RE.match(m["name"])}
        if len(task_ids) >= 2:
            violations.append({"canonical": canonical, "task_ids_merged": sorted(task_ids)})
    return violations


def decision_recommendation(stats: dict) -> str:
    """Apply LTM-0017 § Step 4 decision tree."""
    if stats["merge_ratio_pct"] >= 30.0:
        return "Path 1: bundle_LTM-0015"
    if stats["entity_groups_with_2plus_canonical_count"] >= 5:
        return "Path 3: isolated_LTM-0015"
    return "Path 2: escalate to A2 (topic-clustering)"


def _safe_log_name(name: str) -> str:
    """Truncate + redact entity strings before logging (T5 control)."""
    if SECRET_PAT.search(name):
        return f"<redacted:{len(name)}c>"
    return name[:50]


def build_report(
    entities: list[dict],
    namespace: str,
    mode: str,
    alias_map: dict[str, str] | None,
    scrutator_version: str = "0.3.0",
) -> dict:
    raw = raw_groups(entities)
    canon = group_by_canonical(entities, mode, alias_map=alias_map)
    stats = compute_stats(entities, raw, canon)
    violations = detect_task_id_violations(canon)
    return {
        "namespace": namespace,
        "captured_at": datetime.now(UTC).isoformat(),
        "scrutator_version": scrutator_version,
        "canonicalise_mode": mode,
        "alias_table_present": alias_map is not None,
        **stats,
        "task_id_namespace_violations": violations,
        "decision_recommendation": decision_recommendation(stats),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LTM-0017 entity resolver audit")
    ap.add_argument("--input-json", required=True, help="JSON list of {name, source_chunk_id, ...}")
    ap.add_argument("--canonicalise", default="all", choices=["casefold", "whitespace", "alias", "all"])
    ap.add_argument("--alias-map", default=None, help="Optional JSON {raw_name: canonical}")
    ap.add_argument("--namespace", default="ltm-bench-datarim-kb")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    entities = json.loads(Path(args.input_json).read_text())
    alias_map = json.loads(Path(args.alias_map).read_text()) if args.alias_map else None
    report = build_report(entities, args.namespace, args.canonicalise, alias_map)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
