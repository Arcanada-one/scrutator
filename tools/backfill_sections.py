"""One-shot, idempotent backfill of `section` metadata for pre-SRCH-0021 chunks.

HARD-GATED (SRCH-0021 plan Fork 4): this script performs live writes to the
`chunks` table. It defaults to `--dry-run` (report-only, zero writes); a real
run requires the explicit `--live` flag. Per Fork 4's rollout sequence, both
the dry-run and the live run are operator-executed — this script is authored
here but MUST NOT be invoked against any live namespace (including
`arcanada`) by CI, by tests, or by this task's automation.

Zero embedding-client imports (V-AC-2): `section` is re-derived purely from
each chunk's already-stored `heading_hierarchy` (or a `chunk_index`-ordered
single-root fallback when it is empty — PRD Risk table), never re-embedded.

Idempotent (Fork 7): a chunk whose `metadata.section.schema_version` already
matches `SECTION_SCHEMA_VERSION` is excluded by the WHERE clause, so re-running
against an already-backfilled namespace updates 0 rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

from scrutator.chunker.splitters import SECTION_SCHEMA_VERSION, compute_doc_id, normalize_heading_path
from scrutator.db.connection import close_pool, get_pool

logger = logging.getLogger(__name__)


async def _fetch_stale_chunks(namespace: str) -> list[dict[str, Any]]:
    """Rows in `namespace` whose section.schema_version is absent or stale (Fork 7 gate)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id::text AS chunk_id, c.source_path, c.chunk_index, c.metadata
            FROM chunks c
            JOIN namespaces n ON n.id = c.namespace_id
            WHERE n.name = $1
              AND (
                  c.metadata->'section'->>'schema_version' IS NULL
                  OR (c.metadata->'section'->>'schema_version')::int != $2
              )
            ORDER BY c.source_path, c.chunk_index
            """,
            namespace,
            SECTION_SCHEMA_VERSION,
        )
    return [dict(r) for r in rows]


def _fallback_root_section(doc_id: str) -> dict[str, Any]:
    """No `heading_hierarchy` to derive from — single implicit root (PRD Risk table)."""
    return {
        "doc_id": doc_id,
        "heading_path": [],
        "depth": 1,
        "anchor": "root",
        "anchor_path": ["root"],
        "section_key": "root",
        "schema_version": SECTION_SCHEMA_VERSION,
    }


def compute_section_for_row(namespace: str, source_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Recompute `section` for one stored chunk row from its `heading_hierarchy`."""
    doc_id = compute_doc_id(namespace, source_path)
    heading_hierarchy = metadata.get("heading_hierarchy") or []
    if not heading_hierarchy:
        return _fallback_root_section(doc_id)
    section = normalize_heading_path(heading_hierarchy)
    section["doc_id"] = doc_id
    return section


async def _apply_update(conn: Any, chunk_id: str, section: dict[str, Any]) -> None:
    await conn.execute(
        "UPDATE chunks SET metadata = metadata || $1::jsonb WHERE id = $2::uuid",
        json.dumps({"section": section}),
        chunk_id,
    )


async def run_backfill(namespace: str, dry_run: bool = True) -> dict[str, Any]:
    """Backfill `section` metadata for one namespace.

    dry_run=True (default): report the candidate count, write nothing.
    dry_run=False: issue one parameterized `UPDATE ... metadata = metadata || $1` per
    stale row (additive merge — never overwrites unrelated metadata keys).
    """
    rows = await _fetch_stale_chunks(namespace)
    updated = 0

    if not dry_run and rows:
        pool = await get_pool()
        async with pool.acquire() as conn:
            for row in rows:
                metadata = (
                    json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {})
                )
                section = compute_section_for_row(namespace, row["source_path"], metadata)
                await _apply_update(conn, row["chunk_id"], section)
                updated += 1

    return {
        "namespace": namespace,
        "dry_run": dry_run,
        "candidates": len(rows),
        "updated": updated,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True, help="namespace to backfill (e.g. 'arcanada')")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the candidate count without writing (default behaviour — this flag is a no-op alias)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="perform the live UPDATE — HARD-GATED, operator-run only (see plan Fork 4)",
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = await run_backfill(namespace=args.namespace, dry_run=not args.live)
    print(json.dumps(result, indent=2))
    await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
