#!/usr/bin/env python3
"""SRCH-0038: idempotent offline backfill of the whole-document content hash.

Chunks indexed before SRCH-0038 lack `metadata.section.doc_content_hash`. This one-time,
idempotent script binds the hash OFFLINE for legacy docs by concatenating each document's
chunks in `chunk_index` order and hashing once — an ingest-equivalent one-time bind, NOT a
response-time recompute, so the S1 invariant (hash bound at ingest / backfill, only READ at
fetch) is preserved.

Properties:
- **Idempotent** — a document that already carries `doc_content_hash` is never touched; a second
  run no-ops.
- **Additive** — writes only the JSONB key `metadata.section.doc_content_hash`; no DDL, no other
  field altered.
- **Namespace-agnostic** — processes every namespace (offline operator tool, not a request path).

Until a legacy doc is backfilled or re-indexed, fetch returns its `content_hash` as `""`
(never a silently-recomputed value).

Usage:
    SCRUTATOR_DATABASE_URL=postgresql://scrutator:...@host:5432/scrutator \
      PYTHONPATH=src python scripts/backfill_doc_content_hash.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys

from scrutator.db.connection import close_pool, get_pool


def compute_doc_content_hash(full_content: str) -> str:
    """Whole-document hash in the `sha256:` format, identical to the ingest-side stamp."""
    return "sha256:" + hashlib.sha256(full_content.encode()).hexdigest()


def _needs_backfill(section: dict | None) -> bool:
    """A doc needs backfill iff it has a section (thus a doc_id) but no doc_content_hash yet."""
    if not section:
        return False
    return bool(section.get("doc_id")) and not section.get("doc_content_hash")


async def _distinct_docs_missing_hash(conn) -> list[tuple[int, str]]:
    """Return (namespace_id, doc_id) pairs that carry a doc_id but no doc_content_hash."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT c.namespace_id, c.metadata->'section'->>'doc_id' AS doc_id
        FROM chunks c
        WHERE c.metadata->'section'->>'doc_id' IS NOT NULL
          AND (c.metadata->'section'->>'doc_content_hash') IS NULL
        """
    )
    return [(r["namespace_id"], r["doc_id"]) for r in rows if r["doc_id"]]


async def _backfill_one_doc(conn, namespace_id: int, doc_id: str, dry_run: bool) -> int:
    """Concat the doc's chunks in order, hash once, stamp into every chunk's section. Returns the
    number of chunk rows updated (0 when already stamped / dry-run)."""
    chunk_rows = await conn.fetch(
        """
        SELECT id, chunk_index, content, metadata
        FROM chunks
        WHERE namespace_id = $1 AND metadata->'section'->>'doc_id' = $2
        ORDER BY chunk_index
        """,
        namespace_id,
        doc_id,
    )
    if not chunk_rows:
        return 0
    full_content = "".join(r["content"] for r in chunk_rows)
    doc_hash = compute_doc_content_hash(full_content)

    updated = 0
    for r in chunk_rows:
        meta = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else dict(r["metadata"] or {})
        section = meta.get("section")
        if not _needs_backfill(section):
            continue  # idempotent: already stamped → skip
        section["doc_content_hash"] = doc_hash
        meta["section"] = section
        if not dry_run:
            await conn.execute(
                "UPDATE chunks SET metadata = $1::jsonb, updated_at = NOW() WHERE id = $2",
                json.dumps(meta),
                r["id"],
            )
        updated += 1
    return updated


async def backfill(dry_run: bool = False) -> dict[str, int]:
    pool = await get_pool()
    docs_processed = 0
    chunks_updated = 0
    async with pool.acquire() as conn:
        for namespace_id, doc_id in await _distinct_docs_missing_hash(conn):
            n = await _backfill_one_doc(conn, namespace_id, doc_id, dry_run)
            if n:
                docs_processed += 1
                chunks_updated += n
    return {"docs_processed": docs_processed, "chunks_updated": chunks_updated}


async def _main(dry_run: bool) -> None:
    try:
        result = await backfill(dry_run=dry_run)
    finally:
        await close_pool()
    mode = "DRY-RUN" if dry_run else "APPLIED"
    print(f"[{mode}] docs_processed={result['docs_processed']} chunks_updated={result['chunks_updated']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill metadata.section.doc_content_hash (SRCH-0038).")
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    args = parser.parse_args()
    try:
        asyncio.run(_main(args.dry_run))
    except KeyboardInterrupt:
        sys.exit(130)
