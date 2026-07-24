#!/usr/bin/env python3
"""SRCH-0039: idempotent offline backfill of exact whole-document bytes for the EVIDENCE corpus.

Fresh ingest with ``SCRUTATOR_EVIDENCE_EXACT_BYTES=1`` writes each evidence doc's exact bytes into
``evidence_documents`` inside the ingest transaction (see ``indexer._build_evidence_document``).
This one-time, idempotent script populates rows for the LARGE *existing* corpus indexed before the
flag existed, so the operator can flip the flag knowing the exact-capable docs already have rows
(``content_exact`` goes False→True for them; the rest keep graceful reassembly until re-ingested).

Integrity contract (why this is NOT a lossy fabrication):
    The fetch path advertises the CHUNK-stamped ``metadata.section.doc_content_hash``. This script
    writes an ``evidence_documents`` row ONLY when the chunk reassembly hashes to that SAME stamped
    value — so a later ``content_exact=True`` can never fail ``sha256(content) == content_hash``.
    * A doc whose stamp is a native original-content hash (its lossy reassembly differs) is SKIPPED
      as ``mismatch`` — it cannot be made byte-exact offline; re-ingest it with the flag on.
    * A doc with no ``doc_content_hash`` stamp is SKIPPED as ``unstamped`` — run
      ``backfill_doc_content_hash.py`` first (that binds the stamp to the reassembly hash).
    This is the ingest-equivalent one-time bind for legacy docs (S1 preserved: hash bound once,
    only READ at fetch), never a response-time recompute.

Properties:
- **Idempotent** — a doc whose ``evidence_documents`` row already carries the target ``content_hash``
  is left untouched; a second run no-ops.
- **Additive** — writes only ``evidence_documents`` rows; no DDL, and it never mutates ``chunks``.
- **Content-hash-keyed** — the row's ``content_hash`` IS the chunk-stamped hash; the reassembly must
  hash to it or the doc is skipped.
- **Evidence-scoped** — processes every namespace EXCEPT ``settings.skills_namespace`` (skills keep
  their own ``source_documents`` path).
- **Dry-run by default** — reports what WOULD be written; pass ``--apply`` to actually write. (The
  safer default than the SRCH-0038 hash backfill, because this materialises real content bytes.)

Verification gate (creative condition 5): keep the flag OFF until an ``--apply`` run reports
``mismatch == 0`` and ``unstamped == 0`` (or those residual docs are accepted as graceful-degrade),
i.e. the count of exact-capable docs == rows present.

Usage:
    SCRUTATOR_DATABASE_URL=postgresql://scrutator:...@host:5432/scrutator \
      PYTHONPATH=src python scripts/backfill_evidence_exact_bytes.py            # dry-run (default)
    SCRUTATOR_DATABASE_URL=... PYTHONPATH=src \
      python scripts/backfill_evidence_exact_bytes.py --apply                   # write rows
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys

from scrutator.config import settings
from scrutator.db.connection import close_pool, get_pool

_UPSERT_SQL = """
    INSERT INTO evidence_documents (namespace_id, source_path, doc_id, content_hash, raw_content, updated_at)
    VALUES ($1, $2, $3, $4, $5, NOW())
    ON CONFLICT (namespace_id, source_path)
    DO UPDATE SET
        doc_id = EXCLUDED.doc_id,
        content_hash = EXCLUDED.content_hash,
        raw_content = EXCLUDED.raw_content,
        updated_at = NOW()
"""


def compute_doc_content_hash(full_content: str) -> str:
    """Whole-document hash in the ``sha256:`` format, identical to the ingest-side stamp."""
    return "sha256:" + hashlib.sha256(full_content.encode()).hexdigest()


async def _evidence_docs(conn) -> list[tuple[int, str]]:
    """Return (namespace_id, doc_id) pairs for every NON-skills doc that carries a doc_id."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT c.namespace_id, c.metadata->'section'->>'doc_id' AS doc_id
        FROM chunks c
        JOIN namespaces n ON n.id = c.namespace_id
        WHERE c.metadata->'section'->>'doc_id' IS NOT NULL
          AND n.name <> $1
        """,
        settings.skills_namespace,
    )
    return [(r["namespace_id"], r["doc_id"]) for r in rows if r["doc_id"]]


async def _backfill_one_evidence_doc(conn, namespace_id: int, doc_id: str, dry_run: bool) -> str:
    """Reconstruct one doc's bytes from its chunks and, IFF the reassembly hashes to the chunk-
    stamped ``doc_content_hash``, upsert an ``evidence_documents`` row. Returns a status:
    ``"written"`` (row written, or would-be under dry-run), ``"idempotent"`` (row already present
    with the target hash), ``"mismatch"`` (native-hash doc — needs re-ingest), ``"unstamped"`` (no
    ``doc_content_hash`` — run the hash backfill first), or ``"skipped"`` (no chunks)."""
    chunk_rows = await conn.fetch(
        """
        SELECT chunk_index, content, source_path, metadata
        FROM chunks
        WHERE namespace_id = $1 AND metadata->'section'->>'doc_id' = $2
        ORDER BY chunk_index
        """,
        namespace_id,
        doc_id,
    )
    if not chunk_rows:
        return "skipped"

    full_content = "".join(r["content"] for r in chunk_rows)
    reconstructed_hash = compute_doc_content_hash(full_content)

    first = chunk_rows[0]
    meta = json.loads(first["metadata"]) if isinstance(first["metadata"], str) else dict(first["metadata"] or {})
    section = meta.get("section") or {}
    stamped_hash = section.get("doc_content_hash")
    source_path = first["source_path"]

    # The fetch path advertises `stamped_hash`; only a row whose bytes hash to it is byte-exact.
    if not stamped_hash:
        return "unstamped"
    if reconstructed_hash != stamped_hash:
        return "mismatch"

    existing = await conn.fetchval(
        "SELECT content_hash FROM evidence_documents WHERE namespace_id = $1 AND source_path = $2",
        namespace_id,
        source_path,
    )
    if existing == stamped_hash:
        return "idempotent"

    if not dry_run:
        await conn.execute(_UPSERT_SQL, namespace_id, source_path, doc_id, stamped_hash, full_content)
    return "written"


async def backfill(dry_run: bool = True) -> dict[str, int]:
    pool = await get_pool()
    counts = {"written": 0, "idempotent": 0, "mismatch": 0, "unstamped": 0, "skipped": 0}
    async with pool.acquire() as conn:
        for namespace_id, doc_id in await _evidence_docs(conn):
            status = await _backfill_one_evidence_doc(conn, namespace_id, doc_id, dry_run)
            counts[status] += 1
    return counts


async def _main(dry_run: bool) -> None:
    try:
        result = await backfill(dry_run=dry_run)
    finally:
        await close_pool()
    mode = "DRY-RUN" if dry_run else "APPLIED"
    print(
        f"[{mode}] written={result['written']} idempotent={result['idempotent']} "
        f"mismatch={result['mismatch']} unstamped={result['unstamped']} skipped={result['skipped']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill exact evidence bytes into evidence_documents (SRCH-0039).")
    parser.add_argument(
        "--apply", action="store_true", help="actually write rows (default is a dry-run that writes nothing)"
    )
    args = parser.parse_args()
    try:
        asyncio.run(_main(dry_run=not args.apply))
    except KeyboardInterrupt:
        sys.exit(130)
