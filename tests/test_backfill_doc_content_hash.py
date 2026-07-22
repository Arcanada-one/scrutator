"""SRCH-0038 — offline doc-content-hash backfill: idempotency + S1-preserving one-time bind.

Fully mocked (matching the repo's HARD-GATE pattern): no real DB connection is opened, no
namespace is backfilled by this suite.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts import backfill_doc_content_hash as bf

from .conftest import build_indexed_doc


def test_compute_matches_ingest_stamp():
    content = "# Doc\n\n" + ("word " * 100)
    assert bf.compute_doc_content_hash(content) == "sha256:" + hashlib.sha256(content.encode()).hexdigest()


def test_needs_backfill_predicate():
    assert bf._needs_backfill({"doc_id": "0123456789abcdef"}) is True
    assert bf._needs_backfill({"doc_id": "0123456789abcdef", "doc_content_hash": "sha256:x"}) is False
    assert bf._needs_backfill(None) is False
    assert bf._needs_backfill({}) is False  # no doc_id → not a fetchable doc, skip


class _FakeConn:
    """Minimal asyncpg-conn stand-in: serves the doc's chunk rows and records UPDATEs."""

    def __init__(self, rows: list[dict]):
        # Strip the doc_content_hash to simulate legacy (pre-SRCH-0038) chunks.
        self._rows = []
        for r in rows:
            meta = json.loads(json.dumps(r["metadata"]))  # deep copy
            meta["section"].pop("doc_content_hash", None)
            self._rows.append(
                {"id": r["chunk_id"], "chunk_index": r["chunk_index"], "content": r["content"], "metadata": meta}
            )
        self.updates: dict[str, dict] = {}

    async def fetch(self, _sql, *_params):
        return self._rows

    async def execute(self, _sql, meta_json, chunk_id):
        meta = json.loads(meta_json)
        self.updates[chunk_id] = meta
        # Reflect the write back so a second pass sees the stamped state (idempotency check).
        for row in self._rows:
            if row["id"] == chunk_id:
                row["metadata"] = meta


@pytest.mark.asyncio
async def test_backfill_stamps_then_is_idempotent():
    content = "# Doc\n\n" + ("lorem " * 200)
    doc_id, _original_hash, rows = build_indexed_doc(content, namespace="arcanada")
    conn = _FakeConn(rows)

    # The backfill can only reconstruct from stored chunks (the original source is not persisted),
    # so its whole-doc hash is over the chunk-concatenation — the accepted ingest-equivalent bind
    # for LEGACY docs (D3). This may differ from a native ingest's original-content hash where the
    # chunker normalized whitespace; both are bound-once/read-at-fetch, so S1 holds either way.
    expected_hash = bf.compute_doc_content_hash("".join(r["content"] for r in rows))

    # First pass: stamps every chunk with the whole-doc hash.
    updated = await bf._backfill_one_doc(conn, namespace_id=1, doc_id=doc_id, dry_run=False)
    assert updated == len(rows)
    for meta in conn.updates.values():
        assert meta["section"]["doc_content_hash"] == expected_hash

    # Second pass: everything already stamped → no-op (idempotent).
    updated_again = await bf._backfill_one_doc(conn, namespace_id=1, doc_id=doc_id, dry_run=False)
    assert updated_again == 0


@pytest.mark.asyncio
async def test_backfill_dry_run_writes_nothing():
    content = "# Doc\n\n" + ("lorem " * 100)
    doc_id, _h, rows = build_indexed_doc(content, namespace="arcanada")
    conn = _FakeConn(rows)
    updated = await bf._backfill_one_doc(conn, namespace_id=1, doc_id=doc_id, dry_run=True)
    assert updated == len(rows)  # would-update count reported …
    assert conn.updates == {}  # … but no UPDATE executed.
