"""SRCH-0039 — offline evidence exact-bytes backfill: idempotent, additive, integrity-guarded.

Fully mocked (matching the repo's HARD-GATE pattern): no real DB connection is opened, no
namespace is backfilled by this suite.

Integrity contract exercised here: the fetch path advertises the CHUNK-stamped ``doc_content_hash``,
so the backfill writes an ``evidence_documents`` row ONLY when the chunk-reassembly hashes to that
SAME stamped value. A doc whose stamp is a native original-content hash (reassembly differs) or that
carries no stamp yet cannot be repaired offline — it is SKIPPED (needs a re-ingest with the flag on,
or a prior ``doc_content_hash`` backfill), never written with a row that would make a later
``content_exact=True`` fail ``sha256(content)==content_hash``.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts import backfill_evidence_exact_bytes as bf

from .conftest import build_indexed_doc


def _bare(content_hash: str) -> str:
    return content_hash.split(":", 1)[1] if ":" in content_hash else content_hash


def _restamp_to_reassembly(rows: list[dict]) -> str:
    """Simulate a doc already ``doc_content_hash``-backfilled (SRCH-0038 legacy path): stamp every
    chunk's ``doc_content_hash`` to the hash of the chunk reassembly. Returns that hash."""
    reassembly = "".join(r["content"] for r in rows)
    reassembly_hash = bf.compute_doc_content_hash(reassembly)
    for r in rows:
        r["metadata"] = json.loads(json.dumps(r["metadata"]))  # deep copy
        r["metadata"]["section"]["doc_content_hash"] = reassembly_hash
    return reassembly_hash


def _strip_stamp(rows: list[dict]) -> None:
    for r in rows:
        r["metadata"] = json.loads(json.dumps(r["metadata"]))
        r["metadata"]["section"].pop("doc_content_hash", None)


class _FakeConn:
    """Minimal asyncpg-conn stand-in: serves one doc's chunk rows, records evidence upserts, and
    reflects the write so a second pass observes idempotency."""

    def __init__(self, rows: list[dict], existing: dict | None = None):
        self._rows = [
            {
                "chunk_index": r["chunk_index"],
                "content": r["content"],
                "source_path": r["source_path"],
                "metadata": r["metadata"],
            }
            for r in rows
        ]
        # (namespace_id, source_path) -> content_hash
        self._evidence: dict[tuple[int, str], str] = dict(existing or {})
        self.upserts: list[dict] = []

    async def fetch(self, _sql, *_params):
        return self._rows

    async def fetchval(self, _sql, namespace_id, source_path):
        return self._evidence.get((namespace_id, source_path))

    async def execute(self, _sql, namespace_id, source_path, doc_id, content_hash, raw_content):
        self.upserts.append(
            {
                "namespace_id": namespace_id,
                "source_path": source_path,
                "doc_id": doc_id,
                "content_hash": content_hash,
                "raw_content": raw_content,
            }
        )
        self._evidence[(namespace_id, source_path)] = content_hash


def test_compute_matches_ingest_stamp():
    content = "# Doc\n\n" + ("word " * 100)
    assert bf.compute_doc_content_hash(content) == "sha256:" + hashlib.sha256(content.encode()).hexdigest()


@pytest.mark.asyncio
async def test_backfill_writes_row_when_reassembly_matches_stamp_then_idempotent():
    """A doc whose stamped hash == reassembly hash (already ``doc_content_hash``-backfilled) → the
    backfill writes a byte-exact evidence row, then a second pass no-ops (idempotent)."""
    content = "# Evidence\n\n" + ("lorem " * 200)
    doc_id, _orig_hash, rows = build_indexed_doc(content, namespace="arcanada")
    reassembly = "".join(r["content"] for r in rows)
    reassembly_hash = _restamp_to_reassembly(rows)
    conn = _FakeConn(rows)

    status = await bf._backfill_one_evidence_doc(conn, namespace_id=1, doc_id=doc_id, dry_run=False)
    assert status == "written"
    assert len(conn.upserts) == 1
    up = conn.upserts[0]
    assert up["raw_content"] == reassembly
    assert up["content_hash"] == reassembly_hash
    # By-construction: the stored bytes hash to the advertised (stamped) hash.
    assert hashlib.sha256(up["raw_content"].encode()).hexdigest() == _bare(up["content_hash"])

    status_again = await bf._backfill_one_evidence_doc(conn, namespace_id=1, doc_id=doc_id, dry_run=False)
    assert status_again == "idempotent"
    assert len(conn.upserts) == 1  # no second write


@pytest.mark.asyncio
async def test_backfill_skips_native_hash_mismatch():
    """A doc whose stamp is a native original-content hash (reassembly differs) is NOT written — it
    cannot be made byte-exact offline; it needs a re-ingest with the flag on."""
    content = (
        "---\ntitle: t\n---\n\n# Doc\n\n   \n"  # frontmatter + whitespace → lossy reassembly
        + ("word " * 300)
    )
    doc_id, _orig_hash, rows = build_indexed_doc(content, namespace="arcanada")
    # build_indexed_doc stamps the ORIGINAL-content hash; reassembly differs → mismatch.
    assert "".join(r["content"] for r in rows) != content
    conn = _FakeConn(rows)

    status = await bf._backfill_one_evidence_doc(conn, namespace_id=1, doc_id=doc_id, dry_run=False)
    assert status == "mismatch"
    assert conn.upserts == []


@pytest.mark.asyncio
async def test_backfill_skips_unstamped_doc():
    """A doc with no ``doc_content_hash`` stamp cannot be written (the fetch path would advertise
    "" and break the invariant) — run the ``doc_content_hash`` backfill first."""
    content = "# Doc\n\n" + ("lorem " * 200)
    doc_id, _h, rows = build_indexed_doc(content, namespace="arcanada")
    _strip_stamp(rows)
    conn = _FakeConn(rows)

    status = await bf._backfill_one_evidence_doc(conn, namespace_id=1, doc_id=doc_id, dry_run=False)
    assert status == "unstamped"
    assert conn.upserts == []


@pytest.mark.asyncio
async def test_backfill_dry_run_writes_nothing():
    content = "# Evidence\n\n" + ("lorem " * 200)
    doc_id, _h, rows = build_indexed_doc(content, namespace="arcanada")
    _restamp_to_reassembly(rows)
    conn = _FakeConn(rows)

    status = await bf._backfill_one_evidence_doc(conn, namespace_id=1, doc_id=doc_id, dry_run=True)
    assert status == "written"  # would-write reported …
    assert conn.upserts == []  # … but no upsert executed.
