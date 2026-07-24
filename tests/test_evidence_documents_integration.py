"""SRCH-0039 (Mechanism C) — live-PostgreSQL integration coverage for the isolated
``evidence_documents`` store.

This is the durable confirmation the mock suite CANNOT give: it drives the REAL ingest path
(``replace_source_chunks_atomic`` upserting the evidence row in the SAME transaction as the chunks)
against a real Postgres, then reads it back through ``POST /v1/fetch``'s fetcher.

Proves the three ratified evidence behaviours:
  (a) flag ON + row present  → byte-exact ``content`` for a LARGE doc spanning many chunks, with
      ``sha256(content) == content_hash`` and ``content_exact=True``;
  (b) flag ON + row ABSENT   → graceful degradation to reassembly (``content_exact=False``), NOT a
      409 (the deliberate skills-vs-evidence divergence — evidence row-absence is a pre-backfill
      state, not an integrity failure);
  (c) IDOR                    → a cross-namespace fetch of an evidence doc returns 404 (no oracle).

Skipped unless a disposable Scrutator-schema database is provided — mirroring the repo's other
live-PG integration suites:

    export SCRUTATOR_SRCH0039_TEST_DATABASE_URL=postgresql://user:pw@host:5432/scrutator_srch0039_test
    export SRCH0039_TEST_DB_GO=scrutator_srch0039_test      # must equal the DB name (belt-and-braces)
    .venv-linux/bin/python -m pytest tests/test_evidence_documents_integration.py -q
"""

from __future__ import annotations

import hashlib
import os
import uuid

import asyncpg
import pytest
from fastapi import HTTPException

from scrutator.config import settings
from scrutator.db.connection import apply_schema, close_pool, get_pool
from scrutator.db.models import FetchRequest
from scrutator.search.indexer import compute_doc_content_hash

_DENSE_DIMENSIONS = 1024


def _bare_hash(content_hash: str) -> str:
    return content_hash.split(":", 1)[1] if ":" in content_hash else content_hash


def _evidence_doc(target_bytes: int) -> str:
    """A realistic evidence document: YAML frontmatter + headings (so every chunk carries a
    ``section``, and thus a ``doc_id`` stamp, matching how fetch-by-doc_id resolves rows) + enough
    body to cross ``target_bytes`` and span many chunks with overlap. NO 256 KB cap applies to
    evidence, so this may be arbitrarily large."""
    body = "The incident timeline records each mitigation step and its measured outcome. " * 30 + "\n\n"
    doc = "---\ntitle: big-postmortem\nseverity: sev1\n---\n\n# Big Post-mortem\n\n"
    section = 0
    while len(doc.encode("utf-8")) < target_bytes:
        section += 1
        doc += f"## Phase {section}\n\n{body}"
    return doc


@pytest.fixture
async def evidence_db(monkeypatch):
    dsn = os.environ.get("SCRUTATOR_SRCH0039_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("no disposable PostgreSQL in SCRUTATOR_SRCH0039_TEST_DATABASE_URL")

    # Safety: never touch the configured application database; require an explicitly approved
    # disposable *_test database (same double-approval contract as the SRCH-0038 suite).
    probe = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    database_name = await probe.fetchval("SELECT current_database()")
    approval = os.environ.get("SRCH0039_TEST_DB_GO")
    if not database_name.endswith("_test") or approval != database_name or dsn == settings.database_url:
        await probe.close()
        pytest.fail("SRCH-0039 integration DB must be a separately approved *_test database")
    await probe.close()

    # Repoint the application pool at the disposable DB and ensure the current schema (incl. the new
    # evidence_documents table) is applied — apply_schema() is idempotent (IF NOT EXISTS).
    monkeypatch.setattr(settings, "database_url", dsn)
    await close_pool()
    await apply_schema()

    # Flag ON for the integration proof (default-off in prod until the operator flips it).
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)

    # A per-run unique EVIDENCE namespace (NOT the skills namespace, which keeps its own path), so
    # the exact-bytes evidence path fires and cleanup is isolated from any real data.
    test_ns = f"evidence-srch0039-{uuid.uuid4()}"

    # Deterministic, network-free embeddings so the real ingest path runs without the embed API.
    async def _fake_dense(texts):
        return [[0.001 * (i + 1)] * _DENSE_DIMENSIONS for i in range(len(texts))]

    async def _fake_sparse(texts):
        return [{"tok": 1.0} for _ in texts]

    monkeypatch.setattr("scrutator.search.indexer.embed_texts", _fake_dense)
    monkeypatch.setattr("scrutator.search.indexer.embed_sparse", _fake_sparse)

    pool = await get_pool()
    namespace_id = await pool.fetchval("INSERT INTO namespaces (name) VALUES ($1) RETURNING id", test_ns)
    created_ns_ids = [namespace_id]

    def _register_ns(ns_id: int) -> None:
        created_ns_ids.append(ns_id)

    try:
        yield pool, test_ns, namespace_id, _register_ns
    finally:
        for ns_id in created_ns_ids:
            await pool.execute("DELETE FROM evidence_documents WHERE namespace_id = $1", ns_id)
            await pool.execute("DELETE FROM chunks WHERE namespace_id = $1", ns_id)
            await pool.execute("DELETE FROM namespaces WHERE id = $1", ns_id)
        await close_pool()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_bytes",
    [
        pytest.param(20 * 1024, id="~20KB-multi-chunk"),
        pytest.param(300 * 1024, id="~300KB-over-skills-cap"),  # proves evidence has NO 256 KB cap
    ],
)
async def test_evidence_ingest_then_fetch_is_byte_exact(evidence_db, target_bytes):
    from scrutator.chunker.splitters import compute_doc_id
    from scrutator.search import fetcher, indexer

    pool, test_ns, namespace_id, _register = evidence_db
    source_path = f"evidence/big-{target_bytes}.md"
    doc = _evidence_doc(target_bytes)

    # (a) The REAL ingest transaction writes the evidence_documents row in-transaction with chunks.
    resp = await indexer.index_document(doc, source_path, namespace=test_ns)
    assert resp.chunks_indexed > 1, "a large evidence doc must span >1 chunk"

    expected_hash = compute_doc_content_hash(doc)
    row = await pool.fetchrow(
        "SELECT raw_content, content_hash FROM evidence_documents WHERE namespace_id = $1 AND source_path = $2",
        namespace_id,
        source_path,
    )
    assert row is not None, "evidence_documents row must exist after ingest with the flag ON"
    assert row["raw_content"] == doc
    assert row["content_hash"] == expected_hash
    assert hashlib.sha256(row["raw_content"].encode()).hexdigest() == _bare_hash(row["content_hash"])

    # POST /v1/fetch returns byte-exact content with sha256(content) == content_hash.
    doc_id = compute_doc_id(test_ns, source_path)
    fetched = await fetcher.fetch(
        FetchRequest(by="source_id", id=doc_id, range="full"),
        frozenset({namespace_id}),
    )
    assert fetched.trust_class == "evidence"
    assert fetched.content_exact is True
    assert fetched.content == doc
    assert fetched.content_hash == expected_hash
    assert hashlib.sha256(fetched.content.encode()).hexdigest() == _bare_hash(fetched.content_hash)


@pytest.mark.asyncio
async def test_evidence_row_absent_degrades_to_reassembly_not_409(evidence_db):
    """Flag ON but the evidence_documents row is ABSENT (simulating an un-backfilled doc on the huge
    existing corpus) → the fetcher returns the lossy reassembly with content_exact=False, NEVER a
    409. This is the deliberate skills-vs-evidence divergence."""
    from scrutator.chunker.splitters import compute_doc_id
    from scrutator.search import fetcher, indexer

    pool, test_ns, namespace_id, _register = evidence_db
    source_path = "evidence/unbackfilled.md"
    doc = _evidence_doc(20 * 1024)

    await indexer.index_document(doc, source_path, namespace=test_ns)
    # Delete only the exact-bytes row, leaving the chunks — the pre-backfill state.
    await pool.execute(
        "DELETE FROM evidence_documents WHERE namespace_id = $1 AND source_path = $2",
        namespace_id,
        source_path,
    )

    doc_id = compute_doc_id(test_ns, source_path)
    fetched = await fetcher.fetch(
        FetchRequest(by="source_id", id=doc_id, range="full"),
        frozenset({namespace_id}),
    )
    assert fetched.trust_class == "evidence"
    assert fetched.content_exact is False, "row-absent evidence must gracefully degrade, not fail closed"
    # Reassembly is the lossy chunk concatenation (frontmatter dropped, edges stripped, overlap dup).
    assert fetched.content != doc


@pytest.mark.asyncio
async def test_evidence_cross_namespace_fetch_is_404_idor(evidence_db):
    """IDOR parity: an evidence doc ingested in namespace A is NOT reachable by a caller whose
    allowed-set is a DIFFERENT namespace B → 404 (no existence oracle), the same fail-closed
    ``namespace_id = ANY(allowed)`` predicate the skills path uses (authorize-before-bytes)."""
    from scrutator.chunker.splitters import compute_doc_id
    from scrutator.search import fetcher, indexer

    pool, test_ns, namespace_id, register_ns = evidence_db
    source_path = "evidence/tenant-a-only.md"
    doc = _evidence_doc(20 * 1024)
    await indexer.index_document(doc, source_path, namespace=test_ns)

    # A separate tenant B that must NOT be able to reach tenant A's evidence doc.
    other_ns = f"evidence-srch0039-other-{uuid.uuid4()}"
    other_ns_id = await pool.fetchval("INSERT INTO namespaces (name) VALUES ($1) RETURNING id", other_ns)
    register_ns(other_ns_id)

    doc_id = compute_doc_id(test_ns, source_path)
    with pytest.raises(HTTPException) as excinfo:
        await fetcher.fetch(
            FetchRequest(by="source_id", id=doc_id, range="full"),
            frozenset({other_ns_id}),  # caller scoped to tenant B only
        )
    assert excinfo.value.status_code == 404
