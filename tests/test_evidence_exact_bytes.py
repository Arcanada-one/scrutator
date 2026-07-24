"""SRCH-0039 — evidence-corpus exact bytes via the isolated ``evidence_documents`` table.

Mirror of the SRCH-0038 skills guardrail (``test_fetch_skills_byte_exact.py``) but for the LARGE
evidence corpus, with the deliberate policy divergence ratified in
``creative-SRCH-0039-storage-fork.md``:

- **Skills** (``source_documents``): ALWAYS exact; a missing row → fail-closed 409.
- **Evidence** (``evidence_documents``): flag-gated (``evidence_exact_bytes``, default-off). When ON
  and a row is present → exact bytes (``content_exact=True``). When OFF, or the row is ABSENT
  (expected pre-backfill state on the huge existing corpus) → graceful degradation to lossy
  chunk reassembly (``content_exact=False``), NEVER a 409.

The by-construction invariant still holds: whenever ``content_exact=True``,
``sha256(content) == content_hash``.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from scrutator.config import settings
from scrutator.db.models import FetchRequest

from .conftest import build_indexed_doc

# A realistic evidence doc: frontmatter + headings + a body long enough to force >1 overlapping,
# edge-stripped embedding chunk, so the reassembly is genuinely lossy (the whole reason exact bytes
# are stored separately).
_EVIDENCE_DOC = (
    "---\n"
    "title: Incident Post-mortem\n"
    "date: 2026-07-01\n"
    "---\n"
    "\n"
    "# Post-mortem\n\n"
    "   \n"  # leading body whitespace the per-chunk .strip() drops
    "The outage began at 03:14 UTC. " + ("detail " * 220) + "\n\n"
    "## Timeline\n\nMitigation applied. " + ("event " * 220) + "\n   \n"  # trailing whitespace
)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _bare_hash(content_hash: str) -> str:
    return content_hash.split(":", 1)[1] if ":" in content_hash else content_hash


@pytest.mark.asyncio
async def test_evidence_flag_on_row_present_is_byte_exact(monkeypatch):
    """Flag ON + a present ``evidence_documents`` row → byte-exact source, ``content_exact=True``,
    ``sha256(content) == content_hash``. The lossy reassembly path would FAIL this exact doc
    (proven by the reassembly-hash precondition)."""
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    doc_id, content_hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace="arcanada")
    assert len(rows) > 1, "doc must span >1 chunk to exercise reassembly"

    reassembly = "".join(r["content"] for r in rows)
    # Precondition: reassembly is genuinely lossy — the bug exact-bytes guards against.
    assert reassembly != _EVIDENCE_DOC
    assert _sha256_hex(reassembly) != _bare_hash(content_hash)

    from scrutator.search import fetcher

    # The reader returns (raw_content, content_hash-bound-at-write). Here the stored hash matches the
    # current chunk stamp, so the row is trusted.
    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(
            fetcher, "fetch_evidence_raw_content", new_callable=AsyncMock, return_value=(_EVIDENCE_DOC, content_hash)
        ),
    ):
        resp = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))

    assert resp.trust_class == "evidence"
    assert resp.content_exact is True
    assert resp.content == _EVIDENCE_DOC
    assert resp.content != reassembly
    assert _sha256_hex(resp.content) == _bare_hash(content_hash)


@pytest.mark.asyncio
async def test_evidence_flag_on_row_absent_degrades_gracefully_not_409(monkeypatch):
    """Flag ON but NO ``evidence_documents`` row (expected pre-backfill state) → graceful reassembly
    with ``content_exact=False``. NOT a 409 (that is the skills policy, deliberately not evidence's)."""
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    doc_id, _content_hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace="arcanada")

    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(fetcher, "fetch_evidence_raw_content", new_callable=AsyncMock, return_value=None),
    ):
        resp = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))

    assert resp.trust_class == "evidence"
    assert resp.content_exact is False
    assert resp.content == "".join(r["content"] for r in rows)


@pytest.mark.asyncio
async def test_evidence_stale_row_hash_mismatch_degrades_not_exact(monkeypatch):
    """SECURITY REGRESSION (SRCH-0039 pre-merge review BLOCKER): a STALE ``evidence_documents`` row
    (its ``raw_content`` predates a content change that re-stamped the chunks) must NEVER be returned
    as ``content_exact=True`` with the current, mismatching chunk ``content_hash``. Read-side
    belt-and-braces: the reader returns the row's bound-at-write ``content_hash``; when it differs
    from the current chunk stamp, the fetcher degrades to reassembly (``content_exact=False``) — a
    hash-to-hash comparison of two stored values, NOT a body re-hash. Without the guard the fetcher
    returns ``content=V1`` with ``content_hash=H2`` and ``content_exact=True`` (``sha256(V1)=H1≠H2``)."""
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    # Current generation V2 → chunks stamped with H2 (the hash the fetcher advertises).
    v2 = _EVIDENCE_DOC + "\n\n## Amended\n\nRevised finding. " + ("update " * 220) + "\n"
    doc_id, current_hash, rows = build_indexed_doc(v2, namespace="arcanada")

    # A stale row from generation V1 → its bound content_hash H1 differs from H2.
    stale_bytes = "# Post-mortem V1\n\nOriginal finding.\n"
    stale_hash = "sha256:" + hashlib.sha256(stale_bytes.encode()).hexdigest()
    assert stale_hash != current_hash

    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(
            fetcher, "fetch_evidence_raw_content", new_callable=AsyncMock, return_value=(stale_bytes, stale_hash)
        ),
    ):
        resp = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))

    # The stale row is rejected: fetch degrades to reassembly, never lies with content_exact=True.
    assert resp.content_exact is False
    assert resp.content != stale_bytes
    assert resp.content == "".join(r["content"] for r in rows)
    # The advertised hash stays the current chunk stamp; it must not match the returned (reassembled)
    # content — but crucially content_exact is False, so no integrity claim is made.
    assert resp.content_hash == current_hash


@pytest.mark.asyncio
async def test_evidence_flag_off_skips_exact_lookup(monkeypatch):
    """Flag OFF (default) → the evidence exact-bytes reader is NEVER consulted; reassembly,
    ``content_exact=False``. No behaviour change on merge until the flag flips."""
    monkeypatch.setattr(settings, "evidence_exact_bytes", False)
    doc_id, _content_hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace="arcanada")

    from scrutator.search import fetcher

    reader = AsyncMock(return_value=(_EVIDENCE_DOC, "sha256:whatever"))
    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(fetcher, "fetch_evidence_raw_content", reader),
    ):
        resp = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))

    reader.assert_not_awaited()
    assert resp.content_exact is False
    assert resp.content == "".join(r["content"] for r in rows)


@pytest.mark.asyncio
async def test_skills_path_unchanged_when_evidence_flag_on(monkeypatch):
    """Even with ``evidence_exact_bytes`` ON, the skills namespace keeps its OWN policy: exact bytes
    from ``source_documents`` (never ``evidence_documents``), and a missing skills row still
    fails closed with 409 — the evidence graceful-degradation must not leak into skills."""
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)

    from scrutator.search import fetcher

    # Skills doc WITH a source_documents row → exact, from the skills reader, not the evidence one.
    sk_id, _sk_hash, sk_rows = build_indexed_doc(_EVIDENCE_DOC, namespace="skills")
    evidence_reader = AsyncMock(return_value=("EVIDENCE-LEAK", "sha256:leak"))
    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=sk_rows),
        patch.object(fetcher, "fetch_source_raw_content", new_callable=AsyncMock, return_value=_EVIDENCE_DOC),
        patch.object(fetcher, "fetch_evidence_raw_content", evidence_reader),
    ):
        sk = await fetcher.fetch(FetchRequest(by="source_id", id=sk_id, range="full"), frozenset({1}))
    assert sk.trust_class == "skill"
    assert sk.content_exact is True
    assert sk.content == _EVIDENCE_DOC
    evidence_reader.assert_not_awaited()

    # Skills doc WITHOUT a source_documents row → fail closed 409 (unchanged by the evidence flag).
    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=sk_rows),
        patch.object(fetcher, "fetch_source_raw_content", new_callable=AsyncMock, return_value=None),
        patch.object(fetcher, "fetch_evidence_raw_content", new_callable=AsyncMock, return_value=None),
        pytest.raises(HTTPException) as excinfo,
    ):
        await fetcher.fetch(FetchRequest(by="source_id", id=sk_id, range="full"), frozenset({1}))
    assert excinfo.value.status_code == 409


# ── Indexer: _build_evidence_document gating ─────────────────────────────


def test_build_evidence_document_flag_off_returns_none(monkeypatch):
    """Default-off: no evidence row is built, so ingest behaviour is byte-identical pre-flip."""
    monkeypatch.setattr(settings, "evidence_exact_bytes", False)
    from scrutator.search.indexer import _build_evidence_document

    assert _build_evidence_document("arcanada", "ev.md", "# E\n\nbody") is None


def test_build_evidence_document_skills_namespace_returns_none(monkeypatch):
    """Skills documents keep the ``source_documents`` path — the evidence builder never claims them,
    even with the flag ON (the two policies stay isolated)."""
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    from scrutator.search.indexer import _build_evidence_document

    assert _build_evidence_document(settings.skills_namespace, "s.md", "# S\n\nbody") is None


def test_build_evidence_document_flag_on_evidence_namespace_is_byte_exact(monkeypatch):
    """Flag ON + non-skills namespace → an evidence row whose ``raw_content`` is the SAME string the
    ``content_hash`` covers, so ``sha256(raw_content) == content_hash`` by construction."""
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    from scrutator.chunker.splitters import compute_doc_id
    from scrutator.search.indexer import _build_evidence_document, compute_doc_content_hash

    full_content = "# Evidence\n\n" + ("word " * 300)
    doc = _build_evidence_document("arcanada", "ev.md", full_content)
    assert doc is not None
    assert doc["raw_content"] == full_content
    assert doc["doc_id"] == compute_doc_id("arcanada", "ev.md")
    assert doc["content_hash"] == compute_doc_content_hash(full_content)
    assert _sha256_hex(doc["raw_content"]) == _bare_hash(doc["content_hash"])


def test_build_evidence_document_has_no_256kb_cap(monkeypatch):
    """Unlike the skills builder (256 KB-capped), evidence has NO per-document byte cap — a >256 KB
    evidence doc builds a row rather than raising (the evidence corpus holds large documents)."""
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    from scrutator.db.models import INDEX_BATCH_MAX_DOCUMENT_BYTES
    from scrutator.search.indexer import _build_evidence_document

    oversized = "# Big Evidence\n\n" + ("word " * ((INDEX_BATCH_MAX_DOCUMENT_BYTES // 5) + 200))
    assert len(oversized.encode()) > INDEX_BATCH_MAX_DOCUMENT_BYTES
    doc = _build_evidence_document("arcanada", "big-ev.md", oversized)
    assert doc is not None
    assert doc["raw_content"] == oversized
