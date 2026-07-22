"""SRCH-0038 1b — skills-namespace fetch returns byte-exact source; evidence stays approximate.

This is THE Definition-of-Done guardrail for the re-run: the QA block proved that the shipped
fetcher reassembles a document from embedding chunks (``"".join(chunks)``) while advertising the
pre-chunk ``content_hash``, so ``sha256(returned content) != content_hash`` for every realistic
doc (frontmatter dropped, ``overlap_tokens`` duplication, per-section ``.strip()``) and a
multi-chunk skill plan comes back as invalid JSON — breaking the ARAS-0047 config-pinned blake3
gate. These tests assert the exact scenario that exposed the reassembly lie.

Storage under 1b: the exact source bytes live in the isolated ``source_documents`` table (read via
``repository.fetch_source_raw_content``), NOT in ``chunks.metadata`` — the 1a metadata seam raised
a multi-KB blob into the ``idx_chunks_metadata`` GIN index and hit the ~2704-byte ``jsonb_ops``
entry ceiling on real skills. These mock tests patch that accessor to stand in for the row.

Mutation proof: each skills test also confirms the *reassembly* hash FAILS — i.e. the reassembly
path would fail this test, and only the exact-bytes path passes it.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from scrutator.chunker.models import SectionMeta
from scrutator.db.models import FetchRequest
from scrutator.search.indexer import _stamp_doc_id, compute_doc_content_hash

from ..conftest import build_indexed_doc

# A realistic skill doc: YAML frontmatter + leading/trailing whitespace + a heading + a body long
# enough to force >1 chunk with an overlap region. Reassembling its embedding chunks is lossy
# (frontmatter stripped, chunk-edge `.strip()`, overlap duplication) → the reassembly hash fails.
_SKILL_FRONTMATTER = (
    "---\n"
    "name: build-release\n"
    "version: 7\n"
    "trust: skill\n"
    "---\n"
    "\n"
    "# Build Release\n\n"
    "   \n"  # leading body whitespace the per-chunk .strip() drops
    "Step one: assemble the artifact. " + ("word " * 220) + "\n\n"
    "## Step Two\n\nSign and publish. " + ("token " * 220) + "\n   \n"  # trailing whitespace
)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _bare_hash(content_hash: str) -> str:
    """Strip the `sha256:` prefix the ingest stamp carries."""
    return content_hash.split(":", 1)[1] if ":" in content_hash else content_hash


@pytest.mark.asyncio
async def test_skills_multichunk_fetch_is_byte_exact_and_reassembly_would_fail():
    """THE DoD guardrail: a multi-chunk skills doc with frontmatter + whitespace + overlap, fetched
    at range="full", returns byte-exact source → sha256(content) == content_hash. The pre-1a
    reassembly path FAILS this exact doc (proven by the reassembly-hash assertion)."""
    doc_id, content_hash, rows = build_indexed_doc(_SKILL_FRONTMATTER, namespace="skills")
    assert len(rows) > 1, "doc must span >1 chunk to exercise reassembly"

    reassembly = "".join(r["content"] for r in rows)
    # Precondition: this doc is genuinely lossy under reassembly — the bug being guarded against.
    assert reassembly != _SKILL_FRONTMATTER
    assert _sha256_hex(reassembly) != _bare_hash(content_hash)

    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(fetcher, "fetch_source_raw_content", new_callable=AsyncMock, return_value=_SKILL_FRONTMATTER),
    ):
        resp = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))

    assert resp.trust_class == "skill"
    assert resp.content_exact is True
    # Keystone: the returned bytes hash to the advertised content_hash, by construction.
    assert _sha256_hex(resp.content) == _bare_hash(content_hash)
    # And the returned content is the exact source, not the lossy reassembly.
    assert resp.content == _SKILL_FRONTMATTER
    assert resp.content != reassembly


@pytest.mark.asyncio
async def test_skills_json_plan_fetch_is_parseable_json():
    """A JSON skill plan (the ARAS-0047 artifact) fetched from the skills namespace is byte-exact
    AND `json.loads`-parseable. Drives the REAL ingest stamp (`_stamp_doc_id` /
    `compute_doc_content_hash`) over a JSON payload chunked into >1 overlapping, edge-stripped
    embedding chunks — the reassembly of which is invalid JSON."""
    body = json.dumps({"name": "deploy", "steps": [{"id": i, "do": "x" * 30} for i in range(60)]}, indent=2)
    plan = "  \n" + body + "\n  \n"
    content_hash = compute_doc_content_hash(plan)

    # Simulate the embedding-chunk split: an overlap region + per-chunk `.strip()` (what the real
    # chunker does), so the reassembly is lossy and would be invalid JSON.
    mid = len(plan) // 2
    chunk_texts = [plan[: mid + 40].strip(), plan[mid:].strip()]
    reassembly = "".join(chunk_texts)
    assert reassembly != plan
    with pytest.raises(json.JSONDecodeError):
        json.loads(reassembly)  # the lossy reassembly is NOT valid JSON — the bug

    section = SectionMeta(
        heading_path=["deploy"], depth=1, anchor="deploy", anchor_path=["deploy"], section_key="deploy"
    )
    # 1b: `doc_raw_content` is NO LONGER stamped into metadata — the exact bytes live in
    # source_documents. The section stamp carries only doc_id + doc_content_hash now.
    stamp0 = _stamp_doc_id(section, "skills", "deploy.json", content_hash)
    stamp1 = _stamp_doc_id(section, "skills", "deploy.json", content_hash)
    doc_id = stamp0["doc_id"]
    rows = [
        {
            "chunk_id": "11111111-1111-1111-1111-111111111111",
            "chunk_index": 0,
            "content": chunk_texts[0],
            "content_hash": _sha256_hex(chunk_texts[0]),
            "source_path": "deploy.json",
            "source_type": "json",
            "token_count": 1,
            "metadata": {"section": stamp0},
            "indexed_at": "2026-07-22T10:00:00+00:00",
            "namespace": "skills",
        },
        {
            "chunk_id": "22222222-2222-2222-2222-222222222222",
            "chunk_index": 1,
            "content": chunk_texts[1],
            "content_hash": _sha256_hex(chunk_texts[1]),
            "source_path": "deploy.json",
            "source_type": "json",
            "token_count": 1,
            "metadata": {"section": stamp1},
            "indexed_at": "2026-07-22T10:00:00+00:00",
            "namespace": "skills",
        },
    ]

    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(fetcher, "fetch_source_raw_content", new_callable=AsyncMock, return_value=plan),
    ):
        resp = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))

    assert resp.content_exact is True
    assert _sha256_hex(resp.content) == _bare_hash(content_hash)
    # The keystone payoff: the byte-exact content parses as the intended JSON skill plan.
    parsed = json.loads(resp.content)
    assert parsed["name"] == "deploy"
    assert len(parsed["steps"]) == 60


@pytest.mark.asyncio
async def test_evidence_path_is_approximate_and_skills_path_is_exact():
    """The evidence corpus keeps the (lossy) reassembly, flagged `content_exact=False`; the skills
    namespace returns exact bytes flagged `content_exact=True`."""
    from scrutator.search import fetcher

    # Evidence namespace → reassembly, approximate.
    ev_id, _ev_hash, ev_rows = build_indexed_doc(_SKILL_FRONTMATTER, namespace="arcanada")
    with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=ev_rows):
        ev = await fetcher.fetch(FetchRequest(by="source_id", id=ev_id, range="full"), frozenset({1}))
    assert ev.trust_class == "evidence"
    assert ev.content_exact is False
    assert ev.content == "".join(r["content"] for r in ev_rows)

    # Skills namespace → exact (bytes come from the source_documents accessor, not metadata).
    sk_id, _sk_hash, sk_rows = build_indexed_doc(_SKILL_FRONTMATTER, namespace="skills")
    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=sk_rows),
        patch.object(fetcher, "fetch_source_raw_content", new_callable=AsyncMock, return_value=_SKILL_FRONTMATTER),
    ):
        sk = await fetcher.fetch(FetchRequest(by="source_id", id=sk_id, range="full"), frozenset({1}))
    assert sk.trust_class == "skill"
    assert sk.content_exact is True
    assert sk.content == _SKILL_FRONTMATTER


@pytest.mark.asyncio
async def test_legacy_skill_without_raw_content_fails_closed():
    """A skills doc with NO `source_documents` row (legacy skill indexed before 1b) must fail
    closed (409), never silently return a hash-failing reassembly. The accessor returns None."""
    doc_id, _hash, rows = build_indexed_doc(_SKILL_FRONTMATTER, namespace="skills")

    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(fetcher, "fetch_source_raw_content", new_callable=AsyncMock, return_value=None),
        pytest.raises(HTTPException) as excinfo,
    ):
        await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))
    assert excinfo.value.status_code == 409


def test_skills_exact_bytes_blob_is_size_guarded():
    """The single POST /v1/index path has no per-document byte cap; the exact-bytes source_documents
    write must not let it persist an unbounded skills blob. A skills doc over the 256 KB cap raises
    (in `_build_source_document`, invoked by `_chunk_dicts`); the same oversized doc in a non-skills
    namespace does NOT (the guard is skills-scoped)."""
    from scrutator.db.models import INDEX_BATCH_MAX_DOCUMENT_BYTES
    from scrutator.search.indexer import BatchIndexLimitError, _chunk_dicts, chunk_document

    oversized = "# Big Skill\n\n" + ("word " * ((INDEX_BATCH_MAX_DOCUMENT_BYTES // 5) + 100))
    assert len(oversized.encode()) > INDEX_BATCH_MAX_DOCUMENT_BYTES
    result = chunk_document(oversized, "big.md", max_tokens=64, overlap_tokens=8)

    with pytest.raises(BatchIndexLimitError):
        _chunk_dicts(result, "skills", "big.md", oversized)

    # Non-skills namespace never builds a source_documents row, so it is not size-guarded here.
    _chunk_dicts(result, "arcanada", "big.md", oversized)  # must not raise
