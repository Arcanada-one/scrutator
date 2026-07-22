"""SRCH-0038 S1 / V-AC-4 — content_hash is bound at ingest and READ, never recomputed.

Mutation proof: after ingest, mutate a stored chunk's content while leaving the stored
doc_content_hash intact → fetch → the returned content_hash still equals the ingest-bound
value. A recompute-over-response implementation would return a DIFFERENT hash and fail here.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, patch

import pytest

from scrutator.db.models import FetchRequest

from ..conftest import build_indexed_doc

_DOC = (
    "# Skill Plan\n\nStep one: do the thing. " + ("word " * 120) + "\n\n## Step Two\n\nFinish it. " + ("token " * 120)
)


@pytest.mark.asyncio
async def test_content_hash_is_stored_not_recomputed():
    doc_id, ingest_hash, rows = build_indexed_doc(_DOC, namespace="arcanada")

    # Tamper with a stored chunk's content (as a DB-level corruption / poisoning would), but
    # leave metadata.section.doc_content_hash intact — exactly the scenario S1 must survive.
    rows[1]["content"] = "TAMPERED-BYTES-INJECTED-AFTER-INGEST"
    reassembled_after_tamper = "".join(r["content"] for r in rows)
    recompute_hash = "sha256:" + hashlib.sha256(reassembled_after_tamper.encode()).hexdigest()

    from scrutator.search import fetcher

    with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows):
        resp = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id), frozenset({1}))

    # The returned hash is the ingest-bound stamp — unchanged by the content tamper.
    assert resp.content_hash == ingest_hash
    # And it is NOT what a recompute-over-response would have produced.
    assert resp.content_hash != recompute_hash
    # The tampered content DID flow through to `content` (fetch does not silently "heal" it) —
    # proving the hash's stability is real, not an artifact of returning pristine content.
    assert "TAMPERED-BYTES-INJECTED-AFTER-INGEST" in resp.content


@pytest.mark.asyncio
async def test_offset_slice_returns_whole_doc_hash():
    """S1 corollary: an offset slice never re-hashes — the whole-doc ingest hash is returned."""
    doc_id, ingest_hash, rows = build_indexed_doc(_DOC, namespace="arcanada")
    from scrutator.search import fetcher

    with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows):
        resp = await fetcher.fetch(
            FetchRequest(by="source_id", id=doc_id, range={"offset_start": 5, "offset_end": 25}),
            frozenset({1}),
        )
    assert resp.content_hash == ingest_hash
