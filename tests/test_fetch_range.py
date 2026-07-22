"""SRCH-0038 V-AC-8 — range semantics: full / parent_of_chunk / offset.

The offset slice is a response-time view over ingest-bound content; the returned content_hash
stays the WHOLE-doc ingest hash (S1 — the slice is never re-hashed).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from scrutator.db.models import FetchRequest

from .conftest import build_indexed_doc

_DOC = "# Guide\n\nIntro paragraph here. " + ("alpha " * 120) + "\n\n## Section Two\n\nBody. " + ("beta " * 120)


@pytest.mark.asyncio
async def test_range_full():
    _doc_id, content_hash, rows = build_indexed_doc(_DOC)
    from scrutator.search import fetcher

    with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows):
        resp = await fetcher.fetch(FetchRequest(by="source_id", id=_doc_id, range="full"), frozenset({1}))

    assert resp.content == "".join(r["content"] for r in rows)
    assert resp.content_hash == content_hash
    assert len(resp.chunk_manifest) == len(rows)
    # Manifest offsets are contiguous cumulative char lengths.
    assert resp.chunk_manifest[0].offset_start == 0
    for prev, nxt in zip(resp.chunk_manifest, resp.chunk_manifest[1:], strict=False):
        assert nxt.offset_start == prev.offset_end
    assert resp.chunk_manifest[-1].offset_end == len(resp.content)


@pytest.mark.asyncio
async def test_range_parent_of_chunk():
    """{"parent_of_chunk": cid} returns the chunk's whole parent document (auto-merge-to-parent)."""
    _doc_id, content_hash, rows = build_indexed_doc(_DOC)
    target_chunk = rows[2]["chunk_id"]
    from scrutator.search import fetcher

    with patch.object(fetcher, "fetch_chunks_by_chunk_id", new_callable=AsyncMock, return_value=rows) as m:
        resp = await fetcher.fetch(
            FetchRequest(by="chunk_id", id=target_chunk, range={"parent_of_chunk": target_chunk}),
            frozenset({1}),
        )

    # parent_of_chunk resolves via the chunk→doc lookup, returning the whole parent doc.
    m.assert_awaited_once_with(target_chunk, frozenset({1}))
    assert resp.content == "".join(r["content"] for r in rows)
    assert resp.content_hash == content_hash


@pytest.mark.asyncio
async def test_range_offset_slice_keeps_whole_doc_hash():
    _doc_id, content_hash, rows = build_indexed_doc(_DOC)
    full = "".join(r["content"] for r in rows)
    from scrutator.search import fetcher

    with patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows):
        resp = await fetcher.fetch(
            FetchRequest(by="source_id", id=_doc_id, range={"offset_start": 10, "offset_end": 40}),
            frozenset({1}),
        )

    # The returned content is the [10:40] slice of the reassembled full-doc content …
    assert resp.content == full[10:40]
    assert resp.content != full
    # … while the content_hash remains the whole-doc ingest hash (S1 — slice never re-hashed).
    assert resp.content_hash == content_hash
