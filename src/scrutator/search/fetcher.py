"""Exact whole-document fetch-by-id orchestration (SRCH-0038).

Assembles a closed :class:`FetchResponse` from namespace-scoped chunk rows: reassembles the
document by ``chunk_index`` order, applies the requested ``range``, derives provenance, and
labels a non-authorizing ``trust_class``.

Security posture:
- **S1** — ``content_hash`` is the whole-document sha256 STAMPED AT INGEST
  (``metadata.section.doc_content_hash``) and READ here. It is never recomputed over the
  assembled response; an ``offset`` slice never re-hashes (the whole-doc hash is returned).
- **S2** — every DB read is namespace-scoped in the repository layer
  (``namespace_id = ANY(allowed)``); an unknown / cross-namespace id yields ``[]`` → 404, with
  no existence oracle.
- **S4** — the response is a closed model; document body text can only reach ``content``. Every
  other field is derived server-side from DB columns / config (``trust_class`` from namespace).
"""

from __future__ import annotations

import hashlib

from fastapi import HTTPException

from scrutator.chunker.tokenizer import token_count
from scrutator.config import settings
from scrutator.db.models import (
    ChunkManifestEntry,
    FetchRequest,
    FetchResponse,
    OffsetRange,
    ParentOfChunkRange,
)
from scrutator.db.repository import fetch_chunks_by_chunk_id, fetch_chunks_by_doc_id


def _derive_index_snapshot_id(doc_key: str, max_indexed_at: str) -> str:
    """Deterministic, physical-object-free snapshot id (D7): stable for identical index state,
    changes when the document is re-indexed (``max_indexed_at`` moves)."""
    return hashlib.sha256(f"{doc_key}|{max_indexed_at}".encode()).hexdigest()[:16]


def _trust_class(namespace: str) -> str:
    """D5: namespace-derived, NON-AUTHORIZING hint. ``"skill"`` does not authorize execution —
    the execution gate is the ARAS interpreter's config-pinned blake3 (D8)."""
    return "skill" if namespace == settings.skills_namespace else "evidence"


async def _resolve_rows(request: FetchRequest, allowed_namespace_ids: frozenset[int]) -> list[dict]:
    """Selector → ordered chunk rows (namespace-scoped). ``parent_of_chunk`` overrides the
    top-level selector: it resolves the referenced chunk to its whole parent document (D4)."""
    if isinstance(request.range, ParentOfChunkRange):
        return await fetch_chunks_by_chunk_id(request.range.parent_of_chunk, allowed_namespace_ids)
    if request.by == "chunk_id":
        return await fetch_chunks_by_chunk_id(request.id, allowed_namespace_ids)
    return await fetch_chunks_by_doc_id(request.id, allowed_namespace_ids)


async def fetch(request: FetchRequest, allowed_namespace_ids: frozenset[int]) -> FetchResponse:
    """Resolve a selector to a whole document (or bounded range) and return a closed
    :class:`FetchResponse`. Raises ``HTTPException(404)`` for an unknown / cross-namespace id."""
    rows = await _resolve_rows(request, allowed_namespace_ids)
    if not rows:
        # No existence oracle: unknown id and cross-namespace id are indistinguishable (S2).
        raise HTTPException(status_code=404, detail="document not found")

    # rows are ordered by chunk_index — reassembly is the inverse of the header-split.
    full_content = "".join(row["content"] for row in rows)

    manifest: list[ChunkManifestEntry] = []
    cursor = 0
    for row in rows:
        length = len(row["content"])
        manifest.append(ChunkManifestEntry(chunk_id=row["chunk_id"], offset_start=cursor, offset_end=cursor + length))
        cursor += length

    first = rows[0]  # canonically chunk_index = 0 after ordering
    section = first["metadata"].get("section") or {}
    source_id = section.get("doc_id", "")
    # S1: READ the ingest-bound whole-doc hash; NEVER recompute over the response. Legacy rows
    # lacking the stamp degrade to "" (never a recomputed value) — see backfill script / D3.
    content_hash = section.get("doc_content_hash", "")
    namespace = first["namespace"]
    source_path = first["source_path"]

    indexed_ats = [row["indexed_at"] for row in rows if row.get("indexed_at")]
    max_indexed_at = max(indexed_ats) if indexed_ats else ""
    index_snapshot_id = _derive_index_snapshot_id(source_id or source_path, max_indexed_at)

    # Range application (D4). The content_hash always stays the WHOLE-doc ingest hash (S1) —
    # an offset slice is a response-time view over ingest-bound content, never re-hashed.
    content = full_content
    if isinstance(request.range, OffsetRange):
        content = full_content[request.range.offset_start : request.range.offset_end]

    return FetchResponse(
        source_id=source_id,
        path=source_path,
        content=content,
        content_len_tokens=token_count(content),
        content_hash=content_hash,
        index_snapshot_id=index_snapshot_id,
        indexed_at=max_indexed_at,
        embedding_model_id=settings.embedding_model_id,
        namespace=namespace,
        trust_class=_trust_class(namespace),
        chunk_manifest=manifest,
        stale=False,  # D6 MVP: no live-source access; typed & present for forward-compat.
    )
