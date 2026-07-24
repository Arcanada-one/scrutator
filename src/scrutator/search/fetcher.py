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
    InjectionSignal,
    OffsetRange,
    ParentOfChunkRange,
)
from scrutator.db.repository import (
    fetch_chunks_by_chunk_id,
    fetch_chunks_by_doc_id,
    fetch_evidence_raw_content,
    fetch_source_raw_content,
)
from scrutator.search.ingest_safety import source_trust_tier


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

    manifest: list[ChunkManifestEntry] = []
    cursor = 0
    for row in rows:
        length = len(row["content"])
        manifest.append(ChunkManifestEntry(chunk_id=row["chunk_id"], offset_start=cursor, offset_end=cursor + length))
        cursor += length

    first = rows[0]  # canonically chunk_index = 0 after ordering
    section = first["metadata"].get("section") or {}
    source_id = section.get("doc_id", "")
    # ARAS-0055: READ the ingest-stamped injection signal (server-computed at index time; the doc
    # body can never forge it — it only lands in `content`). Absent/legacy stamp ⇒ zero signal.
    injection = InjectionSignal(**(first["metadata"].get("injection") or {}))
    # S1: READ the ingest-bound whole-doc hash; NEVER recompute over the response. Legacy rows
    # lacking the stamp degrade to "" (never a recomputed value) — see backfill script / D3.
    content_hash = section.get("doc_content_hash", "")
    namespace = first["namespace"]
    source_path = first["source_path"]

    # SRCH-0038 1b: skills-namespace documents return the EXACT stored source bytes from the
    # isolated `source_documents` table (upserted at ingest inside the same transaction as the
    # chunks) so `sha256(content) == content_hash` by construction — the reassembly path below is
    # lossy (frontmatter/whitespace stripped, overlap duplicated) and would break the ARAS blake3
    # gate. The exact bytes are stored OUT of `chunks.metadata` because that GIN-indexed column
    # hit the ~2704-byte jsonb_ops entry ceiling on real multi-KB skills (the 1a seam).
    if namespace == settings.skills_namespace:
        raw_content = await fetch_source_raw_content(source_id, allowed_namespace_ids)
        if raw_content is None:
            # Fail closed / typed: a legacy skill with no source_documents row has no exact bytes.
            # Returning a hash-failing reassembly would be the exact integrity-theater this removes.
            raise HTTPException(
                status_code=409,
                detail="skills document missing exact source bytes (no source_documents row); re-index required",
            )
        full_content = raw_content
        content_exact = True
    else:
        # SRCH-0039 (Mechanism C): the LARGE evidence corpus gets exact whole-document bytes from
        # the isolated `evidence_documents` table, flag-gated and gracefully-degrading. When
        # `evidence_exact_bytes` is ON and a row is present, return its exact `raw_content`
        # (`content_exact=True`, `sha256(content) == content_hash` by construction). When the flag
        # is OFF (default — no behaviour change on merge) or the row is ABSENT (an expected
        # pre-backfill state on the huge existing corpus that is filled gradually), fall back to the
        # lossy chunk reassembly (`content_exact=False`). This absence path GRACEFULLY DEGRADES —
        # deliberately NOT the skills fail-closed 409 above — because evidence row-absence is a
        # transient migration state, not an integrity failure (the ratified skills-vs-evidence
        # policy divergence). The by-construction invariant still holds: whenever
        # `content_exact=True`, `sha256(content) == content_hash`.
        evidence_raw = None
        if settings.evidence_exact_bytes:
            evidence_row = await fetch_evidence_raw_content(source_id, allowed_namespace_ids)
            # Read-side belt-and-braces (SRCH-0039 pre-merge review): trust the exact-bytes row ONLY
            # when its bound-at-write `content_hash` still equals the current chunk stamp
            # (`content_hash` above). A stale row — one whose bytes predate a content change that
            # re-stamped the chunks — is REJECTED here and degrades to reassembly, so fetch can never
            # return stale bytes as `content_exact=True`. This compares two stored hashes (both bound
            # at write time), NOT a re-hash of the body, so it does not recompute integrity at read.
            if evidence_row is not None and evidence_row[1] == content_hash:
                evidence_raw = evidence_row[0]
        if evidence_raw is not None:
            full_content = evidence_raw
            content_exact = True
        else:
            # rows are ordered by chunk_index — reassembly is the inverse of the header-split.
            full_content = "".join(row["content"] for row in rows)
            content_exact = False

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
        # ARAS-0055: `trust_tier` (provenance) and `injection` (ingest scan) COMPOSE with
        # `trust_class` above; neither is an input to `_trust_class`, so a raw-tier or flagged
        # document keeps exactly the namespace-derived class — no cross-promotion to skill/exec.
        trust_tier=source_trust_tier(source_path),
        injection=injection,
        chunk_manifest=manifest,
        stale=False,  # D6 MVP: no live-source access; typed & present for forward-compat.
        content_exact=content_exact,
    )
