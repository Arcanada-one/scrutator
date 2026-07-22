"""Shared test fixtures (SRCH-0023): FastAPI auth-dependency override helper.

Pre-SRCH-0023 route tests exercised `/v1/*` and `/v1/ltm/*` endpoints with no credential,
relying on the (now-closed) unauthenticated-default behaviour. Those tests use
`override_tenant_context` to simulate an authenticated caller scoped to a known namespace,
matching the endpoints' new `Depends(require_tenant_context)` requirement.
"""

from __future__ import annotations

from contextlib import contextmanager

from scrutator.auth.dependency import require_tenant_context
from scrutator.auth.models import TenantContext

# ── SRCH-0038: fetch-endpoint test helper ────────────────────────────
#
# The whole suite is mock-based (no live Postgres); `build_indexed_doc` reproduces what the
# indexer stamps — real chunking + the real whole-doc `doc_content_hash` — in the row shape
# `repository.fetch_chunks_by_doc_id` / `fetch_chunks_by_chunk_id` return, so fetch tests
# exercise the genuine ingest→read hash path with the DB bypassed (patch the repository fns).

_DEFAULT_INDEXED_AT = "2026-07-22T10:00:00+00:00"


def build_indexed_doc(
    content: str,
    namespace: str = "arcanada",
    source_path: str = "doc.md",
    indexed_at: str = _DEFAULT_INDEXED_AT,
):
    """Return ``(doc_id, content_hash, rows)`` for ``content`` exactly as the indexer would stamp
    them (real ``chunk_document`` + real ``compute_doc_content_hash``), in fetch-row shape.

    ``content`` MUST begin with a markdown heading so every chunk carries a ``section`` (and thus
    the ``doc_id`` / ``doc_content_hash`` stamp) — matching how fetch-by-doc_id resolves rows.
    """
    from scrutator.chunker.splitters import compute_doc_id
    from scrutator.search.indexer import _chunk_dicts, chunk_document, compute_doc_content_hash

    result = chunk_document(content, source_path, max_tokens=64, overlap_tokens=8)
    chunk_dicts = _chunk_dicts(result, namespace, source_path, content)
    doc_id = compute_doc_id(namespace, source_path)
    content_hash = compute_doc_content_hash(content)
    rows = [
        {
            "chunk_id": cd["id"],
            "chunk_index": cd["chunk_index"],
            "content": cd["content"],
            "content_hash": cd["content_hash"],
            "source_path": cd["source_path"],
            "source_type": cd["source_type"],
            "token_count": cd["token_count"],
            "metadata": cd["metadata"],
            "indexed_at": indexed_at,
            "namespace": namespace,
        }
        for cd in chunk_dicts
    ]
    return doc_id, content_hash, rows


def make_tenant_context(
    namespace_ids: frozenset[int] = frozenset({1}),
    namespace_names: frozenset[str] = frozenset({"arcanada"}),
    principal_id: str = "test-principal",
) -> TenantContext:
    return TenantContext(
        principal_id=principal_id,
        principal_type="service",
        allowed_namespace_ids=namespace_ids,
        allowed_namespace_names=namespace_names,
    )


@contextmanager
def override_tenant_context(app, ctx: TenantContext | None = None):
    """Override the `require_tenant_context` FastAPI dependency for the duration of the block."""
    ctx = ctx or make_tenant_context()
    app.dependency_overrides[require_tenant_context] = lambda: ctx
    try:
        yield ctx
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)
