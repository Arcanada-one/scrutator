"""Pydantic models for search & index API contracts."""

from __future__ import annotations

import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class NamespaceCreate(BaseModel):
    """Request to create a namespace."""

    name: str
    description: str | None = None


class NamespaceInfo(BaseModel):
    """Namespace with aggregate stats."""

    id: int
    name: str
    description: str | None = None
    chunk_count: int = 0


class NamespaceStats(BaseModel):
    """Per-namespace statistics."""

    name: str
    chunk_count: int
    project_count: int


class IndexStats(BaseModel):
    """Overall index statistics."""

    total_chunks: int
    total_namespaces: int
    total_projects: int
    namespaces: list[NamespaceStats] = Field(default_factory=list)


class IndexRequest(BaseModel):
    """API request for POST /v1/index."""

    content: str
    source_path: str
    source_type: str | None = None
    namespace: str = "arcanada"
    project: str | None = None
    stream: str | None = None
    max_tokens: int = 512
    overlap_tokens: int = 50

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


class IndexResponse(BaseModel):
    """API response for POST /v1/index."""

    chunks_indexed: int
    source_path: str
    namespace: str
    strategy_used: str


INDEX_BATCH_MAX_DOCUMENT_BYTES = 262_144
INDEX_BATCH_MAX_REQUEST_BYTES = 1_048_576


class BatchIndexRequest(BaseModel):
    """Bounded request for POST /v1/index/batch."""

    documents: list[IndexRequest] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def one_namespace_and_unique_paths(self) -> BatchIndexRequest:
        namespaces = {document.namespace for document in self.documents}
        if len(namespaces) != 1:
            raise ValueError("all documents must use one namespace")
        paths = [document.source_path for document in self.documents]
        if len(paths) != len(set(paths)):
            raise ValueError("source_path values must be unique")
        if any(len(document.content.encode("utf-8")) > INDEX_BATCH_MAX_DOCUMENT_BYTES for document in self.documents):
            raise ValueError("document content exceeds batch byte limit")
        return self


class BatchIndexSucceeded(BaseModel):
    source_path: str
    status: Literal["succeeded"] = "succeeded"
    chunks_indexed: int


BatchIndexErrorCode = Literal[
    "chunking_failed",
    "dense_embedding_failed",
    "sparse_embedding_failed",
    "invalid_dense_embeddings",
    "invalid_sparse_embeddings",
    "persistence_failed",
]


class BatchIndexFailed(BaseModel):
    source_path: str
    status: Literal["failed"] = "failed"
    error_code: BatchIndexErrorCode


class BatchIndexResponse(BaseModel):
    results: list[BatchIndexSucceeded | BatchIndexFailed]


class DeleteSourceRequest(BaseModel):
    """Namespace-scoped tombstone request used by the audited Feeder."""

    namespace: str
    source_path: str


class DeleteSourceResponse(BaseModel):
    namespace: str
    source_path: str
    chunks_deleted: int


_MAX_SEARCH_LIMIT = 50


class SearchRequest(BaseModel):
    """API request for POST /v1/search."""

    query: str
    namespace: str | None = None
    project: str | None = None
    source_type: str | None = None
    limit: int = 10
    min_score: float = 0.0
    include_content: bool = True
    group_by: Literal["document", "section"] | None = None  # SRCH-0021, opt-in, default off

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("limit")
    @classmethod
    def limit_range(cls, v: int) -> int:
        if v < 1:
            raise ValueError("limit must be >= 1")
        return min(v, _MAX_SEARCH_LIMIT)


class ChunkLookupResult(BaseModel):
    """Result of chunk lookup by source_path."""

    chunk_id: str
    chunk_index: int
    source_path: str
    source_type: str
    content_preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Per-chunk source attribution. FROZEN interface contract consumed by
    ARCA-0180 (answer side). Version with `schema_version`; additive-only.

    score_kind disambiguates the scale of relevance_score:
    - 'rrf': RRF fused score (~[0, 0.05]); rerank_enabled=False
    - 'colbert_rerank': ColBERT MaxSim score (unbounded above); rerank_enabled=True
    """

    schema_version: int = 1  # bump only on breaking shape change
    chunk_id: str
    source_path: str  # relative KB path, e.g. "concepts/architecture.md"
    source_type: str  # "md" | "pdf" | "code"
    chunk_index: int  # ordinal in source
    heading_hierarchy: list[str] = Field(default_factory=list)
    relevance_score: float  # score that produced the FINAL ordering
    score_kind: Literal["rrf", "colbert_rerank"]  # which score relevance_score holds


class SearchResult(BaseModel):
    """A single search result with source attribution."""

    chunk_id: str
    content: str = ""
    source_path: str
    source_type: str
    chunk_index: int
    score: float
    namespace: str
    project: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    heading_hierarchy: list[str] = Field(default_factory=list)
    citation: Citation | None = None  # M1 (SRCH-0029): typed source attribution; None until populated by searcher
    # SRCH-0038 (D2): additive, defaulted → non-breaking for the frozen search-baseline contract.
    # `source_id` is the opaque doc id (fetch selector); `content_hash` is the document-level
    # sha256 stamped at ingest — equal to the whole-doc hash returned by POST /v1/fetch (roundtrip).
    content_hash: str = ""
    source_id: str = ""


def doc_fields_from_metadata(metadata: dict[str, Any] | None) -> tuple[str, str]:
    """SRCH-0038 (D2): project ``(source_id, content_hash)`` from a chunk's stored
    ``metadata.section``. Returns ``("", "")`` when the section or keys are absent (un-backfilled
    legacy rows) — never a recomputed value. Used by both searcher projection sites so the
    /v1/search hit's ``content_hash`` equals the whole-doc hash returned by /v1/fetch."""
    section = (metadata or {}).get("section") or {}
    return section.get("doc_id", ""), section.get("doc_content_hash", "")


class GroupedSearchResult(BaseModel):
    """A `group_by`-folded search result (SRCH-0021, D-REQ-05). Only present when
    `SearchRequest.group_by` is set; the default (absent) search path never builds these."""

    group_key: str
    doc_id: str = ""
    score: float
    representative: SearchResult
    member_chunk_ids: list[str] = Field(default_factory=list)
    member_count: int = 0


class SearchResponse(BaseModel):
    """API response for POST /v1/search.

    `results` is `list[SearchResult]` when `group_by` is absent (default, byte-identical
    to pre-SRCH-0021 behaviour — V-AC-6) or `list[GroupedSearchResult]` when set (D-REQ-05).
    """

    results: list[SearchResult] | list[GroupedSearchResult]
    total: int
    query: str
    search_time_ms: float


# ── SRCH-0038: exact whole-document fetch-by-id ──────────────────────

# Opaque document identity: compute_doc_id() → sha256(...)[:16] → 16 lowercase-hex chars.
# `document_id` and `source_id` selectors validate against this; anything path-like or
# malformed is rejected here (S3), pre-DB, before any query is constructed.
_DOC_ID_RE = re.compile(r"[0-9a-f]{16}")


def _is_opaque_doc_id(value: str) -> bool:
    return bool(_DOC_ID_RE.fullmatch(value))


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


class ParentOfChunkRange(BaseModel):
    """`range` variant: return the whole parent document of a given chunk (auto-merge-to-parent)."""

    parent_of_chunk: str
    model_config = {"extra": "forbid"}


class OffsetRange(BaseModel):
    """`range` variant: a `[offset_start:offset_end]` character slice of the reassembled full-doc
    content. The returned `content_hash` remains the WHOLE-doc ingest hash — the slice is never
    re-hashed (S1). Offsets are relative to Scrutator's reassembled concatenation (MVP, D4)."""

    offset_start: int = Field(ge=0)
    offset_end: int = Field(ge=0)
    model_config = {"extra": "forbid"}


class FetchRequest(BaseModel):
    """API request for POST /v1/fetch (SRCH-0038). Closed model (S4).

    Selectors are opaque-only (S3): `document_id`/`source_id` are aliases for the same 16-hex
    opaque doc id; `chunk_id` is a UUID. No selector accepts a filesystem path — a path-like or
    malformed id raises ValidationError (422) before any DB access.
    """

    by: Literal["document_id", "source_id", "chunk_id"]
    id: str
    range: Literal["full"] | ParentOfChunkRange | OffsetRange = "full"
    include: list[Literal["content", "provenance"]] = Field(default_factory=lambda: ["content", "provenance"])
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_selector(self) -> FetchRequest:
        if self.by in ("document_id", "source_id"):
            if not _is_opaque_doc_id(self.id):
                raise ValueError("id must be a 16-character lowercase-hex opaque document id")
        elif not _is_uuid(self.id):  # chunk_id
            raise ValueError("chunk_id must be a UUID")
        if isinstance(self.range, ParentOfChunkRange) and not _is_uuid(self.range.parent_of_chunk):
            raise ValueError("parent_of_chunk must be a UUID")
        if isinstance(self.range, OffsetRange) and self.range.offset_end < self.range.offset_start:
            raise ValueError("offset_end must be >= offset_start")
        return self


class ChunkManifestEntry(BaseModel):
    """One chunk's position within the reassembled document (cumulative character offsets)."""

    chunk_id: str
    offset_start: int
    offset_end: int


class FetchResponse(BaseModel):
    """API response for POST /v1/fetch (SRCH-0038). Closed model (S4/D9).

    `content` is the ONLY free-text sink — document bytes can land nowhere else. Every other
    field is derived server-side from DB columns / config (never from document body text), so
    a document whose body literally contains e.g. `"trust_class":"skill"` cannot forge metadata.

    `content_hash` is the whole-document sha256 stamped at ingest and READ here — never
    recomputed over this response (S1). `trust_class` is a namespace-derived, NON-AUTHORIZING
    hint (D5): returning `"skill"` does NOT authorize execution — the execution gate is the ARAS
    interpreter's config-pinned blake3 (D8); Scrutator remains untrusted transport.
    """

    source_id: str
    path: str
    content: str
    content_len_tokens: int
    content_hash: str
    index_snapshot_id: str
    indexed_at: str
    embedding_model_id: str
    namespace: str
    trust_class: Literal["skill", "evidence"]
    chunk_manifest: list[ChunkManifestEntry] = Field(default_factory=list)
    stale: bool = False


# ── SRCH-0021: hierarchical navigation ───────────────────────────────


class OutlineNode(BaseModel):
    """One node of a document's table-of-contents tree."""

    title: str
    anchor: str
    depth: int
    section_key: str
    chunk_ids: list[str] = Field(default_factory=list)
    children: list[OutlineNode] = Field(default_factory=list)


class OutlineResponse(BaseModel):
    """API response for GET /v1/navigate/outline."""

    source_path: str
    namespace: str
    doc_id: str
    total_chunks: int
    outline: list[OutlineNode] = Field(default_factory=list)


class SectionBreadcrumb(BaseModel):
    """A lightweight section reference (used for ancestors/siblings/children)."""

    title: str
    section_key: str
    depth: int


class SectionSelf(SectionBreadcrumb):
    """The target section itself, additionally carrying its chunk ids."""

    chunk_ids: list[str] = Field(default_factory=list)


class SectionContext(BaseModel):
    """API response for GET /v1/navigate/section."""

    chunk_id: str
    doc_id: str
    section_key: str
    ancestors: list[SectionBreadcrumb] = Field(default_factory=list)
    self_: SectionSelf = Field(alias="self")
    siblings: list[SectionBreadcrumb] = Field(default_factory=list)
    children: list[SectionBreadcrumb] = Field(default_factory=list)

    model_config = {"populate_by_name": True}
