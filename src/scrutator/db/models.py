"""Pydantic models for search & index API contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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
