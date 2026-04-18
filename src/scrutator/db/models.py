"""Pydantic models for search & index API contracts."""

from __future__ import annotations

from typing import Any

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


class SearchResponse(BaseModel):
    """API response for POST /v1/search."""

    results: list[SearchResult]
    total: int
    query: str
    search_time_ms: float
