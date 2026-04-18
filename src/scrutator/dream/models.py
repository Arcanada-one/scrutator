"""Pydantic models for dream analysis requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class DreamAnalysisRequest(BaseModel):
    """Request for dream analysis on a namespace."""

    namespace: str
    min_similarity: float = 0.7
    dedup_threshold: float = 0.92
    max_results_per_type: int = 50
    stale_days: int = 90
    include_boost: bool = True

    @field_validator("namespace")
    @classmethod
    def namespace_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("namespace must not be empty")
        return v.strip()

    @field_validator("min_similarity", "dedup_threshold")
    @classmethod
    def threshold_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        return v

    @field_validator("max_results_per_type")
    @classmethod
    def max_results_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_results_per_type must be >= 1")
        return min(v, 200)


class DuplicatePair(BaseModel):
    """Two chunks with very high semantic similarity (likely duplicates)."""

    chunk_id_a: str
    chunk_id_b: str
    similarity: float
    source_path_a: str
    source_path_b: str
    content_preview_a: str
    content_preview_b: str


class CrossReference(BaseModel):
    """Two chunks that are semantically related but not yet linked."""

    chunk_id_a: str
    chunk_id_b: str
    similarity: float
    source_path_a: str
    source_path_b: str
    suggested_edge_type: str = "related"


class OrphanChunk(BaseModel):
    """A chunk with no graph edges — isolated knowledge."""

    chunk_id: str
    source_path: str
    edge_count: int
    created_at: str


class StaleChunk(BaseModel):
    """A chunk that hasn't been updated in a long time."""

    chunk_id: str
    source_path: str
    days_since_update: int
    edge_count: int


class BoostScore(BaseModel):
    """Relevance boost based on edge connectivity."""

    chunk_id: str
    source_path: str
    edge_count: int
    avg_edge_weight: float
    boost_score: float


class EdgeCreate(BaseModel):
    """Request to create a graph edge."""

    source_chunk_id: str
    target_chunk_id: str
    edge_type: str
    weight: float = 1.0
    created_by: str = "dreamer"

    @field_validator("edge_type")
    @classmethod
    def edge_type_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("edge_type must not be empty")
        return v.strip()


class EdgeCreateByPath(BaseModel):
    """Request to create a graph edge using source_paths instead of chunk UUIDs."""

    source_path: str
    target_path: str
    edge_type: str
    weight: float = 1.0
    created_by: str = "dreamer"
    source_chunk_index: int = 0
    target_chunk_index: int = 0

    @field_validator("edge_type")
    @classmethod
    def edge_type_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("edge_type must not be empty")
        return v.strip()

    @field_validator("source_path", "target_path")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("path must not be empty")
        return v.strip()


class EdgeCreateByPathResponse(BaseModel):
    """Response for edge creation by path."""

    created: int
    not_found: list[str]


class EdgeInfo(BaseModel):
    """Graph edge with full info."""

    id: int
    source_chunk_id: str
    target_chunk_id: str
    edge_type: str
    weight: float
    created_by: str
    created_at: str


class DreamAnalysisResult(BaseModel):
    """Full result of dream analysis for a namespace."""

    namespace: str
    duplicates: list[DuplicatePair]
    cross_references: list[CrossReference]
    orphans: list[OrphanChunk]
    stale: list[StaleChunk]
    boosts: list[BoostScore]
    stats: dict
