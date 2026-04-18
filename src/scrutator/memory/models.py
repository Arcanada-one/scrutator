"""Pydantic models for memory indexing and recall API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

_VALID_MEMORY_TYPES = {"fact", "preference", "decision", "event", "observation"}
_MAX_CONTENT_LENGTH = 10_000
_MAX_BULK_SIZE = 100
_MAX_RECALL_LIMIT = 50


class MemoryRecord(BaseModel):
    """A single memory to index."""

    content: str
    actor: str
    memory_type: str = "fact"
    namespace: str = "arcanada"
    project: str | None = None
    tags: list[str] = Field(default_factory=list)
    importance: float = 0.5
    valid_from: str | None = None
    valid_until: str | None = None
    source_ref: str | None = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content must not be empty")
        if len(v) > _MAX_CONTENT_LENGTH:
            raise ValueError(f"content exceeds {_MAX_CONTENT_LENGTH} characters")
        return v

    @field_validator("actor")
    @classmethod
    def actor_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("actor must not be empty")
        return v.strip()

    @field_validator("memory_type")
    @classmethod
    def valid_memory_type(cls, v: str) -> str:
        if v not in _VALID_MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {_VALID_MEMORY_TYPES}")
        return v

    @field_validator("importance")
    @classmethod
    def importance_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("importance must be between 0.0 and 1.0")
        return v


class MemoryIndexResponse(BaseModel):
    """Response for POST /v1/memories."""

    memory_id: str
    chunk_id: str
    namespace: str


class MemoryBulkRequest(BaseModel):
    """Request for POST /v1/memories/bulk."""

    memories: list[MemoryRecord]

    @field_validator("memories")
    @classmethod
    def bulk_size_limit(cls, v: list[MemoryRecord]) -> list[MemoryRecord]:
        if len(v) > _MAX_BULK_SIZE:
            raise ValueError(f"bulk size exceeds {_MAX_BULK_SIZE}")
        if not v:
            raise ValueError("memories list must not be empty")
        return v


class MemoryBulkResponse(BaseModel):
    """Response for POST /v1/memories/bulk."""

    indexed: int
    memory_ids: list[str]


class MemoryRecallRequest(BaseModel):
    """Request for POST /v1/memories/recall."""

    query: str
    namespace: str | None = None
    project: str | None = None
    actor: str | None = None
    memory_type: str | None = None
    limit: int = 10
    min_score: float = 0.0
    include_expired: bool = False
    importance_boost: bool = True

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
        return min(v, _MAX_RECALL_LIMIT)

    @field_validator("memory_type")
    @classmethod
    def valid_memory_type(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {_VALID_MEMORY_TYPES}")
        return v


class MemoryRecallResult(BaseModel):
    """A single recalled memory."""

    memory_id: str
    content: str
    actor: str
    memory_type: str
    importance: float
    score: float
    namespace: str
    project: str | None = None
    tags: list[str] = Field(default_factory=list)
    valid_from: str | None = None
    valid_until: str | None = None
    source_ref: str | None = None
    created_at: str | None = None


class MemoryRecallResponse(BaseModel):
    """Response for POST /v1/memories/recall."""

    results: list[MemoryRecallResult]
    total: int
    query: str
    search_time_ms: float


class MemoryStats(BaseModel):
    """Memory statistics."""

    total_memories: int
    by_namespace: dict[str, int] = Field(default_factory=dict)
    by_actor: dict[str, int] = Field(default_factory=dict)
    by_type: dict[str, int] = Field(default_factory=dict)


class MemoryDeleteRequest(BaseModel):
    """Query params for DELETE /v1/memories."""

    actor: str
    namespace: str | None = None

    @field_validator("actor")
    @classmethod
    def actor_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("actor must not be empty")
        return v.strip()
