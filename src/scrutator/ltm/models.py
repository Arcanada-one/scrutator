"""Pydantic models for the LTM pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

_MAX_CONTENT_LENGTH = 500_000
_MAX_RECALL_LIMIT = 50


class JobStatus(StrEnum):
    PENDING = "pending"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    DEDUPING = "deduping"
    DONE = "done"
    FAILED = "failed"


class IngestRequest(BaseModel):
    """Request for POST /v1/ltm/ingest."""

    content: str
    source_path: str
    namespace: str = "arcanada"
    project: str | None = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content must not be empty")
        if len(v) > _MAX_CONTENT_LENGTH:
            raise ValueError(f"content exceeds {_MAX_CONTENT_LENGTH} characters")
        return v

    @field_validator("source_path")
    @classmethod
    def source_path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_path must not be empty")
        return v.strip()


class IngestResponse(BaseModel):
    """Response for POST /v1/ltm/ingest."""

    job_id: str
    status: JobStatus


class Entity(BaseModel):
    """An extracted named entity."""

    name: str
    entity_type: str
    description: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty")
        return v.strip()

    @field_validator("entity_type")
    @classmethod
    def type_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("entity_type must not be empty")
        return v.strip()


class EntityEdge(BaseModel):
    """A relationship between two entities."""

    source: str
    target: str
    relation: str
    weight: float = 1.0

    @field_validator("relation")
    @classmethod
    def relation_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("relation must not be empty")
        return v.strip()


class LtmJob(BaseModel):
    """Pipeline job state."""

    id: str
    namespace: str
    source_path: str
    status: JobStatus = JobStatus.PENDING
    current_step: str | None = None
    total_chunks: int = 0
    processed_chunks: int = 0
    error: str | None = None


class RecallRequest(BaseModel):
    """Request for POST /v1/ltm/recall."""

    query: str
    namespace: str | None = None
    limit: int = 10
    expand_entities: bool = True
    min_score: float = 0.0

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


class RecallResult(BaseModel):
    """A single recall result enriched with entity context."""

    chunk_id: str
    content: str
    source_path: str
    score: float
    namespace: str
    project: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[EntityEdge] = Field(default_factory=list)


class RecallResponse(BaseModel):
    """Response for POST /v1/ltm/recall."""

    results: list[RecallResult]
    total: int
    query: str
    search_time_ms: float
