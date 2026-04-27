"""Pydantic models for the LTM pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

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
    # LTM-0012 temporal filtering (all optional — backward compatible)
    as_of: datetime | None = None
    time_range: tuple[datetime, datetime] | None = None
    temporal_boost: float = 0.3

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

    @field_validator("temporal_boost")
    @classmethod
    def boost_range(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("temporal_boost must be in [0.0, 1.0]")
        return v

    @model_validator(mode="after")
    def time_range_ordered(self) -> RecallRequest:
        if self.time_range is not None:
            start, end = self.time_range
            if start >= end:
                raise ValueError("time_range start must be before end")
        return self


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


class EventType(StrEnum):
    """Canonical event types — matches `format_event_extraction` system prompt."""

    ARCHIVED = "archived"
    CREATED = "created"
    COMPLETED = "completed"
    STARTED = "started"
    RELEASED = "released"
    DEPLOYED = "deployed"
    UPDATED = "updated"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


class EntityEvent(BaseModel):
    """A temporal event attached to an entity (LTM-0012)."""

    entity_name: str
    event_type: str  # not enum — accept extras with WARN logged at pipeline layer
    when_t: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    description: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entity_name", "event_type")
    @classmethod
    def required_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def description_truncate(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        return v[:500] if len(v) > 500 else v

    @model_validator(mode="after")
    def at_least_one_timestamp(self) -> EntityEvent:
        if self.when_t is None and self.valid_from is None:
            raise ValueError("at least one of when_t / valid_from must be set")
        return self

    @model_validator(mode="after")
    def valid_period_ordered(self) -> EntityEvent:
        if self.valid_from is not None and self.valid_to is not None and self.valid_from >= self.valid_to:
            raise ValueError("valid_from must be before valid_to")
        return self


# ---- LTM-0013: Reflect layer ------------------------------------------------

_MAX_META_FACT_CONTENT = 4000


class FactType(StrEnum):
    """Canonical meta-fact types — see reflect prompt schema."""

    SUMMARY = "summary"
    CONTRADICTION = "contradiction"
    DERIVED_RELATION = "derived_relation"


class MetaFact(BaseModel):
    """LLM-derived meta-fact with provenance (LTM-0013)."""

    id: str | None = None
    namespace: str
    fact_type: FactType
    content: str
    source_chunk_ids: list[str]
    entity_ids: list[str] = Field(default_factory=list)
    depth: int = 1
    derived_at: datetime | None = None
    model_used: str
    reflect_run_id: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("namespace")
    @classmethod
    def namespace_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("namespace must not be empty")
        return v.strip()

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content must not be empty")
        if len(v) > _MAX_META_FACT_CONTENT:
            raise ValueError(f"content exceeds {_MAX_META_FACT_CONTENT} characters")
        return v

    @field_validator("source_chunk_ids")
    @classmethod
    def at_least_one_source(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one source_chunk_id required")
        return v

    @field_validator("depth")
    @classmethod
    def depth_one_only(cls, v: int) -> int:
        if v != 1:
            raise ValueError("depth must equal 1 (no meta-of-meta)")
        return v

    @field_validator("model_used")
    @classmethod
    def model_used_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_used must not be empty")
        return v.strip()


class ReflectRequest(BaseModel):
    """Request for POST /v1/ltm/reflect."""

    namespace: str = "arcanada"
    since: datetime | None = None
    max_chunks: int | None = None
    dry_run: bool = False

    @field_validator("namespace")
    @classmethod
    def namespace_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("namespace must not be empty")
        return v.strip()

    @field_validator("max_chunks")
    @classmethod
    def max_chunks_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("max_chunks must be >= 1")
        return v


class ReflectRunSummary(BaseModel):
    """Summary of a single reflect run — returned by POST /v1/ltm/reflect."""

    run_id: str
    status: str
    chunks_scanned: int
    meta_facts_created: int
    cost_usd: float
    req_count: int
    abort_reason: str | None = None
    duration_ms: float


class ReflectResponse(BaseModel):
    """Response for POST /v1/ltm/reflect."""

    summary: ReflectRunSummary
    preview: list[MetaFact] | None = None
