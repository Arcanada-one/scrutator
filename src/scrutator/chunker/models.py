"""Pydantic models for chunking: Chunk, ChunkMetadata, ChunkRequest, ChunkResponse, ChunkResult."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChunkMetadata(BaseModel):
    """Metadata extracted from the source document."""

    source_path: str
    source_type: str  # 'markdown', 'python', 'text'
    heading_hierarchy: list[str] = Field(default_factory=list)
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    wikilinks: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    language: str | None = None


class Chunk(BaseModel):
    """A single chunk of a document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    chunk_index: int
    parent_id: str | None = None
    token_count: int = 0
    metadata: ChunkMetadata
    content_hash: str = ""

    def model_post_init(self, _context: Any) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()


class ChunkResult(BaseModel):
    """Result of chunking a document."""

    chunks: list[Chunk]
    total_chunks: int = 0
    total_tokens: int = 0
    strategy_used: str = ""

    def model_post_init(self, _context: Any) -> None:
        if not self.total_chunks:
            self.total_chunks = len(self.chunks)
        if not self.total_tokens:
            self.total_tokens = sum(c.token_count for c in self.chunks)


_MAX_CONTENT_BYTES = 1_048_576  # 1 MB


class ChunkRequest(BaseModel):
    """API request body for POST /v1/chunk."""

    content: str
    source_path: str = "unknown"
    source_type: str | None = None
    max_tokens: int = 512
    overlap_tokens: int = 50
    embed: bool = False

    @field_validator("content")
    @classmethod
    def content_max_size(cls, v: str) -> str:
        if len(v.encode()) > _MAX_CONTENT_BYTES:
            raise ValueError(f"content exceeds maximum size of {_MAX_CONTENT_BYTES} bytes (1 MB)")
        return v


class ChunkResponse(BaseModel):
    """API response body for POST /v1/chunk."""

    chunks: list[Chunk]
    total_chunks: int
    total_tokens: int
    strategy_used: str
