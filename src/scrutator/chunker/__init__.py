"""Chunking Engine — adaptive semantic document splitting."""

from scrutator.chunker.engine import chunk_document
from scrutator.chunker.models import Chunk, ChunkMetadata, ChunkResult

__all__ = ["chunk_document", "Chunk", "ChunkMetadata", "ChunkResult"]
