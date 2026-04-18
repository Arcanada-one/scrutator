"""Adaptive multi-strategy chunking engine."""

from __future__ import annotations

import os

from scrutator.chunker.metadata import detect_language, extract_frontmatter, extract_tags, extract_wikilinks
from scrutator.chunker.models import Chunk, ChunkMetadata, ChunkResult
from scrutator.chunker.splitters import semantic_split, split_by_headers, split_code
from scrutator.chunker.tokenizer import token_count

# File extension to source type mapping
_TYPE_MAP: dict[str, str] = {
    ".md": "markdown",
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".txt": "text",
}


def detect_type(source_path: str) -> str:
    """Detect source type from file extension."""
    _, ext = os.path.splitext(source_path)
    return _TYPE_MAP.get(ext.lower(), "text")


def chunk_document(
    content: str,
    source_path: str,
    source_type: str | None = None,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> ChunkResult:
    """Adaptive multi-strategy chunker (PRD Adaptive Chunking Strategy)."""
    file_type = source_type or detect_type(source_path)

    if file_type == "markdown":
        return _chunk_markdown(content, source_path, max_tokens, overlap_tokens)
    elif file_type in ("python", "typescript", "javascript"):
        return _chunk_code(content, source_path, file_type, max_tokens)
    elif token_count(content) <= max_tokens // 2:
        return _single_chunk(content, source_path, file_type)
    else:
        return _chunk_sliding_window(content, source_path, file_type, max_tokens, overlap_tokens)


def _make_chunk(
    content: str,
    index: int,
    source_path: str,
    source_type: str,
    heading_hierarchy: list[str] | None = None,
    frontmatter: dict | None = None,
    parent_id: str | None = None,
) -> Chunk:
    """Create a Chunk with extracted metadata."""
    return Chunk(
        content=content,
        chunk_index=index,
        parent_id=parent_id,
        token_count=token_count(content),
        metadata=ChunkMetadata(
            source_path=source_path,
            source_type=source_type,
            heading_hierarchy=heading_hierarchy or [],
            frontmatter=frontmatter or {},
            wikilinks=extract_wikilinks(content),
            tags=extract_tags(content),
            language=detect_language(content),
        ),
    )


def _chunk_markdown(content: str, source_path: str, max_tokens: int, overlap_tokens: int) -> ChunkResult:
    """Chunk markdown using header-based splitting with adaptive sub-splitting."""
    frontmatter, body = extract_frontmatter(content)

    sections = split_by_headers(body)

    if not sections:
        # No headers — treat as plain text
        return _chunk_sliding_window(content, source_path, "markdown", max_tokens, overlap_tokens)

    chunks: list[Chunk] = []
    idx = 0

    for hierarchy, section_content in sections:
        section_tokens = token_count(section_content)

        if section_tokens <= max_tokens:
            # Fits in one chunk
            chunks.append(_make_chunk(section_content, idx, source_path, "markdown", hierarchy, frontmatter))
            idx += 1

        elif section_tokens <= max_tokens * 4:
            # Medium section — sliding window sub-split
            sub_chunks = semantic_split(section_content, max_tokens, overlap_tokens)
            for sub in sub_chunks:
                chunks.append(_make_chunk(sub, idx, source_path, "markdown", hierarchy, frontmatter))
                idx += 1

        else:
            # Giant section — parent-child hierarchy
            parent = _make_chunk(
                section_content[:200] + "...",
                idx,
                source_path,
                "markdown",
                hierarchy,
                frontmatter,
            )
            chunks.append(parent)
            idx += 1

            sub_chunks = semantic_split(section_content, max_tokens, overlap_tokens)
            for sub in sub_chunks:
                chunks.append(
                    _make_chunk(sub, idx, source_path, "markdown", hierarchy, frontmatter, parent_id=parent.id)
                )
                idx += 1

    return ChunkResult(chunks=chunks, strategy_used="markdown_headers")


def _chunk_code(content: str, source_path: str, file_type: str, max_tokens: int) -> ChunkResult:
    """Chunk source code by function/class boundaries."""
    code_chunks = split_code(content, max_tokens)
    chunks = [_make_chunk(c, i, source_path, file_type) for i, c in enumerate(code_chunks)]
    return ChunkResult(chunks=chunks, strategy_used="code_boundaries")


def _single_chunk(content: str, source_path: str, file_type: str) -> ChunkResult:
    """Short document — no chunking needed."""
    chunk = _make_chunk(content, 0, source_path, file_type)
    return ChunkResult(chunks=[chunk], strategy_used="single")


def _chunk_sliding_window(
    content: str, source_path: str, file_type: str, max_tokens: int, overlap_tokens: int
) -> ChunkResult:
    """Generic text — sliding window with overlap."""
    text_chunks = semantic_split(content, max_tokens, overlap_tokens)
    chunks = [_make_chunk(c, i, source_path, file_type) for i, c in enumerate(text_chunks)]
    return ChunkResult(chunks=chunks, strategy_used="sliding_window")
