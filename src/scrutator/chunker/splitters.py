"""Core splitting functions: header-based, sliding window, code-aware."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from scrutator.chunker.tokenizer import token_count

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# The shared embedding provider rejects any individual input over 24,000
# characters, independently of Scrutator's approximate token budget.
MAX_EMBEDDING_INPUT_CHARS = 24_000

# SRCH-0021: hierarchical navigation — section metadata schema version.
# Bump when normalize_heading_path()'s output shape changes; backfill_sections.py
# re-derives any chunk whose stored section.schema_version != this constant.
SECTION_SCHEMA_VERSION = 1

_SLUG_WHITESPACE_RE = re.compile(r"[\s_]+")


def slugify(text: str) -> str:
    """GitHub-style deterministic slug: strip leading '#', lowercase, keep unicode
    word characters (cyrillic + latin + digits), collapse whitespace to '-'."""
    text = text.lstrip("#").strip().lower()
    text = unicodedata.normalize("NFKC", text)
    kept = [ch for ch in text if ch.isalnum() or ch in (" ", "-", "_")]
    slug = _SLUG_WHITESPACE_RE.sub("-", "".join(kept))
    return slug.strip("-")


def compute_doc_id(namespace: str, source_path: str) -> str:
    """Stable document identity for grouping/outline, scoped to a namespace."""
    return hashlib.sha256(f"{namespace}|{source_path}".encode()).hexdigest()[:16]


def normalize_heading_path(heading_hierarchy: list[str]) -> dict:
    """Normalize a '#'-prefixed heading stack into the `section` metadata shape.

    `doc_id` is left empty here — the chunker has no namespace context;
    the indexer (which knows the namespace) stamps it in via compute_doc_id()
    before writing the metadata dict to the DB.
    """
    heading_path = [h.lstrip("#").strip() for h in heading_hierarchy]
    anchor_path = [slugify(h) for h in heading_path]
    return {
        "doc_id": "",
        "heading_path": heading_path,
        "depth": len(heading_path),
        "anchor": anchor_path[-1] if anchor_path else "",
        "anchor_path": anchor_path,
        "section_key": "/".join(anchor_path),
        "schema_version": SECTION_SCHEMA_VERSION,
    }


def split_by_headers(text: str) -> list[tuple[list[str], str]]:
    """Split markdown by headers. Returns list of (heading_hierarchy, section_content).

    Each section includes its header line and all content until the next header
    of equal or higher level.
    """
    lines = text.split("\n")
    sections: list[tuple[list[str], str]] = []
    current_hierarchy: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        match = _HEADER_RE.match(line)
        if match:
            # Save previous section
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append((list(current_hierarchy), content))

            level = len(match.group(1))
            header_text = line.strip()

            # Update hierarchy: truncate to current level, then append
            current_hierarchy = [h for h in current_hierarchy if len(h) - len(h.lstrip("#")) < level]
            current_hierarchy.append(header_text)
            current_lines = [line]
        else:
            current_lines.append(line)

    # Last section
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((list(current_hierarchy), content))

    return sections


def _within_embedding_bounds(text: str, max_tokens: int) -> bool:
    return token_count(text) <= max_tokens and len(text) <= MAX_EMBEDDING_INPUT_CHARS


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Split one dense segment without dropping or normalizing any characters."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        upper = min(start + MAX_EMBEDDING_INPUT_CHARS, len(text))
        if token_count(text[start:upper]) <= max_tokens:
            end = upper
        else:
            low = start + 1
            high = upper
            end = start
            while low <= high:
                midpoint = (low + high) // 2
                if token_count(text[start:midpoint]) <= max_tokens:
                    end = midpoint
                    low = midpoint + 1
                else:
                    high = midpoint - 1

            if end == start:
                raise ValueError("max_tokens is too small for non-empty input")

        if end < len(text):
            whitespace = list(re.finditer(r"\s+", text[start:end]))
            if whitespace:
                boundary = start + whitespace[-1].end()
                if boundary > start:
                    end = boundary

        chunks.append(text[start:end])
        start = end

    return chunks


def semantic_split(text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> list[str]:
    """Split text within both token and provider character limits."""
    if _within_embedding_bounds(text, max_tokens):
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current_parts: list[str] = []

    for para in paragraphs:
        if not _within_embedding_bounds(para, max_tokens):
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
            chunks.extend(_hard_split(para, max_tokens))
            continue

        candidate = "\n\n".join([*current_parts, para])
        if current_parts and not _within_embedding_bounds(candidate, max_tokens):
            chunks.append("\n\n".join(current_parts))

            # Overlap: keep last parts that fit within overlap_tokens
            overlap_parts: list[str] = []
            overlap_count = 0
            for part in reversed(current_parts):
                pt = token_count(part)
                if overlap_count + pt > overlap_tokens:
                    break
                overlap_parts.insert(0, part)
                overlap_count += pt

            current_parts = overlap_parts
            while current_parts and not _within_embedding_bounds("\n\n".join([*current_parts, para]), max_tokens):
                current_parts.pop(0)

        current_parts.append(para)

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


_PYTHON_FUNC_RE = re.compile(r"^((?:async\s+)?def\s+\w+|class\s+\w+)", re.MULTILINE)


def split_code(text: str, max_tokens: int = 512) -> list[str]:
    """Split Python code by function/class boundaries."""
    matches = list(_PYTHON_FUNC_RE.finditer(text))

    if not matches:
        # No functions/classes found — return as single chunk or sliding window
        if _within_embedding_bounds(text, max_tokens):
            return [text]
        return semantic_split(text, max_tokens, overlap_tokens=0)

    chunks: list[str] = []

    # Content before first function/class
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            if _within_embedding_bounds(preamble, max_tokens):
                chunks.append(preamble)
            else:
                chunks.extend(semantic_split(preamble, max_tokens, overlap_tokens=0))

    # Each function/class
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if block:
            if not _within_embedding_bounds(block, max_tokens):
                # Large block — split further
                sub_chunks = semantic_split(block, max_tokens, overlap_tokens=0)
                chunks.extend(sub_chunks)
            else:
                chunks.append(block)

    return chunks
