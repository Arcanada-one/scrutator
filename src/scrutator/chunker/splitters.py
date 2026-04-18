"""Core splitting functions: header-based, sliding window, code-aware."""

from __future__ import annotations

import re

from scrutator.chunker.tokenizer import token_count

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


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


def semantic_split(text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> list[str]:
    """Split text by sliding window with overlap, respecting paragraph boundaries."""
    if token_count(text) <= max_tokens:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = token_count(para)

        if current_tokens + para_tokens > max_tokens and current_parts:
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
            current_tokens = overlap_count

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


_PYTHON_FUNC_RE = re.compile(r"^((?:async\s+)?def\s+\w+|class\s+\w+)", re.MULTILINE)


def split_code(text: str, max_tokens: int = 512) -> list[str]:
    """Split Python code by function/class boundaries."""
    matches = list(_PYTHON_FUNC_RE.finditer(text))

    if not matches:
        # No functions/classes found — return as single chunk or sliding window
        if token_count(text) <= max_tokens:
            return [text]
        return semantic_split(text, max_tokens, overlap_tokens=0)

    chunks: list[str] = []

    # Content before first function/class
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            chunks.append(preamble)

    # Each function/class
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if block:
            if token_count(block) > max_tokens:
                # Large block — split further
                sub_chunks = semantic_split(block, max_tokens, overlap_tokens=0)
                chunks.extend(sub_chunks)
            else:
                chunks.append(block)

    return chunks
