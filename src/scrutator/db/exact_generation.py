"""Pure bounded verification of an existing evidence generation; no embeddings or writes."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from scrutator.chunker.engine import chunk_document
from scrutator.chunker.splitters import compute_doc_id
from scrutator.config import settings


def metadata(row: Any) -> dict:
    value = row["metadata"]
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def matches_generation(rows: list, namespace: str, path: str, doc_id: str, content_hash: str, raw: str) -> bool:
    """Retain stamped acceptance; prove the complete producer for one legacy preamble.

    Stored stamps are witnesses, never minted here. Replay UUIDs are compared only via
    parent topology; neither replay output nor any stored chunk is persisted.
    """
    if not rows or doc_id != compute_doc_id(namespace, path) or not isinstance(raw, str):
        return False
    if content_hash != "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest():
        return False
    try:
        if any(row["chunk_index"] != index for index, row in enumerate(rows)):
            return False
        metas = [metadata(row) for row in rows]
        sections = [m.get("section") for m in metas]
        stamped = [section for section in sections if isinstance(section, dict)]
        if not stamped or any(
            section.get("doc_id") != doc_id or section.get("doc_content_hash") != content_hash for section in stamped
        ):
            return False
        if len(stamped) == len(rows):
            return True
        if (
            len(stamped) != len(rows) - 1
            or "section" not in metas[0]
            or sections[0] is not None
            or metas[0].get("heading_hierarchy") != []
            or rows[0]["chunk_index"] != 0
            or rows[0]["parent_id"] is not None
            or len(raw.encode("utf-8")) > settings.evidence_population_max_bytes
        ):
            return False
        if settings.evidence_producer_overlap_tokens >= settings.evidence_producer_max_tokens:
            return False
        if any(row["source_type"] != "markdown" or row["chunk_index"] != index for index, row in enumerate(rows)):
            return False
        replay = chunk_document(
            raw,
            path,
            source_type="markdown",
            max_tokens=settings.evidence_producer_max_tokens,
            overlap_tokens=settings.evidence_producer_overlap_tokens,
        ).chunks
        if len(replay) != len(rows) or replay[0].metadata.section is not None:
            return False
        stored_indices = {str(row["chunk_id"]): row["chunk_index"] for row in rows}
        replay_indices = {chunk.id: chunk.chunk_index for chunk in replay}
        for row, meta, section, chunk in zip(rows, metas, sections, replay, strict=True):
            expected_section = chunk.metadata.section.model_dump() if chunk.metadata.section else None
            if expected_section is not None:
                expected_section.pop("doc_id", None)
            actual_section = (
                {k: v for k, v in section.items() if k not in ("doc_id", "doc_content_hash")}
                if isinstance(section, dict)
                else section
            )
            parent = row["parent_id"]
            if parent is not None and str(parent) not in stored_indices:
                return False
            if (
                row["content"] != chunk.content
                or row["content_hash"] != chunk.content_hash
                or row["token_count"] != chunk.token_count
                or meta.get("heading_hierarchy") != chunk.metadata.heading_hierarchy
                or meta.get("frontmatter") != chunk.metadata.frontmatter
                or meta.get("wikilinks") != chunk.metadata.wikilinks
                or meta.get("tags") != chunk.metadata.tags
                or meta.get("language") != chunk.metadata.language
                or actual_section != expected_section
                or stored_indices.get(str(parent)) != replay_indices.get(chunk.parent_id)
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
