"""Hierarchical navigation — outline + section-context assembly (SRCH-0021).

Read-only query-time assembly over the existing `chunks` table and the
normalized `metadata.section` keys written by the chunker/indexer. Mirrors
the codebase's existing read patterns (`get_chunks_by_source_path`) but MUST
NOT call `upsert_namespace` — namespace resolution here is read-only via
`get_namespaces()`, since these are read endpoints, not indexing endpoints.
"""

from __future__ import annotations

import uuid

from fastapi import HTTPException

from scrutator.chunker.splitters import compute_doc_id
from scrutator.db.models import (
    ChunkLookupResult,
    OutlineNode,
    OutlineResponse,
    SectionBreadcrumb,
    SectionContext,
    SectionSelf,
)
from scrutator.db.repository import get_chunks_by_source_path, get_namespaces, get_section_siblings_children

# Fork 3 (SRCH-0021 plan): default cap on outline tree size, hard-capped again
# server-side regardless of the caller-supplied max_nodes.
DEFAULT_MAX_NODES = 2000
HARD_MAX_NODES_CEILING = 10000


async def _resolve_namespace_id(namespace: str) -> int | None:
    """Read-only namespace resolve — unlike search(), this MUST NOT auto-create."""
    namespaces = await get_namespaces()
    for ns in namespaces:
        if ns.name == namespace:
            return ns.id
    return None


def _section_of(row: ChunkLookupResult) -> dict | None:
    return row.metadata.get("section")


def _fallback_section(doc_id: str = "") -> dict:
    """Un-backfilled / no-header doc → single implicit root section (PRD Risk table)."""
    return {
        "doc_id": doc_id,
        "heading_path": [],
        "depth": 1,
        "anchor": "root",
        "anchor_path": ["root"],
        "section_key": "root",
        "schema_version": 0,
    }


async def build_outline(namespace: str, source_path: str, max_nodes: int = DEFAULT_MAX_NODES) -> OutlineResponse:
    """Assemble the hierarchical TOC tree for a (namespace, source_path).

    404 if source_path unknown in namespace; 422 if the flat chunk count
    exceeds max_nodes (checked BEFORE tree assembly — Fork 3).
    """
    max_nodes = min(max_nodes, HARD_MAX_NODES_CEILING)

    namespace_id = await _resolve_namespace_id(namespace)
    if namespace_id is None:
        raise HTTPException(status_code=404, detail=f"unknown namespace: {namespace}")

    rows = await get_chunks_by_source_path(source_path, namespace_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"unknown source_path: {source_path}")

    if len(rows) > max_nodes:
        raise HTTPException(
            status_code=422,
            detail="document exceeds max_nodes; use /v1/navigate/section for a narrower view",
        )

    doc_id = compute_doc_id(namespace, source_path)
    node_index: dict[str, dict] = {}
    root_children: dict[str, dict] = {}

    def _get_or_create(anchor_path: list[str], heading_path: list[str]) -> dict:
        key = "/".join(anchor_path)
        if key in node_index:
            return node_index[key]
        node = {
            "title": heading_path[-1] if heading_path else anchor_path[-1],
            "anchor": anchor_path[-1],
            "depth": len(anchor_path),
            "section_key": key,
            "chunk_ids": [],
            "children": [],
        }
        node_index[key] = node
        if len(anchor_path) <= 1:
            root_children[key] = node
        else:
            parent = _get_or_create(anchor_path[:-1], heading_path[:-1])
            parent["children"].append(node)
        return node

    for row in rows:
        section = _section_of(row) or _fallback_section(doc_id)
        anchor_path = section.get("anchor_path") or ["root"]
        heading_path = section.get("heading_path") or ["root"]
        node = _get_or_create(anchor_path, heading_path)
        node["chunk_ids"].append(row.chunk_id)

    return OutlineResponse(
        source_path=source_path,
        namespace=namespace,
        doc_id=doc_id,
        total_chunks=len(rows),
        outline=[OutlineNode(**n) for n in root_children.values()],
    )


async def build_section_context(chunk_id: str) -> SectionContext:
    """Assemble ancestors/self/siblings/children for a chunk.

    422 if chunk_id is not a UUID; 404 if the chunk does not exist.
    """
    try:
        uuid.UUID(chunk_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"chunk_id is not a valid UUID: {chunk_id}") from exc

    result = await get_section_siblings_children(chunk_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"chunk not found: {chunk_id}")

    doc_rows: list[ChunkLookupResult] = result["doc_rows"]
    self_row = next((r for r in doc_rows if r.chunk_id == chunk_id), None)
    if self_row is None:
        raise HTTPException(status_code=404, detail=f"chunk not found: {chunk_id}")

    rows_sections = [(row, _section_of(row) or _fallback_section()) for row in doc_rows]
    self_section = next(sec for row, sec in rows_sections if row.chunk_id == chunk_id)

    self_anchor_path: list[str] = self_section.get("anchor_path") or ["root"]
    self_heading_path: list[str] = self_section.get("heading_path") or ["root"]
    self_depth: int = self_section.get("depth", len(self_anchor_path))
    self_key: str = self_section.get("section_key", "/".join(self_anchor_path))
    doc_id: str = self_section.get("doc_id", "")
    parent_prefix = "/".join(self_anchor_path[:-1])

    def _title(sec: dict, key: str) -> str:
        heading_path = sec.get("heading_path")
        return heading_path[-1] if heading_path else key

    ancestors: list[SectionBreadcrumb] = []
    for depth in range(1, self_depth):
        prefix = self_anchor_path[:depth]
        key = "/".join(prefix)
        match = next((sec for _, sec in rows_sections if sec.get("section_key") == key), None)
        title = _title(match, key) if match else (prefix[-1] if prefix else key)
        ancestors.append(SectionBreadcrumb(title=title, section_key=key, depth=depth))

    siblings: list[SectionBreadcrumb] = []
    seen_siblings: set[str] = set()
    for _, sec in rows_sections:
        key = sec.get("section_key", "")
        if key == self_key or key in seen_siblings:
            continue
        depth = sec.get("depth", 0)
        sec_anchor_path = sec.get("anchor_path", [])
        if depth == self_depth and "/".join(sec_anchor_path[:-1]) == parent_prefix:
            seen_siblings.add(key)
            siblings.append(SectionBreadcrumb(title=_title(sec, key), section_key=key, depth=depth))

    children: list[SectionBreadcrumb] = []
    seen_children: set[str] = set()
    for _, sec in rows_sections:
        key = sec.get("section_key", "")
        if key in seen_children:
            continue
        depth = sec.get("depth", 0)
        sec_anchor_path = sec.get("anchor_path", [])
        if depth == self_depth + 1 and sec_anchor_path[:-1] == self_anchor_path:
            seen_children.add(key)
            children.append(SectionBreadcrumb(title=_title(sec, key), section_key=key, depth=depth))

    self_chunk_ids = [row.chunk_id for row, sec in rows_sections if sec.get("section_key") == self_key]

    return SectionContext(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_key=self_key,
        ancestors=ancestors,
        self_=SectionSelf(
            title=self_heading_path[-1] if self_heading_path else self_key,
            section_key=self_key,
            depth=self_depth,
            chunk_ids=self_chunk_ids,
        ),
        siblings=siblings,
        children=children,
    )
