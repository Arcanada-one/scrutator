"""Edge creation by source_path — resolves paths to chunk UUIDs server-side."""

from __future__ import annotations

from scrutator.db.repository import get_chunks_by_source_path, insert_edges
from scrutator.dream.models import EdgeCreateByPath, EdgeCreateByPathResponse


async def create_edges_by_path(edges: list[EdgeCreateByPath]) -> EdgeCreateByPathResponse:
    """Resolve source_path → chunk_id for each edge, then batch-insert."""
    # Collect unique paths
    all_paths: set[str] = set()
    for edge in edges:
        all_paths.add(edge.source_path)
        all_paths.add(edge.target_path)

    # Batch lookup: path → {chunk_index: chunk_id}
    path_map: dict[str, dict[int, str]] = {}
    for path in all_paths:
        chunks = await get_chunks_by_source_path(path)
        if chunks:
            path_map[path] = {c.chunk_index: c.chunk_id for c in chunks}

    # Resolve edges
    resolved: list[dict] = []
    not_found: set[str] = set()

    for edge in edges:
        source_chunks = path_map.get(edge.source_path)
        target_chunks = path_map.get(edge.target_path)

        if source_chunks is None:
            not_found.add(edge.source_path)
            continue
        if target_chunks is None:
            not_found.add(edge.target_path)
            continue

        source_id = source_chunks.get(edge.source_chunk_index)
        target_id = target_chunks.get(edge.target_chunk_index)

        if source_id is None:
            not_found.add(f"{edge.source_path}[{edge.source_chunk_index}]")
            continue
        if target_id is None:
            not_found.add(f"{edge.target_path}[{edge.target_chunk_index}]")
            continue

        resolved.append(
            {
                "source_chunk_id": source_id,
                "target_chunk_id": target_id,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "created_by": edge.created_by,
            }
        )

    created = await insert_edges(resolved) if resolved else 0
    return EdgeCreateByPathResponse(created=created, not_found=sorted(not_found))
