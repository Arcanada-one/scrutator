"""Index pipeline — chunk document, embed chunks, store in database."""

from __future__ import annotations

import logging

from scrutator.chunker.engine import chunk_document
from scrutator.db.models import IndexResponse
from scrutator.db.repository import (
    delete_by_source,
    get_chunk_ids_by_source,
    insert_chunks,
    insert_sparse_vectors,
    upsert_namespace,
    upsert_project,
)
from scrutator.search.embedder import embed_sparse, embed_texts

logger = logging.getLogger(__name__)


async def index_document(
    content: str,
    source_path: str,
    namespace: str = "arcanada",
    project: str | None = None,
    source_type: str | None = None,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> IndexResponse:
    """Full index pipeline: chunk → embed → store."""
    # 1. Chunk the document
    chunk_result = chunk_document(
        content=content,
        source_path=source_path,
        source_type=source_type,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    if not chunk_result.chunks:
        return IndexResponse(chunks_indexed=0, source_path=source_path, namespace=namespace, strategy_used="empty")

    # 2. Embed all chunks
    texts = [c.content for c in chunk_result.chunks]
    embeddings = await embed_texts(texts)

    # 3. Ensure namespace (and project) exist
    namespace_id = await upsert_namespace(namespace)
    project_id = await upsert_project(namespace_id, project) if project else None

    # 4. Delete old chunks for this source (re-index)
    await delete_by_source(source_path)

    # 5. Store chunks with embeddings
    chunk_dicts = [
        {
            "source_path": source_path,
            "source_type": c.metadata.source_type,
            "chunk_index": c.chunk_index,
            "parent_id": c.parent_id,
            "content": c.content,
            "content_hash": c.content_hash,
            "token_count": c.token_count,
            "metadata": {
                "heading_hierarchy": c.metadata.heading_hierarchy,
                "frontmatter": c.metadata.frontmatter,
                "wikilinks": c.metadata.wikilinks,
                "tags": c.metadata.tags,
                "language": c.metadata.language,
            },
        }
        for c in chunk_result.chunks
    ]

    inserted = await insert_chunks(chunk_dicts, embeddings, namespace_id, project_id)

    # 6. Get sparse embeddings and store them
    try:
        sparse_weights = await embed_sparse(texts)
        chunk_ids = await get_chunk_ids_by_source(source_path)
        if chunk_ids and sparse_weights:
            await insert_sparse_vectors(chunk_ids, sparse_weights)
    except Exception:
        logger.warning("Sparse indexing failed for %s", source_path, exc_info=True)

    return IndexResponse(
        chunks_indexed=inserted,
        source_path=source_path,
        namespace=namespace,
        strategy_used=chunk_result.strategy_used,
    )
