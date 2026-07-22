"""Index pipeline — chunk document, embed chunks, store in database."""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass

from scrutator.chunker.engine import chunk_document
from scrutator.chunker.models import SectionMeta
from scrutator.chunker.splitters import compute_doc_id
from scrutator.config import settings
from scrutator.db.models import (
    INDEX_BATCH_MAX_DOCUMENT_BYTES,
    BatchIndexErrorCode,
    BatchIndexFailed,
    BatchIndexSucceeded,
    IndexRequest,
    IndexResponse,
)
from scrutator.db.repository import (
    replace_source_chunks_atomic,
    upsert_namespace,
    upsert_project,
)
from scrutator.search.embedder import embed_sparse, embed_texts

logger = logging.getLogger(__name__)

INDEX_BATCH_MAX_CHUNKS = 256
INDEX_BATCH_MAX_TOKENS = 131_072
_DENSE_DIMENSIONS = 1024


class BatchIndexLimitError(ValueError):
    """Raised before embedding when a packed batch crosses a resource cap."""


class _BatchEmbeddingError(Exception):
    def __init__(self, code: BatchIndexErrorCode):
        self.code = code


@dataclass(frozen=True)
class _PreparedDocument:
    position: int
    document: IndexRequest
    chunks: list[dict]
    offset: int
    source_document: dict | None = None

    @property
    def end(self) -> int:
        return self.offset + len(self.chunks)


def compute_doc_content_hash(full_content: str) -> str:
    """Whole-document content hash bound at ingest (SRCH-0038 D3 / S1).

    Bound ONCE over the full pre-chunk source content and stored in each chunk's
    `metadata.section.doc_content_hash`. The fetch path only READS this value — it is never
    recomputed over the assembled response, so integrity verification is not theater.
    """
    return "sha256:" + hashlib.sha256(full_content.encode()).hexdigest()


def _stamp_doc_id(
    section: SectionMeta | None,
    namespace: str,
    source_path: str,
    doc_content_hash: str,
) -> dict | None:
    """Finalize a chunk's section dict with its namespace-scoped doc_id and the whole-document
    content hash (SRCH-0038 D3). Both are indexer-only context — the chunker has neither the
    namespace nor the full pre-chunk content.

    ``doc_content_hash`` is ~71 bytes, safely under the ``jsonb_ops`` GIN entry ceiling, and is
    what ``SearchHit.content_hash`` / the fetch path READ. The exact pre-chunk bytes are NOT
    stamped here (that was the SRCH-0038 1a seam, which raised the multi-KB blob into
    ``idx_chunks_metadata`` GIN and hit the ~2704-byte ``jsonb_ops`` entry ceiling on real
    skills). Under 1b they live in the isolated, un-indexed ``source_documents`` table instead
    (see ``_build_source_document`` / ``replace_source_chunks_atomic``)."""
    if section is None:
        return None
    return {
        **section.model_dump(),
        "doc_id": compute_doc_id(namespace, source_path),
        "doc_content_hash": doc_content_hash,
    }


def _build_source_document(namespace: str, source_path: str, full_content: str) -> dict | None:
    """SRCH-0038 1b: build the isolated exact-source row for the skills namespace ONLY, so a skill
    fetch is byte-exact against ``content_hash``. Returns ``None`` for every other namespace (the
    evidence corpus keeps chunk-reassembly, ``content_exact=False``).

    By-construction byte-exactness: ``raw_content`` is the SAME ``full_content`` string that
    ``content_hash`` is computed over — ``compute_doc_content_hash(full_content)`` — with no
    re-encode or normalization between them, so ``sha256(raw_content) == content_hash`` holds.
    The blob lands in ``source_documents`` (NOT ``chunks.metadata``), so it never enters the
    ``idx_chunks_metadata`` GIN index and the ~2704-byte ``jsonb_ops`` entry ceiling never applies.

    Size guard: the single ``POST /v1/index`` path has no per-document byte cap, so bound the
    exact-source blob at the same 256 KB cap the batch endpoint enforces."""
    if namespace != settings.skills_namespace:
        return None
    if len(full_content.encode("utf-8")) > INDEX_BATCH_MAX_DOCUMENT_BYTES:
        raise BatchIndexLimitError(f"skills document exceeds {INDEX_BATCH_MAX_DOCUMENT_BYTES}-byte exact-source cap")
    return {
        "doc_id": compute_doc_id(namespace, source_path),
        "source_path": source_path,
        "content_hash": compute_doc_content_hash(full_content),
        "raw_content": full_content,
    }


def _chunk_dicts(chunk_result, namespace: str, source_path: str, full_content: str) -> list[dict]:
    doc_content_hash = compute_doc_content_hash(full_content)
    # SRCH-0038 1b: keep only the ~71-byte `doc_content_hash` in `metadata.section` (safely under
    # the jsonb_ops GIN entry ceiling); the exact bytes go to `source_documents`, never the GIN
    # -indexed `chunks.metadata`. The skills-blob size guard now lives in `_build_source_document`,
    # invoked here so an oversized skills doc is rejected before persistence even if the caller
    # ignores the source-document payload.
    _build_source_document(namespace, source_path, full_content)
    return [
        {
            "id": chunk.id,
            "source_path": source_path,
            "source_type": chunk.metadata.source_type,
            "chunk_index": chunk.chunk_index,
            "parent_id": chunk.parent_id,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "token_count": chunk.token_count,
            "metadata": {
                "heading_hierarchy": chunk.metadata.heading_hierarchy,
                "frontmatter": chunk.metadata.frontmatter,
                "wikilinks": chunk.metadata.wikilinks,
                "tags": chunk.metadata.tags,
                "language": chunk.metadata.language,
                "section": _stamp_doc_id(chunk.metadata.section, namespace, source_path, doc_content_hash),
            },
        }
        for chunk in chunk_result.chunks
    ]


async def index_documents(documents: list[IndexRequest]) -> list[BatchIndexSucceeded | BatchIndexFailed]:
    """Chunk and embed a bounded document pack before storing each source."""
    prepared, texts, results = _prepare_documents(documents)
    _enforce_pack_caps(prepared, texts)
    if not prepared:
        return _complete_results(results)

    positions = [(item.position, item.document.source_path) for item in prepared]
    try:
        embeddings, sparse_weights = await _embed_batch(texts)
    except _BatchEmbeddingError as exc:
        _set_failures(results, positions, exc.code)
        return _complete_results(results)

    try:
        namespace_id = await upsert_namespace(documents[0].namespace)
    except Exception:
        logger.error("Batch namespace persistence failed")
        _set_failures(results, positions, "persistence_failed")
        return _complete_results(results)

    for item in prepared:
        results[item.position] = await _persist_prepared(item, embeddings, sparse_weights, namespace_id)
    return _complete_results(results)


def _prepare_documents(
    documents: list[IndexRequest],
) -> tuple[list[_PreparedDocument], list[str], list[BatchIndexSucceeded | BatchIndexFailed | None]]:
    prepared: list[_PreparedDocument] = []
    texts: list[str] = []
    results: list[BatchIndexSucceeded | BatchIndexFailed | None] = [None] * len(documents)
    for position, document in enumerate(documents):
        try:
            chunk_result = chunk_document(
                content=document.content,
                source_path=document.source_path,
                source_type=document.source_type,
                max_tokens=document.max_tokens,
                overlap_tokens=document.overlap_tokens,
            )
        except Exception:
            logger.error("Batch chunking failed for one source")
            results[position] = BatchIndexFailed(source_path=document.source_path, error_code="chunking_failed")
            continue
        chunk_dicts = _chunk_dicts(chunk_result, document.namespace, document.source_path, document.content)
        source_document = _build_source_document(document.namespace, document.source_path, document.content)
        prepared.append(_PreparedDocument(position, document, chunk_dicts, len(texts), source_document))
        texts.extend(chunk["content"] for chunk in chunk_dicts)
    return prepared, texts, results


def _enforce_pack_caps(prepared: list[_PreparedDocument], texts: list[str]) -> None:
    if len(texts) > INDEX_BATCH_MAX_CHUNKS:
        raise BatchIndexLimitError("batch chunk limit exceeded")
    if sum(chunk["token_count"] for item in prepared for chunk in item.chunks) > INDEX_BATCH_MAX_TOKENS:
        raise BatchIndexLimitError("batch token limit exceeded")


async def _embed_batch(texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
    try:
        embeddings = await embed_texts(texts)
    except Exception as exc:
        _log_embedding_failure("Dense", exc)
        raise _BatchEmbeddingError("dense_embedding_failed") from None
    if not _valid_dense_embeddings(embeddings, len(texts)):
        raise _BatchEmbeddingError("invalid_dense_embeddings")

    try:
        sparse_weights = await embed_sparse(texts)
    except Exception as exc:
        _log_embedding_failure("Sparse", exc)
        raise _BatchEmbeddingError("sparse_embedding_failed") from None
    if not _valid_sparse_embeddings(sparse_weights, len(texts)):
        raise _BatchEmbeddingError("invalid_sparse_embeddings")
    return embeddings, sparse_weights


def _log_embedding_failure(stage: str, exception: Exception) -> None:
    status_code = getattr(exception, "status_code", None)
    if not isinstance(status_code, int):
        status_code = "none"
    logger.error(
        "%s embedding failed for batch: error_type=%s status_code=%s",
        stage,
        type(exception).__name__,
        status_code,
    )


async def _persist_prepared(
    item: _PreparedDocument,
    embeddings: list[list[float]],
    sparse_weights: list[dict[str, float]],
    namespace_id: int,
) -> BatchIndexSucceeded | BatchIndexFailed:
    try:
        project_id = await upsert_project(namespace_id, item.document.project) if item.document.project else None
        inserted = await replace_source_chunks_atomic(
            item.chunks,
            embeddings[item.offset : item.end],
            sparse_weights[item.offset : item.end],
            namespace_id,
            project_id,
            source_document=item.source_document,
        )
        return BatchIndexSucceeded(source_path=item.document.source_path, chunks_indexed=inserted)
    except Exception:
        logger.error("Batch persistence failed for one source")
        return BatchIndexFailed(source_path=item.document.source_path, error_code="persistence_failed")


def _set_failures(
    results: list[BatchIndexSucceeded | BatchIndexFailed | None],
    positions: list[tuple[int, str]],
    code: BatchIndexErrorCode,
) -> None:
    for position, path in positions:
        results[position] = BatchIndexFailed(source_path=path, error_code=code)


def _complete_results(
    results: list[BatchIndexSucceeded | BatchIndexFailed | None],
) -> list[BatchIndexSucceeded | BatchIndexFailed]:
    if any(result is None for result in results):
        raise RuntimeError("batch result mapping incomplete")
    return [result for result in results if result is not None]


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _valid_dense_embeddings(embeddings: object, expected_count: int) -> bool:
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        return False
    return all(
        isinstance(vector, list) and len(vector) == _DENSE_DIMENSIONS and all(_finite_number(value) for value in vector)
        for vector in embeddings
    )


def _valid_sparse_embeddings(embeddings: object, expected_count: int) -> bool:
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        return False
    return all(
        isinstance(vector, dict)
        and all(isinstance(token, str) and _finite_number(weight) for token, weight in vector.items())
        for vector in embeddings
    )


async def _embed_single_document(texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
    embeddings = await embed_texts(texts)
    if not _valid_dense_embeddings(embeddings, len(texts)):
        raise ValueError("invalid dense embeddings")
    try:
        sparse_weights = await embed_sparse(texts)
        if not _valid_sparse_embeddings(sparse_weights, len(texts)):
            raise ValueError("invalid sparse embeddings")
    except Exception:
        # The legacy endpoint has always treated sparse indexing as optional.
        # Persist explicit empty weights so dense replacement remains atomic.
        logger.warning("Sparse indexing unavailable for single-source request")
        sparse_weights = [{} for _ in texts]
    return embeddings, sparse_weights


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
    embeddings, sparse_weights = await _embed_single_document(texts)

    # 3. Ensure namespace (and project) exist
    namespace_id = await upsert_namespace(namespace)
    project_id = await upsert_project(namespace_id, project) if project else None

    # 4. Replace dense and sparse rows as one source generation.
    chunk_dicts = _chunk_dicts(chunk_result, namespace, source_path, content)
    source_document = _build_source_document(namespace, source_path, content)
    inserted = await replace_source_chunks_atomic(
        chunk_dicts,
        embeddings,
        sparse_weights,
        namespace_id,
        project_id,
        source_document=source_document,
    )

    return IndexResponse(
        chunks_indexed=inserted,
        source_path=source_path,
        namespace=namespace,
        strategy_used=chunk_result.strategy_used,
    )
