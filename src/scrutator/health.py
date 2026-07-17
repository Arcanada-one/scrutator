"""Health check endpoint + API routers — minimal FastAPI app."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from scrutator import __version__
from scrutator.auth.capabilities import (
    NamespaceCapability,
    require_feeder_capability,
    require_rollback_capability,
)
from scrutator.auth.dependency import require_tenant_context, resolve_namespace_selector
from scrutator.auth.models import TenantContext
from scrutator.chunker.engine import chunk_document
from scrutator.chunker.models import ChunkRequest, ChunkResponse
from scrutator.config import settings
from scrutator.db.connection import apply_schema, close_pool, get_pool
from scrutator.db.models import (
    INDEX_BATCH_MAX_REQUEST_BYTES,
    BatchIndexRequest,
    BatchIndexResponse,
    ChunkLookupResult,
    DeleteSourceRequest,
    DeleteSourceResponse,
    IndexRequest,
    IndexResponse,
    IndexStats,
    NamespaceCreate,
    NamespaceInfo,
    OutlineResponse,
    SearchRequest,
    SearchResponse,
    SectionContext,
)
from scrutator.db.repository import (
    delete_by_source,
    delete_edges_by_creator,
    get_chunks_by_source_path,
    get_edges_for_chunk,
    get_namespaces,
    get_stats,
    insert_edges,
    upsert_namespace,
)
from scrutator.dream.analyzer import analyze as dream_analyze
from scrutator.dream.edges import create_edges_by_path
from scrutator.dream.models import (
    DreamAnalysisRequest,
    DreamAnalysisResult,
    EdgeCreate,
    EdgeCreateByPath,
    EdgeCreateByPathResponse,
    EdgeInfo,
)
from scrutator.memory.models import (
    MemoryBulkRequest,
    MemoryBulkResponse,
    MemoryIndexResponse,
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemoryRecord,
    MemoryStats,
)
from scrutator.memory.service import (
    bulk_index as memory_bulk_index,
)
from scrutator.memory.service import (
    get_memory_stats,
    index_memory,
)
from scrutator.memory.service import (
    recall as memory_recall,
)
from scrutator.request_limits import BoundedRequestBodyMiddleware
from scrutator.search.embedder import close_client as close_embedding_client
from scrutator.search.indexer import BatchIndexLimitError, index_document, index_documents
from scrutator.search.navigator import build_outline, build_section_context
from scrutator.search.searcher import search

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to DB and apply schema. Shutdown: close pool."""
    try:
        await get_pool()
        await apply_schema()
    except Exception:
        pass  # DB optional — chunking works without it
    yield
    await close_embedding_client()
    await close_pool()


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(
    BoundedRequestBodyMiddleware,
    path="/v1/index/batch",
    max_bytes=INDEX_BATCH_MAX_REQUEST_BYTES,
)


# LTM router
from scrutator.ltm.router import router as ltm_router  # noqa: E402

app.include_router(ltm_router)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": __version__,
    }


@app.post("/v1/chunk", response_model=ChunkResponse)
async def chunk_endpoint(request: ChunkRequest, ctx: TenantContext = Depends(require_tenant_context)) -> ChunkResponse:
    result = chunk_document(
        content=request.content,
        source_path=request.source_path,
        source_type=request.source_type,
        max_tokens=request.max_tokens,
        overlap_tokens=request.overlap_tokens,
    )
    return ChunkResponse(
        chunks=result.chunks,
        total_chunks=result.total_chunks,
        total_tokens=result.total_tokens,
        strategy_used=result.strategy_used,
    )


@app.post("/v1/index", response_model=IndexResponse)
async def index_endpoint(
    request: IndexRequest,
    capability: NamespaceCapability = Depends(require_feeder_capability),
) -> IndexResponse:
    # Reader namespace grants never imply mutation authority. Indexing is
    # intentionally restricted to the dedicated, namespace-scoped Feeder
    # credential even when the bearer principal may read this namespace.
    if request.namespace not in capability.namespaces:
        raise HTTPException(status_code=403, detail="namespace outside feeder scope")
    try:
        return await index_document(
            content=request.content,
            source_path=request.source_path,
            namespace=request.namespace,
            project=request.project,
            source_type=request.source_type,
            max_tokens=request.max_tokens,
            overlap_tokens=request.overlap_tokens,
        )
    except Exception as e:
        logger.exception("Index failed for %s", request.source_path)
        raise HTTPException(status_code=503, detail=f"Index failed: {type(e).__name__}: {e}") from e


@app.post("/v1/index/batch", response_model=BatchIndexResponse)
async def batch_index_endpoint(
    request: BatchIndexRequest,
    capability: NamespaceCapability = Depends(require_feeder_capability),
) -> BatchIndexResponse:
    if request.documents[0].namespace not in capability.namespaces:
        raise HTTPException(status_code=403, detail="namespace outside feeder scope")
    try:
        return BatchIndexResponse(results=await index_documents(request.documents))
    except BatchIndexLimitError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/v1/index", response_model=DeleteSourceResponse)
async def delete_source_endpoint(
    request: DeleteSourceRequest,
    capability: NamespaceCapability = Depends(require_rollback_capability),
) -> DeleteSourceResponse:
    """Tombstone one source inside a namespace granted to the caller."""
    # Read-capable bearer principals cannot delete. Only the two dedicated
    # rollback credentials carry mutation authority.
    if not capability.operator and request.namespace not in capability.namespaces:
        raise HTTPException(status_code=403, detail="namespace outside rollback scope")
    pool = await get_pool()
    async with pool.acquire() as conn:
        namespace_id = await conn.fetchval("SELECT id FROM namespaces WHERE name=$1", request.namespace)
    if namespace_id is None:
        raise HTTPException(status_code=404, detail="namespace not found")
    try:
        deleted = await delete_by_source(request.source_path, namespace_id)
        return DeleteSourceResponse(
            namespace=request.namespace,
            source_path=request.source_path,
            chunks_deleted=deleted,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Delete failed: {e}") from e


@app.post("/v1/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest, ctx: TenantContext = Depends(require_tenant_context)
) -> SearchResponse:
    namespace_id = await resolve_namespace_selector(ctx, request.namespace)
    try:
        return await search(
            query=request.query,
            namespace_id=namespace_id,
            project=request.project,
            source_type=request.source_type,
            limit=request.limit,
            min_score=request.min_score,
            include_content=request.include_content,
            group_by=request.group_by,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Search failed: {e}") from e


# ── Navigation endpoints (SRCH-0021) ─────────────────────────────────


@app.get("/v1/navigate/outline", response_model=OutlineResponse)
async def navigate_outline(
    namespace: str,
    source_path: str,
    max_nodes: int = 2000,
    ctx: TenantContext = Depends(require_tenant_context),
) -> OutlineResponse:
    await resolve_namespace_selector(ctx, namespace)
    return await build_outline(namespace=namespace, source_path=source_path, max_nodes=max_nodes)


@app.get("/v1/navigate/section", response_model=SectionContext)
async def navigate_section(
    chunk_id: str,
    ctx: TenantContext = Depends(require_tenant_context),
) -> SectionContext:
    return await build_section_context(chunk_id, ctx.allowed_namespace_ids)


@app.get("/v1/chunks", response_model=list[ChunkLookupResult])
async def get_chunks(
    source_path: str,
    namespace: str | None = None,
    ctx: TenantContext = Depends(require_tenant_context),
) -> list[ChunkLookupResult]:
    namespace_id = await resolve_namespace_selector(ctx, namespace)
    try:
        return await get_chunks_by_source_path(source_path, namespace_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Chunk lookup failed: {e}") from e


@app.post("/v1/namespaces", response_model=NamespaceInfo)
async def create_namespace(
    request: NamespaceCreate, ctx: TenantContext = Depends(require_tenant_context)
) -> NamespaceInfo:
    # Privileged write: namespace creation requires a verified principal — never permitted
    # for the grace-window anonymous context, even while SCRUTATOR_AUTH_ENFORCE=False.
    if ctx.principal_id == "anonymous":
        raise HTTPException(status_code=401, detail="namespace creation requires an authenticated principal")
    if request.name not in ctx.allowed_namespace_names:
        raise HTTPException(status_code=403, detail="namespace outside caller scope")
    try:
        ns_id = await upsert_namespace(request.name, request.description)
        return NamespaceInfo(id=ns_id, name=request.name, description=request.description, chunk_count=0)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Namespace creation failed: {e}") from e


@app.get("/v1/namespaces", response_model=list[NamespaceInfo])
async def list_namespaces(ctx: TenantContext = Depends(require_tenant_context)) -> list[NamespaceInfo]:
    try:
        return await get_namespaces(namespace_ids=ctx.allowed_namespace_ids)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to list namespaces: {e}") from e


@app.get("/v1/stats", response_model=IndexStats)
async def stats_endpoint(ctx: TenantContext = Depends(require_tenant_context)) -> IndexStats:
    try:
        data = await get_stats(namespace_ids=ctx.allowed_namespace_ids)
        return IndexStats(**data)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to get stats: {e}") from e


# ── Dream endpoints ─────────────────────────────────────────────────


@app.post("/v1/dream/analyze", response_model=DreamAnalysisResult)
async def dream_analyze_endpoint(
    request: DreamAnalysisRequest, ctx: TenantContext = Depends(require_tenant_context)
) -> DreamAnalysisResult:
    try:
        return await dream_analyze(request, namespace_ids=ctx.allowed_namespace_ids)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Dream analysis failed: {e}") from e


@app.post("/v1/edges")
async def create_edges(edges: list[EdgeCreate], ctx: TenantContext = Depends(require_tenant_context)) -> dict:
    try:
        count = await insert_edges([e.model_dump() for e in edges], ctx.allowed_namespace_ids)
        return {"created": count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Edge creation failed: {e}") from e


@app.get("/v1/edges/{chunk_id}", response_model=list[EdgeInfo])
async def get_edges(chunk_id: str, ctx: TenantContext = Depends(require_tenant_context)) -> list[EdgeInfo]:
    try:
        rows = await get_edges_for_chunk(chunk_id, ctx.allowed_namespace_ids)
        return [EdgeInfo(**r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to get edges: {e}") from e


@app.delete("/v1/edges")
async def delete_edges(
    created_by: str,
    namespace: str | None = None,
    ctx: TenantContext = Depends(require_tenant_context),
) -> dict:
    namespace_id = await resolve_namespace_selector(ctx, namespace)
    try:
        count = await delete_edges_by_creator(created_by, namespace_id)
        return {"deleted": count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Edge deletion failed: {e}") from e


@app.post("/v1/edges/by-path", response_model=EdgeCreateByPathResponse)
async def create_edges_by_path_endpoint(
    edges: list[EdgeCreateByPath],
    namespace: str | None = None,
    ctx: TenantContext = Depends(require_tenant_context),
) -> EdgeCreateByPathResponse:
    namespace_id = await resolve_namespace_selector(ctx, namespace)
    try:
        return await create_edges_by_path(edges, namespace_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Edge creation by path failed: {e}") from e


# ── Memory endpoints ───────────────────────────────────────────────


@app.post("/v1/memories", response_model=MemoryIndexResponse)
async def create_memory(
    record: MemoryRecord, ctx: TenantContext = Depends(require_tenant_context)
) -> MemoryIndexResponse:
    namespace_id = await resolve_namespace_selector(ctx, record.namespace)
    try:
        return await index_memory(record, namespace_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory index failed: {e}") from e


@app.post("/v1/memories/bulk", response_model=MemoryBulkResponse)
async def create_memories_bulk(
    request: MemoryBulkRequest, ctx: TenantContext = Depends(require_tenant_context)
) -> MemoryBulkResponse:
    namespace_ids = {
        namespace: await resolve_namespace_selector(ctx, namespace)
        for namespace in {record.namespace for record in request.memories}
    }
    try:
        return await memory_bulk_index(request.memories, namespace_ids)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Bulk memory index failed: {e}") from e


@app.post("/v1/memories/recall", response_model=MemoryRecallResponse)
async def recall_memories(
    request: MemoryRecallRequest, ctx: TenantContext = Depends(require_tenant_context)
) -> MemoryRecallResponse:
    namespace_id = await resolve_namespace_selector(ctx, request.namespace)
    try:
        return await memory_recall(request, namespace_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory recall failed: {e}") from e


@app.get("/v1/memories/stats", response_model=MemoryStats)
async def memory_stats_endpoint(ctx: TenantContext = Depends(require_tenant_context)) -> MemoryStats:
    try:
        return await get_memory_stats(ctx.allowed_namespace_ids)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory stats failed: {e}") from e


@app.delete("/v1/memories")
async def delete_memories(
    actor: str,
    namespace: str | None = None,
    ctx: TenantContext = Depends(require_tenant_context),
) -> dict:
    from scrutator.db.repository import delete_memories_by_actor

    namespace_id = await resolve_namespace_selector(ctx, namespace)
    try:
        count = await delete_memories_by_actor(actor, namespace_id)
        return {"deleted": count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory deletion failed: {e}") from e
