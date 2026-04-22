"""Health check endpoint + API routers — minimal FastAPI app."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from scrutator import __version__
from scrutator.chunker.engine import chunk_document
from scrutator.chunker.models import ChunkRequest, ChunkResponse
from scrutator.config import settings
from scrutator.db.connection import apply_schema, close_pool, get_pool
from scrutator.db.models import (
    ChunkLookupResult,
    IndexRequest,
    IndexResponse,
    IndexStats,
    NamespaceCreate,
    NamespaceInfo,
    SearchRequest,
    SearchResponse,
)
from scrutator.db.repository import (
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
from scrutator.search.indexer import index_document
from scrutator.search.searcher import search


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to DB and apply schema. Shutdown: close pool."""
    try:
        await get_pool()
        await apply_schema()
    except Exception:
        pass  # DB optional — chunking works without it
    yield
    await close_pool()


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

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
async def chunk_endpoint(request: ChunkRequest) -> ChunkResponse:
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
async def index_endpoint(request: IndexRequest) -> IndexResponse:
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
        raise HTTPException(status_code=503, detail=f"Index failed: {e}") from e


@app.post("/v1/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest) -> SearchResponse:
    try:
        return await search(
            query=request.query,
            namespace=request.namespace,
            project=request.project,
            source_type=request.source_type,
            limit=request.limit,
            min_score=request.min_score,
            include_content=request.include_content,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Search failed: {e}") from e


@app.get("/v1/chunks", response_model=list[ChunkLookupResult])
async def get_chunks(source_path: str, namespace: str | None = None) -> list[ChunkLookupResult]:
    try:
        namespace_id = None
        if namespace:
            namespaces = await get_namespaces()
            for ns in namespaces:
                if ns.name == namespace:
                    namespace_id = ns.id
                    break
        return await get_chunks_by_source_path(source_path, namespace_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Chunk lookup failed: {e}") from e


@app.post("/v1/namespaces", response_model=NamespaceInfo)
async def create_namespace(request: NamespaceCreate) -> NamespaceInfo:
    try:
        ns_id = await upsert_namespace(request.name, request.description)
        return NamespaceInfo(id=ns_id, name=request.name, description=request.description, chunk_count=0)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Namespace creation failed: {e}") from e


@app.get("/v1/namespaces", response_model=list[NamespaceInfo])
async def list_namespaces() -> list[NamespaceInfo]:
    try:
        return await get_namespaces()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to list namespaces: {e}") from e


@app.get("/v1/stats", response_model=IndexStats)
async def stats_endpoint() -> IndexStats:
    try:
        data = await get_stats()
        return IndexStats(**data)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to get stats: {e}") from e


# ── Dream endpoints ─────────────────────────────────────────────────


@app.post("/v1/dream/analyze", response_model=DreamAnalysisResult)
async def dream_analyze_endpoint(request: DreamAnalysisRequest) -> DreamAnalysisResult:
    try:
        return await dream_analyze(request)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Dream analysis failed: {e}") from e


@app.post("/v1/edges")
async def create_edges(edges: list[EdgeCreate]) -> dict:
    try:
        count = await insert_edges([e.model_dump() for e in edges])
        return {"created": count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Edge creation failed: {e}") from e


@app.get("/v1/edges/{chunk_id}", response_model=list[EdgeInfo])
async def get_edges(chunk_id: str) -> list[EdgeInfo]:
    try:
        rows = await get_edges_for_chunk(chunk_id)
        return [EdgeInfo(**r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to get edges: {e}") from e


@app.delete("/v1/edges")
async def delete_edges(created_by: str, namespace: str | None = None) -> dict:
    try:
        namespace_id = None
        if namespace:
            namespaces = await get_namespaces()
            for ns in namespaces:
                if ns.name == namespace:
                    namespace_id = ns.id
                    break
        count = await delete_edges_by_creator(created_by, namespace_id)
        return {"deleted": count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Edge deletion failed: {e}") from e


@app.post("/v1/edges/by-path", response_model=EdgeCreateByPathResponse)
async def create_edges_by_path_endpoint(edges: list[EdgeCreateByPath]) -> EdgeCreateByPathResponse:
    try:
        return await create_edges_by_path(edges)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Edge creation by path failed: {e}") from e


# ── Memory endpoints ───────────────────────────────────────────────


@app.post("/v1/memories", response_model=MemoryIndexResponse)
async def create_memory(record: MemoryRecord) -> MemoryIndexResponse:
    try:
        return await index_memory(record)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory index failed: {e}") from e


@app.post("/v1/memories/bulk", response_model=MemoryBulkResponse)
async def create_memories_bulk(request: MemoryBulkRequest) -> MemoryBulkResponse:
    try:
        return await memory_bulk_index(request.memories)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Bulk memory index failed: {e}") from e


@app.post("/v1/memories/recall", response_model=MemoryRecallResponse)
async def recall_memories(request: MemoryRecallRequest) -> MemoryRecallResponse:
    try:
        return await memory_recall(request)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory recall failed: {e}") from e


@app.get("/v1/memories/stats", response_model=MemoryStats)
async def memory_stats_endpoint() -> MemoryStats:
    try:
        return await get_memory_stats()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory stats failed: {e}") from e


@app.delete("/v1/memories")
async def delete_memories(actor: str, namespace: str | None = None) -> dict:
    from scrutator.db.repository import delete_memories_by_actor

    try:
        namespace_id = None
        if namespace:
            namespaces = await get_namespaces()
            for ns in namespaces:
                if ns.name == namespace:
                    namespace_id = ns.id
                    break
        count = await delete_memories_by_actor(actor, namespace_id)
        return {"deleted": count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory deletion failed: {e}") from e
