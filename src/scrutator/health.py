"""Health check endpoint + API routers — minimal FastAPI app."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from scrutator import __version__
from scrutator.chunker.engine import chunk_document
from scrutator.chunker.models import ChunkRequest, ChunkResponse
from scrutator.config import settings
from scrutator.db.connection import apply_schema, close_pool, get_pool
from scrutator.db.models import (
    IndexRequest,
    IndexResponse,
    IndexStats,
    NamespaceCreate,
    NamespaceInfo,
    SearchRequest,
    SearchResponse,
)
from scrutator.db.repository import get_namespaces, get_stats, upsert_namespace
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
            limit=request.limit,
            min_score=request.min_score,
            include_content=request.include_content,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Search failed: {e}") from e


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
