"""Health check endpoint + API routers — minimal FastAPI app."""

from fastapi import FastAPI

from scrutator import __version__
from scrutator.chunker.engine import chunk_document
from scrutator.chunker.models import ChunkRequest, ChunkResponse
from scrutator.config import settings

app = FastAPI(title=settings.app_name, version=settings.app_version)


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
