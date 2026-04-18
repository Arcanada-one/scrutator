"""Health check endpoint — minimal FastAPI app."""

from fastapi import FastAPI

from scrutator import __version__
from scrutator.config import settings

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": __version__,
    }
