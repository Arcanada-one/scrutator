"""Embedding client — async httpx calls to Embedding API (arcana-db:8300)."""

from __future__ import annotations

import httpx

from scrutator.config import settings


class EmbeddingError(Exception):
    """Raised when the Embedding API returns an error."""


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Get dense embeddings for a list of texts from the Embedding API.

    Returns list of 1024-dim vectors.
    """
    if not texts:
        return []

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.embedding_api_url}/v1/embeddings",
            json={"input": texts},
        )
        if response.status_code != 200:
            raise EmbeddingError(f"Embedding API returned {response.status_code}: {response.text}")

        data = response.json()
        return [item["embedding"] for item in data["data"]]


async def embed_sparse(texts: list[str]) -> list[dict[str, float]]:
    """Get sparse (lexical weight) embeddings from the Embedding API.

    Returns list of {token_id: weight} dicts.
    """
    if not texts:
        return []

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.embedding_api_url}/v1/embeddings/sparse",
            json={"input": texts},
        )
        if response.status_code != 200:
            raise EmbeddingError(f"Sparse Embedding API returned {response.status_code}: {response.text}")

        data = response.json()
        return [item["sparse_weights"] for item in data["data"]]


async def embed_single(text: str) -> list[float]:
    """Get dense embedding for a single text."""
    results = await embed_texts([text])
    return results[0]
