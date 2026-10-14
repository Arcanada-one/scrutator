"""KC2 consumers rely on the ingest-stamped whole-document hash."""

import hashlib
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from scrutator.config import settings
from scrutator.db.models import FetchRequest

from .conftest import build_indexed_doc
from .test_evidence_exact_bytes import _EVIDENCE_DOC


@pytest.mark.asyncio
async def test_kc2_store_refuses_fetch_when_exact_bytes_are_disabled(monkeypatch):
    monkeypatch.setattr(settings, "evidence_exact_bytes", False)
    doc_id, _content_hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace="kc2-store")
    exact_reader = AsyncMock(return_value=(_EVIDENCE_DOC, "sha256:unused"))

    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(fetcher, "fetch_evidence_raw_content", exact_reader),
        pytest.raises(HTTPException) as excinfo,
    ):
        await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))

    assert excinfo.value.status_code == 409
    exact_reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_kc2_store_refuses_fetch_when_exact_bytes_are_missing(monkeypatch):
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", ["kc2-store"])
    doc_id, _content_hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace="kc2-store")

    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(fetcher, "fetch_evidence_raw_content", new_callable=AsyncMock, return_value=None),
        pytest.raises(HTTPException) as excinfo,
    ):
        await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))

    assert excinfo.value.status_code == 409


@pytest.mark.asyncio
async def test_kc2_store_returns_exact_bytes_when_present(monkeypatch):
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", ["kc2-store"])
    doc_id, content_hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace="kc2-store")

    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(
            fetcher,
            "fetch_evidence_raw_content",
            new_callable=AsyncMock,
            return_value=(_EVIDENCE_DOC, content_hash),
        ),
    ):
        response = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))

    assert response.content_exact is True
    assert response.content == _EVIDENCE_DOC
    assert "sha256:" + hashlib.sha256(response.content.encode()).hexdigest() == response.content_hash


@pytest.mark.asyncio
async def test_kc2_store_refuses_fetch_when_exact_bytes_are_stale(monkeypatch):
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", ["kc2-store"])
    doc_id, current_hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace="kc2-store")
    stale_bytes = "stale source generation"
    stale_hash = "sha256:" + hashlib.sha256(stale_bytes.encode()).hexdigest()
    assert stale_hash != current_hash

    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(
            fetcher,
            "fetch_evidence_raw_content",
            new_callable=AsyncMock,
            return_value=(stale_bytes, stale_hash),
        ),
        pytest.raises(HTTPException) as excinfo,
    ):
        await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))

    assert excinfo.value.status_code == 409
