"""KB-CONS-02 AC5: an incomplete kc-corpus generation is never served as accepted.

A document whose exact row is missing, stale, tampered or disabled (an interrupted or partial projection)
is refused with 409 for a namespace in `evidence_exact_required_namespaces`. The legacy default
(`kc2-store` only) and every other namespace keep their behaviour.
"""

import hashlib
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from scrutator.config import Settings, settings
from scrutator.db.models import FetchRequest

from .conftest import build_indexed_doc
from .test_evidence_exact_bytes import _EVIDENCE_DOC

NS = "kc-corpus"


def _configure(monkeypatch, *, exact_bytes=True, exact_namespaces=(NS,), required=(NS,)):
    monkeypatch.setattr(settings, "evidence_exact_bytes", exact_bytes)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", list(exact_namespaces))
    monkeypatch.setattr(settings, "evidence_exact_required_namespaces", list(required))


async def _fetch(rows, doc_id, exact_row):
    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(fetcher, "fetch_evidence_raw_content", new_callable=AsyncMock, return_value=exact_row),
    ):
        return await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))


def test_default_required_set_is_unchanged_and_private_namespace_is_not_in_it():
    assert Settings().evidence_exact_required_namespaces == ["kc2-store"]
    assert Settings().evidence_exact_required_for("kc2-store")
    assert not Settings().evidence_exact_required_for(NS)


@pytest.mark.parametrize("bad", ['"kc-corpus"', "[1]", '["a","a"]', '["bad name"]'])
def test_required_namespaces_reject_malformed_values(monkeypatch, bad):
    monkeypatch.setenv("SCRUTATOR_EVIDENCE_EXACT_REQUIRED_NAMESPACES", bad)
    with pytest.raises(Exception):  # noqa: B017 - pydantic settings parse/validation error
        Settings()


def test_required_namespaces_parse_from_json_env(monkeypatch):
    monkeypatch.setenv("SCRUTATOR_EVIDENCE_EXACT_REQUIRED_NAMESPACES", '["kc-corpus"]')
    assert Settings().evidence_exact_required_namespaces == ["kc-corpus"]


@pytest.mark.asyncio
async def test_missing_exact_row_is_refused(monkeypatch):
    _configure(monkeypatch)
    doc_id, _hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace=NS)
    with pytest.raises(HTTPException) as excinfo:
        await _fetch(rows, doc_id, None)
    assert excinfo.value.status_code == 409
    assert NS in excinfo.value.detail


@pytest.mark.asyncio
async def test_disabled_exact_evidence_is_refused_before_any_read(monkeypatch):
    _configure(monkeypatch, exact_bytes=False)
    doc_id, _hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace=NS)
    reader = AsyncMock(return_value=(_EVIDENCE_DOC, "sha256:unused"))
    from scrutator.search import fetcher

    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", new_callable=AsyncMock, return_value=rows),
        patch.object(fetcher, "fetch_evidence_raw_content", reader),
        pytest.raises(HTTPException) as excinfo,
    ):
        await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))
    assert excinfo.value.status_code == 409
    reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_exact_row_is_refused(monkeypatch):
    _configure(monkeypatch)
    doc_id, _hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace=NS)
    with pytest.raises(HTTPException) as excinfo:
        await _fetch(rows, doc_id, (_EVIDENCE_DOC, "sha256:" + "0" * 64))
    assert excinfo.value.status_code == 409


@pytest.mark.asyncio
async def test_tampered_exact_bytes_with_matching_stored_hash_are_refused(monkeypatch):
    _configure(monkeypatch)
    doc_id, content_hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace=NS)
    with pytest.raises(HTTPException) as excinfo:
        await _fetch(rows, doc_id, (_EVIDENCE_DOC + "tampered", content_hash))
    assert excinfo.value.status_code == 409


@pytest.mark.asyncio
async def test_complete_generation_is_served_exact(monkeypatch):
    _configure(monkeypatch)
    doc_id, content_hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace=NS)
    response = await _fetch(rows, doc_id, (_EVIDENCE_DOC, content_hash))
    assert response.content_exact is True
    assert response.content == _EVIDENCE_DOC
    assert "sha256:" + hashlib.sha256(response.content.encode()).hexdigest() == response.content_hash


@pytest.mark.asyncio
async def test_other_namespace_keeps_graceful_degradation(monkeypatch):
    _configure(monkeypatch, exact_namespaces=(NS, "other"))
    doc_id, _hash, rows = build_indexed_doc(_EVIDENCE_DOC, namespace="other")
    response = await _fetch(rows, doc_id, None)
    assert response.content_exact is False
