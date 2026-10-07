"""Namespace-bounded exact receiving, retaining legacy compatibility and corrupt-body refusal."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from scrutator.config import Settings, settings
from scrutator.db.models import FetchRequest
from scrutator.search import fetcher
from scrutator.search.indexer import _build_evidence_document

from .conftest import build_indexed_doc


@pytest.mark.parametrize("scope,selected", [(None, True), ([], False), (["kc2-store"], True), (["other"], False)])
def test_write_scope_compatibility(scope, selected, monkeypatch):
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", scope)
    assert (_build_evidence_document("kc2-store", "note.md", "exact\n") is not None) is selected
    assert _build_evidence_document(settings.skills_namespace, "skill.md", "exact\n") is None
    monkeypatch.setattr(settings, "evidence_exact_bytes", False)
    assert _build_evidence_document("kc2-store", "note.md", "exact\n") is None


@pytest.mark.parametrize("scope", ["kc2-store", ["*"], [""], [" kc2-store"], [1], ["kc2-store", "kc2-store"], {}])
def test_invalid_scope_never_widens(scope):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, evidence_exact_namespaces=scope)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,corrupt,exact",
    [(["kc2-store"], False, True), ([], False, False), (["other"], False, False), (["kc2-store"], True, False)],
)
async def test_read_scope_and_actual_bytes(scope, corrupt, exact, monkeypatch):
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", scope)
    body = "---\r\ntitle: Exact\r\n---\r\n\n# Heading\n\n  Unicode: λ. " + "detail " * 350 + "\n  \n"
    doc_id, digest, rows = build_indexed_doc(body, namespace="kc2-store")
    assert "".join(row["content"] for row in rows) != body
    raw = body + "tampered" if corrupt else body
    reader = AsyncMock(return_value=(raw, digest))
    with (
        patch.object(fetcher, "fetch_chunks_by_doc_id", AsyncMock(return_value=rows)),
        patch.object(fetcher, "fetch_evidence_raw_content", reader),
    ):
        response = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({1}))
    assert response.content_exact is exact
    assert response.content_hash == digest
    assert (response.content == body) is exact
    assert reader.await_count == (1 if scope == ["kc2-store"] else 0)


@pytest.mark.asyncio
async def test_zero_chunk_target_replacement_refuses_before_embedding_and_store(monkeypatch):
    from scrutator.chunker.models import ChunkResult
    from scrutator.db.models import IndexRequest
    from scrutator.search import indexer

    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", ["kc2-store"])
    with (
        patch.object(indexer, "chunk_document", return_value=ChunkResult(chunks=[], strategy_used="empty")),
        patch.object(indexer, "_embed_single_document", AsyncMock()) as embed,
        patch.object(indexer, "replace_source_chunks_atomic", AsyncMock()) as store,
    ):
        with pytest.raises(indexer.BatchIndexLimitError, match="indexable chunk"):
            await indexer.index_document("# Heading", "empty.md", namespace="kc2-store")
        with pytest.raises(indexer.BatchIndexLimitError, match="indexable chunk"):
            indexer._prepare_documents(
                [IndexRequest(content="# Heading", source_path="empty.md", namespace="kc2-store")]
            )
    embed.assert_not_awaited()
    store.assert_not_awaited()
