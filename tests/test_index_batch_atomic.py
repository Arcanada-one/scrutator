"""Atomic source replacement tests for SRCH-0026 batch indexing."""

import asyncio
import copy
from unittest.mock import AsyncMock, patch

import pytest

from scrutator.db import repository


class _Transaction:
    def __init__(self, conn):
        self.conn = conn
        self.snapshot = None

    async def __aenter__(self):
        self.snapshot = copy.deepcopy(self.conn.state)
        self.conn.events.append("transaction-enter")

    async def __aexit__(self, exc_type, _exc, _tb):
        if exc_type is not None:
            self.conn.state = self.snapshot
            self.conn.events.append("transaction-rollback")
        else:
            self.conn.events.append("transaction-commit")


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_args):
        return None


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


class _FaithfulConnection:
    def __init__(self, fail_sparse: bool = False):
        self.fail_sparse = fail_sparse
        self.events = []
        self.state = {0: {"content": "old", "sparse": {"old": 1.0}}}

    def transaction(self):
        return _Transaction(self)

    async def execute(self, sql, *args):
        compact = " ".join(sql.split())
        if "pg_advisory_xact_lock" in compact:
            self.events.append("advisory-lock")
        elif "INSERT INTO sparse_vectors" in compact:
            self.events.append(f"sparse-{args[0]}")
            if self.fail_sparse:
                raise RuntimeError("injected sparse crash")
            self.state[args[0]]["sparse"] = args[1]
        elif compact.startswith("DELETE FROM chunks"):
            self.events.append("delete-generation")
            self.state = {}
        return "OK"

    async def fetchval(self, sql, *args):
        assert "INSERT INTO chunks" in sql
        chunk_index = args[4]
        self.events.append(f"dense-{chunk_index}")
        self.state[chunk_index] = {"content": args[6], "sparse": None}
        return chunk_index


def _chunks():
    return [
        {
            "source_path": "reflection.md",
            "source_type": "markdown",
            "chunk_index": 0,
            "parent_id": None,
            "content": "new",
            "content_hash": "hash",
            "metadata": {},
            "token_count": 1,
        }
    ]


@pytest.mark.asyncio
async def test_atomic_replace_rolls_back_dense_write_when_sparse_insert_crashes():
    replace = getattr(repository, "replace_source_chunks_atomic", None)
    assert callable(replace), "atomic source replacement primitive is required"
    conn = _FaithfulConnection(fail_sparse=True)
    with (
        patch("scrutator.db.repository.get_pool", return_value=_Pool(conn)),
        pytest.raises(RuntimeError, match="injected sparse crash"),
    ):
        await replace(_chunks(), [[0.1] * 1024], [{"1": 0.5}], namespace_id=7, project_id=None)

    assert conn.state == {0: {"content": "old", "sparse": {"old": 1.0}}}
    assert conn.events[:2] == ["transaction-enter", "advisory-lock"]
    assert conn.events[-1] == "transaction-rollback"


@pytest.mark.asyncio
async def test_atomic_replace_deletes_prior_generation_inside_transaction_before_new_inserts():
    replace = getattr(repository, "replace_source_chunks_atomic", None)
    assert callable(replace), "atomic source replacement primitive is required"
    conn = _FaithfulConnection()
    conn.state[1] = {"content": "obsolete", "sparse": {"old": 0.5}}
    with patch("scrutator.db.repository.get_pool", return_value=_Pool(conn)):
        count = await replace(_chunks(), [[0.1] * 1024], [{"1": 0.5}], namespace_id=7, project_id=None)

    assert count == 1
    assert conn.state[0]["content"] == "new"
    assert conn.events.index("delete-generation") < conn.events.index("dense-0")
    assert conn.events[-1] == "transaction-commit"


@pytest.mark.asyncio
async def test_single_document_pipeline_uses_the_atomic_replace_primitive():
    from scrutator.search.indexer import index_document

    with (
        patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 1024]),
        patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock, return_value=[{"1": 0.5}]),
        patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock, return_value=7),
        patch(
            "scrutator.search.indexer.replace_source_chunks_atomic",
            new_callable=AsyncMock,
            return_value=1,
        ) as replace,
    ):
        response = await index_document(
            content="single lesson",
            source_path="one.md",
            namespace="self-improvement",
        )

    assert response.chunks_indexed == 1
    replace.assert_awaited_once()


@pytest.mark.asyncio
async def test_single_document_pipeline_rejects_malformed_vectors_before_atomic_replace():
    from scrutator.search.indexer import index_document

    with (
        patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock, return_value=[[float("nan")] * 1024]),
        patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock, return_value=[{"1": 0.5}]),
        patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as replace,
        pytest.raises(ValueError, match="invalid dense embeddings"),
    ):
        await index_document(content="single lesson", source_path="one.md", namespace="self-improvement")

    replace.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_document_keeps_sparse_soft_fail_contract_with_atomic_dense_write():
    from scrutator.search.indexer import index_document

    with (
        patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 1024]),
        patch(
            "scrutator.search.indexer.embed_sparse",
            new_callable=AsyncMock,
            side_effect=RuntimeError("sparse unavailable"),
        ),
        patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock, return_value=7),
        patch(
            "scrutator.search.indexer.replace_source_chunks_atomic",
            new_callable=AsyncMock,
            return_value=1,
        ) as replace,
    ):
        response = await index_document(content="single lesson", source_path="one.md", namespace="self-improvement")

    assert response.chunks_indexed == 1
    assert replace.await_args.args[2] == [{}]


@pytest.mark.asyncio
async def test_atomic_replace_is_idempotent_on_replay():
    conn = _FaithfulConnection()
    with patch("scrutator.db.repository.get_pool", return_value=_Pool(conn)):
        first = await repository.replace_source_chunks_atomic(_chunks(), [[0.1] * 1024], [{"1": 0.5}], namespace_id=7)
        second = await repository.replace_source_chunks_atomic(_chunks(), [[0.1] * 1024], [{"1": 0.5}], namespace_id=7)

    assert (first, second) == (1, 1)
    assert list(conn.state) == [0]
    assert conn.state[0]["content"] == "new"


class _ConcurrentTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return None

    async def __aexit__(self, *_args):
        if self.conn.lock_held:
            self.conn.shared.lock.release()
            self.conn.lock_held = False


class _ConcurrentConnection:
    def __init__(self, shared):
        self.shared = shared
        self.lock_held = False

    def transaction(self):
        return _ConcurrentTransaction(self)

    async def execute(self, sql, *args):
        compact = " ".join(sql.split())
        if "pg_advisory_xact_lock" in compact:
            await self.shared.lock.acquire()
            self.lock_held = True
        elif "INSERT INTO sparse_vectors" in compact:
            self.shared.events.append(f"sparse:{args[0]}")
        elif compact.startswith("DELETE FROM chunks"):
            self.shared.events.append("delete-generation")
        return "OK"

    async def fetchval(self, sql, *args):
        assert "INSERT INTO chunks" in sql
        content = args[6]
        self.shared.events.append(f"dense:{content}")
        await asyncio.sleep(0)
        return f"00000000-0000-0000-0000-00000000000{args[4]}"


class _ConcurrentAcquire:
    def __init__(self, shared):
        self.conn = _ConcurrentConnection(shared)

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_args):
        return None


class _ConcurrentPool:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.events = []

    def acquire(self):
        return _ConcurrentAcquire(self)


def _generation(prefix: str):
    chunks = []
    for index in range(2):
        chunks.append(
            {
                **_chunks()[0],
                "chunk_index": index,
                "content": f"{prefix}{index}",
                "content_hash": f"hash-{prefix}{index}",
            }
        )
    return chunks


@pytest.mark.asyncio
async def test_concurrent_same_source_replacements_do_not_interleave_generations():
    pool = _ConcurrentPool()
    with patch("scrutator.db.repository.get_pool", return_value=pool):
        await asyncio.gather(
            repository.replace_source_chunks_atomic(
                _generation("A"), [[0.1] * 1024] * 2, [{"1": 0.1}] * 2, namespace_id=7
            ),
            repository.replace_source_chunks_atomic(
                _generation("B"), [[0.2] * 1024] * 2, [{"2": 0.2}] * 2, namespace_id=7
            ),
        )

    dense_order = [event for event in pool.events if event.startswith("dense:")]
    assert dense_order in (
        ["dense:A0", "dense:A1", "dense:B0", "dense:B1"],
        ["dense:B0", "dense:B1", "dense:A0", "dense:A1"],
    )
