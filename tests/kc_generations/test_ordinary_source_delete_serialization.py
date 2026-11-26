"""Native regression candidate: ordinary source deletion shares replacement's lock.

Uses this directory's disposable PostgreSQL/pgvector ``pool`` fixture. No KC scope
is registered: private-generation triggers must not mask the ordinary-path race.
"""

from __future__ import annotations

import asyncio
import hashlib
from contextvars import ContextVar
from uuid import uuid4

import pytest

from scrutator.db import repository

_ROLE: ContextVar[str | None] = ContextVar("source_serialization_role", default=None)


class _Transaction:
    def __init__(self, connection, role, control, *args, **kwargs):
        self.connection = connection
        self.role = role
        self.control = control
        self.transaction = connection.transaction(*args, **kwargs)

    async def __aenter__(self):
        await self.transaction.__aenter__()
        isolation = await self.connection.fetchval("SHOW transaction_isolation")
        self.control.isolations[self.role] = isolation
        assert isolation == "read committed", f"fixture isolation differs from race precondition: {isolation}"

    async def __aexit__(self, *args):
        return await self.transaction.__aexit__(*args)


class _Connection:
    def __init__(self, connection, role, control):
        self.connection = connection
        self.role = role
        self.control = control

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def transaction(self, *args, **kwargs):
        return _Transaction(self.connection, self.role, self.control, *args, **kwargs)

    async def execute(self, sql, *args):
        result = await self.connection.execute(sql, *args)
        if self.role == self.control.held_role and " ".join(sql.split()).startswith("DELETE FROM chunks"):
            # The SQL statement really completed. The surrounding real database
            # transaction (and, after the fix, advisory lock) remains open.
            self.control.first_delete_completed.set()
            await self.control.release.wait()
        return result


class _Acquire:
    def __init__(self, pool, control):
        self.pool = pool
        self.control = control

    async def __aenter__(self):
        self.connection = await self.pool.acquire()
        role = _ROLE.get()
        self.control.pids[role] = await self.connection.fetchval("SELECT pg_backend_pid()")
        self.control.acquired[role].set()
        return _Connection(self.connection, role, self.control)

    async def __aexit__(self, *_args):
        await self.pool.release(self.connection)


class _Pool:
    def __init__(self, pool, control):
        self.pool = pool
        self.control = control

    def acquire(self):
        return _Acquire(self.pool, self.control)


class _Control:
    def __init__(self, held_role):
        self.held_role = held_role
        self.first_delete_completed = asyncio.Event()
        self.release = asyncio.Event()
        self.acquired = {role: asyncio.Event() for role in ("delete", "replace")}
        self.pids = {}
        self.isolations = {}


async def _with_role(role, operation):
    token = _ROLE.set(role)
    try:
        return await operation()
    finally:
        _ROLE.reset(token)


async def _observe_advisory_wait_or_completion(pool, control, role, task):
    """Return an observed database condition; timeout is only a failure bound."""
    async with asyncio.timeout(10):
        await control.acquired[role].wait()
        while True:
            waiting = await pool.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE pid=$1 AND locktype='advisory' AND NOT granted)",
                control.pids[role],
            )
            if waiting:
                return "advisory_wait"
            if task.done():
                # Surface unexpected database exceptions rather than classifying
                # them as the expected red-baseline early completion.
                task.result()
                return "completed"
            await asyncio.sleep(0.01)  # poll an explicit lock observation, not a timing barrier


async def _replace(namespace_id, path, label):
    text = f"# Synthetic {label}\n\nOwned regression fixture.\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return await repository.replace_source_chunks_atomic(
        [
            {
                "id": str(uuid4()),
                "source_path": path,
                "source_type": "markdown",
                "chunk_index": 0,
                "content": text,
                "content_hash": digest,
                "metadata": {},
                "token_count": 12,
            }
        ],
        [[0.0] * 1024],
        [{"1": 1.0}],
        namespace_id,
        evidence_document={
            "doc_id": hashlib.sha256(f"{namespace_id}:{path}".encode()).hexdigest()[:16],
            "content_hash": "sha256:" + digest,
            "raw_content": text,
        },
    )


async def _snapshot(pool, namespace_id, path):
    chunks = await pool.fetch(
        "SELECT c.id::text, c.content, c.content_hash, s.token_weights::text AS weights "
        "FROM chunks c LEFT JOIN sparse_vectors s ON s.chunk_id=c.id "
        "WHERE c.namespace_id=$1 AND c.source_path=$2 ORDER BY c.chunk_index",
        namespace_id,
        path,
    )
    evidence = await pool.fetchrow(
        "SELECT doc_id, content_hash, raw_content FROM evidence_documents WHERE namespace_id=$1 AND source_path=$2",
        namespace_id,
        path,
    )
    return [dict(row) for row in chunks], dict(evidence) if evidence else None


@pytest.mark.asyncio
@pytest.mark.parametrize("held_role", ["delete", "replace"], ids=["absent-path-delete-first", "replacement-first"])
async def test_ordinary_delete_and_replacement_share_source_lock(pool, monkeypatch, held_role, record_property):
    suffix = uuid4().hex
    namespace_id = await pool.fetchval("INSERT INTO namespaces(name) VALUES($1) RETURNING id", f"c366-{suffix}")
    other_namespace_id = await pool.fetchval(
        "INSERT INTO namespaces(name) VALUES($1) RETURNING id", f"c366-other-{suffix}"
    )
    target_path = "ordinary/target.md"
    other_path = "ordinary/unrelated.md"
    await _replace(namespace_id, other_path, "other-source")
    await _replace(other_namespace_id, target_path, "other-namespace")
    untouched = {
        (namespace_id, other_path): await _snapshot(pool, namespace_id, other_path),
        (other_namespace_id, target_path): await _snapshot(pool, other_namespace_id, target_path),
    }
    assert await _snapshot(pool, namespace_id, target_path) == ([], None)
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM kc_corpus_scopes WHERE namespace_id=ANY($1::int[])",
            [namespace_id, other_namespace_id],
        )
        == 0
    )

    control = _Control(held_role)

    async def controlled_pool():
        return _Pool(pool, control)

    monkeypatch.setattr(repository, "get_pool", controlled_pool)

    async def delete():
        return await repository.delete_by_source(target_path, namespace_id)

    async def replace():
        return await _replace(namespace_id, target_path, "G2")

    operations = {"delete": delete, "replace": replace}
    waiting_role = "replace" if held_role == "delete" else "delete"
    tasks = {}
    observed = None
    try:
        tasks[held_role] = asyncio.create_task(_with_role(held_role, operations[held_role]))
        async with asyncio.timeout(10):
            await control.first_delete_completed.wait()
        tasks[waiting_role] = asyncio.create_task(_with_role(waiting_role, operations[waiting_role]))
        observed = await _observe_advisory_wait_or_completion(pool, control, waiting_role, tasks[waiting_role])
    finally:
        control.release.set()
        try:
            async with asyncio.timeout(10):
                await asyncio.gather(*tasks.values())
        finally:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    target = await _snapshot(pool, namespace_id, target_path)
    record_property("transaction_isolation", str(control.isolations))
    assert control.isolations == {"delete": "read committed", "replace": "read committed"}
    for (ns, path), before in untouched.items():
        assert await _snapshot(pool, ns, path) == before
    assert tasks["replace"].result() == 1
    if held_role == "delete":
        assert tasks["delete"].result() == 0
        assert len(target[0]) == 1 and target[0][0]["weights"] is not None
        assert target[1] is not None, "concurrent replacement lost its exact-evidence row"
        assert target[1]["raw_content"] == target[0][0]["content"]
        assert target[1]["content_hash"] == "sha256:" + target[0][0]["content_hash"]
    else:
        assert tasks["delete"].result() == 1, "delete returned before the replacement committed"
        assert target == ([], None)
    assert observed == "advisory_wait", "second operation did not wait on the shared source lock"
