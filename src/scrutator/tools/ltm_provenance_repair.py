"""Bounded, operator-gated provenance repair for legacy LTM entities (LTM-0014).

``prepare`` is the only phase that calls the configured LLM. It never mutates
the graph. ``apply`` and ``rollback`` use one serializable transaction plus a
namespace advisory lock and compare-and-set predicates. The default ``audit``
phase is read-only. No command is an implicit production authorization.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import stat
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scrutator.config import settings
from scrutator.db.connection import close_pool, get_pool
from scrutator.ltm.llm import LtmLlmClient
from scrutator.ltm.pipeline import IngestPipeline

TASK_ID = "LTM-0014"
SCHEMA_VERSION = 1
EXTRACTION_CONTRACT_VERSION = 1
DEFAULT_STATE_ROOT = Path("/var/lib/scrutator/ltm-backfill/LTM-0014")
_APPROVAL_MESSAGE = "LTM-0014 guard: paid extraction and graph mutation remain operator-gated"


class PlanError(RuntimeError):
    """A plan or live snapshot failed a fail-closed safety check."""


class ApprovalError(RuntimeError):
    """A mutation or paid-call approval token was absent or incorrect."""


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    source_path: str
    chunk_index: int
    content_hash: str
    content: str = field(repr=False)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntityRecord:
    id: str
    name: str
    entity_type: str
    source_chunk_id: str | None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntitySourceRecord:
    entity_id: str
    namespace_id: int
    source_path: str
    content_hash: str
    source_chunk_id: str | None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    entity_type: str


@dataclass(frozen=True)
class Repair:
    entity_id: str
    entity_name: str
    entity_type: str
    chunk_id: str
    source_path: str
    content_hash: str


@dataclass(frozen=True)
class RepairDecisions:
    repairs: list[Repair]
    ambiguous_entity_ids: list[str]
    absent_entity_ids: list[str]


@dataclass(frozen=True)
class Snapshot:
    namespace: str
    namespace_id: int
    chunks: list[ChunkRecord]
    entities: list[EntityRecord]
    entity_sources: list[EntitySourceRecord]
    digest: str


@dataclass(frozen=True)
class RepairPlan:
    run_id: str
    namespace: str
    namespace_id: int
    snapshot_digest: str
    connector: str
    model: str
    max_entities_per_chunk: int
    extraction_contract_version: int
    repairs: list[Repair]
    ambiguous_entity_ids: list[str]
    absent_entity_ids: list[str]
    created_at: str
    plan_sha256: str

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Snapshot,
        run_id: str,
        repairs: Sequence[Repair],
        ambiguous_entity_ids: Sequence[str] = (),
        absent_entity_ids: Sequence[str] = (),
    ) -> RepairPlan:
        body = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "run_id": run_id,
            "namespace": snapshot.namespace,
            "namespace_id": snapshot.namespace_id,
            "snapshot_digest": snapshot.digest,
            "connector": settings.ltm_connector,
            "model": settings.ltm_model,
            "max_entities_per_chunk": settings.ltm_max_entities_per_chunk,
            "extraction_contract_version": EXTRACTION_CONTRACT_VERSION,
            "created_at": _utc_now(),
            "repairs": [asdict(item) for item in repairs],
            "ambiguous_entity_ids": sorted(ambiguous_entity_ids),
            "absent_entity_ids": sorted(absent_entity_ids),
        }
        return cls(
            run_id=run_id,
            namespace=snapshot.namespace,
            namespace_id=snapshot.namespace_id,
            snapshot_digest=snapshot.digest,
            connector=body["connector"],
            model=body["model"],
            max_entities_per_chunk=body["max_entities_per_chunk"],
            extraction_contract_version=body["extraction_contract_version"],
            repairs=list(repairs),
            ambiguous_entity_ids=body["ambiguous_entity_ids"],
            absent_entity_ids=body["absent_entity_ids"],
            created_at=body["created_at"],
            plan_sha256=_digest(body),
        )

    def to_dict(self) -> dict[str, Any]:
        body = self.body()
        body["plan_sha256"] = self.plan_sha256
        return body

    def body(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "run_id": self.run_id,
            "namespace": self.namespace,
            "namespace_id": self.namespace_id,
            "snapshot_digest": self.snapshot_digest,
            "connector": self.connector,
            "model": self.model,
            "max_entities_per_chunk": self.max_entities_per_chunk,
            "extraction_contract_version": self.extraction_contract_version,
            "created_at": self.created_at,
            "repairs": [asdict(item) for item in self.repairs],
            "ambiguous_entity_ids": self.ambiguous_entity_ids,
            "absent_entity_ids": self.absent_entity_ids,
        }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _snapshot_body(
    namespace: str,
    namespace_id: int,
    chunks: Sequence[ChunkRecord],
    entities: Sequence[EntityRecord],
    entity_sources: Sequence[EntitySourceRecord],
) -> dict[str, Any]:
    return {
        "namespace": namespace,
        "namespace_id": namespace_id,
        "chunks": [
            {
                "id": item.id,
                "source_path": item.source_path,
                "chunk_index": item.chunk_index,
                "content_hash": item.content_hash,
                "content_sha256": hashlib.sha256(item.content.encode()).hexdigest(),
            }
            for item in chunks
        ],
        "entities": [asdict(item) for item in entities],
        "entity_sources": [asdict(item) for item in entity_sources],
    }


def build_snapshot(
    namespace: str,
    namespace_id: int,
    chunks: Iterable[ChunkRecord],
    entities: Iterable[EntityRecord],
    entity_sources: Iterable[EntitySourceRecord],
) -> Snapshot:
    """Build a stable whole-namespace snapshot bound to content bytes and provenance."""
    ordered_chunks = sorted(chunks, key=lambda item: (item.source_path, item.chunk_index, item.id))
    ordered_entities = sorted(entities, key=lambda item: (item.name, item.entity_type, item.id))
    ordered_sources = sorted(entity_sources, key=lambda item: (item.entity_id, item.source_path))
    body = _snapshot_body(namespace, namespace_id, ordered_chunks, ordered_entities, ordered_sources)
    return Snapshot(namespace, namespace_id, ordered_chunks, ordered_entities, ordered_sources, _digest(body))


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def contains_whole_phrase(content: str, phrase: str) -> bool:
    """Match a normalized phrase only when adjacent characters are not word characters."""
    normalized_content = _normalize(content)
    normalized_phrase = _normalize(phrase)
    if not normalized_phrase:
        return False
    pattern = rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)"
    return re.search(pattern, normalized_content, flags=re.UNICODE) is not None


def _evidence_by_entity(
    snapshot: Snapshot,
    extracted_by_chunk: Mapping[str, Sequence[ExtractedEntity]],
) -> dict[str, set[str]]:
    chunks = {item.id: item for item in snapshot.chunks}
    null_entities = {(item.name, item.entity_type): item for item in snapshot.entities if item.source_chunk_id is None}
    evidence: dict[str, set[str]] = {item.id: set() for item in null_entities.values()}
    for chunk_id, extracted in extracted_by_chunk.items():
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise PlanError(f"extraction references unknown chunk {chunk_id}")
        for item in extracted:
            entity = null_entities.get((item.name, item.entity_type))
            if entity and contains_whole_phrase(chunk.content, entity.name):
                evidence[entity.id].add(chunk_id)
    return evidence


def _decisions_from_evidence(snapshot: Snapshot, evidence: Mapping[str, set[str]]) -> RepairDecisions:
    chunks = {item.id: item for item in snapshot.chunks}
    entities = {item.id: item for item in snapshot.entities if item.source_chunk_id is None}
    sourced_entities = {item.entity_id for item in snapshot.entity_sources}
    repairs: list[Repair] = []
    ambiguous: list[str] = []
    absent: list[str] = []
    for entity_id, entity in sorted(entities.items()):
        matches = sorted(evidence.get(entity_id, set()))
        if entity_id in sourced_entities or not matches:
            absent.append(entity_id)
        elif len(matches) > 1:
            ambiguous.append(entity_id)
        else:
            chunk = chunks[matches[0]]
            repairs.append(
                Repair(entity.id, entity.name, entity.entity_type, chunk.id, chunk.source_path, chunk.content_hash)
            )
    return RepairDecisions(repairs, ambiguous, absent)


def build_repair_decisions(
    snapshot: Snapshot,
    extracted_by_chunk: Mapping[str, Sequence[ExtractedEntity]],
) -> RepairDecisions:
    """Select only exact-key entities supported by one literal whole-phrase chunk."""
    return _decisions_from_evidence(snapshot, _evidence_by_entity(snapshot, extracted_by_chunk))


def validate_plan(encoded: Mapping[str, Any]) -> RepairPlan:
    """Parse a plan and reject schema, task, shape, or digest drift."""
    if encoded.get("schema_version") != SCHEMA_VERSION or encoded.get("task_id") != TASK_ID:
        raise PlanError("unsupported LTM-0014 plan schema")
    supplied_digest = encoded.get("plan_sha256")
    body = {key: value for key, value in encoded.items() if key != "plan_sha256"}
    if not isinstance(supplied_digest, str) or supplied_digest != _digest(body):
        raise PlanError("plan digest mismatch")
    try:
        repairs = [Repair(**item) for item in encoded["repairs"]]
        plan = RepairPlan(
            run_id=encoded["run_id"],
            namespace=encoded["namespace"],
            namespace_id=encoded["namespace_id"],
            snapshot_digest=encoded["snapshot_digest"],
            connector=encoded["connector"],
            model=encoded["model"],
            max_entities_per_chunk=encoded["max_entities_per_chunk"],
            extraction_contract_version=encoded["extraction_contract_version"],
            repairs=repairs,
            ambiguous_entity_ids=list(encoded["ambiguous_entity_ids"]),
            absent_entity_ids=list(encoded["absent_entity_ids"]),
            created_at=encoded["created_at"],
            plan_sha256=supplied_digest,
        )
        _validate_plan_shape(plan)
        return plan
    except (KeyError, TypeError) as exc:
        raise PlanError("invalid LTM-0014 plan shape") from exc


def _validate_plan_shape(plan: RepairPlan) -> None:
    repaired = [item.entity_id for item in plan.repairs]
    if len(repaired) != len(set(repaired)):
        raise PlanError("duplicate entity repair in plan")
    classified = [*repaired, *plan.ambiguous_entity_ids, *plan.absent_entity_ids]
    if len(classified) != len(set(classified)):
        raise PlanError("entity appears in multiple plan classifications")
    for repair in plan.repairs:
        if not repair.source_path or re.fullmatch(r"[0-9a-f]{64}", repair.content_hash) is None:
            raise PlanError(f"invalid chunk metadata for {repair.entity_id}")
    if plan.max_entities_per_chunk <= 0 or plan.extraction_contract_version != EXTRACTION_CONTRACT_VERSION:
        raise PlanError("invalid extraction contract in plan")


def _validate_plan_integrity(plan: RepairPlan) -> None:
    _validate_plan_shape(plan)
    if plan.plan_sha256 != _digest(plan.body()):
        raise PlanError("plan digest mismatch")


def _validate_repairs_against_snapshot(plan: RepairPlan, snapshot: Snapshot) -> None:
    chunks = {item.id: item for item in snapshot.chunks}
    for repair in plan.repairs:
        chunk = chunks.get(repair.chunk_id)
        if chunk is None or (chunk.source_path, chunk.content_hash) != (repair.source_path, repair.content_hash):
            raise PlanError(f"chunk metadata drift for {repair.entity_id}")


def require_approval(env_name: str, expected: str) -> None:
    """Fail closed unless a process-environment approval exactly matches the bound value."""
    if not expected or os.environ.get(env_name) != expected:
        raise ApprovalError(f"{_APPROVAL_MESSAGE}; set {env_name} to the exact bound value")


def validated_run_dir(state_root: Path, run_id: str) -> Path:
    """Resolve one flat operator run directory without traversal or separator aliases."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise PlanError("run-id must be one safe path component")
    return state_root / run_id


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a private JSON state file without a mode-0644 window."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _private_json(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        raise PlanError(f"refusing non-private or non-regular state file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PlanError("state file must contain a JSON object")
    return data


def load_plan(path: Path) -> RepairPlan:
    return validate_plan(_private_json(path))


def _chunk_from_row(row: Mapping[str, Any]) -> ChunkRecord:
    return ChunkRecord(
        id=str(row["id"]),
        source_path=row["source_path"],
        chunk_index=row["chunk_index"],
        content_hash=row["content_hash"],
        content=row["content"],
    )


def _entity_from_row(row: Mapping[str, Any]) -> EntityRecord:
    source = row["source_chunk_id"]
    return EntityRecord(str(row["id"]), row["name"], row["entity_type"], str(source) if source else None)


def _source_from_row(row: Mapping[str, Any]) -> EntitySourceRecord:
    source = row["source_chunk_id"]
    return EntitySourceRecord(
        entity_id=str(row["entity_id"]),
        namespace_id=row["namespace_id"],
        source_path=row["source_path"],
        content_hash=row["content_hash"],
        source_chunk_id=str(source) if source else None,
    )


async def load_snapshot(conn: Any, namespace: str) -> Snapshot:
    """Read the exact namespace state used by prepare/apply validation."""
    ns = await conn.fetchrow("SELECT id FROM namespaces WHERE name = $1", namespace)
    if ns is None:
        raise PlanError(f"namespace {namespace!r} does not exist")
    namespace_id = ns["id"]
    await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(namespace_id))
    chunks = await conn.fetch(
        """
        SELECT id::text AS id, source_path, chunk_index, content_hash, content
        FROM chunks WHERE namespace_id = $1
        ORDER BY source_path, chunk_index, id
        """,
        namespace_id,
    )
    entities = await conn.fetch(
        """
        SELECT id::text AS id, name, entity_type, source_chunk_id::text AS source_chunk_id
        FROM entities WHERE namespace_id = $1
        ORDER BY name, entity_type, id
        """,
        namespace_id,
    )
    sources = await conn.fetch(
        """
        SELECT entity_id::text AS entity_id, namespace_id, source_path, content_hash,
               source_chunk_id::text AS source_chunk_id
        FROM entity_sources WHERE namespace_id = $1
        ORDER BY entity_id, source_path
        """,
        namespace_id,
    )
    return build_snapshot(
        namespace,
        namespace_id,
        (_chunk_from_row(row) for row in chunks),
        (_entity_from_row(row) for row in entities),
        (_source_from_row(row) for row in sources),
    )


async def read_snapshot(pool: Any, namespace: str) -> Snapshot:
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        return await load_snapshot(conn, namespace)


def audit_snapshot(snapshot: Snapshot) -> dict[str, Any]:
    linked = sum(item.source_chunk_id is not None for item in snapshot.entities)
    null_count = len(snapshot.entities) - linked
    return {
        "task_id": TASK_ID,
        "namespace": snapshot.namespace,
        "namespace_id": snapshot.namespace_id,
        "snapshot_digest": snapshot.digest,
        "chunks": len(snapshot.chunks),
        "entities": len(snapshot.entities),
        "entities_linked": linked,
        "entities_null_source_chunk_id": null_count,
        "entity_sources": len(snapshot.entity_sources),
    }


def _parse_command_count(status: str, expected_command: str) -> int:
    parts = status.split()
    if not parts or parts[0] != expected_command:
        raise PlanError(f"unexpected database status {status!r}")
    return int(parts[-1])


async def _set_tenant(conn: Any, plan: RepairPlan) -> None:
    await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(plan.namespace_id))


async def _acquire_session_lock(conn: Any, plan: RepairPlan) -> None:
    await conn.execute("SELECT pg_advisory_lock(hashtextextended($1, 0))", f"ltm0014:{plan.namespace_id}")


async def _release_session_lock(conn: Any, plan: RepairPlan) -> None:
    await conn.execute("SELECT pg_advisory_unlock(hashtextextended($1, 0))", f"ltm0014:{plan.namespace_id}")


def _repair_current_state(snapshot: Snapshot, repair: Repair) -> tuple[EntityRecord, EntitySourceRecord | None]:
    entity = next((item for item in snapshot.entities if item.id == repair.entity_id), None)
    if entity is None or (entity.name, entity.entity_type) != (repair.entity_name, repair.entity_type):
        raise PlanError(f"entity identity drift for {repair.entity_id}")
    source = next(
        (
            item
            for item in snapshot.entity_sources
            if item.entity_id == repair.entity_id and item.source_path == repair.source_path
        ),
        None,
    )
    return entity, source


def _is_exact_applied(
    entity: EntityRecord,
    source: EntitySourceRecord | None,
    repair: Repair,
    namespace_id: int,
) -> bool:
    return bool(
        entity.source_chunk_id == repair.chunk_id
        and source
        and source.namespace_id == namespace_id
        and source.content_hash == repair.content_hash
        and source.source_chunk_id == repair.chunk_id
    )


async def _apply_one(conn: Any, plan: RepairPlan, repair: Repair) -> None:
    updated = await conn.execute(
        """
        UPDATE entities
        SET source_chunk_id = $2::uuid, updated_at = NOW()
        WHERE id = $1::uuid AND namespace_id = $3 AND name = $4 AND entity_type = $5
          AND source_chunk_id IS NULL
        """,
        repair.entity_id,
        repair.chunk_id,
        plan.namespace_id,
        repair.entity_name,
        repair.entity_type,
    )
    if _parse_command_count(updated, "UPDATE") != 1:
        raise PlanError(f"NULL-only entity CAS failed for {repair.entity_id}")
    inserted = await conn.execute(
        """
        INSERT INTO entity_sources (
            entity_id, namespace_id, source_path, content_hash, source_chunk_id, updated_at
        ) VALUES ($1::uuid, $2, $3, $4, $5::uuid, NOW())
        """,
        repair.entity_id,
        plan.namespace_id,
        repair.source_path,
        repair.content_hash,
        repair.chunk_id,
    )
    if _parse_command_count(inserted, "INSERT") != 1:
        raise PlanError(f"entity_sources insert failed for {repair.entity_id}")


async def apply_plan(pool: Any, plan: RepairPlan) -> dict[str, int]:
    """Apply an unchanged prepared plan in one locked serializable transaction."""
    _validate_plan_integrity(plan)
    require_approval("LTM0014_APPLY_GO", plan.plan_sha256)
    async with pool.acquire() as conn:
        await _acquire_session_lock(conn, plan)
        try:
            async with conn.transaction(isolation="serializable"):
                await _set_tenant(conn, plan)
                current = await load_snapshot(conn, plan.namespace)
                _validate_repairs_against_snapshot(plan, current)
                exact_applied = []
                for repair in plan.repairs:
                    entity, source = _repair_current_state(current, repair)
                    exact_applied.append(_is_exact_applied(entity, source, repair, plan.namespace_id))
                if exact_applied and all(exact_applied):
                    return {"applied": 0, "already_applied": len(plan.repairs)}
                if current.digest != plan.snapshot_digest:
                    raise PlanError("namespace snapshot changed since prepare")
                if any(exact_applied):
                    raise PlanError("partially applied plan requires audit or rollback")
                for repair in plan.repairs:
                    await _apply_one(conn, plan, repair)
        finally:
            await _release_session_lock(conn, plan)
    return {"applied": len(plan.repairs), "already_applied": 0}


async def _rollback_one(conn: Any, plan: RepairPlan, repair: Repair) -> bool:
    deleted = await conn.execute(
        """
        DELETE FROM entity_sources
        WHERE entity_id = $1::uuid AND namespace_id = $2 AND source_path = $3
          AND content_hash = $4 AND source_chunk_id = $5::uuid
        """,
        repair.entity_id,
        plan.namespace_id,
        repair.source_path,
        repair.content_hash,
        repair.chunk_id,
    )
    updated = await conn.execute(
        """
        UPDATE entities SET source_chunk_id = NULL, updated_at = NOW()
        WHERE id = $1::uuid AND namespace_id = $2 AND source_chunk_id = $3::uuid
          AND name = $4 AND entity_type = $5
        """,
        repair.entity_id,
        plan.namespace_id,
        repair.chunk_id,
        repair.entity_name,
        repair.entity_type,
    )
    deleted_count = _parse_command_count(deleted, "DELETE")
    updated_count = _parse_command_count(updated, "UPDATE")
    if deleted_count != updated_count or updated_count not in {0, 1}:
        raise PlanError(f"rollback CAS mismatch for {repair.entity_id}")
    return updated_count == 1


async def rollback_plan(pool: Any, plan: RepairPlan) -> dict[str, int]:
    """Rollback only exact plan-owned values; never overwrite newer provenance."""
    _validate_plan_integrity(plan)
    require_approval("LTM0014_ROLLBACK_GO", plan.plan_sha256)
    rolled_back = 0
    already = 0
    async with pool.acquire() as conn:
        await _acquire_session_lock(conn, plan)
        try:
            async with conn.transaction(isolation="serializable"):
                await _set_tenant(conn, plan)
                current = await load_snapshot(conn, plan.namespace)
                _validate_repairs_against_snapshot(plan, current)
                for repair in plan.repairs:
                    entity, source = _repair_current_state(current, repair)
                    if entity.source_chunk_id is None and source is None:
                        already += 1
                        continue
                    if not _is_exact_applied(entity, source, repair, plan.namespace_id):
                        raise PlanError(f"rollback refused non-plan provenance for {repair.entity_id}")
                    rolled_back += int(await _rollback_one(conn, plan, repair))
        finally:
            await _release_session_lock(conn, plan)
    return {"rolled_back": rolled_back, "already_rolled_back": already}


def _usage_receipt(state: Mapping[str, Any], snapshot: Snapshot) -> dict[str, Any]:
    records = state["usage_records"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "run_id": state["run_id"],
        "namespace": snapshot.namespace,
        "snapshot_digest": snapshot.digest,
        "connector": state["connector"],
        "model": state["model"],
        "max_entities_per_chunk": state["max_entities_per_chunk"],
        "extraction_contract_version": state["extraction_contract_version"],
        "request_count": len(records),
        "input_tokens": sum(int(record.get("input_tokens") or 0) for record in records),
        "output_tokens": sum(int(record.get("output_tokens") or 0) for record in records),
        "total_tokens": sum(int(record.get("total_tokens") or 0) for record in records),
        "cost_usd": sum(float(record.get("cost_usd") or 0) for record in records),
        "success_count": sum(record.get("status") == "success" for record in records),
        "failure_count": sum(record.get("status") != "success" for record in records),
        "updated_at": _utc_now(),
    }


def _new_prepare_state(run_id: str, snapshot: Snapshot) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "run_id": run_id,
        "namespace": snapshot.namespace,
        "namespace_id": snapshot.namespace_id,
        "snapshot_digest": snapshot.digest,
        "connector": settings.ltm_connector,
        "model": settings.ltm_model,
        "max_entities_per_chunk": settings.ltm_max_entities_per_chunk,
        "extraction_contract_version": EXTRACTION_CONTRACT_VERSION,
        "completed_chunk_ids": [],
        "inflight_chunk_id": None,
        "evidence_by_entity": {},
        "usage_records": [],
        "updated_at": _utc_now(),
    }
    body["state_sha256"] = _digest(body)
    return body


def _validate_prepare_state(state: Mapping[str, Any], run_id: str, snapshot: Snapshot) -> dict[str, Any]:
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    if state.get("state_sha256") != _digest(body):
        raise PlanError("prepare state digest mismatch")
    expected = (TASK_ID, run_id, snapshot.namespace, snapshot.namespace_id, snapshot.digest)
    actual = (
        state.get("task_id"),
        state.get("run_id"),
        state.get("namespace"),
        state.get("namespace_id"),
        state.get("snapshot_digest"),
    )
    if actual != expected:
        raise PlanError("prepare state is not bound to the current namespace snapshot")
    extraction_expected = (
        settings.ltm_connector,
        settings.ltm_model,
        settings.ltm_max_entities_per_chunk,
        EXTRACTION_CONTRACT_VERSION,
    )
    extraction_actual = (
        state.get("connector"),
        state.get("model"),
        state.get("max_entities_per_chunk"),
        state.get("extraction_contract_version"),
    )
    if extraction_actual != extraction_expected:
        raise PlanError("prepare state extraction configuration changed")
    if state.get("inflight_chunk_id") is not None:
        raise PlanError("uncertain paid request is still in flight; refusing automatic rebill")
    return dict(state)


def _save_prepare_state(run_dir: Path, state: dict[str, Any], snapshot: Snapshot) -> None:
    state["updated_at"] = _utc_now()
    state.pop("state_sha256", None)
    state["state_sha256"] = _digest(state)
    atomic_write_json(run_dir / "prepare-state.json", state)
    receipt = _usage_receipt(state, snapshot)
    atomic_write_json(run_dir / "cost-receipt.json", receipt)


def _restore_or_create_state(run_dir: Path, run_id: str, snapshot: Snapshot) -> dict[str, Any]:
    state_path = run_dir / "prepare-state.json"
    if not state_path.exists():
        return _new_prepare_state(run_id, snapshot)
    return _validate_prepare_state(_private_json(state_path), run_id, snapshot)


def _record_evidence(state: dict[str, Any], snapshot: Snapshot, chunk: ChunkRecord, extracted: Sequence[Any]) -> None:
    null_entities = {(item.name, item.entity_type): item for item in snapshot.entities if item.source_chunk_id is None}
    evidence = state["evidence_by_entity"]
    for item in extracted:
        entity = null_entities.get((item.name, item.entity_type))
        if entity is None or not contains_whole_phrase(chunk.content, entity.name):
            continue
        chunk_ids = evidence.setdefault(entity.id, [])
        if chunk.id not in chunk_ids:
            chunk_ids.append(chunk.id)


def _state_decisions(snapshot: Snapshot, state: Mapping[str, Any]) -> RepairDecisions:
    evidence = {entity_id: set(chunk_ids) for entity_id, chunk_ids in state["evidence_by_entity"].items()}
    return _decisions_from_evidence(snapshot, evidence)


def _prepared_result(plan: RepairPlan, run_dir: Path) -> dict[str, Any]:
    return {
        "status": "prepared",
        "repairs": len(plan.repairs),
        "ambiguous": len(plan.ambiguous_entity_ids),
        "absent": len(plan.absent_entity_ids),
        "plan_sha256": plan.plan_sha256,
        "run_dir": str(run_dir),
    }


def _load_completed_plan(plan_path: Path, run_id: str, snapshot: Snapshot) -> RepairPlan | None:
    if not plan_path.exists():
        return None
    plan = load_plan(plan_path)
    expected = (run_id, snapshot.namespace, snapshot.namespace_id, snapshot.digest)
    actual = (plan.run_id, plan.namespace, plan.namespace_id, plan.snapshot_digest)
    if actual != expected:
        raise PlanError("completed plan is not bound to the current namespace snapshot")
    extraction_expected = (
        settings.ltm_connector,
        settings.ltm_model,
        settings.ltm_max_entities_per_chunk,
        EXTRACTION_CONTRACT_VERSION,
    )
    extraction_actual = (
        plan.connector,
        plan.model,
        plan.max_entities_per_chunk,
        plan.extraction_contract_version,
    )
    if extraction_actual != extraction_expected:
        raise PlanError("completed plan extraction configuration changed")
    return plan


def _extraction_pipeline(snapshot: Snapshot, state: dict[str, Any]) -> IngestPipeline:
    llm = LtmLlmClient(
        mc_url=settings.ltm_mc_url,
        connector=settings.ltm_connector,
        model=settings.ltm_model,
        api_key=settings.ltm_mc_api_key,
        usage_sink=state["usage_records"].append,
    )
    return IngestPipeline(llm, snapshot.namespace, snapshot.namespace_id, settings.ltm_max_entities_per_chunk)


async def _process_prepare_batch(
    pipeline: IngestPipeline,
    batch: Sequence[ChunkRecord],
    state: dict[str, Any],
    snapshot: Snapshot,
    run_dir: Path,
) -> None:
    for chunk in batch:
        state["inflight_chunk_id"] = chunk.id
        _save_prepare_state(run_dir, state, snapshot)
        try:
            extracted = await pipeline.extract_entities(chunk.content)
        except Exception:
            _save_prepare_state(run_dir, state, snapshot)
            raise
        _record_evidence(state, snapshot, chunk, extracted)
        state["completed_chunk_ids"].append(chunk.id)
        state["inflight_chunk_id"] = None
        _save_prepare_state(run_dir, state, snapshot)


async def prepare_plan(
    pool: Any,
    namespace: str,
    run_id: str,
    state_root: Path = DEFAULT_STATE_ROOT,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """Run resumable extraction only; write a final plan after every chunk is complete."""
    if max_chunks is not None and max_chunks <= 0:
        raise PlanError("max_chunks must be positive")
    require_approval("LTM0014_PREPARE_GO", run_id)
    snapshot = await read_snapshot(pool, namespace)
    run_dir = validated_run_dir(state_root, run_id)
    plan_path = run_dir / "plan.json"
    completed_plan = _load_completed_plan(plan_path, run_id, snapshot)
    if completed_plan is not None:
        return _prepared_result(completed_plan, run_dir)
    state = _restore_or_create_state(run_dir, run_id, snapshot)
    _save_prepare_state(run_dir, state, snapshot)
    completed = set(state["completed_chunk_ids"])
    remaining = [item for item in snapshot.chunks if item.id not in completed]
    batch = remaining[:max_chunks] if max_chunks is not None else remaining
    await _process_prepare_batch(_extraction_pipeline(snapshot, state), batch, state, snapshot, run_dir)
    remaining_count = len(snapshot.chunks) - len(state["completed_chunk_ids"])
    if remaining_count:
        return {"status": "continuation_required", "remaining_chunks": remaining_count, "run_dir": str(run_dir)}
    decisions = _state_decisions(snapshot, state)
    plan = RepairPlan.from_snapshot(
        snapshot,
        run_id,
        decisions.repairs,
        decisions.ambiguous_entity_ids,
        decisions.absent_entity_ids,
    )
    atomic_write_json(plan_path, plan.to_dict())
    return _prepared_result(plan, run_dir)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="read-only namespace counts and digest")
    audit.add_argument("--namespace", required=True)
    prepare = subparsers.add_parser("prepare", help="paid extraction only; no graph mutation")
    prepare.add_argument("--namespace", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    prepare.add_argument("--max-chunks", type=int)
    for command in ("apply", "rollback"):
        mutation = subparsers.add_parser(command, help=f"operator-gated {command}")
        mutation.add_argument("--plan", type=Path, required=True)
        mutation.add_argument("--live", action="store_true", help="required mutation opt-in")
    return parser.parse_args(argv)


async def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "audit":
        pool = await get_pool()
        return audit_snapshot(await read_snapshot(pool, args.namespace))
    if args.command == "prepare":
        if args.max_chunks is not None and args.max_chunks <= 0:
            raise PlanError("--max-chunks must be positive")
        require_approval("LTM0014_PREPARE_GO", args.run_id)
        pool = await get_pool()
        return await prepare_plan(pool, args.namespace, args.run_id, args.state_root, args.max_chunks)
    if not args.live:
        raise ApprovalError(f"{_APPROVAL_MESSAGE}; {args.command} also requires --live")
    plan = load_plan(args.plan)
    env_name = "LTM0014_APPLY_GO" if args.command == "apply" else "LTM0014_ROLLBACK_GO"
    require_approval(env_name, plan.plan_sha256)
    pool = await get_pool()
    result = await (apply_plan(pool, plan) if args.command == "apply" else rollback_plan(pool, plan))
    receipt_name = "apply-receipt.json" if args.command == "apply" else "rollback-receipt.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "run_id": plan.run_id,
        "namespace": plan.namespace,
        "plan_sha256": plan.plan_sha256,
        "at": _utc_now(),
        **result,
    }
    atomic_write_json(args.plan.parent / receipt_name, receipt)
    return result


async def _main(argv: Sequence[str] | None = None) -> None:
    try:
        result = await _run_cli(_parse_args(argv))
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
