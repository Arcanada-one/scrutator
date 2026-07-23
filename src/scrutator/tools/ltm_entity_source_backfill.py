"""Deterministic, reversible historical entity-provenance repair (LTM-0019).

The planner makes no LLM, embedding, or external API calls. It repairs only
NULL entities supported by one structural provenance chunk and one unique,
normalized whole-name occurrence in that chunk.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scrutator.db.connection import close_pool, get_pool

TASK_ID = "LTM-0019"
SCHEMA_VERSION = 2
METHOD = "structural_plus_unique_literal_v1"
DEFAULT_STATE_ROOT = Path("/var/lib/scrutator/ltm-backfill/LTM-0019")
TARGET_NAMESPACE = "ltm-bench-datarim-kb"
TARGET_NAMESPACE_ID = 825


class PlanError(RuntimeError):
    """The live snapshot, plan, receipt, or filesystem state is unsafe."""


class ApprovalError(RuntimeError):
    """A live mutation was not explicitly bound to the exact plan digest."""


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    source_path: str
    content_hash: str
    content: str


@dataclass(frozen=True)
class DatabaseIdentity:
    database: str
    database_oid: str
    system_identifier: str
    in_recovery: bool


@dataclass(frozen=True)
class EntityRecord:
    id: str
    name: str
    entity_type: str
    source_chunk_id: str | None
    updated_at: str


@dataclass(frozen=True)
class Repair:
    entity_id: str
    entity_name: str
    entity_type: str
    original_updated_at: str
    chunk_id: str
    source_path: str
    content_hash: str
    source_updated_at: str = ""


@dataclass(frozen=True)
class Classification:
    repairs: list[Repair]
    counts: dict[str, int]


@dataclass(frozen=True)
class Snapshot:
    database_identity: DatabaseIdentity
    namespace: str
    namespace_id: int
    chunks: list[ChunkRecord]
    entities: list[EntityRecord]
    preowned_entity_ids: frozenset[str]
    evidence_by_entity: dict[str, frozenset[str]]
    digest: str

    @classmethod
    def build(
        cls,
        *,
        database_identity: DatabaseIdentity,
        namespace: str,
        namespace_id: int,
        chunks: Iterable[ChunkRecord],
        entities: Iterable[EntityRecord],
        preowned_entity_ids: frozenset[str],
        evidence_by_entity: Mapping[str, Iterable[str]],
    ) -> Snapshot:
        ordered_chunks = sorted(chunks, key=lambda item: (item.source_path, item.id))
        ordered_entities = sorted(entities, key=lambda item: (item.name, item.entity_type, item.id))
        ordered_evidence = {
            entity_id: frozenset(chunk_ids) for entity_id, chunk_ids in sorted(evidence_by_entity.items()) if chunk_ids
        }
        body = {
            "database_identity": asdict(database_identity),
            "namespace": namespace,
            "namespace_id": namespace_id,
            "chunks": [
                {
                    "id": item.id,
                    "source_path": item.source_path,
                    "content_hash": item.content_hash,
                    "content_sha256": hashlib.sha256(item.content.encode()).hexdigest(),
                }
                for item in ordered_chunks
            ],
            "entities": [asdict(item) for item in ordered_entities],
            "preowned_entity_ids": sorted(preowned_entity_ids),
            "evidence_by_entity": {entity_id: sorted(chunk_ids) for entity_id, chunk_ids in ordered_evidence.items()},
        }
        return cls(
            database_identity=database_identity,
            namespace=namespace,
            namespace_id=namespace_id,
            chunks=ordered_chunks,
            entities=ordered_entities,
            preowned_entity_ids=preowned_entity_ids,
            evidence_by_entity=ordered_evidence,
            digest=_digest(body),
        )


@dataclass(frozen=True)
class RepairPlan:
    task_id: str
    schema_version: int
    method: str
    run_id: str
    database_identity: DatabaseIdentity
    namespace: str
    namespace_id: int
    snapshot_digest: str
    selection: str
    parent_plan_sha256: str | None
    classification_counts: dict[str, int]
    repairs: list[Repair]
    created_at: str
    plan_sha256: str

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Snapshot,
        run_id: str,
        classification: Classification,
        *,
        created_at: str | None = None,
    ) -> RepairPlan:
        validated_run_dir(Path("."), run_id)
        plan_created_at = created_at or _utc_now()
        repairs = [replace(item, source_updated_at=plan_created_at) for item in classification.repairs]
        body = {
            "task_id": TASK_ID,
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "run_id": run_id,
            "database_identity": asdict(snapshot.database_identity),
            "namespace": snapshot.namespace,
            "namespace_id": snapshot.namespace_id,
            "snapshot_digest": snapshot.digest,
            "selection": "all",
            "parent_plan_sha256": None,
            "classification_counts": dict(sorted(classification.counts.items())),
            "repairs": [asdict(item) for item in repairs],
            "created_at": plan_created_at,
        }
        return cls(
            task_id=TASK_ID,
            schema_version=SCHEMA_VERSION,
            method=METHOD,
            run_id=run_id,
            database_identity=snapshot.database_identity,
            namespace=snapshot.namespace,
            namespace_id=snapshot.namespace_id,
            snapshot_digest=snapshot.digest,
            selection="all",
            parent_plan_sha256=None,
            classification_counts=body["classification_counts"],
            repairs=repairs,
            created_at=body["created_at"],
            plan_sha256=_digest(body),
        )

    def body(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "schema_version": self.schema_version,
            "method": self.method,
            "run_id": self.run_id,
            "database_identity": asdict(self.database_identity),
            "namespace": self.namespace,
            "namespace_id": self.namespace_id,
            "snapshot_digest": self.snapshot_digest,
            "selection": self.selection,
            "parent_plan_sha256": self.parent_plan_sha256,
            "classification_counts": self.classification_counts,
            "repairs": [asdict(item) for item in self.repairs],
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "plan_sha256": self.plan_sha256}

    def validate_against(self, snapshot: Snapshot) -> None:
        if (
            snapshot.database_identity != self.database_identity
            or snapshot.namespace != self.namespace
            or snapshot.namespace_id != self.namespace_id
            or snapshot.digest != self.snapshot_digest
        ):
            raise PlanError("snapshot drift")
        classification = classify_snapshot(snapshot)
        if self.classification_counts != classification.counts:
            raise PlanError("classification drift")
        if self.selection == "all":
            expected = RepairPlan.from_snapshot(
                snapshot,
                self.run_id,
                classification,
                created_at=self.created_at,
            )
        else:
            if not self.run_id.endswith("-canary"):
                raise PlanError("canary run-id is not derived from a full plan")
            full = RepairPlan.from_snapshot(
                snapshot,
                self.run_id.removesuffix("-canary"),
                classification,
                created_at=self.created_at,
            )
            if full.plan_sha256 != self.parent_plan_sha256:
                raise PlanError("canary parent plan drift")
            expected = derive_canary_plan(full)
        if self.to_dict() != expected.to_dict():
            raise PlanError("repair selection is not the exact recomputed eligible set")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def count_whole_phrase(content: str, phrase: str) -> int:
    normalized_phrase = _normalize(phrase)
    if not normalized_phrase:
        return 0
    pattern = rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)"
    return len(re.findall(pattern, _normalize(content), flags=re.UNICODE))


def classify_snapshot(snapshot: Snapshot) -> Classification:
    chunks = {item.id: item for item in snapshot.chunks}
    categories = (
        "eligible",
        "ambiguous_structural",
        "literal_absent",
        "literal_nonunique",
        "no_structural",
        "preowned",
        "invalid_metadata",
    )
    counts = dict.fromkeys(categories, 0)
    repairs: list[Repair] = []

    for entity in snapshot.entities:
        if entity.source_chunk_id is not None:
            continue
        if entity.id in snapshot.preowned_entity_ids:
            counts["preowned"] += 1
            continue
        evidence = snapshot.evidence_by_entity.get(entity.id, frozenset())
        if not evidence:
            counts["no_structural"] += 1
            continue
        if len(evidence) > 1:
            counts["ambiguous_structural"] += 1
            continue
        chunk_id = next(iter(evidence))
        selected = chunks.get(chunk_id)
        if (
            selected is None
            or not selected.source_path
            or re.fullmatch(r"[0-9a-f]{64}", selected.content_hash or "") is None
        ):
            counts["invalid_metadata"] += 1
            continue

        occurrences = {chunk.id: count_whole_phrase(chunk.content, entity.name) for chunk in snapshot.chunks}
        selected_count = occurrences[chunk_id]
        total_count = sum(occurrences.values())
        if selected_count == 0:
            counts["literal_absent"] += 1
            continue
        if selected_count != 1 or total_count != 1:
            counts["literal_nonunique"] += 1
            continue

        counts["eligible"] += 1
        repairs.append(
            Repair(
                entity_id=entity.id,
                entity_name=entity.name,
                entity_type=entity.entity_type,
                original_updated_at=entity.updated_at,
                chunk_id=selected.id,
                source_path=selected.source_path,
                content_hash=selected.content_hash,
            )
        )

    repairs.sort(key=lambda item: item.entity_id)
    return Classification(repairs=repairs, counts=counts)


def validate_plan(encoded: Mapping[str, Any]) -> RepairPlan:
    body = {key: value for key, value in encoded.items() if key != "plan_sha256"}
    supplied_digest = encoded.get("plan_sha256")
    if not isinstance(supplied_digest, str) or supplied_digest != _digest(body):
        raise PlanError("plan digest mismatch")
    try:
        plan = RepairPlan(
            task_id=encoded["task_id"],
            schema_version=encoded["schema_version"],
            method=encoded["method"],
            run_id=encoded["run_id"],
            database_identity=DatabaseIdentity(**encoded["database_identity"]),
            namespace=encoded["namespace"],
            namespace_id=encoded["namespace_id"],
            snapshot_digest=encoded["snapshot_digest"],
            selection=encoded["selection"],
            parent_plan_sha256=encoded["parent_plan_sha256"],
            classification_counts=dict(encoded["classification_counts"]),
            repairs=[Repair(**item) for item in encoded["repairs"]],
            created_at=encoded["created_at"],
            plan_sha256=supplied_digest,
        )
    except (KeyError, TypeError) as exc:
        raise PlanError("invalid plan shape") from exc
    if plan.task_id != TASK_ID or plan.schema_version != SCHEMA_VERSION or plan.method != METHOD:
        raise PlanError("unsupported LTM-0019 plan contract")
    if plan.namespace != TARGET_NAMESPACE or plan.namespace_id != TARGET_NAMESPACE_ID:
        raise PlanError("LTM-0019 plans are restricted to the exact benchmark namespace")
    identity = plan.database_identity
    if (
        not identity.database
        or not identity.database_oid.isdigit()
        or not identity.system_identifier.isdigit()
        or identity.in_recovery
    ):
        raise PlanError("unsafe or invalid database identity")
    validated_run_dir(Path("."), plan.run_id)
    if len({item.entity_id for item in plan.repairs}) != len(plan.repairs):
        raise PlanError("duplicate entity repair")
    if plan.selection == "all":
        if plan.parent_plan_sha256 is not None:
            raise PlanError("full plan cannot have a parent plan")
    elif plan.selection == "canary":
        if (
            len(plan.repairs) != 1
            or not isinstance(plan.parent_plan_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", plan.parent_plan_sha256) is None
        ):
            raise PlanError("invalid canary selection contract")
    else:
        raise PlanError("unsupported plan selection")
    for repair in plan.repairs:
        if (
            not repair.source_path
            or re.fullmatch(r"[0-9a-f]{64}", repair.content_hash) is None
            or not repair.original_updated_at
            or repair.source_updated_at != plan.created_at
        ):
            raise PlanError(f"invalid repair metadata for {repair.entity_id}")
    return plan


def derive_canary_plan(full_plan: RepairPlan) -> RepairPlan:
    if full_plan.selection != "all" or full_plan.parent_plan_sha256 is not None:
        raise PlanError("canary must be derived from a full plan")
    if not full_plan.repairs:
        raise PlanError("cannot derive a canary from an empty plan")
    chosen = min(
        full_plan.repairs,
        key=lambda repair: hashlib.sha256(f"{full_plan.plan_sha256}{repair.entity_id}".encode()).hexdigest(),
    )
    run_id = f"{full_plan.run_id}-canary"
    validated_run_dir(Path("."), run_id)
    body = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "run_id": run_id,
        "database_identity": asdict(full_plan.database_identity),
        "namespace": full_plan.namespace,
        "namespace_id": full_plan.namespace_id,
        "snapshot_digest": full_plan.snapshot_digest,
        "selection": "canary",
        "parent_plan_sha256": full_plan.plan_sha256,
        "classification_counts": full_plan.classification_counts,
        "repairs": [asdict(chosen)],
        "created_at": full_plan.created_at,
    }
    return RepairPlan(
        task_id=TASK_ID,
        schema_version=SCHEMA_VERSION,
        method=METHOD,
        run_id=run_id,
        database_identity=full_plan.database_identity,
        namespace=full_plan.namespace,
        namespace_id=full_plan.namespace_id,
        snapshot_digest=full_plan.snapshot_digest,
        selection="canary",
        parent_plan_sha256=full_plan.plan_sha256,
        classification_counts=full_plan.classification_counts,
        repairs=[chosen],
        created_at=full_plan.created_at,
        plan_sha256=_digest(body),
    )


def validated_run_dir(state_root: Path, run_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise PlanError("run-id must be one safe path component")
    return state_root / run_id


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _open_private_directory(path: Path, *, create: bool) -> tuple[int, Path]:
    absolute = _absolute_path(path)
    if any(part in (".", "..") for part in absolute.parts[1:]):
        raise PlanError(f"state directory contains unsafe path components: {absolute}")
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current /= part
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=descriptor)
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise PlanError(f"refusing symlinked or non-directory state path component: {current}") from exc
                raise
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise PlanError(f"state directory is not owner-private or owned by this process: {absolute}")
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _safe_state_name(path: Path) -> str:
    name = path.name
    if not name or name in (".", "..") or "/" in name:
        raise PlanError(f"unsafe state filename: {path}")
    return name


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    parent_fd, _ = _open_private_directory(path.parent, create=True)
    target_name = _safe_state_name(path)
    temporary_name = f".{target_name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError as exc:
            raise PlanError("private state temporary-name collision") from exc
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(_canonical_json(payload))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.chmod(target_name, 0o600, dir_fd=parent_fd, follow_symlinks=False)
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_fd)
        os.close(parent_fd)


def _private_bytes(path: Path) -> bytes:
    parent_fd, _ = _open_private_directory(path.parent, create=False)
    target_name = _safe_state_name(path)
    descriptor = os.open(target_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise PlanError(f"refusing non-private, non-owned, or non-regular state file: {path}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _private_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_private_bytes(path))
    if not isinstance(payload, dict):
        raise PlanError("state file must contain a JSON object")
    return payload


def load_plan(path: Path) -> RepairPlan:
    return validate_plan(_private_json(path))


_STRUCTURAL_EVIDENCE_SQL = """
SELECT ee.source_entity_id AS entity_id, ee.source_chunk_id AS chunk_id
FROM entity_edges ee
JOIN entities e ON e.id = ee.source_entity_id
WHERE e.namespace_id = $1 AND ee.source_chunk_id IS NOT NULL
UNION ALL
SELECT ee.target_entity_id, ee.source_chunk_id
FROM entity_edges ee
JOIN entities e ON e.id = ee.target_entity_id
WHERE e.namespace_id = $1 AND ee.source_chunk_id IS NOT NULL
UNION ALL
SELECT ev.entity_id, ev.source_chunk_id
FROM entity_events ev
JOIN entities e ON e.id = ev.entity_id
WHERE e.namespace_id = $1 AND ev.source_chunk_id IS NOT NULL
UNION ALL
SELECT ee.source_entity_id, ees.source_chunk_id
FROM entity_edge_sources ees
JOIN entity_edges ee ON ee.id = ees.edge_id
JOIN entities e ON e.id = ee.source_entity_id
WHERE e.namespace_id = $1 AND ees.source_chunk_id IS NOT NULL
UNION ALL
SELECT ee.target_entity_id, ees.source_chunk_id
FROM entity_edge_sources ees
JOIN entity_edges ee ON ee.id = ees.edge_id
JOIN entities e ON e.id = ee.target_entity_id
WHERE e.namespace_id = $1 AND ees.source_chunk_id IS NOT NULL
"""


async def load_snapshot(conn: Any, namespace: str) -> Snapshot:
    identity_row = await conn.fetchrow(
        """
        SELECT current_database() AS database,
               (SELECT oid::text FROM pg_database WHERE datname = current_database()) AS database_oid,
               (SELECT system_identifier::text FROM pg_control_system()) AS system_identifier,
               pg_is_in_recovery() AS in_recovery
        """
    )
    if identity_row is None:
        raise PlanError("database identity is unavailable")
    database_identity = DatabaseIdentity(**dict(identity_row))
    if database_identity.in_recovery:
        raise PlanError("LTM-0019 refuses a recovery/replica database")
    namespace_row = await conn.fetchrow("SELECT id FROM namespaces WHERE name = $1", namespace)
    if namespace_row is None:
        raise PlanError(f"namespace {namespace!r} does not exist")
    namespace_id = namespace_row["id"]
    await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(namespace_id))
    chunk_rows = await conn.fetch(
        """
        SELECT id::text AS id, source_path, content_hash, content
        FROM chunks
        WHERE namespace_id = $1
        ORDER BY source_path, chunk_index, id
        """,
        namespace_id,
    )
    entity_rows = await conn.fetch(
        """
        SELECT id::text AS id, name, entity_type,
               source_chunk_id::text AS source_chunk_id, updated_at
        FROM entities
        WHERE namespace_id = $1
        ORDER BY name, entity_type, id
        """,
        namespace_id,
    )
    source_rows = await conn.fetch(
        "SELECT entity_id::text AS entity_id FROM entity_sources WHERE namespace_id = $1 ORDER BY entity_id",
        namespace_id,
    )
    evidence_rows = await conn.fetch(_STRUCTURAL_EVIDENCE_SQL, namespace_id)
    evidence: dict[str, set[str]] = defaultdict(set)
    for row in evidence_rows:
        if row["entity_id"] is not None and row["chunk_id"] is not None:
            evidence[str(row["entity_id"])].add(str(row["chunk_id"]))
    return Snapshot.build(
        database_identity=database_identity,
        namespace=namespace,
        namespace_id=namespace_id,
        chunks=[
            ChunkRecord(str(row["id"]), row["source_path"], row["content_hash"], row["content"]) for row in chunk_rows
        ],
        entities=[
            EntityRecord(
                str(row["id"]),
                row["name"],
                row["entity_type"],
                str(row["source_chunk_id"]) if row["source_chunk_id"] else None,
                row["updated_at"].isoformat(),
            )
            for row in entity_rows
        ],
        preowned_entity_ids=frozenset(row["entity_id"] for row in source_rows),
        evidence_by_entity=evidence,
    )


async def read_snapshot(pool: Any, namespace: str) -> Snapshot:
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        return await load_snapshot(conn, namespace)


def audit_snapshot(snapshot: Snapshot) -> dict[str, Any]:
    classification = classify_snapshot(snapshot)
    linked = sum(item.source_chunk_id is not None for item in snapshot.entities)
    null_count = len(snapshot.entities) - linked
    return {
        "task_id": TASK_ID,
        "method": METHOD,
        "namespace": snapshot.namespace,
        "namespace_id": snapshot.namespace_id,
        "snapshot_digest": snapshot.digest,
        "chunks": len(snapshot.chunks),
        "entities": len(snapshot.entities),
        "entities_linked": linked,
        "entities_null_source_chunk_id": null_count,
        "classification_counts": classification.counts,
    }


def require_approval(env_name: str, expected: str) -> None:
    if not expected or os.environ.get(env_name) != expected:
        raise ApprovalError(f"set {env_name} to the exact LTM-0019 plan digest")


def _parse_command_count(status: str, expected_command: str) -> int:
    parts = status.split()
    if not parts or parts[0] != expected_command:
        raise PlanError(f"unexpected database status {status!r}")
    return int(parts[-1])


async def _acquire_session_lock(conn: Any, namespace_id: int) -> None:
    await conn.execute("SELECT pg_advisory_lock(hashtextextended($1, 0))", f"ltm0019:{namespace_id}")


async def _release_session_lock(conn: Any, namespace_id: int) -> None:
    await conn.execute("SELECT pg_advisory_unlock(hashtextextended($1, 0))", f"ltm0019:{namespace_id}")


async def _configure_mutation_transaction(conn: Any, namespace_id: int) -> None:
    await conn.execute("SET LOCAL lock_timeout = '5s'")
    await conn.execute("SET LOCAL statement_timeout = '120s'")
    await conn.execute("SELECT set_config('application_name', $1, true)", "scrutator-ltm0019-backfill")
    await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(namespace_id))


async def _lock_plan_rows(conn: Any, plan: RepairPlan) -> None:
    entity_ids = [item.entity_id for item in plan.repairs]
    chunk_ids = [item.chunk_id for item in plan.repairs]
    entity_rows = await conn.fetch(
        """
        SELECT id
        FROM entities
        WHERE namespace_id = $1 AND id = ANY($2::uuid[])
        ORDER BY id
        FOR UPDATE
        """,
        plan.namespace_id,
        entity_ids,
    )
    chunk_rows = await conn.fetch(
        """
        SELECT id
        FROM chunks
        WHERE namespace_id = $1 AND id = ANY($2::uuid[])
        ORDER BY id
        FOR KEY SHARE
        """,
        plan.namespace_id,
        chunk_ids,
    )
    if len(entity_rows) != len(set(entity_ids)) or len(chunk_rows) != len(set(chunk_ids)):
        raise PlanError("plan row lock cardinality mismatch")


async def _current_repair_state(conn: Any, plan: RepairPlan, repair: Repair) -> tuple[Any, list[Any]]:
    entity = await conn.fetchrow(
        """
        SELECT id::text AS id, name, entity_type,
               source_chunk_id::text AS source_chunk_id, updated_at
        FROM entities
        WHERE namespace_id = $1 AND id = $2::uuid
        """,
        plan.namespace_id,
        repair.entity_id,
    )
    sources = await conn.fetch(
        """
        SELECT namespace_id, source_path, content_hash,
               source_chunk_id::text AS source_chunk_id, updated_at
        FROM entity_sources
        WHERE namespace_id = $1 AND entity_id = $2::uuid
        ORDER BY source_path
        """,
        plan.namespace_id,
        repair.entity_id,
    )
    return entity, list(sources)


def _state_is_exact_applied(entity: Any, sources: list[Any], plan: RepairPlan, repair: Repair) -> bool:
    if entity is None or len(sources) != 1:
        return False
    source = sources[0]
    return bool(
        entity["name"] == repair.entity_name
        and entity["entity_type"] == repair.entity_type
        and str(entity["source_chunk_id"]) == repair.chunk_id
        and entity["updated_at"].isoformat() == repair.original_updated_at
        and source["namespace_id"] == plan.namespace_id
        and source["source_path"] == repair.source_path
        and source["content_hash"] == repair.content_hash
        and str(source["source_chunk_id"]) == repair.chunk_id
        and source["updated_at"].isoformat() == repair.source_updated_at
    )


def _virtual_preapply_snapshot(snapshot: Snapshot, plan: RepairPlan) -> Snapshot:
    target_ids = {item.entity_id for item in plan.repairs}
    entities = [replace(item, source_chunk_id=None) if item.id in target_ids else item for item in snapshot.entities]
    return Snapshot.build(
        database_identity=snapshot.database_identity,
        namespace=snapshot.namespace,
        namespace_id=snapshot.namespace_id,
        chunks=snapshot.chunks,
        entities=entities,
        preowned_entity_ids=snapshot.preowned_entity_ids - target_ids,
        evidence_by_entity=snapshot.evidence_by_entity,
    )


async def _assert_exact_postwrite_readback(conn: Any, plan: RepairPlan) -> None:
    for repair in plan.repairs:
        rows = await conn.fetch(
            """
            SELECT e.name, e.entity_type,
                   e.source_chunk_id::text AS entity_chunk_id,
                   e.updated_at AS entity_updated_at,
                   es.source_path, es.content_hash,
                   es.source_chunk_id::text AS source_chunk_id,
                   es.updated_at AS source_updated_at,
                   c.source_path AS chunk_source_path,
                   c.content_hash AS chunk_content_hash
            FROM entities e
            JOIN entity_sources es
              ON es.entity_id = e.id AND es.namespace_id = e.namespace_id
            JOIN chunks c
              ON c.id = es.source_chunk_id AND c.namespace_id = e.namespace_id
            WHERE e.namespace_id = $1 AND e.id = $2::uuid
            """,
            plan.namespace_id,
            repair.entity_id,
        )
        if len(rows) != 1:
            raise PlanError(f"post-write readback cardinality mismatch for {repair.entity_id}")
        row = rows[0]
        if (
            row["name"],
            row["entity_type"],
            str(row["entity_chunk_id"]),
            row["entity_updated_at"].isoformat(),
            row["source_path"],
            row["content_hash"],
            str(row["source_chunk_id"]),
            row["source_updated_at"].isoformat(),
            row["chunk_source_path"],
            row["chunk_content_hash"],
        ) != (
            repair.entity_name,
            repair.entity_type,
            repair.chunk_id,
            repair.original_updated_at,
            repair.source_path,
            repair.content_hash,
            repair.chunk_id,
            repair.source_updated_at,
            repair.source_path,
            repair.content_hash,
        ):
            raise PlanError(f"post-write readback mismatch for {repair.entity_id}")


async def apply_plan(pool: Any, plan: RepairPlan) -> dict[str, Any]:
    if plan.plan_sha256 != _digest(plan.body()):
        raise PlanError("plan digest mismatch")
    require_approval("LTM0019_APPLY_GO", plan.plan_sha256)
    source_timestamps: dict[str, str] = {}
    async with pool.acquire() as conn:
        await _acquire_session_lock(conn, plan.namespace_id)
        try:
            async with conn.transaction(isolation="serializable"):
                await _configure_mutation_transaction(conn, plan.namespace_id)
                await _lock_plan_rows(conn, plan)
                states = [await _current_repair_state(conn, plan, repair) for repair in plan.repairs]
                exact_applied = [
                    _state_is_exact_applied(entity, sources, plan, repair)
                    for repair, (entity, sources) in zip(plan.repairs, states, strict=True)
                ]
                if exact_applied and all(exact_applied):
                    current = await load_snapshot(conn, plan.namespace)
                    plan.validate_against(_virtual_preapply_snapshot(current, plan))
                    await _assert_exact_postwrite_readback(conn, plan)
                    return {
                        "applied": 0,
                        "already_applied": len(plan.repairs),
                        "source_updated_at": {repair.entity_id: repair.source_updated_at for repair in plan.repairs},
                    }
                if any(exact_applied):
                    raise PlanError("partially applied plan requires audit")
                current = await load_snapshot(conn, plan.namespace)
                plan.validate_against(current)
                for repair, (entity, sources) in zip(plan.repairs, states, strict=True):
                    if entity is None or entity["source_chunk_id"] is not None or sources:
                        raise PlanError(f"repair precondition changed for {repair.entity_id}")
                    updated = await conn.execute(
                        """
                        UPDATE entities
                        SET source_chunk_id = $2::uuid
                        WHERE namespace_id = $3 AND id = $1::uuid
                          AND name = $4 AND entity_type = $5
                          AND source_chunk_id IS NULL AND updated_at = $6::timestamptz
                        """,
                        repair.entity_id,
                        repair.chunk_id,
                        plan.namespace_id,
                        repair.entity_name,
                        repair.entity_type,
                        repair.original_updated_at,
                    )
                    if _parse_command_count(updated, "UPDATE") != 1:
                        raise PlanError(f"NULL-only entity CAS failed for {repair.entity_id}")
                    source_updated_at = await conn.fetchval(
                        """
                        INSERT INTO entity_sources (
                            entity_id, namespace_id, source_path, content_hash,
                            source_chunk_id, updated_at
                        ) VALUES ($1::uuid, $2, $3, $4, $5::uuid, $6::timestamptz)
                        RETURNING updated_at
                        """,
                        repair.entity_id,
                        plan.namespace_id,
                        repair.source_path,
                        repair.content_hash,
                        repair.chunk_id,
                        repair.source_updated_at,
                    )
                    if source_updated_at.isoformat() != repair.source_updated_at:
                        raise PlanError(f"source timestamp readback mismatch for {repair.entity_id}")
                    source_timestamps[repair.entity_id] = repair.source_updated_at
                await _assert_exact_postwrite_readback(conn, plan)
        finally:
            await _release_session_lock(conn, plan.namespace_id)
    return {
        "applied": len(plan.repairs),
        "already_applied": 0,
        "source_updated_at": source_timestamps,
    }


def _validate_apply_receipt(receipt: Mapping[str, Any], plan: RepairPlan) -> dict[str, str]:
    if receipt.get("task_id") != TASK_ID or receipt.get("plan_sha256") != plan.plan_sha256:
        raise PlanError("apply receipt is not bound to the plan")
    timestamps = receipt.get("source_updated_at")
    if not isinstance(timestamps, dict) or set(timestamps) != {item.entity_id for item in plan.repairs}:
        raise PlanError("apply receipt source timestamp set mismatch")
    expected = {item.entity_id: item.source_updated_at for item in plan.repairs}
    if timestamps != expected:
        raise PlanError("apply receipt does not contain the plan-owned source timestamps")
    return dict(timestamps)


async def rollback_plan(pool: Any, plan: RepairPlan, receipt: Mapping[str, Any]) -> dict[str, int]:
    if plan.plan_sha256 != _digest(plan.body()):
        raise PlanError("plan digest mismatch")
    require_approval("LTM0019_ROLLBACK_GO", plan.plan_sha256)
    timestamps = _validate_apply_receipt(receipt, plan)
    rolled_back = 0
    async with pool.acquire() as conn:
        await _acquire_session_lock(conn, plan.namespace_id)
        try:
            async with conn.transaction(isolation="serializable"):
                await _configure_mutation_transaction(conn, plan.namespace_id)
                await _lock_plan_rows(conn, plan)
                current = await load_snapshot(conn, plan.namespace)
                plan.validate_against(_virtual_preapply_snapshot(current, plan))
                for repair in plan.repairs:
                    entity, sources = await _current_repair_state(conn, plan, repair)
                    if not _state_is_exact_applied(entity, sources, plan, repair):
                        raise PlanError(f"rollback refused non-plan provenance for {repair.entity_id}")
                    if sources[0]["updated_at"].isoformat() != timestamps[repair.entity_id]:
                        raise PlanError(f"rollback refused newer source ownership for {repair.entity_id}")
                    deleted = await conn.execute(
                        """
                        DELETE FROM entity_sources
                        WHERE entity_id = $1::uuid AND namespace_id = $2
                          AND source_path = $3 AND content_hash = $4
                          AND source_chunk_id = $5::uuid
                          AND updated_at = $6::timestamptz
                        """,
                        repair.entity_id,
                        plan.namespace_id,
                        repair.source_path,
                        repair.content_hash,
                        repair.chunk_id,
                        timestamps[repair.entity_id],
                    )
                    updated = await conn.execute(
                        """
                        UPDATE entities
                        SET source_chunk_id = NULL, updated_at = $6::timestamptz
                        WHERE namespace_id = $2 AND id = $1::uuid
                          AND name = $3 AND entity_type = $4
                          AND source_chunk_id = $5::uuid
                          AND updated_at = $6::timestamptz
                        """,
                        repair.entity_id,
                        plan.namespace_id,
                        repair.entity_name,
                        repair.entity_type,
                        repair.chunk_id,
                        repair.original_updated_at,
                    )
                    if _parse_command_count(deleted, "DELETE") != 1 or _parse_command_count(updated, "UPDATE") != 1:
                        raise PlanError(f"rollback CAS mismatch for {repair.entity_id}")
                    rolled_back += 1
                restored = await load_snapshot(conn, plan.namespace)
                if restored.digest != plan.snapshot_digest:
                    raise PlanError("rollback did not restore the exact snapshot")
        finally:
            await _release_session_lock(conn, plan.namespace_id)
    return {"rolled_back": rolled_back}


async def prepare_plan(
    pool: Any,
    namespace: str,
    run_id: str,
    state_root: Path = DEFAULT_STATE_ROOT,
    *,
    expected_namespace_id: int | None = None,
) -> dict[str, Any]:
    snapshot = await read_snapshot(pool, namespace)
    if expected_namespace_id is not None and snapshot.namespace_id != expected_namespace_id:
        raise PlanError(
            f"namespace id drift: {namespace!r} is {snapshot.namespace_id}, expected {expected_namespace_id}"
        )
    classification = classify_snapshot(snapshot)
    plan = RepairPlan.from_snapshot(snapshot, run_id, classification)
    run_dir = validated_run_dir(state_root, run_id)
    atomic_write_json(run_dir / "plan.json", plan.to_dict())
    atomic_write_json(
        run_dir / "prepare-receipt.json",
        {
            "task_id": TASK_ID,
            "run_id": run_id,
            "namespace": namespace,
            "namespace_id": snapshot.namespace_id,
            "snapshot_digest": snapshot.digest,
            "plan_sha256": plan.plan_sha256,
            "classification_counts": classification.counts,
            "repairs": len(classification.repairs),
            "prepared_at": _utc_now(),
        },
    )
    return {
        "status": "prepared",
        "repairs": len(classification.repairs),
        "classification_counts": classification.counts,
        "snapshot_digest": snapshot.digest,
        "plan_sha256": plan.plan_sha256,
        "run_dir": str(run_dir),
    }


async def backup_plan(pool: Any, plan: RepairPlan, backup_root: Path) -> dict[str, Any]:
    target_ids = [item.entity_id for item in plan.repairs]
    chunk_ids = [item.chunk_id for item in plan.repairs]
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        snapshot = await load_snapshot(conn, plan.namespace)
        plan.validate_against(snapshot)
        entity_rows = await conn.fetch(
            """
            SELECT to_jsonb(e)::text AS row_json
            FROM entities e
            WHERE e.namespace_id = $1 AND e.id = ANY($2::uuid[])
            ORDER BY e.id
            """,
            plan.namespace_id,
            target_ids,
        )
        source_rows = await conn.fetch(
            """
            SELECT to_jsonb(es)::text AS row_json
            FROM entity_sources es
            WHERE es.namespace_id = $1 AND es.entity_id = ANY($2::uuid[])
            ORDER BY es.entity_id, es.source_path
            """,
            plan.namespace_id,
            target_ids,
        )
        chunk_rows = await conn.fetch(
            """
            SELECT (
                to_jsonb(c) - 'content' - 'embedding_dense' - 'textsearch_ru' - 'textsearch_en'
            )::text AS row_json
            FROM chunks c
            WHERE c.namespace_id = $1 AND c.id = ANY($2::uuid[])
            ORDER BY c.id
            """,
            plan.namespace_id,
            chunk_ids,
        )
    if len(entity_rows) != len(set(target_ids)) or len(chunk_rows) != len(set(chunk_ids)) or source_rows:
        raise PlanError("backup row cardinality does not match the unapplied plan")
    backup = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "snapshot_digest": plan.snapshot_digest,
        "namespace": plan.namespace,
        "namespace_id": plan.namespace_id,
        "database_identity": asdict(snapshot.database_identity),
        "entities": [json.loads(row["row_json"]) for row in entity_rows],
        "entity_sources": [json.loads(row["row_json"]) for row in source_rows],
        "chunks_without_content_or_vectors": [json.loads(row["row_json"]) for row in chunk_rows],
        "created_at": _utc_now(),
    }
    backup_path = backup_root / "backup.json"
    atomic_write_json(backup_path, backup)
    encoded = _private_bytes(backup_path)
    parsed = json.loads(encoded)
    if (
        len(parsed["entities"]) != len(set(target_ids))
        or len(parsed["chunks_without_content_or_vectors"]) != len(set(chunk_ids))
        or parsed["entity_sources"]
    ):
        raise PlanError("backup parse/recount failed")
    manifest = {
        "task_id": TASK_ID,
        "plan_sha256": plan.plan_sha256,
        "snapshot_digest": plan.snapshot_digest,
        "backup_file": backup_path.name,
        "backup_sha256": hashlib.sha256(encoded).hexdigest(),
        "entity_rows": len(entity_rows),
        "entity_source_rows": len(source_rows),
        "chunk_metadata_rows": len(chunk_rows),
        "restore_contract": (
            "Use the LTM-0019 rollback command while every applied source tuple "
            "matches its receipt; otherwise reconcile from this private backup."
        ),
        "created_at": _utc_now(),
    }
    atomic_write_json(backup_root / "manifest.json", manifest)
    return {**manifest, "backup_root": str(backup_root)}


async def audit_plan(pool: Any, plan: RepairPlan) -> dict[str, Any]:
    exact = 0
    missing = 0
    mismatched = 0
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        for repair in plan.repairs:
            entity, sources = await _current_repair_state(conn, plan, repair)
            if _state_is_exact_applied(entity, sources, plan, repair):
                chunk_exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM chunks
                        WHERE namespace_id = $1 AND id = $2::uuid
                          AND source_path = $3 AND content_hash = $4
                    )
                    """,
                    plan.namespace_id,
                    repair.chunk_id,
                    repair.source_path,
                    repair.content_hash,
                )
                if chunk_exists:
                    exact += 1
                else:
                    mismatched += 1
            elif entity is not None and entity["source_chunk_id"] is None and not sources:
                missing += 1
            else:
                mismatched += 1
    return {
        "task_id": TASK_ID,
        "plan_sha256": plan.plan_sha256,
        "repairs": len(plan.repairs),
        "exact_applied": exact,
        "not_applied": missing,
        "mismatched": mismatched,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="read-only namespace or plan audit")
    audit.add_argument("--namespace", default=TARGET_NAMESPACE)
    audit.add_argument("--plan", type=Path)
    prepare = commands.add_parser("prepare", help="zero-call deterministic plan")
    prepare.add_argument("--namespace", default=TARGET_NAMESPACE)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    backup = commands.add_parser("backup", help="private target-row backup")
    backup.add_argument("--plan", type=Path, required=True)
    backup.add_argument("--backup-root", type=Path, required=True)
    canary = commands.add_parser("canary", help="derive one deterministic repair from a full plan")
    canary.add_argument("--plan", type=Path, required=True)
    canary.add_argument("--output-dir", type=Path, required=True)
    for command in ("apply", "rollback"):
        mutation = commands.add_parser(command, help=f"live {command} bound to a plan digest")
        mutation.add_argument("--plan", type=Path, required=True)
        mutation.add_argument("--live", action="store_true")
        if command == "rollback":
            mutation.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


async def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "canary":
        full_plan = load_plan(args.plan)
        canary_plan = derive_canary_plan(full_plan)
        atomic_write_json(args.output_dir / "plan.json", canary_plan.to_dict())
        receipt = {
            "task_id": TASK_ID,
            "parent_plan_sha256": full_plan.plan_sha256,
            "canary_plan_sha256": canary_plan.plan_sha256,
            "canary_entity_id": canary_plan.repairs[0].entity_id,
            "derived_at": _utc_now(),
        }
        atomic_write_json(args.output_dir / "canary-selection-receipt.json", receipt)
        return receipt
    pool = await get_pool()
    if args.command == "audit":
        if args.plan:
            return await audit_plan(pool, load_plan(args.plan))
        return audit_snapshot(await read_snapshot(pool, args.namespace))
    if args.command == "prepare":
        if args.namespace != TARGET_NAMESPACE:
            raise PlanError(f"live LTM-0019 prepare is restricted to {TARGET_NAMESPACE}")
        return await prepare_plan(
            pool,
            args.namespace,
            args.run_id,
            args.state_root,
            expected_namespace_id=TARGET_NAMESPACE_ID,
        )
    plan = load_plan(args.plan)
    if args.command == "backup":
        return await backup_plan(pool, plan, args.backup_root)
    if not args.live:
        raise ApprovalError(f"{args.command} requires --live and an exact plan-digest approval")
    if args.command == "apply":
        result = await apply_plan(pool, plan)
        receipt = {
            "task_id": TASK_ID,
            "plan_sha256": plan.plan_sha256,
            "run_id": plan.run_id,
            "applied_at": _utc_now(),
            **result,
        }
        atomic_write_json(args.plan.parent / "apply-receipt.json", receipt)
        return result
    receipt_path = args.receipt or args.plan.parent / "apply-receipt.json"
    receipt = _private_json(receipt_path)
    result = await rollback_plan(pool, plan, receipt)
    atomic_write_json(
        args.plan.parent / "rollback-receipt.json",
        {
            "task_id": TASK_ID,
            "plan_sha256": plan.plan_sha256,
            "run_id": plan.run_id,
            "rolled_back_at": _utc_now(),
            **result,
        },
    )
    return result


async def _main(argv: Sequence[str] | None = None) -> None:
    try:
        print(json.dumps(await _run_cli(_parse_args(argv)), indent=2, sort_keys=True))
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
