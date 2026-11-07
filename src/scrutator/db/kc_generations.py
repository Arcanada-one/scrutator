"""KB-CONS-01-N: native private-generation selection primitive (synthetic-fixture slice).

What this module is
-------------------
A durable PostgreSQL selection row per authorized corpus scope, updated by a conditional ``UPDATE``
inside one ``asyncpg`` transaction (revision + incarnation + prior-generation CAS), with immutable
scope / generation / member records, one retained read context (lease), generation-membership
eligibility for every read path, and a fail-closed authority seam.

What it is NOT (kept honest on purpose)
---------------------------------------
* There is NO real private-use authority backend here. The default backend refuses with
  ``CURRENT_AUTHORITY_UNAVAILABLE``; tests inject a clearly labelled synthetic backend.
* Destructive GC is not implemented. ``mark_gc_pending`` is the non-destructive first half
  (lock order + lease recheck); ``collect_generation`` always refuses ``GC_DISABLED``.
* The triggers in schema.sql are a fail-closed backstop, not a privilege boundary: any session able
  to ``set_config`` can reach the stage-writer GUC. Role separation is NOT measured here.

Lock order (documented, used by every path; nothing takes a selection lock after a generation lock)
  1. ``kc_selections`` row of the scope  ->  2. ``kc_generations`` row  ->  3. member/source/chunk rows.
Staging (generation-first wrapper) only ever takes generation FOR SHARE -> member FOR UPDATE.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID, uuid4

import asyncpg
import numpy as np

from scrutator.chunker.splitters import compute_doc_id
from scrutator.db.connection import get_pool

KC_BUILD = "kc-generations/1"
NATIVE_PREFIX = "kc-generations/"
MANIFEST_SCHEMA = "KCSyntheticManifest/v1"
SEAL_SCHEMA = "KCSealReport/v1"
LOCK_TIMEOUT_MS = 5000
AUTHORITY_TIMEOUT_S = 2.0
MAX_BIGINT = 2**63 - 1
# Destructive GC is disabled in this slice: there is no code path that deletes generation bytes.
DESTRUCTIVE_GC_ENABLED = False

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MEMBER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}$")
_ROOT_KEYS = {"schema", "generation_id", "purpose", "roots", "members", "required_edges"}
_MEMBER_KEYS = {"member_id", "semantic_revision", "representation_id", "bytes", "sha256"}
_EDGE_KEYS = {"from", "to", "relation"}
_KNOWN_RELATIONS = {"requires"}
_MAX_EDGES = 20000


class KcGenerationError(Exception):
    """Typed refusal. ``code`` is the stable machine-readable outcome."""

    def __init__(self, code: str, message: str = "", **detail: Any) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message
        self.detail = detail


# ── authority seam ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuthorityObservation:
    """A current-authority observation with the backend's own ordering witness."""

    backend: str
    principal_id: str
    scope_id: str
    purpose: str
    revision: str
    witness: str

    def as_json(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "principal_id": self.principal_id,
            "scope_id": self.scope_id,
            "purpose": self.purpose,
            "revision": self.revision,
            "witness": self.witness,
        }


class AuthorityBackend(Protocol):
    name: str

    async def check(
        self, conn: asyncpg.Connection | None, *, principal_id: str, scope_id: str, purpose: str
    ) -> AuthorityObservation: ...


class UnavailableAuthority:
    """Default: no real private-use authority exists, so every check refuses."""

    name = "unavailable"

    async def check(
        self, conn: asyncpg.Connection | None, *, principal_id: str, scope_id: str, purpose: str
    ) -> AuthorityObservation:
        raise KcGenerationError(
            "CURRENT_AUTHORITY_UNAVAILABLE", "no explicit private-use authority backend is configured"
        )


_backend: AuthorityBackend = UnavailableAuthority()


def configure_authority_backend(backend: AuthorityBackend | None) -> None:
    """Install an explicit authority backend (None restores the refusing default)."""
    global _backend  # noqa: PLW0603
    _backend = backend if backend is not None else UnavailableAuthority()


def get_authority_backend() -> AuthorityBackend:
    return _backend


async def _check_authority(
    backend: AuthorityBackend | None,
    conn: asyncpg.Connection | None,
    *,
    principal_id: str,
    scope_id: str,
    purpose: str,
) -> AuthorityObservation:
    backend = backend or _backend
    try:
        obs = await asyncio.wait_for(
            backend.check(conn, principal_id=principal_id, scope_id=scope_id, purpose=purpose),
            AUTHORITY_TIMEOUT_S,
        )
    except KcGenerationError:
        raise
    except Exception as exc:  # outage, timeout, malformed backend: never a silent allow
        raise KcGenerationError(
            "CURRENT_AUTHORITY_UNAVAILABLE", f"authority lookup failed: {type(exc).__name__}"
        ) from exc
    if (
        not isinstance(obs, AuthorityObservation)
        or obs.backend in ("", "unavailable")
        or obs.principal_id != principal_id
        or obs.scope_id != scope_id
        or obs.purpose != purpose
        or not obs.revision
        or not obs.witness
    ):
        raise KcGenerationError("CURRENT_AUTHORITY_UNAVAILABLE", "authority observation lacks an ordering witness")
    return obs


# ── tokens, transactions ────────────────────────────────────────────────────────────────────────


def parse_revision(value: Any) -> int:
    """Decimal-string (or int) BIGINT token; refuses overflow/negatives instead of wrapping."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise KcGenerationError("TOKEN_INVALID", "revision must be a decimal string or integer")
    if isinstance(value, str) and not re.fullmatch(r"0|[1-9][0-9]{0,18}", value):
        raise KcGenerationError("TOKEN_INVALID", "revision must be a canonical decimal string")
    n = int(value)
    if n < 0 or n > MAX_BIGINT:
        raise KcGenerationError("TOKEN_INVALID", "revision outside BIGINT range")
    return n


@dataclass(frozen=True)
class SelectionToken:
    scope_id: str
    incarnation_id: str
    revision: int
    generation_id: str | None
    manifest_sha256: str | None

    def as_json(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "incarnation_id": self.incarnation_id,
            "revision": str(self.revision),  # decimal string at JSON/JS boundaries
            "generation_id": self.generation_id,
            "manifest_sha256": self.manifest_sha256,
        }


def _token(row: Any) -> SelectionToken:
    return SelectionToken(
        scope_id=str(row["scope_id"]),
        incarnation_id=str(row["incarnation_id"]),
        revision=int(row["revision"]),
        generation_id=row["generation_id"],
        manifest_sha256=row["manifest_sha256"],
    )


def _db_error(exc: BaseException) -> KcGenerationError | None:
    if isinstance(exc, (asyncpg.DeadlockDetectedError, asyncpg.LockNotAvailableError)):
        return KcGenerationError("LOCK_FAILURE", type(exc).__name__)
    if isinstance(exc, asyncpg.RaiseError):
        msg = str(exc)
        m = re.match(r"(KC_[A-Z_]+)", msg)
        if m:
            return KcGenerationError(m.group(1).removeprefix("KC_"), msg)
    if isinstance(exc, asyncpg.UniqueViolationError):
        return KcGenerationError("UNIQUE_VIOLATION", str(exc))
    return None


@asynccontextmanager
async def _tx(
    pool: asyncpg.Pool | None = None, *, readonly: bool = False, isolation: str | None = None
) -> AsyncIterator[asyncpg.Connection]:
    """One short transaction. Failures (deadlock, lock timeout, trigger refusal) surface as typed
    ``KcGenerationError``; a failed operation always rolls back and never reports partial success."""
    pool = pool or await get_pool()
    try:
        async with pool.acquire() as conn, conn.transaction(readonly=readonly, isolation=isolation):
            await conn.execute(f"SET LOCAL lock_timeout = {LOCK_TIMEOUT_MS}")
            yield conn
    except KcGenerationError:
        raise
    except Exception as exc:
        typed = _db_error(exc)
        if typed is not None:
            raise typed from exc
        raise


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── manifest ────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ManifestMember:
    member_id: str
    semantic_revision: str
    representation_id: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ParsedManifest:
    generation_id: str
    purpose: str
    roots: tuple[str, ...]
    members: tuple[ManifestMember, ...]
    edges: tuple[tuple[str, str], ...]

    @property
    def member_ids(self) -> frozenset[str]:
        return frozenset(m.member_id for m in self.members)

    def closure(self) -> frozenset[str]:
        """Required closure from the roots over ``requires`` edges (bounded: <= members steps)."""
        adj: dict[str, list[str]] = {}
        for a, b in self.edges:
            adj.setdefault(a, []).append(b)
        seen: set[str] = set()
        stack = list(self.roots)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adj.get(node, ()))
        return frozenset(seen)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise KcGenerationError("MANIFEST_INVALID", "duplicate JSON key")
        out[key] = value
    return out


def parse_manifest(raw: bytes, *, max_members: int, max_manifest_bytes: int, max_total_bytes: int) -> ParsedManifest:
    """Strict parser of the synthetic profile manifest. Unknown versions, duplicate identities,
    conflicting revisions, unknown required relations and limit exhaustion all refuse."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) == 0:
        raise KcGenerationError("MANIFEST_INVALID", "manifest must be non-empty bytes")
    if len(raw) > max_manifest_bytes:
        raise KcGenerationError("LIMIT_EXCEEDED", "manifest exceeds max_manifest_bytes")
    try:
        doc = json.loads(bytes(raw).decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as exc:
        raise KcGenerationError("MANIFEST_INVALID", "manifest is not strict UTF-8 JSON") from exc
    if not isinstance(doc, dict) or set(doc) != _ROOT_KEYS:
        raise KcGenerationError("MANIFEST_INVALID", "unexpected manifest keys")
    if doc["schema"] != MANIFEST_SCHEMA:
        raise KcGenerationError("MANIFEST_INVALID", "unknown manifest schema/version")
    gen, purpose = doc["generation_id"], doc["purpose"]
    if not isinstance(gen, str) or not _ID.match(gen) or not isinstance(purpose, str):
        raise KcGenerationError("MANIFEST_INVALID", "bad generation_id/purpose")
    members_raw, edges_raw, roots_raw = doc["members"], doc["required_edges"], doc["roots"]
    if not all(isinstance(x, list) for x in (members_raw, edges_raw, roots_raw)):
        raise KcGenerationError("MANIFEST_INVALID", "members/required_edges/roots must be arrays")
    if len(members_raw) > max_members or len(edges_raw) > _MAX_EDGES:
        raise KcGenerationError("LIMIT_EXCEEDED", "member or edge count exceeds the profile limit")
    members: list[ManifestMember] = []
    ids: set[str] = set()
    total = 0
    for m in members_raw:
        if not isinstance(m, dict) or set(m) != _MEMBER_KEYS:
            raise KcGenerationError("MANIFEST_INVALID", "unexpected member keys")
        mid, rev, rep, nbytes, digest = (
            m[k] for k in ("member_id", "semantic_revision", "representation_id", "bytes", "sha256")
        )
        if not (isinstance(mid, str) and _MEMBER_ID.match(mid)):
            raise KcGenerationError("MANIFEST_INVALID", "bad member_id")
        if mid in ids:
            raise KcGenerationError("MANIFEST_INVALID", "duplicate member identity / conflicting revision")
        if not (
            isinstance(rev, str)
            and 1 <= len(rev.encode()) <= 128
            and isinstance(rep, str)
            and 1 <= len(rep.encode()) <= 128
        ):
            raise KcGenerationError("MANIFEST_INVALID", "bad revision/representation")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0 or nbytes > 2**31 - 1:
            raise KcGenerationError("MANIFEST_INVALID", "bad byte count")
        if not (isinstance(digest, str) and _HEX64.match(digest)):
            raise KcGenerationError("MANIFEST_INVALID", "bad sha256")
        ids.add(mid)
        total += nbytes
        members.append(ManifestMember(mid, rev, rep, nbytes, digest))
    if total > max_total_bytes:
        raise KcGenerationError("LIMIT_EXCEEDED", "member byte total exceeds max_total_bytes")
    edges: list[tuple[str, str]] = []
    for e in edges_raw:
        if not isinstance(e, dict) or set(e) != _EDGE_KEYS:
            raise KcGenerationError("MANIFEST_INVALID", "unexpected edge keys")
        if e["relation"] not in _KNOWN_RELATIONS:
            raise KcGenerationError("MANIFEST_INVALID", "unknown required relation")
        if e["from"] not in ids or e["to"] not in ids:
            raise KcGenerationError("MANIFEST_INVALID", "edge references an unknown member")
        edges.append((e["from"], e["to"]))
    if not roots_raw or any(r not in ids for r in roots_raw) or len(set(roots_raw)) != len(roots_raw):
        raise KcGenerationError("MANIFEST_INVALID", "roots must be a non-empty unique subset of members")
    return ParsedManifest(gen, purpose, tuple(roots_raw), tuple(members), tuple(edges))


def build_manifest_bytes(
    generation_id: str,
    purpose: str,
    members: list[dict[str, Any]],
    roots: list[str],
    required_edges: list[tuple[str, str]],
) -> bytes:
    """Fixture helper: deterministic manifest bytes for the synthetic profile."""
    doc = {
        "schema": MANIFEST_SCHEMA,
        "generation_id": generation_id,
        "purpose": purpose,
        "roots": roots,
        "members": members,
        "required_edges": [{"from": a, "to": b, "relation": "requires"} for a, b in required_edges],
    }
    return json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# ── scope / candidate ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScopeInfo:
    scope_id: str
    namespace_id: int
    project_id: int
    purpose: str
    profile: str
    max_generations: int
    max_members: int
    max_manifest_bytes: int
    max_total_bytes: int
    max_active_leases: int
    lease_ttl_seconds: int


def _scope(row: Any) -> ScopeInfo:
    return ScopeInfo(
        scope_id=str(row["scope_id"]),
        namespace_id=row["namespace_id"],
        project_id=row["project_id"],
        purpose=row["purpose"],
        profile=row["profile"],
        max_generations=row["max_generations"],
        max_members=row["max_members"],
        max_manifest_bytes=row["max_manifest_bytes"],
        max_total_bytes=row["max_total_bytes"],
        max_active_leases=row["max_active_leases"],
        lease_ttl_seconds=row["lease_ttl_seconds"],
    )


async def create_scope(
    pool: asyncpg.Pool | None,
    namespace_id: int,
    project_id: int,
    purpose: str,
    profile: str = "kc-synthetic/v1",
    **limits: int,
) -> str:
    """Administrative, owner-run registration of one scope + its pre-created selection row (revision 0,
    no selection). Never called on a read path; never provisions a namespace or project."""
    allowed = {
        "max_generations",
        "max_members",
        "max_manifest_bytes",
        "max_total_bytes",
        "max_active_leases",
        "lease_ttl_seconds",
    }
    if set(limits) - allowed:
        raise KcGenerationError("SCOPE_INVALID", "unknown limit")
    cols = ["namespace_id", "project_id", "purpose", "profile", *limits]
    vals = [namespace_id, project_id, purpose, profile, *limits.values()]
    placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
    try:
        async with _tx(pool) as conn:
            scope_id = await conn.fetchval(
                f"INSERT INTO kc_corpus_scopes ({', '.join(cols)}) VALUES ({placeholders}) RETURNING scope_id", *vals
            )
            await conn.execute("INSERT INTO kc_selections (scope_id) VALUES ($1)", scope_id)
    except (asyncpg.ForeignKeyViolationError, asyncpg.CheckViolationError, asyncpg.UniqueViolationError) as exc:
        raise KcGenerationError("SCOPE_INVALID", type(exc).__name__) from exc
    except KcGenerationError as exc:
        if exc.code == "UNIQUE_VIOLATION":
            raise KcGenerationError("SCOPE_INVALID", "scope already registered") from exc
        raise
    return str(scope_id)


async def get_scope(pool: asyncpg.Pool | None, scope_id: str) -> ScopeInfo | None:
    pool = pool or await get_pool()
    row = await pool.fetchrow("SELECT * FROM kc_corpus_scopes WHERE scope_id = $1::uuid", scope_id)
    return _scope(row) if row else None


async def find_scope(pool: asyncpg.Pool | None, namespace_id: int, project_id: int, purpose: str) -> ScopeInfo | None:
    """Resolve a registered scope by its server-side key. Never creates one."""
    pool = pool or await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM kc_corpus_scopes WHERE namespace_id=$1 AND project_id=$2 AND purpose=$3",
        namespace_id,
        project_id,
        purpose,
    )
    return _scope(row) if row else None


async def get_selection(pool: asyncpg.Pool | None, scope_id: str) -> SelectionToken:
    pool = pool or await get_pool()
    row = await pool.fetchrow("SELECT * FROM kc_selections WHERE scope_id = $1::uuid", scope_id)
    if row is None:
        raise KcGenerationError("SCOPE_NOT_FOUND")
    return _token(row)


@dataclass(frozen=True)
class CandidateResult:
    scope_id: str
    generation_id: str
    manifest_sha256: str
    created: bool


async def create_candidate(
    pool: asyncpg.Pool | None, scope_id: str, manifest: bytes, *, admission_ref: str, profile_ref: str
) -> CandidateResult:
    """Insert the immutable manifest and its expected member rows once.

    An existing upstream generation key is idempotent only when the complete manifest bytes and the
    admission/profile binding are identical; otherwise ``GENERATION_ID_CONFLICT`` is raised BEFORE any
    member, manifest or selection row is modified."""
    async with _tx(pool) as conn:
        srow = await conn.fetchrow("SELECT * FROM kc_corpus_scopes WHERE scope_id = $1::uuid FOR SHARE", scope_id)
        if srow is None:
            raise KcGenerationError("SCOPE_NOT_FOUND")
        scope = _scope(srow)
        parsed = parse_manifest(
            manifest,
            max_members=scope.max_members,
            max_manifest_bytes=scope.max_manifest_bytes,
            max_total_bytes=scope.max_total_bytes,
        )
        if parsed.purpose != scope.purpose:
            raise KcGenerationError("MANIFEST_INVALID", "manifest purpose does not match the registered scope")
        digest = sha256_hex(manifest)
        existing = await conn.fetchrow(
            "SELECT manifest, admission_ref, profile_ref, manifest_sha256 FROM kc_generations "
            "WHERE scope_id=$1::uuid AND generation_id=$2 FOR SHARE",
            scope_id,
            parsed.generation_id,
        )
        if existing is None:
            live = await conn.fetchrow(
                "SELECT count(*) AS n, COALESCE(sum(m.expected_bytes), 0)::bigint AS b "
                "FROM kc_generations g LEFT JOIN kc_generation_members m USING (scope_id, generation_id) "
                "WHERE g.scope_id=$1::uuid AND g.state <> 'collected'",
                scope_id,
            )
            gens = await conn.fetchval(
                "SELECT count(*) FROM kc_generations WHERE scope_id=$1::uuid AND state <> 'collected'", scope_id
            )
            incoming = sum(m.bytes for m in parsed.members)
            if gens >= scope.max_generations or int(live["b"]) + incoming > scope.max_total_bytes:
                raise KcGenerationError(
                    "LIMIT_EXCEEDED", "retention budget exhausted (storage never evicts selected data)"
                )
            inserted = await conn.fetchval(
                "INSERT INTO kc_generations (scope_id, generation_id, manifest, manifest_bytes, manifest_sha256, "
                "admission_ref, profile_ref) VALUES ($1::uuid,$2,$3,$4,$5,$6,$7) "
                "ON CONFLICT (scope_id, generation_id) DO NOTHING RETURNING generation_id",
                scope_id,
                parsed.generation_id,
                bytes(manifest),
                len(manifest),
                digest,
                admission_ref,
                profile_ref,
            )
            if inserted is None:  # a concurrent identical creator won; re-read below
                existing = await conn.fetchrow(
                    "SELECT manifest, admission_ref, profile_ref, manifest_sha256 FROM kc_generations "
                    "WHERE scope_id=$1::uuid AND generation_id=$2 FOR SHARE",
                    scope_id,
                    parsed.generation_id,
                )
            else:
                ns_name = await conn.fetchval("SELECT name FROM namespaces WHERE id=$1", scope.namespace_id)
                for m in parsed.members:
                    member_uuid = uuid4()
                    path = f"{NATIVE_PREFIX}{scope_id}/{member_uuid}"
                    await conn.execute(
                        "INSERT INTO kc_generation_members (scope_id, generation_id, member_id, semantic_revision, "
                        "representation_id, expected_bytes, expected_sha256, member_uuid, namespace_id, project_id, "
                        "source_path, doc_id) VALUES ($1::uuid,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)",
                        scope_id,
                        parsed.generation_id,
                        m.member_id,
                        m.semantic_revision,
                        m.representation_id,
                        m.bytes,
                        m.sha256,
                        member_uuid,
                        scope.namespace_id,
                        scope.project_id,
                        path,
                        compute_doc_id(ns_name, path),
                    )
                return CandidateResult(scope_id, parsed.generation_id, digest, True)
        if (
            bytes(existing["manifest"]) != bytes(manifest)
            or existing["admission_ref"] != admission_ref
            or existing["profile_ref"] != profile_ref
        ):
            raise KcGenerationError("GENERATION_ID_CONFLICT", "generation identity exists with different bytes/binding")
        return CandidateResult(scope_id, parsed.generation_id, digest, False)


# ── staging (immutable, retry-safe) ─────────────────────────────────────────────────────────────

_CHUNK_INSERT_SQL = """
    INSERT INTO chunks (id, namespace_id, project_id, source_path, source_type, chunk_index, parent_id,
                        content, content_hash, embedding_dense, metadata, token_count, indexed_at)
    VALUES ($1::uuid, $2, $3, $4, $5, $6, $7::uuid, $8, $9, $10, $11::jsonb, $12, NOW())
"""


def chunk_set_digest(chunks: Iterable[dict[str, Any]]) -> str:
    """Digest of the ordered (chunk_index, sha256(content)) set — ids are intentionally excluded so an
    identical retry with fresh random chunk ids is recognised as identical."""
    items = [
        [c["chunk_index"], sha256_hex(c["content"].encode("utf-8"))]
        for c in sorted(chunks, key=lambda c: c["chunk_index"])
    ]
    return sha256_hex(json.dumps(items, separators=(",", ":")).encode())


@dataclass(frozen=True)
class MemberBinding:
    member_uuid: str
    namespace: str
    namespace_id: int
    project_id: int
    source_path: str
    doc_id: str
    expected_bytes: int
    expected_sha256: str


async def get_member_binding(
    pool: asyncpg.Pool | None, scope_id: str, generation_id: str, member_id: str
) -> MemberBinding:
    pool = pool or await get_pool()
    row = await pool.fetchrow(
        "SELECT m.*, n.name AS ns_name FROM kc_generation_members m JOIN namespaces n ON n.id = m.namespace_id "
        "WHERE m.scope_id=$1::uuid AND m.generation_id=$2 AND m.member_id=$3",
        scope_id,
        generation_id,
        member_id,
    )
    if row is None:
        raise KcGenerationError("MEMBER_NOT_FOUND")
    return MemberBinding(
        str(row["member_uuid"]),
        row["ns_name"],
        row["namespace_id"],
        row["project_id"],
        row["source_path"],
        row["doc_id"],
        row["expected_bytes"],
        row["expected_sha256"],
    )


def prepare_member_chunks(namespace: str, native_path: str, text: str) -> list[dict[str, Any]]:
    """Chunk a member with the EXISTING chunker/indexer helpers, stamped for its native path."""
    from scrutator.search.indexer import _chunk_dicts, chunk_document

    result = chunk_document(text, native_path, max_tokens=64, overlap_tokens=8)
    return _chunk_dicts(result, namespace, native_path, text)


async def stage_member(
    pool: asyncpg.Pool | None,
    scope_id: str,
    generation_id: str,
    member_id: str,
    *,
    body: bytes,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    sparse_weights: list[dict[str, float]],
    producer: str,
) -> str:
    """Populate one expected member of a STAGING generation. Returns ``created`` or ``unchanged``.

    Embeddings are computed by the caller BEFORE the transaction (no model call is held inside it).
    A populated member is never replaced: an identical retry is a no-op (no delete/reinsert, no chunk
    id churn) and a different body/chunk set is ``MEMBER_CONFLICT``. Bodies that are not strict UTF-8
    or contain U+0000 are refused before any write (never stripped or normalised)."""
    from scrutator.db.repository import _validate_atomic_replacement
    from scrutator.search.indexer import compute_doc_content_hash

    if not chunks:
        raise KcGenerationError("MEMBER_INVALID", "a member needs at least one chunk")
    try:
        text = bytes(body).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KcGenerationError("MEMBER_INVALID", "body is not strict UTF-8") from exc
    if "\x00" in text:
        raise KcGenerationError("NUL_IN_BODY", "PostgreSQL TEXT cannot store U+0000; refused, not stripped")
    try:
        _validate_atomic_replacement(chunks, embeddings, sparse_weights)
    except ValueError as exc:
        raise KcGenerationError("MEMBER_INVALID", str(exc)) from exc
    indexes = sorted(c["chunk_index"] for c in chunks)
    if indexes != list(range(len(chunks))):
        raise KcGenerationError("MEMBER_INVALID", "chunk indexes must be contiguous from 0")
    for c in chunks:
        if c["content_hash"] != sha256_hex(c["content"].encode("utf-8")) or "\x00" in c["content"]:
            raise KcGenerationError("MEMBER_INVALID", "chunk stamp does not match its bytes")
    digest = chunk_set_digest(chunks)

    async with _tx(pool) as conn:
        gen = await conn.fetchrow(
            "SELECT state FROM kc_generations WHERE scope_id=$1::uuid AND generation_id=$2 FOR SHARE",
            scope_id,
            generation_id,
        )
        if gen is None:
            raise KcGenerationError("GENERATION_NOT_FOUND")
        m = await conn.fetchrow(
            "SELECT * FROM kc_generation_members WHERE scope_id=$1::uuid AND generation_id=$2 AND member_id=$3 FOR UPDATE",  # noqa: E501
            scope_id,
            generation_id,
            member_id,
        )
        if m is None:
            raise KcGenerationError("MEMBER_NOT_FOUND")
        body_ok = len(body) == m["expected_bytes"] and sha256_hex(bytes(body)) == m["expected_sha256"]
        if not body_ok:
            raise KcGenerationError("MEMBER_MISMATCH", "body bytes/digest differ from the manifest member")
        if m["chunk_set_sha256"] is not None:
            if m["chunk_set_sha256"] == digest and m["producer"] == producer:
                return "unchanged"
            raise KcGenerationError("MEMBER_CONFLICT", "a populated member cannot be replaced with different content")
        if gen["state"] != "staging":
            raise KcGenerationError("GENERATION_NOT_STAGING", gen["state"])
        if any(c["source_path"] != m["source_path"] for c in chunks):
            raise KcGenerationError("MEMBER_INVALID", "chunks must carry the server-controlled native path")
        await conn.execute("SELECT set_config('scrutator.kc_stage_writer', $1, true)", f"{scope_id}/{generation_id}")
        await conn.execute(
            "INSERT INTO evidence_documents (namespace_id, source_path, doc_id, content_hash, raw_content) "
            "VALUES ($1,$2,$3,$4,$5)",
            m["namespace_id"],
            m["source_path"],
            m["doc_id"],
            compute_doc_content_hash(text),
            text,
        )
        for chunk, emb, weights in zip(chunks, embeddings, sparse_weights, strict=True):
            await conn.execute(
                _CHUNK_INSERT_SQL,
                chunk["id"],
                m["namespace_id"],
                m["project_id"],
                m["source_path"],
                chunk["source_type"],
                chunk["chunk_index"],
                chunk.get("parent_id"),
                chunk["content"],
                chunk["content_hash"],
                np.asarray(emb, dtype=np.float32),
                json.dumps(chunk.get("metadata", {})),
                chunk.get("token_count", 0),
            )
            await conn.execute(
                "INSERT INTO sparse_vectors (chunk_id, token_weights) VALUES ($1::uuid, $2::jsonb)",
                chunk["id"],
                json.dumps(weights),
            )
        await conn.execute("SELECT set_config('scrutator.kc_stage_writer', '', true)")
        await conn.execute(
            "UPDATE kc_generation_members SET chunk_set_sha256=$4, chunk_count=$5, producer=$6 "
            "WHERE scope_id=$1::uuid AND generation_id=$2 AND member_id=$3",
            scope_id,
            generation_id,
            member_id,
            digest,
            len(chunks),
            producer,
        )
        return "created"


# ── seal ────────────────────────────────────────────────────────────────────────────────────────


async def seal_generation(
    pool: asyncpg.Pool | None, scope_id: str, generation_id: str, *, oracle_closure: Iterable[str]
) -> dict[str, Any]:
    """Seal only after exact expected-member equality, required-edge closure equal to the independent
    oracle, and a FRESH full-byte readback (UTF-8 byte lengths + recomputed hashes, per-chunk recomputed
    hashes and the bound chunk-set digest). A refusal leaves the generation in ``staging`` unchanged."""
    oracle = frozenset(oracle_closure)
    async with _tx(pool) as conn:
        # order: selection (share) -> generation (update) -> members/chunks
        if await conn.fetchval("SELECT 1 FROM kc_selections WHERE scope_id=$1::uuid FOR SHARE", scope_id) is None:
            raise KcGenerationError("SCOPE_NOT_FOUND")
        gen = await conn.fetchrow(
            "SELECT * FROM kc_generations WHERE scope_id=$1::uuid AND generation_id=$2 FOR UPDATE",
            scope_id,
            generation_id,
        )
        if gen is None:
            raise KcGenerationError("GENERATION_NOT_FOUND")
        if gen["state"] == "sealed":
            return json.loads(gen["seal_report"]) if isinstance(gen["seal_report"], str) else dict(gen["seal_report"])
        if gen["state"] != "staging":
            raise KcGenerationError("GENERATION_NOT_STAGING", gen["state"])
        scope = _scope(await conn.fetchrow("SELECT * FROM kc_corpus_scopes WHERE scope_id=$1::uuid", scope_id))
        parsed = parse_manifest(
            bytes(gen["manifest"]),
            max_members=scope.max_members,
            max_manifest_bytes=scope.max_manifest_bytes,
            max_total_bytes=scope.max_total_bytes,
        )
        rows = await conn.fetch(
            "SELECT * FROM kc_generation_members WHERE scope_id=$1::uuid AND generation_id=$2 ORDER BY member_id",
            scope_id,
            generation_id,
        )
        by_id = {r["member_id"]: r for r in rows}
        if set(by_id) != set(parsed.member_ids):
            raise KcGenerationError("SEAL_REFUSED", "stored member set differs from the manifest", reason="MEMBER_SET")
        for pm in parsed.members:
            r = by_id[pm.member_id]
            if (r["semantic_revision"], r["representation_id"], r["expected_bytes"], r["expected_sha256"]) != (
                pm.semantic_revision,
                pm.representation_id,
                pm.bytes,
                pm.sha256,
            ):
                raise KcGenerationError("SEAL_REFUSED", "member row differs from manifest", reason="MEMBER_FIELDS")
        closure = parsed.closure()
        if closure != parsed.member_ids:
            raise KcGenerationError("SEAL_REFUSED", "members are not exactly the required closure", reason="CLOSURE")
        if closure != oracle:
            raise KcGenerationError("SEAL_REFUSED", "closure differs from the independent oracle", reason="ORACLE")
        report_members: list[dict[str, Any]] = []
        for member_id in sorted(by_id):
            r = by_id[member_id]
            if r["chunk_set_sha256"] is None:
                raise KcGenerationError("SEAL_REFUSED", f"member {member_id} is not populated", reason="MISSING_MEMBER")
            doc = await conn.fetchrow(
                "SELECT doc_id, content_hash, raw_content FROM evidence_documents WHERE namespace_id=$1 AND source_path=$2",  # noqa: E501
                r["namespace_id"],
                r["source_path"],
            )
            if doc is None:
                raise KcGenerationError("SEAL_REFUSED", f"member {member_id} has no stored body", reason="MISSING_BODY")
            raw = doc["raw_content"].encode("utf-8")
            if len(raw) != r["expected_bytes"] or sha256_hex(raw) != r["expected_sha256"]:
                raise KcGenerationError("SEAL_REFUSED", f"member {member_id} body bytes/digest mismatch", reason="BODY")
            if doc["content_hash"] != "sha256:" + r["expected_sha256"] or doc["doc_id"] != r["doc_id"]:
                raise KcGenerationError("SEAL_REFUSED", f"member {member_id} stamp mismatch", reason="STAMP")
            crow = await conn.fetch(
                "SELECT c.chunk_index, c.content, c.content_hash, c.namespace_id, c.project_id, "
                "(c.embedding_dense IS NOT NULL) AS has_dense, (sv.chunk_id IS NOT NULL) AS has_sparse "
                "FROM chunks c LEFT JOIN sparse_vectors sv ON sv.chunk_id = c.id "
                "WHERE c.namespace_id=$1 AND c.source_path=$2 ORDER BY c.chunk_index",
                r["namespace_id"],
                r["source_path"],
            )
            if [c["chunk_index"] for c in crow] != list(range(r["chunk_count"])):
                raise KcGenerationError("SEAL_REFUSED", f"member {member_id} chunk set incomplete", reason="CHUNK_SET")
            for c in crow:
                if (
                    c["namespace_id"] != r["namespace_id"]
                    or c["project_id"] != r["project_id"]
                    or not c["has_dense"]
                    or not c["has_sparse"]
                    or c["content_hash"] != sha256_hex(c["content"].encode("utf-8"))
                ):
                    raise KcGenerationError("SEAL_REFUSED", f"member {member_id} chunk mismatch", reason="CHUNK_BYTES")
            if (
                chunk_set_digest([{"chunk_index": c["chunk_index"], "content": c["content"]} for c in crow])
                != r["chunk_set_sha256"]
            ):
                raise KcGenerationError(
                    "SEAL_REFUSED", f"member {member_id} chunk-set digest mismatch", reason="CHUNK_SET_DIGEST"
                )
            report_members.append(
                {
                    "member_id": member_id,
                    "bytes": len(raw),
                    "sha256": r["expected_sha256"],
                    "chunk_count": r["chunk_count"],
                    "chunk_set_sha256": r["chunk_set_sha256"],
                    "producer": r["producer"],
                }
            )
        report = {
            "schema": SEAL_SCHEMA,
            "build": KC_BUILD,
            "scope_id": scope_id,
            "generation_id": generation_id,
            "manifest_sha256": gen["manifest_sha256"],
            "closure": sorted(closure),
            "total_bytes": sum(m["bytes"] for m in report_members),
            "members": report_members,
        }
        await conn.execute(
            "UPDATE kc_generations SET state='sealed', seal_report=$3::jsonb, sealed_at=NOW() "
            "WHERE scope_id=$1::uuid AND generation_id=$2",
            scope_id,
            generation_id,
            json.dumps(report),
        )
        return report


# ── promotion (CAS) ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PromotionResult:
    operation_id: str
    outcome: str  # "committed" (this call) | "replayed" (historical result of an earlier commit)
    committed: SelectionToken  # the token this operation produced
    current: SelectionToken  # separate observation of the selection now
    is_current: bool  # False when a later promotion has superseded this operation
    authority: dict[str, str]


def _request_digest(candidate: str, manifest_sha: str, inc: str, rev: int, prior: str | None) -> str:
    return sha256_hex(json.dumps([candidate, manifest_sha, inc, rev, prior], separators=(",", ":")).encode())


async def promote(
    pool: asyncpg.Pool | None,
    scope_id: str,
    *,
    operation_id: str,
    candidate_generation_id: str,
    expected_incarnation: str,
    expected_revision: Any,
    expected_generation: str | None,
    principal_id: str,
    backend: AuthorityBackend | None = None,
) -> PromotionResult:
    """Short transaction: selection row lock -> operation lookup -> sealed-candidate lock -> current
    authority -> conditional UPDATE (incarnation + revision + prior generation) -> operation row.
    Zero rows from the UPDATE is ``SELECTION_CONFLICT`` — never a silent last-writer retry."""
    if not isinstance(operation_id, str) or not _ID.match(operation_id):
        raise KcGenerationError("TOKEN_INVALID", "operation_id")
    exp_rev = parse_revision(expected_revision)
    try:
        UUID(expected_incarnation)
    except (ValueError, AttributeError, TypeError) as exc:
        raise KcGenerationError("TOKEN_INVALID", "incarnation") from exc
    async with _tx(pool) as conn:
        sel = await conn.fetchrow("SELECT * FROM kc_selections WHERE scope_id=$1::uuid FOR UPDATE", scope_id)
        if sel is None:
            raise KcGenerationError("SCOPE_NOT_FOUND")
        gen = await conn.fetchrow(
            "SELECT * FROM kc_generations WHERE scope_id=$1::uuid AND generation_id=$2 FOR SHARE",
            scope_id,
            candidate_generation_id,
        )
        if gen is None:
            raise KcGenerationError("GENERATION_NOT_FOUND")
        request_sha = _request_digest(
            candidate_generation_id, gen["manifest_sha256"], expected_incarnation, exp_rev, expected_generation
        )
        op = await conn.fetchrow(
            "SELECT * FROM kc_selection_operations WHERE scope_id=$1::uuid AND operation_id=$2", scope_id, operation_id
        )
        if op is not None:
            if op["request_sha256"] != request_sha:
                raise KcGenerationError("OPERATION_CONFLICT", "operation id reused with different arguments")
            current = _token(sel)
            committed = SelectionToken(
                scope_id,
                str(op["committed_incarnation"]),
                int(op["committed_revision"]),
                op["candidate_generation_id"],
                op["candidate_manifest_sha256"],
            )
            authority = op["authority_ref"]
            return PromotionResult(
                operation_id,
                "replayed",
                committed,
                current,
                current.incarnation_id == committed.incarnation_id and current.revision == committed.revision,
                json.loads(authority) if isinstance(authority, str) else dict(authority),
            )
        if gen["state"] != "sealed" or gen["seal_report"] is None:
            raise KcGenerationError("GENERATION_NOT_SERVABLE", f"candidate state is {gen['state']}")
        report = json.loads(gen["seal_report"]) if isinstance(gen["seal_report"], str) else dict(gen["seal_report"])
        if report.get("manifest_sha256") != gen["manifest_sha256"] or report.get("build") != KC_BUILD:
            raise KcGenerationError("GENERATION_NOT_SERVABLE", "seal report is not bound to this manifest/build")
        obs = await _check_authority(backend, conn, principal_id=principal_id, scope_id=scope_id, purpose="select")
        row = await conn.fetchrow(
            "UPDATE kc_selections SET generation_id=$4, manifest_sha256=$5, revision=revision+1, updated_at=NOW() "
            "WHERE scope_id=$1::uuid AND incarnation_id=$2::uuid AND revision=$3 "
            "AND generation_id IS NOT DISTINCT FROM $6 "
            "RETURNING scope_id, incarnation_id, revision, generation_id, manifest_sha256",
            scope_id,
            expected_incarnation,
            exp_rev,
            candidate_generation_id,
            gen["manifest_sha256"],
            expected_generation,
        )
        if row is None:
            raise KcGenerationError(
                "SELECTION_CONFLICT", "expected selection token is stale", current=_token(sel).as_json()
            )
        committed = _token(row)
        await conn.execute(
            "INSERT INTO kc_selection_operations (scope_id, operation_id, request_sha256, candidate_generation_id, "
            "candidate_manifest_sha256, expected_incarnation, expected_revision, expected_generation_id, "
            "committed_incarnation, committed_revision, principal_id, authority_ref) "
            "VALUES ($1::uuid,$2,$3,$4,$5,$6::uuid,$7,$8,$9::uuid,$10,$11,$12::jsonb)",
            scope_id,
            operation_id,
            request_sha,
            candidate_generation_id,
            gen["manifest_sha256"],
            expected_incarnation,
            exp_rev,
            expected_generation,
            committed.incarnation_id,
            committed.revision,
            principal_id,
            json.dumps(obs.as_json()),
        )
        return PromotionResult(operation_id, "committed", committed, committed, True, obs.as_json())


# ── retained read context ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerationEligibility:
    """The only thing the SQL layer needs: captured scope/generation + server-side namespace/project."""

    scope_id: str
    generation_id: str
    namespace_id: int
    project_id: int


@dataclass(frozen=True)
class GenerationReadContext:
    lease_id: str
    operation_id: str
    principal_id: str
    scope_id: str
    namespace_id: int
    project_id: int
    generation_id: str
    manifest_sha256: str
    selection_incarnation: str
    selection_revision: int
    authority: dict[str, str]

    @property
    def eligibility(self) -> GenerationEligibility:
        return GenerationEligibility(self.scope_id, self.generation_id, self.namespace_id, self.project_id)


def _ctx_from_lease(row: Any, scope: Any) -> GenerationReadContext:
    auth = row["authority_ref"]
    return GenerationReadContext(
        lease_id=str(row["lease_id"]),
        operation_id=row["operation_id"],
        principal_id=row["principal_id"],
        scope_id=str(row["scope_id"]),
        namespace_id=scope["namespace_id"],
        project_id=scope["project_id"],
        generation_id=row["generation_id"],
        manifest_sha256=row["manifest_sha256"],
        selection_incarnation=str(row["selection_incarnation"]),
        selection_revision=int(row["selection_revision"]),
        authority=json.loads(auth) if isinstance(auth, str) else dict(auth),
    )


async def begin_read(
    pool: asyncpg.Pool | None,
    scope_id: str,
    *,
    principal_id: str,
    operation_id: str,
    backend: AuthorityBackend | None = None,
) -> GenerationReadContext:
    """Capture ONE selection and install its retention lease atomically with respect to promotion and GC
    (selection FOR SHARE blocks the promoter's FOR UPDATE; generation FOR SHARE blocks GC's FOR UPDATE)."""
    if not isinstance(operation_id, str) or not _ID.match(operation_id):
        raise KcGenerationError("TOKEN_INVALID", "operation_id")
    async with _tx(pool) as conn:
        scope = await conn.fetchrow("SELECT * FROM kc_corpus_scopes WHERE scope_id=$1::uuid", scope_id)
        if scope is None:
            raise KcGenerationError("SCOPE_NOT_FOUND")
        obs = await _check_authority(backend, conn, principal_id=principal_id, scope_id=scope_id, purpose="read")
        sel = await conn.fetchrow("SELECT * FROM kc_selections WHERE scope_id=$1::uuid FOR SHARE", scope_id)
        if sel["generation_id"] is None:
            raise KcGenerationError("NO_SELECTION", "the scope has no selected generation")
        gen = await conn.fetchrow(
            "SELECT state FROM kc_generations WHERE scope_id=$1::uuid AND generation_id=$2 FOR SHARE",
            scope_id,
            sel["generation_id"],
        )
        if gen is None or gen["state"] != "sealed":
            raise KcGenerationError("GENERATION_NOT_SERVABLE", "selected generation is not sealed")
        existing = await conn.fetchrow(
            "SELECT *, (clock_timestamp() >= expires_at) AS expired FROM kc_read_leases "
            "WHERE scope_id=$1::uuid AND operation_id=$2 AND principal_id=$3 FOR UPDATE",
            scope_id,
            operation_id,
            principal_id,
        )
        if existing is not None:
            if existing["status"] != "active" or existing["expired"]:
                raise KcGenerationError("LEASE_EXPIRED", "operation id already used; start a new operation")
            return _ctx_from_lease(existing, scope)
        active = await conn.fetchval(
            "SELECT count(*) FROM kc_read_leases WHERE scope_id=$1::uuid AND status='active' AND expires_at > clock_timestamp()",  # noqa: E501
            scope_id,
        )
        if active >= scope["max_active_leases"]:
            raise KcGenerationError("LIMIT_EXCEEDED", "active lease limit reached")
        lease = await conn.fetchrow(
            "INSERT INTO kc_read_leases (scope_id, operation_id, principal_id, generation_id, manifest_sha256, "
            "selection_incarnation, selection_revision, authority_ref, expires_at) "
            "VALUES ($1::uuid,$2,$3,$4,$5,$6,$7,$8::jsonb, clock_timestamp() + make_interval(secs => $9)) RETURNING *",
            scope_id,
            operation_id,
            principal_id,
            sel["generation_id"],
            sel["manifest_sha256"],
            sel["incarnation_id"],
            sel["revision"],
            json.dumps(obs.as_json()),
            float(scope["lease_ttl_seconds"]),
        )
        return _ctx_from_lease(lease, scope)


async def load_context(pool: asyncpg.Pool | None, lease_id: str, *, principal_id: str) -> GenerationReadContext:
    """Rebuild the server-side context from a lease id. The id is NOT a bearer capability: the lease's own
    principal must equal the authenticated caller, and use still re-checks lease state + authority."""
    pool = pool or await get_pool()
    try:
        UUID(lease_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise KcGenerationError("LEASE_NOT_FOUND") from exc
    row = await pool.fetchrow("SELECT * FROM kc_read_leases WHERE lease_id=$1::uuid", lease_id)
    if row is None or row["principal_id"] != principal_id:
        raise KcGenerationError("LEASE_NOT_FOUND")  # identical for unknown and foreign (no existence oracle)
    scope = await pool.fetchrow("SELECT * FROM kc_corpus_scopes WHERE scope_id=$1", row["scope_id"])
    return _ctx_from_lease(row, scope)


async def authorize_use(
    pool: asyncpg.Pool | None,
    ctx: GenerationReadContext,
    *,
    principal_id: str,
    backend: AuthorityBackend | None = None,
    purpose: str = "use",
) -> AuthorityObservation:
    """Re-check lease state, captured-generation applicability AND current authority. Does not change the
    captured generation. Call at operation start, before any external model sees content, and at the
    response/use commit point."""
    if principal_id != ctx.principal_id:
        raise KcGenerationError("LEASE_NOT_FOUND")
    async with _tx(pool, readonly=True) as conn:
        row = await conn.fetchrow(
            "SELECT l.status, l.principal_id, l.scope_id, l.generation_id, l.manifest_sha256, g.state, "
            "(clock_timestamp() >= l.expires_at) AS expired "
            "FROM kc_read_leases l JOIN kc_generations g USING (scope_id, generation_id) "
            "WHERE l.lease_id=$1::uuid",
            ctx.lease_id,
        )
        if (
            row is None
            or row["principal_id"] != principal_id
            or str(row["scope_id"]) != ctx.scope_id
            or row["generation_id"] != ctx.generation_id
            or row["manifest_sha256"] != ctx.manifest_sha256
            or row["status"] != "active"
        ):
            raise KcGenerationError("LEASE_NOT_FOUND")
        if row["expired"]:
            raise KcGenerationError("LEASE_EXPIRED", "lease expired by database time")
        if row["state"] != "sealed":
            raise KcGenerationError("GENERATION_NOT_SERVABLE", row["state"])
    return await _check_authority(backend, None, principal_id=principal_id, scope_id=ctx.scope_id, purpose=purpose)


async def renew_lease(
    pool: asyncpg.Pool | None, ctx: GenerationReadContext, *, principal_id: str, backend: AuthorityBackend | None = None
) -> None:
    """One atomic step: selection(share) -> generation(share) -> lease(update); rechecks lease validity,
    sealed/not-gc_pending state, caller binding and current authority against a FRESH clock_timestamp()."""
    if principal_id != ctx.principal_id:
        raise KcGenerationError("LEASE_NOT_FOUND")
    async with _tx(pool) as conn:
        await conn.fetchval("SELECT 1 FROM kc_selections WHERE scope_id=$1::uuid FOR SHARE", ctx.scope_id)
        gen = await conn.fetchrow(
            "SELECT state FROM kc_generations WHERE scope_id=$1::uuid AND generation_id=$2 FOR SHARE",
            ctx.scope_id,
            ctx.generation_id,
        )
        lease = await conn.fetchrow(
            "SELECT l.*, (clock_timestamp() >= l.expires_at) AS expired, s.lease_ttl_seconds "
            "FROM kc_read_leases l JOIN kc_corpus_scopes s USING (scope_id) WHERE l.lease_id=$1::uuid FOR UPDATE OF l",
            ctx.lease_id,
        )
        if lease is None or lease["principal_id"] != principal_id or lease["status"] != "active":
            raise KcGenerationError("LEASE_NOT_FOUND")
        if lease["expired"]:
            raise KcGenerationError("LEASE_EXPIRED", "an expired lease cannot be revived")
        if gen is None or gen["state"] != "sealed":
            raise KcGenerationError("GENERATION_NOT_SERVABLE", gen["state"] if gen else "missing")
        await _check_authority(backend, conn, principal_id=principal_id, scope_id=ctx.scope_id, purpose="read")
        await conn.execute(
            "UPDATE kc_read_leases SET expires_at = clock_timestamp() + make_interval(secs => $2) WHERE lease_id=$1::uuid",  # noqa: E501
            ctx.lease_id,
            float(lease["lease_ttl_seconds"]),
        )


async def release_lease(pool: asyncpg.Pool | None, ctx: GenerationReadContext, *, principal_id: str) -> None:
    if principal_id != ctx.principal_id:
        raise KcGenerationError("LEASE_NOT_FOUND")
    async with _tx(pool) as conn:
        await conn.execute(
            "UPDATE kc_read_leases SET status='released' WHERE lease_id=$1::uuid AND principal_id=$2",
            ctx.lease_id,
            principal_id,
        )


# ── retention / GC (non-destructive half only) ──────────────────────────────────────────────────


async def mark_gc_pending(pool: asyncpg.Pool | None, scope_id: str, generation_id: str) -> str:
    """Lock selection(share) -> generation(update); recheck unselected + unleased; mark ``gc_pending`` (new
    leases are refused from then on). Deletes NOTHING."""
    async with _tx(pool) as conn:
        sel = await conn.fetchrow("SELECT * FROM kc_selections WHERE scope_id=$1::uuid FOR SHARE", scope_id)
        if sel is None:
            raise KcGenerationError("SCOPE_NOT_FOUND")
        gen = await conn.fetchrow(
            "SELECT state FROM kc_generations WHERE scope_id=$1::uuid AND generation_id=$2 FOR UPDATE",
            scope_id,
            generation_id,
        )
        if gen is None:
            raise KcGenerationError("GENERATION_NOT_FOUND")
        if gen["state"] == "gc_pending":
            return "gc_pending"
        if gen["state"] != "sealed":
            raise KcGenerationError("GC_REFUSED", f"state {gen['state']}")
        if sel["generation_id"] == generation_id:
            raise KcGenerationError("GC_REFUSED", "generation is selected")
        leased = await conn.fetchval(
            "SELECT count(*) FROM kc_read_leases WHERE scope_id=$1::uuid AND generation_id=$2 "
            "AND status='active' AND expires_at > clock_timestamp()",
            scope_id,
            generation_id,
        )
        if leased:
            raise KcGenerationError("GC_REFUSED", "generation has valid in-flight leases")
        await conn.execute(
            "UPDATE kc_generations SET state='gc_pending' WHERE scope_id=$1::uuid AND generation_id=$2",
            scope_id,
            generation_id,
        )
        return "gc_pending"


async def collect_generation(pool: asyncpg.Pool | None, scope_id: str, generation_id: str) -> None:
    """Destructive collection is DISABLED in this slice and implemented nowhere."""
    raise KcGenerationError("GC_DISABLED", "destructive GC is disabled; fixture generations stay within finite budgets")


# ── eligibility predicate (shared by every ranking arm and direct path) ─────────────────────────


def eligibility_cte(e: GenerationEligibility, start_index: int) -> tuple[str, list[Any]]:
    """``eligible`` relation: authorized namespace + project + captured SEALED generation membership.

    Returned fragment starts with ``eligible AS MATERIALIZED (...), `` so callers splice it directly after
    ``WITH``; every ranking arm then reads ``eligible`` instead of ``chunks`` (no outer post-filter)."""
    i = start_index
    sql = (
        "eligible AS MATERIALIZED ("
        "SELECT c.* FROM chunks c "
        "JOIN kc_generation_members m ON m.namespace_id = c.namespace_id AND m.project_id = c.project_id "
        "AND m.source_path = c.source_path "
        f"JOIN kc_generations g ON g.scope_id = m.scope_id AND g.generation_id = m.generation_id AND g.state = 'sealed' "  # noqa: E501
        f"WHERE m.scope_id = ${i}::uuid AND m.generation_id = ${i + 1} "
        f"AND c.namespace_id = ${i + 2} AND c.project_id = ${i + 3}), "
    )
    return sql, [e.scope_id, e.generation_id, e.namespace_id, e.project_id]


# ── context-bound direct paths (exact fetch, chunk, related, siblings) ──────────────────────────


async def read_member_exact(
    pool: asyncpg.Pool | None,
    ctx: GenerationReadContext,
    *,
    principal_id: str,
    member_id: str,
    backend: AuthorityBackend | None = None,
) -> dict[str, Any]:
    """Exact fetch of one member of the CAPTURED generation. Missing and foreign identifiers are the same
    ``NOT_FOUND``. Bytes are re-hashed at read time; a mismatch refuses (never repaired from another generation)."""
    await authorize_use(pool, ctx, principal_id=principal_id, backend=backend)
    async with _tx(pool, readonly=True, isolation="repeatable_read") as conn:
        row = await conn.fetchrow(
            "SELECT m.member_id, m.expected_bytes, m.expected_sha256, e.raw_content "
            "FROM kc_generation_members m JOIN kc_generations g USING (scope_id, generation_id) "
            "JOIN evidence_documents e ON e.namespace_id = m.namespace_id AND e.source_path = m.source_path "
            "WHERE m.scope_id=$1::uuid AND m.generation_id=$2 AND m.member_id=$3 AND g.state='sealed' "
            "AND m.namespace_id=$4 AND m.project_id=$5",
            ctx.scope_id,
            ctx.generation_id,
            member_id,
            ctx.namespace_id,
            ctx.project_id,
        )
    if row is None:
        raise KcGenerationError("NOT_FOUND")
    raw = row["raw_content"].encode("utf-8")
    if len(raw) != row["expected_bytes"] or sha256_hex(raw) != row["expected_sha256"]:
        raise KcGenerationError("CONTENT_MISMATCH", "stored bytes no longer match the sealed member")
    await authorize_use(pool, ctx, principal_id=principal_id, backend=backend)  # use-commit point
    return {
        "member_id": row["member_id"],
        "bytes": len(raw),
        "sha256": row["expected_sha256"],
        "text": row["raw_content"],
    }


async def read_chunk(
    pool: asyncpg.Pool | None,
    ctx: GenerationReadContext,
    *,
    principal_id: str,
    chunk_id: str,
    backend: AuthorityBackend | None = None,
) -> dict[str, Any]:
    await authorize_use(pool, ctx, principal_id=principal_id, backend=backend)
    try:
        UUID(chunk_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise KcGenerationError("NOT_FOUND") from exc
    e = ctx.eligibility
    sql, params = eligibility_cte(e, 2)
    async with _tx(pool, readonly=True, isolation="repeatable_read") as conn:
        row = await conn.fetchrow(
            f"WITH {sql} dummy AS (SELECT 1) SELECT c.id::text AS chunk_id, c.source_path, c.chunk_index, c.content, c.content_hash "  # noqa: E501
            "FROM eligible c WHERE c.id = $1::uuid",
            chunk_id,
            *params,
        )
    if row is None:
        raise KcGenerationError("NOT_FOUND")
    await authorize_use(pool, ctx, principal_id=principal_id, backend=backend)
    return dict(row)


async def read_related(
    pool: asyncpg.Pool | None,
    ctx: GenerationReadContext,
    *,
    principal_id: str,
    chunk_id: str,
    backend: AuthorityBackend | None = None,
) -> list[dict[str, Any]]:
    """Graph/related view: the requested chunk AND both endpoints of every exposed edge must be eligible."""
    await authorize_use(pool, ctx, principal_id=principal_id, backend=backend)
    try:
        UUID(chunk_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise KcGenerationError("NOT_FOUND") from exc
    sql, params = eligibility_cte(ctx.eligibility, 2)
    async with _tx(pool, readonly=True, isolation="repeatable_read") as conn:
        if (
            await conn.fetchval(
                f"WITH {sql} dummy AS (SELECT 1) SELECT 1 FROM eligible WHERE id=$1::uuid", chunk_id, *params
            )
            is None
        ):
            raise KcGenerationError("NOT_FOUND")
        rows = await conn.fetch(
            f"WITH {sql} dummy AS (SELECT 1) "
            "SELECT g.id, g.source_chunk_id::text AS source_chunk_id, g.target_chunk_id::text AS target_chunk_id, "
            "g.edge_type, g.weight FROM graph_edges g "
            "JOIN eligible s ON s.id = g.source_chunk_id JOIN eligible t ON t.id = g.target_chunk_id "
            "WHERE g.source_chunk_id = $1::uuid OR g.target_chunk_id = $1::uuid ORDER BY g.id",
            chunk_id,
            *params,
        )
    await authorize_use(pool, ctx, principal_id=principal_id, backend=backend)
    return [dict(r) for r in rows]


async def read_siblings(
    pool: asyncpg.Pool | None,
    ctx: GenerationReadContext,
    *,
    principal_id: str,
    chunk_id: str,
    backend: AuthorityBackend | None = None,
) -> list[dict[str, Any]]:
    """Parent/sibling navigation inside the captured generation only."""
    await authorize_use(pool, ctx, principal_id=principal_id, backend=backend)
    try:
        UUID(chunk_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise KcGenerationError("NOT_FOUND") from exc
    sql, params = eligibility_cte(ctx.eligibility, 2)
    async with _tx(pool, readonly=True, isolation="repeatable_read") as conn:
        me = await conn.fetchrow(
            f"WITH {sql} dummy AS (SELECT 1) SELECT parent_id, source_path FROM eligible WHERE id=$1::uuid",
            chunk_id,
            *params,
        )
        if me is None:
            raise KcGenerationError("NOT_FOUND")
        sib_sql, sib_params = eligibility_cte(ctx.eligibility, 3)
        rows = await conn.fetch(
            f"WITH {sib_sql} dummy AS (SELECT 1) "
            "SELECT id::text AS chunk_id, chunk_index, content FROM eligible "
            "WHERE source_path = $1 AND (parent_id IS NOT DISTINCT FROM $2::uuid) ORDER BY chunk_index",
            me["source_path"],
            me["parent_id"],
            *sib_params,
        )
    await authorize_use(pool, ctx, principal_id=principal_id, backend=backend)
    return [dict(r) for r in rows]


def refuse_unsupported_private_view(view: str) -> None:
    """Graph/LTM/derived entity views merge sources by (namespace, name, type); until a version-bound
    reconstruction exists they are refused for a private generation without leaking bodies."""
    raise KcGenerationError("UNSUPPORTED_PRIVATE_OPERATION", f"{view} is not supported for private generations")


# ── legacy-route bypass closure ─────────────────────────────────────────────────────────────────


async def private_namespace_ids(pool: asyncpg.Pool | None = None) -> frozenset[int]:
    pool = pool or await get_pool()
    rows = await pool.fetch("SELECT DISTINCT namespace_id FROM kc_corpus_scopes")
    return frozenset(r["namespace_id"] for r in rows)
