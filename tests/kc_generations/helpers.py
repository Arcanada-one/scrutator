"""Shared helpers for the KB-CONS-01-N native PostgreSQL tests (SYNTHETIC data only)."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field

import numpy as np

from scrutator.db import kc_generations as kc

DIM = 1024


def _token_vec(token: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


def embed(text: str) -> list[float]:
    """Deterministic bag-of-token embedding: texts sharing tokens are close in cosine space."""
    tokens = re.findall(r"\w+", text.lower()) or ["_"]
    v = sum(_token_vec(t) for t in tokens)
    return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32).tolist()


def sparse(text: str) -> dict[str, float]:
    return {t: 1.0 for t in set(re.findall(r"\w+", text.lower()))}


class SyntheticAuthority:
    """TEST-ONLY authority backend. NOT a real private-use authority: it models a grant table with a
    revision counter so ordering of a revoke against a use can be exercised. Labelled synthetic."""

    name = "synthetic-test-authority"

    def __init__(self, granted: set[str] | None = None) -> None:
        self.granted = set(granted or set())
        self.revision = 1
        self.outage = False
        self.calls: list[tuple[str, str]] = []
        self.on_check = None  # optional async hook(purpose) for ordering tests

    def grant(self, principal: str) -> None:
        self.granted.add(principal)
        self.revision += 1

    def revoke(self, principal: str) -> None:
        self.granted.discard(principal)
        self.revision += 1

    async def check(self, conn, *, principal_id: str, scope_id: str, purpose: str) -> kc.AuthorityObservation:
        self.calls.append((principal_id, purpose))
        if self.on_check is not None:
            await self.on_check(purpose)
        if self.outage:
            raise ConnectionError("synthetic authority outage")
        if principal_id not in self.granted:
            raise kc.KcGenerationError("AUTHORITY_REVOKED", "grant not present")
        return kc.AuthorityObservation(
            self.name, principal_id, scope_id, purpose, str(self.revision), f"synthetic-ordering:{self.revision}"
        )


@dataclass
class World:
    pool: object
    namespace_id: int
    namespace: str
    project_id: int
    other_project_id: int
    scope_id: str
    purpose: str = "pilot"
    extra: dict = field(default_factory=dict)


async def make_world(pool, **limits) -> World:
    ns = f"ns-{uuid.uuid4().hex[:10]}"
    async with pool.acquire() as c:
        nid = await c.fetchval("INSERT INTO namespaces (name) VALUES ($1) RETURNING id", ns)
        pid = await c.fetchval("INSERT INTO projects (namespace_id, name) VALUES ($1,'pilot') RETURNING id", nid)
        oid = await c.fetchval("INSERT INTO projects (namespace_id, name) VALUES ($1,'other') RETURNING id", nid)
    scope_id = await kc.create_scope(pool, nid, pid, "pilot", **limits)
    return World(pool, nid, ns, pid, oid, scope_id)


def member_spec(member_id: str, text: str, revision: str = "1", representation: str = "md-utf8") -> dict:
    raw = text.encode("utf-8")
    return {
        "member_id": member_id,
        "semantic_revision": revision,
        "representation_id": representation,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


@dataclass
class Gen:
    generation_id: str
    manifest: bytes
    texts: dict[str, str]
    roots: list[str]
    edges: list[tuple[str, str]]


def make_gen(
    generation_id: str, texts: dict[str, str], roots: list[str], edges=(), purpose="pilot", revisions=None
) -> Gen:
    revisions = revisions or {}
    members = [member_spec(m, t, revisions.get(m, "1")) for m, t in texts.items()]
    manifest = kc.build_manifest_bytes(generation_id, purpose, members, roots, list(edges))
    return Gen(generation_id, manifest, texts, roots, list(edges))


async def stage_one(world: World, generation_id: str, member_id: str, text: str, *, producer="synthetic-chunker/1"):
    b = await kc.get_member_binding(world.pool, world.scope_id, generation_id, member_id)
    chunks = kc.prepare_member_chunks(b.namespace, b.source_path, text)
    return await kc.stage_member(
        world.pool,
        world.scope_id,
        generation_id,
        member_id,
        body=text.encode("utf-8"),
        chunks=chunks,
        embeddings=[embed(c["content"]) for c in chunks],
        sparse_weights=[sparse(c["content"]) for c in chunks],
        producer=producer,
    )


async def stage_all(world: World, gen: Gen, *, skip=frozenset(), producer="synthetic-chunker/1") -> None:
    for member_id, text in gen.texts.items():
        if member_id not in skip:
            await stage_one(world, gen.generation_id, member_id, text, producer=producer)


async def build_sealed(world: World, gen: Gen, *, oracle=None, admission="adm-1", profile="kc-synthetic/v1") -> dict:
    cand = await kc.create_candidate(
        world.pool, world.scope_id, gen.manifest, admission_ref=admission, profile_ref=profile
    )
    assert cand.created
    await stage_all(world, gen)
    return await kc.seal_generation(
        world.pool, world.scope_id, gen.generation_id, oracle_closure=oracle or gen.texts.keys()
    )


async def promote_simple(world: World, gen_id: str, op: str, backend, principal="p1"):
    sel = await kc.get_selection(world.pool, world.scope_id)
    return await kc.promote(
        world.pool,
        world.scope_id,
        operation_id=op,
        candidate_generation_id=gen_id,
        expected_incarnation=sel.incarnation_id,
        expected_revision=str(sel.revision),
        expected_generation=sel.generation_id,
        principal_id=principal,
        backend=backend,
    )


async def count(pool, sql: str, *args) -> int:
    async with pool.acquire() as c:
        return await c.fetchval(sql, *args)
