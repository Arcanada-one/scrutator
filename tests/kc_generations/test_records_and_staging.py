"""KB-CONS-01-N: schema installer, immutable records, sealed-member protection, staging (SYNTHETIC, native PostgreSQL)."""  # noqa: E501

from __future__ import annotations

from pathlib import Path

import asyncpg
import pytest

from scrutator.db import kc_generations as kc
from scrutator.db import repository
from tests.kc_generations.helpers import build_sealed, count, make_gen, make_world, stage_one

SCHEMA = Path(kc.__file__).parent / "schema.sql"
SNAPSHOT = Path(kc.__file__).parent / "migrations" / "007_kc_generations.sql"

TEXTS = {
    "root-a": "# Alpha\n\nAlpha root document with multibyte text: Привет, мир — 你好 🌍 café naïve (precomposed é).\n",
    "root-b": "# Beta\n\nBeta root document refers to the shared dependency.\n",
    "shared": "# Shared\n\nShared dependency used by both roots.\n",
}
EDGES = [("root-a", "shared"), ("root-b", "shared")]


def _gen(gid="g1", texts=None, roots=("root-a", "root-b"), edges=EDGES):
    return make_gen(gid, texts or TEXTS, list(roots), edges)


async def tamper(pool, sql: str, *args) -> None:
    """Simulate a faulty writer that got past the backstop (replica role skips user triggers + FKs)."""
    async with pool.acquire() as c, c.transaction():
        await c.execute("SET LOCAL session_replication_role = replica")
        await c.execute(sql, *args)


async def err_code(coro) -> str:
    with pytest.raises(kc.KcGenerationError) as ei:
        await coro
    return ei.value.code


# ── installer ───────────────────────────────────────────────────────────────────────────────────


async def test_R_SCH_01_real_installer_creates_native_tables(pool):
    names = {
        r["table_name"]
        for r in await pool.fetch("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'kc\\_%'")
    }
    assert {
        "kc_corpus_scopes",
        "kc_generations",
        "kc_generation_members",
        "kc_selections",
        "kc_selection_operations",
        "kc_read_leases",
    } <= names


async def test_R_SCH_02_installer_is_idempotent_with_existing_data(pool, world):
    gen = _gen("g-up")
    await build_sealed(world, gen)
    before = await count(pool, "SELECT count(*) FROM kc_generation_members")
    sql = SCHEMA.read_text()
    async with pool.acquire() as c:
        await c.execute(sql)  # second application over a populated database == upgrade of existing data
        await c.execute(sql)
    assert await count(pool, "SELECT count(*) FROM kc_generation_members") == before
    sealed = await count(
        pool, "SELECT count(*) FROM kc_generations WHERE state='sealed' AND scope_id=$1::uuid", world.scope_id
    )
    assert sealed == 1


def test_R_SCH_03_review_snapshot_equals_installer_block():
    def block(path: Path) -> str:
        t = path.read_text()
        return t[t.index("-- KC_GENERATION_SCHEMA_BEGIN") : t.index("-- KC_GENERATION_SCHEMA_END")]

    assert block(SCHEMA) == block(SNAPSHOT)


async def test_R_SCH_04_constraints_at_the_database_boundary(pool, world):
    # a project of ANOTHER namespace cannot be paired with this namespace (composite FK)
    async with pool.acquire() as c:
        n2 = await c.fetchval("INSERT INTO namespaces (name) VALUES ('ns-x-' || gen_random_uuid()::text) RETURNING id")
        p2 = await c.fetchval("INSERT INTO projects (namespace_id, name) VALUES ($1,'px') RETURNING id", n2)
    assert await err_code(kc.create_scope(pool, world.namespace_id, p2, "x")) == "SCOPE_INVALID"
    # unique (namespace, project, purpose)
    assert await err_code(kc.create_scope(pool, world.namespace_id, world.project_id, "pilot")) == "SCOPE_INVALID"
    # partially-NULL selection reference and negative revision are rejected by CHECK
    for sql in (
        "UPDATE kc_selections SET generation_id='x' WHERE scope_id=$1::uuid",
        "UPDATE kc_selections SET manifest_sha256=repeat('a',64) WHERE scope_id=$1::uuid",
    ):
        with pytest.raises(asyncpg.PostgresError):
            async with pool.acquire() as c:
                await c.execute(sql, world.scope_id)
    with pytest.raises(asyncpg.PostgresError):
        await tamper(pool, "UPDATE kc_selections SET revision=-1 WHERE scope_id=$1::uuid", world.scope_id)


async def test_R_SCH_05_selection_row_precreated_at_revision_zero(pool, world):
    sel = await kc.get_selection(pool, world.scope_id)
    assert (sel.revision, sel.generation_id, sel.manifest_sha256) == (0, None, None)
    assert sel.as_json()["revision"] == "0"  # decimal string at JSON boundary


# ── immutable scope / generation / member records ───────────────────────────────────────────────


async def test_R_REC_01_scope_rows_are_immutable(pool, world):
    for sql in (
        "UPDATE kc_corpus_scopes SET purpose='zz' WHERE scope_id=$1::uuid",
        "DELETE FROM kc_corpus_scopes WHERE scope_id=$1::uuid",
    ):
        with pytest.raises(asyncpg.PostgresError):
            async with pool.acquire() as c:
                await c.execute(sql, world.scope_id)


async def test_R_REC_02_generation_identity_and_manifest_immutable_and_never_deleted(pool, world):
    gen = _gen("g-imm")
    await build_sealed(world, gen)
    for sql in (
        "UPDATE kc_generations SET manifest=decode('00','hex') WHERE scope_id=$1::uuid AND generation_id='g-imm'",
        "UPDATE kc_generations SET admission_ref='other' WHERE scope_id=$1::uuid AND generation_id='g-imm'",
        "UPDATE kc_generations SET seal_report='{}'::jsonb WHERE scope_id=$1::uuid AND generation_id='g-imm'",
        "UPDATE kc_generations SET state='staging' WHERE scope_id=$1::uuid AND generation_id='g-imm'",
        "DELETE FROM kc_generations WHERE scope_id=$1::uuid AND generation_id='g-imm'",
    ):
        with pytest.raises(asyncpg.PostgresError):
            async with pool.acquire() as c:
                await c.execute(sql, world.scope_id)


async def test_R_REC_03_member_rows_immutable_and_populated_member_cannot_be_replaced(pool, world):
    gen = _gen("g-mem")
    await build_sealed(world, gen)
    for sql in (
        "UPDATE kc_generation_members SET chunk_set_sha256=repeat('b',64) WHERE scope_id=$1::uuid AND member_id='shared'",  # noqa: E501
        "UPDATE kc_generation_members SET expected_sha256=repeat('b',64) WHERE scope_id=$1::uuid AND member_id='shared'",  # noqa: E501
        "DELETE FROM kc_generation_members WHERE scope_id=$1::uuid AND member_id='shared'",
    ):
        with pytest.raises(asyncpg.PostgresError):
            async with pool.acquire() as c:
                await c.execute(sql, world.scope_id)


async def test_R_REC_04_overwrite_of_a_sealed_member_via_stage_is_refused(pool, world):
    gen = _gen("g-ovw")
    await build_sealed(world, gen)
    b = await kc.get_member_binding(pool, world.scope_id, "g-ovw", "shared")
    other = "# Shared\n\nDIFFERENT body of the same length? no: just different.\n"
    chunks = kc.prepare_member_chunks(b.namespace, b.source_path, other)
    # different bytes than the manifest member -> mismatch; and with the right bytes but another chunk set -> conflict
    code = await err_code(
        kc.stage_member(
            pool,
            world.scope_id,
            "g-ovw",
            "shared",
            body=other.encode(),
            chunks=chunks,
            embeddings=[[0.0] * 1024] * len(chunks),
            sparse_weights=[{}] * len(chunks),
            producer="x",
        )
    )
    assert code == "MEMBER_MISMATCH"
    right = TEXTS["shared"]
    alt_chunks = kc.prepare_member_chunks(b.namespace, b.source_path, right)
    alt_chunks[0] = {
        **alt_chunks[0],
        "content": alt_chunks[0]["content"] + " ",
        "content_hash": kc.sha256_hex((alt_chunks[0]["content"] + " ").encode()),
    }
    code = await err_code(
        kc.stage_member(
            pool,
            world.scope_id,
            "g-ovw",
            "shared",
            body=right.encode(),
            chunks=alt_chunks,
            embeddings=[[0.0] * 1024] * len(alt_chunks),
            sparse_weights=[{}] * len(alt_chunks),
            producer="synthetic-chunker/1",
        )
    )
    assert code == "MEMBER_CONFLICT"


async def test_R_REC_05_sealed_rows_protected_against_actual_mutation_entrypoints(pool, world):
    gen = _gen("g-prot")
    await build_sealed(world, gen)
    b = await kc.get_member_binding(pool, world.scope_id, "g-prot", "root-a")
    ns, path = b.namespace_id, b.source_path
    before = await count(pool, "SELECT count(*) FROM chunks WHERE namespace_id=$1 AND source_path=$2", ns, path)
    digest_before = await pool.fetchval(
        "SELECT md5(string_agg(content, '|' ORDER BY chunk_index)) FROM chunks WHERE namespace_id=$1 AND source_path=$2",  # noqa: E501
        ns,
        path,
    )

    # direct SQL on chunks / evidence_documents / sparse_vectors
    for sql, args in (
        ("UPDATE chunks SET content='tampered' WHERE namespace_id=$1 AND source_path=$2", (ns, path)),
        ("DELETE FROM chunks WHERE namespace_id=$1 AND source_path=$2", (ns, path)),
        ("UPDATE evidence_documents SET raw_content='x' WHERE namespace_id=$1 AND source_path=$2", (ns, path)),
        ("DELETE FROM evidence_documents WHERE namespace_id=$1 AND source_path=$2", (ns, path)),
        (
            "UPDATE sparse_vectors SET token_weights='{}'::jsonb WHERE chunk_id IN (SELECT id FROM chunks WHERE namespace_id=$1 AND source_path=$2)",  # noqa: E501
            (ns, path),
        ),
        (
            "DELETE FROM sparse_vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE namespace_id=$1 AND source_path=$2)",  # noqa: E501
            (ns, path),
        ),
        ("DELETE FROM namespaces WHERE id=$1", (ns,)),
    ):
        with pytest.raises(asyncpg.PostgresError):
            async with pool.acquire() as c:
                await c.execute(sql, *args)

    # legacy repository entrypoints
    from scrutator.chunker.splitters import compute_doc_id

    chunk = {
        "id": "00000000-0000-4000-8000-0000000000aa",
        "chunk_index": 0,
        "source_path": path,
        "source_type": "doc",
        "content": "x",
        "content_hash": "h",
        "metadata": {},
        "token_count": 1,
    }
    with pytest.raises(asyncpg.PostgresError):
        await repository.replace_source_chunks_atomic([chunk], [[0.0] * 1024], [{}], ns, world.project_id)
    with pytest.raises(asyncpg.PostgresError):
        await repository.delete_by_source(path, ns)
    with pytest.raises(asyncpg.PostgresError):
        await repository.delete_ltm_source(ns, path)
    assert compute_doc_id(b.namespace, path) == b.doc_id
    assert await count(pool, "SELECT count(*) FROM chunks WHERE namespace_id=$1 AND source_path=$2", ns, path) == before
    assert (
        await pool.fetchval(
            "SELECT md5(string_agg(content, '|' ORDER BY chunk_index)) FROM chunks WHERE namespace_id=$1 AND source_path=$2",  # noqa: E501
            ns,
            path,
        )
        == digest_before
    )


async def test_R_REC_06_reserved_path_without_registered_member_is_refused(pool, world):
    with pytest.raises(asyncpg.PostgresError):
        async with pool.acquire() as c:
            await c.execute(
                "INSERT INTO evidence_documents (namespace_id, source_path, doc_id, content_hash, raw_content) "
                "VALUES ($1, 'kc-generations/forged/path', 'd', 'h', 'x')",
                world.namespace_id,
            )


async def test_R_REC_07_generic_writes_to_ordinary_paths_of_the_same_namespace_unaffected(pool, world):
    chunk = {
        "id": "00000000-0000-4000-8000-0000000000bb",
        "chunk_index": 0,
        "source_path": "docs/ordinary.md",
        "source_type": "doc",
        "content": "ordinary",
        "content_hash": "h",
        "metadata": {},
        "token_count": 1,
    }
    n = await repository.replace_source_chunks_atomic(
        [chunk], [[0.0] * 1024], [{}], world.namespace_id, world.project_id
    )
    assert n == 1
    assert await repository.delete_by_source("docs/ordinary.md", world.namespace_id) == 1


# ── staging ─────────────────────────────────────────────────────────────────────────────────────


async def test_R_STG_01_create_candidate_idempotent_only_for_identical_bytes_and_binding(pool, world):
    gen = _gen("g-idem")
    a = await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="adm", profile_ref="p")
    members_before = await count(
        pool, "SELECT count(*) FROM kc_generation_members WHERE scope_id=$1::uuid", world.scope_id
    )
    b = await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="adm", profile_ref="p")
    assert (a.created, b.created) == (True, False)
    assert (
        await count(pool, "SELECT count(*) FROM kc_generation_members WHERE scope_id=$1::uuid", world.scope_id)
        == members_before
    )


async def test_R_STG_02_same_generation_id_different_manifest_conflicts_before_any_change(pool, world):
    gen = _gen("g-conf")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="adm", profile_ref="p")
    snap = await pool.fetch(
        "SELECT member_id, member_uuid, source_path FROM kc_generation_members WHERE scope_id=$1::uuid ORDER BY 1",
        world.scope_id,
    )
    changed = _gen("g-conf", {**TEXTS, "shared": TEXTS["shared"] + "x"})
    assert (
        await err_code(
            kc.create_candidate(pool, world.scope_id, changed.manifest, admission_ref="adm", profile_ref="p")
        )
        == "GENERATION_ID_CONFLICT"
    )
    assert (
        await err_code(kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="OTHER", profile_ref="p"))
        == "GENERATION_ID_CONFLICT"
    )
    assert snap == await pool.fetch(
        "SELECT member_id, member_uuid, source_path FROM kc_generation_members WHERE scope_id=$1::uuid ORDER BY 1",
        world.scope_id,
    )
    assert await count(pool, "SELECT count(*) FROM kc_generations WHERE scope_id=$1::uuid", world.scope_id) == 1


async def test_R_STG_03_identical_member_retry_is_a_noop_without_chunk_id_churn(pool, world):
    gen = _gen("g-retry")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    assert await stage_one(world, "g-retry", "shared", TEXTS["shared"]) == "created"
    b = await kc.get_member_binding(pool, world.scope_id, "g-retry", "shared")
    ids = [
        r["id"]
        for r in await pool.fetch(
            "SELECT id FROM chunks WHERE namespace_id=$1 AND source_path=$2 ORDER BY chunk_index",
            b.namespace_id,
            b.source_path,
        )
    ]
    assert await stage_one(world, "g-retry", "shared", TEXTS["shared"]) == "unchanged"
    assert ids == [
        r["id"]
        for r in await pool.fetch(
            "SELECT id FROM chunks WHERE namespace_id=$1 AND source_path=$2 ORDER BY chunk_index",
            b.namespace_id,
            b.source_path,
        )
    ]


async def test_R_STG_04_nul_and_non_utf8_bodies_refused_before_any_write(pool, world):
    nul = "# N\n\nbad\x00body\n"
    gen = make_gen("g-nul", {"only": nul}, ["only"])
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    b = await kc.get_member_binding(pool, world.scope_id, "g-nul", "only")
    chunks = kc.prepare_member_chunks(b.namespace, b.source_path, "# N\n\nclean\n")
    emb = [[0.0] * 1024] * len(chunks)
    assert (
        await err_code(
            kc.stage_member(
                pool,
                world.scope_id,
                "g-nul",
                "only",
                body=nul.encode(),
                chunks=chunks,
                embeddings=emb,
                sparse_weights=[{}] * len(chunks),
                producer="p",
            )
        )
        == "NUL_IN_BODY"
    )
    assert (
        await err_code(
            kc.stage_member(
                pool,
                world.scope_id,
                "g-nul",
                "only",
                body=b"\xff\xfe\xfa",
                chunks=chunks,
                embeddings=emb,
                sparse_weights=[{}] * len(chunks),
                producer="p",
            )
        )
        == "MEMBER_INVALID"
    )
    assert await count(pool, "SELECT count(*) FROM evidence_documents WHERE namespace_id=$1", world.namespace_id) == 0
    assert await count(pool, "SELECT count(*) FROM chunks WHERE namespace_id=$1", world.namespace_id) == 0


async def test_R_STG_05_manifest_digest_mismatch_refused_and_nothing_written(pool, world):
    gen = _gen("g-dig")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    b = await kc.get_member_binding(pool, world.scope_id, "g-dig", "shared")
    wrong = TEXTS["shared"].replace("Shared", "Sharee")  # same byte length, different digest
    chunks = kc.prepare_member_chunks(b.namespace, b.source_path, wrong)
    code = await err_code(
        kc.stage_member(
            pool,
            world.scope_id,
            "g-dig",
            "shared",
            body=wrong.encode(),
            chunks=chunks,
            embeddings=[[0.0] * 1024] * len(chunks),
            sparse_weights=[{}] * len(chunks),
            producer="p",
        )
    )
    assert code == "MEMBER_MISMATCH"
    assert await count(pool, "SELECT count(*) FROM chunks WHERE namespace_id=$1", world.namespace_id) == 0


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda d: d.update(schema="KCSyntheticManifest/v2"), "unknown manifest schema"),
        (lambda d: d["members"].append(dict(d["members"][0])), "duplicate member"),
        (
            lambda d: d["required_edges"].append({"from": "root-a", "to": "shared", "relation": "optional"}),
            "unknown required relation",
        ),
        (
            lambda d: d["required_edges"].append({"from": "root-a", "to": "ghost", "relation": "requires"}),
            "unknown member",
        ),
        (lambda d: d.update(extra=1), "unexpected manifest keys"),
        (lambda d: d.update(roots=[]), "roots"),
    ],
)
def test_R_STG_06_manifest_refusals(mutate, expected):
    import json

    d = json.loads(_gen().manifest)
    mutate(d)
    with pytest.raises(kc.KcGenerationError) as ei:
        kc.parse_manifest(json.dumps(d).encode(), max_members=64, max_manifest_bytes=10**6, max_total_bytes=10**9)
    assert ei.value.code == "MANIFEST_INVALID" and expected in ei.value.message


def test_R_STG_07_duplicate_json_keys_and_limits_refused():
    raw = b'{"schema":"KCSyntheticManifest/v1","schema":"x"}'
    with pytest.raises(kc.KcGenerationError) as ei:
        kc.parse_manifest(raw, max_members=4, max_manifest_bytes=10**6, max_total_bytes=10**9)
    assert ei.value.code == "MANIFEST_INVALID"
    with pytest.raises(kc.KcGenerationError) as ei:
        kc.parse_manifest(_gen().manifest, max_members=2, max_manifest_bytes=10**6, max_total_bytes=10**9)
    assert ei.value.code == "LIMIT_EXCEEDED"
    with pytest.raises(kc.KcGenerationError) as ei:
        kc.parse_manifest(_gen().manifest, max_members=64, max_manifest_bytes=50, max_total_bytes=10**9)
    assert ei.value.code == "LIMIT_EXCEEDED"


async def test_R_STG_08_finite_retention_limits_refuse_staging_and_never_evict(pool):
    w = await make_world(pool, max_generations=1)
    await build_sealed(w, _gen("g-l1"))
    assert (
        await err_code(kc.create_candidate(pool, w.scope_id, _gen("g-l2").manifest, admission_ref="a", profile_ref="p"))
        == "LIMIT_EXCEEDED"
    )
    assert (
        await count(pool, "SELECT count(*) FROM kc_generations WHERE scope_id=$1::uuid AND state='sealed'", w.scope_id)
        == 1
    )
    w2 = await make_world(pool, max_total_bytes=50)
    assert (
        await err_code(
            kc.create_candidate(pool, w2.scope_id, _gen("g-big").manifest, admission_ref="a", profile_ref="p")
        )
        == "LIMIT_EXCEEDED"
    )


async def test_R_STG_09_scope_purpose_must_match_manifest(pool, world):
    gen = make_gen("g-purp", TEXTS, ["root-a", "root-b"], EDGES, purpose="other-purpose")
    assert (
        await err_code(kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p"))
        == "MANIFEST_INVALID"
    )


async def test_R_STG_10_doc_id_collision_cannot_alias_two_members(pool, world):
    gen = _gen("g-doc")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    rows = await pool.fetch(
        "SELECT doc_id, source_path FROM kc_generation_members WHERE scope_id=$1::uuid", world.scope_id
    )
    assert len({r["doc_id"] for r in rows}) == len(rows) == len({r["source_path"] for r in rows})
    with pytest.raises(asyncpg.UniqueViolationError):  # the unique index (not only the guard trigger) refuses aliasing
        await tamper(
            pool,
            "UPDATE kc_generation_members SET doc_id=(SELECT doc_id FROM kc_generation_members "
            "WHERE scope_id=$1::uuid AND member_id='shared') WHERE scope_id=$1::uuid AND member_id='root-a'",
            world.scope_id,
        )


async def test_R_SCH_06_partial_schema_failure_is_atomic_and_leaves_no_half_installed_scope_layer(kc_dsn):
    """A failure after the native block rolls back the whole installer run: no scope table, no silent re-creation."""
    import os
    import uuid

    admin = os.environ["KC_TEST_PG_ADMIN_DSN"]
    db = f"kcatomic_{uuid.uuid4().hex[:8]}"
    c = await asyncpg.connect(admin)
    try:
        await c.execute(f'CREATE DATABASE "{db}"')
    finally:
        await c.close()
    try:
        s = await asyncpg.connect(f"{admin.rsplit('/', 1)[0]}/{db}")
        try:
            await s.execute("CREATE EXTENSION IF NOT EXISTS vector")
            with pytest.raises(asyncpg.PostgresError):
                await s.execute(SCHEMA.read_text() + "\nSELECT 1/0;\n")  # fails AFTER the native block
            assert await s.fetchval("SELECT to_regclass('kc_corpus_scopes')") is None
            assert await s.fetchval("SELECT to_regclass('kc_selections')") is None
            await s.execute(SCHEMA.read_text())  # a clean re-run installs it completely
            assert await s.fetchval("SELECT to_regclass('kc_selections')") is not None
        finally:
            await s.close()
    finally:
        c = await asyncpg.connect(admin)
        try:  # removes only the scratch database this test created above
            await c.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
        finally:
            await c.close()
