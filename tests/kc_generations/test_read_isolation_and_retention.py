"""KB-CONS-01-N: one retained read context, generation+project eligibility in every arm and direct path,
authority re-check, lease/GC lock order (SYNTHETIC data, native PostgreSQL)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid
from unittest.mock import patch

import pytest

from scrutator.auth import dependency as dep
from scrutator.auth.models import VerifiedPrincipal
from scrutator.config import settings
from scrutator.db import kc_generations as kc
from scrutator.db import repository
from scrutator.search import searcher
from tests.kc_generations.helpers import (
    SyntheticAuthority,
    build_sealed,
    count,
    embed,
    make_gen,
    make_world,
    promote_simple,
    sparse,
    stage_all,
)
from tests.kc_generations.test_records_and_staging import err_code

ELIG = "# Eligible\n\nharbor lantern quartz zebra eligible material about ships.\n"
CANARY = "# Canary\n\ncanaryonlytoken quartz zebra quartz zebra canary canary dominates ranking.\n"


async def _insert_plain_chunk(pool, ns_id, project_id, path, text, doc_id=None):
    cid = str(uuid.uuid4())
    async with pool.acquire() as c:
        await c.execute(
            "INSERT INTO chunks (id, namespace_id, project_id, source_path, source_type, chunk_index, content, content_hash, "  # noqa: E501
            "embedding_dense, metadata, token_count) VALUES ($1::uuid,$2,$3,$4,'doc',0,$5,$6,$7,'{}'::jsonb,5)",
            cid,
            ns_id,
            project_id,
            path,
            text,
            kc.sha256_hex(text.encode()),
            __import__("numpy").asarray(embed(text), dtype="float32"),
        )
        await c.execute(
            "INSERT INTO sparse_vectors (chunk_id, token_weights) VALUES ($1::uuid, $2::jsonb)",
            cid,
            json.dumps(sparse(text)),
        )
    return cid


async def _chunk_ids(pool, ns_id, path):
    return [
        str(r["id"])
        for r in await pool.fetch(
            "SELECT id FROM chunks WHERE namespace_id=$1 AND source_path=$2 ORDER BY chunk_index", ns_id, path
        )
    ]


class Iso:
    """Selected generation G1 (eligible) + every canary class the spec names."""


async def isolation_world(pool) -> Iso:
    w = await make_world(pool)
    iso = Iso()
    iso.w, iso.auth = w, SyntheticAuthority({"p1", "p2"})
    # superseded generation G0 (sealed, then replaced by G1) holds a canary
    g0 = make_gen("g0", {"old": CANARY}, ["old"])
    await build_sealed(w, g0)
    await promote_simple(w, "g0", "op-g0", iso.auth)
    g1 = make_gen("g1", {"doc": ELIG}, ["doc"])
    await build_sealed(w, g1)
    await promote_simple(w, "g1", "op-g1", iso.auth)
    # staged (never sealed) candidate G2 in the same scope with canary bytes
    g2 = make_gen("g2", {"staged": CANARY}, ["staged"])
    await kc.create_candidate(pool, w.scope_id, g2.manifest, admission_ref="a", profile_ref="p")
    await stage_all(w, g2)
    b = {
        m: await kc.get_member_binding(pool, w.scope_id, gid, m)
        for gid, m in (("g0", "old"), ("g1", "doc"), ("g2", "staged"))
    }
    iso.paths = {k: v.source_path for k, v in b.items()}
    iso.elig_ids = await _chunk_ids(pool, w.namespace_id, b["doc"].source_path)
    iso.canaries = {
        "superseded": await _chunk_ids(pool, w.namespace_id, b["old"].source_path),
        "staged": await _chunk_ids(pool, w.namespace_id, b["staged"].source_path),
        "other-project": [await _insert_plain_chunk(pool, w.namespace_id, w.other_project_id, "docs/other.md", CANARY)],
        "unregistered-same-project": [
            await _insert_plain_chunk(pool, w.namespace_id, w.project_id, "docs/loose.md", CANARY)
        ],
    }
    async with pool.acquire() as c:
        n2 = await c.fetchval(
            "INSERT INTO namespaces (name) VALUES ('ns-foreign-' || gen_random_uuid()::text) RETURNING id"
        )
        p2 = await c.fetchval("INSERT INTO projects (namespace_id, name) VALUES ($1,'pilot') RETURNING id", n2)
    iso.foreign_ns = n2
    iso.canaries["foreign-namespace"] = [await _insert_plain_chunk(pool, n2, p2, "docs/foreign.md", CANARY)]
    iso.all_canary = {c for v in iso.canaries.values() for c in v}
    iso.ctx = await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-iso", backend=iso.auth)
    return iso


def _ids(results) -> set[str]:
    return {r.chunk_id for r in results}


# ── AC5-07: every ranking arm ───────────────────────────────────────────────────────────────────


async def test_AC5_07_control_unconstrained_search_does_surface_the_canaries(pool):
    iso = await isolation_world(pool)
    got = await repository.hybrid_search(
        embed(CANARY), "canaryonlytoken", iso.w.namespace_id, limit=10, query_sparse=sparse(CANARY)
    )
    assert _ids(got) & iso.all_canary, "control failed: canaries must dominate the unconstrained candidate set"


@pytest.mark.parametrize("arm", ["semantic", "fulltext", "sparse"])
async def test_AC5_07_each_ranking_arm_is_constrained_before_rank_and_limit(pool, arm):
    iso = await isolation_world(pool)
    elig = iso.ctx.eligibility
    # drive ONE arm toward the canaries; the other arms are neutralised
    if arm == "semantic":
        kwargs = dict(query_embedding=embed(CANARY), query_text="zzzznomatchzzzz", query_sparse=None)
    elif arm == "fulltext":
        kwargs = dict(
            query_embedding=embed("unrelated words entirely"), query_text="canaryonlytoken", query_sparse=None
        )
    else:
        kwargs = dict(
            query_embedding=embed("unrelated words entirely"),
            query_text="zzzznomatchzzzz",
            query_sparse={"canaryonlytoken": 1.0},
        )
    got = await repository.hybrid_search(namespace_id=iso.w.namespace_id, limit=10, generation=elig, **kwargs)
    assert got, "eligible content must still be returned"
    assert _ids(got) <= set(iso.elig_ids), f"{arm} arm leaked an ineligible candidate"
    assert not (_ids(got) & iso.all_canary)
    # limit=1 with a canary-dominated query must yield the ELIGIBLE chunk, not a displaced empty/other result
    one = await repository.hybrid_search(namespace_id=iso.w.namespace_id, limit=1, generation=elig, **kwargs)
    assert _ids(one) <= set(iso.elig_ids) and len(one) == 1


async def test_AC5_07_filtered_search_arms_are_constrained_too(pool):
    iso = await isolation_world(pool)

    async def fake_embed(_):
        return embed(CANARY)

    with patch("scrutator.search.embedder.embed_single", fake_embed):
        got = await repository.search_with_filters(
            "canaryonlytoken", iso.w.namespace_id, source_type="doc", limit=10, generation=iso.ctx.eligibility
        )
        control = await repository.search_with_filters(
            "canaryonlytoken", iso.w.namespace_id, source_type="doc", limit=10
        )
    assert {r["chunk_id"] for r in control} & iso.all_canary
    assert {r["chunk_id"] for r in got} <= set(iso.elig_ids) and not ({r["chunk_id"] for r in got} & iso.all_canary)


async def test_AC5_07_reranker_input_grouped_results_and_ignored_project_parameter(pool):
    iso = await isolation_world(pool)
    seen: list[list[str]] = []

    async def fake_rerank(query, candidates, top_k):
        seen.append([c.chunk_id for c in candidates])
        for c in candidates:
            c.citation = searcher._build_citation(c, "rrf")
        return candidates[:top_k]

    async def fake_embed(_):
        return embed(CANARY)

    async def fake_sparse(_):
        return [{"canaryonlytoken": 1.0}]

    with (
        patch.object(settings, "rerank_enabled", True),
        patch.object(searcher, "rerank", fake_rerank),
        patch.object(searcher, "embed_single", fake_embed),
        patch.object(searcher, "embed_sparse", fake_sparse),
    ):
        res = await searcher.search(
            "canaryonlytoken", iso.w.namespace_id, project="other", limit=5, generation=iso.ctx.eligibility
        )
        grouped = await searcher.search(
            "canaryonlytoken", iso.w.namespace_id, limit=5, group_by="document", generation=iso.ctx.eligibility
        )
    assert seen and all(set(batch) <= set(iso.elig_ids) for batch in seen), "reranker received an ineligible body"
    assert _ids(res.results) <= set(iso.elig_ids)  # request `project=other` never chose the project
    assert {g.representative.chunk_id for g in grouped.results} <= set(iso.elig_ids)
    assert all(set(g.member_chunk_ids) <= set(iso.elig_ids) for g in grouped.results)


# ── direct paths ────────────────────────────────────────────────────────────────────────────────


async def test_AC5_07_direct_fetch_chunk_related_siblings_cannot_reach_ineligible_content(pool):
    iso = await isolation_world(pool)
    ctx, auth = iso.ctx, iso.auth
    # unknown identifier vs real-but-ineligible identifier: byte-for-byte the same refusal
    unknown = await err_code(kc.read_chunk(pool, ctx, principal_id="p1", chunk_id=str(uuid.uuid4()), backend=auth))
    for klass, ids in iso.canaries.items():
        for cid in ids:
            assert (
                await err_code(kc.read_chunk(pool, ctx, principal_id="p1", chunk_id=cid, backend=auth))
                == unknown
                == "NOT_FOUND"
            ), klass
            assert (
                await err_code(kc.read_related(pool, ctx, principal_id="p1", chunk_id=cid, backend=auth)) == "NOT_FOUND"
            )
            assert (
                await err_code(kc.read_siblings(pool, ctx, principal_id="p1", chunk_id=cid, backend=auth))
                == "NOT_FOUND"
            )
    assert (
        await err_code(kc.read_member_exact(pool, ctx, principal_id="p1", member_id="staged", backend=auth))
        == "NOT_FOUND"
    )
    assert (
        await err_code(kc.read_member_exact(pool, ctx, principal_id="p1", member_id="old", backend=auth)) == "NOT_FOUND"
    )
    ok = await kc.read_chunk(pool, ctx, principal_id="p1", chunk_id=iso.elig_ids[0], backend=auth)
    assert ok["chunk_id"] == iso.elig_ids[0]
    exact = await kc.read_member_exact(pool, ctx, principal_id="p1", member_id="doc", backend=auth)
    assert exact["text"] == ELIG and exact["bytes"] == len(ELIG.encode())


async def test_AC5_07_related_view_needs_both_endpoints_inside_the_generation(pool):
    iso = await isolation_world(pool)
    me = iso.elig_ids[0]
    leak = iso.canaries["other-project"][0]
    async with pool.acquire() as c:
        await c.execute(
            "INSERT INTO graph_edges (source_chunk_id, target_chunk_id, edge_type) VALUES ($1::uuid,$2::uuid,'related')",  # noqa: E501
            me,
            leak,
        )
        await c.execute(
            "INSERT INTO graph_edges (source_chunk_id, target_chunk_id, edge_type) VALUES ($1::uuid,$2::uuid,'related')",  # noqa: E501
            leak,
            me,
        )
        if len(iso.elig_ids) > 1:
            await c.execute(
                "INSERT INTO graph_edges (source_chunk_id, target_chunk_id, edge_type) VALUES ($1::uuid,$2::uuid,'next')",  # noqa: E501
                me,
                iso.elig_ids[1],
            )
    edges = await kc.read_related(pool, iso.ctx, principal_id="p1", chunk_id=me, backend=iso.auth)
    touched = {e["source_chunk_id"] for e in edges} | {e["target_chunk_id"] for e in edges}
    assert leak not in touched and touched <= set(iso.elig_ids)


async def test_AC5_07_entity_graph_and_ltm_views_are_unsupported_for_private_generations(pool):
    with pytest.raises(kc.KcGenerationError) as ei:
        kc.refuse_unsupported_private_view("entity graph")
    assert ei.value.code == "UNSUPPORTED_PRIVATE_OPERATION"


# ── legacy bypass closure + AC5-12 cross-namespace ──────────────────────────────────────────────


class _Req:
    def __init__(self):
        self.headers = {"authorization": "Bearer x"}
        self.method, self.url = "POST", type("U", (), {"path": "/v1/search"})()


async def test_AC5_12_legacy_routes_never_see_a_private_namespace_but_ordinary_ones_are_unchanged(pool):
    iso = await isolation_world(pool)
    ordinary = iso.foreign_ns
    principal = VerifiedPrincipal("p1", "service", frozenset({"kb:ltm.read"}))

    async def verify(_):
        return principal

    async def allowed(_):
        return frozenset({iso.w.namespace_id, ordinary})

    with patch.object(dep, "verify_bearer_token", verify), patch.object(dep, "resolve_allowed_namespaces", allowed):
        legacy = await dep.require_tenant_context(_Req())
        private = await dep.require_private_tenant_context(_Req())
    assert iso.w.namespace_id not in legacy.allowed_namespace_ids and ordinary in legacy.allowed_namespace_ids
    assert iso.w.namespace not in legacy.allowed_namespace_names
    assert iso.w.namespace_id in private.allowed_namespace_ids
    # the legacy fetch helpers, given the legacy allowed set, cannot read a private chunk
    assert await repository.fetch_chunks_by_chunk_id(iso.elig_ids[0], legacy.allowed_namespace_ids) == []
    assert await repository.get_edges_for_chunk(iso.elig_ids[0], legacy.allowed_namespace_ids) == []


async def test_AC5_12_foreign_principal_and_foreign_lease_get_identical_refusals_and_nothing_is_provisioned(pool):
    iso = await isolation_world(pool)
    tables = ("namespaces", "projects", "kc_corpus_scopes")
    before = [await count(pool, f"SELECT count(*) FROM {t}") for t in tables]
    # principal without a grant cannot open a context
    assert (
        await err_code(
            kc.begin_read(pool, iso.w.scope_id, principal_id="stranger", operation_id="op-x", backend=iso.auth)
        )
        == "AUTHORITY_REVOKED"
    )
    # a known lease id is not a bearer capability: unknown and foreign are the same refusal
    foreign = await err_code(kc.load_context(pool, iso.ctx.lease_id, principal_id="p2"))
    unknown = await err_code(kc.load_context(pool, str(uuid.uuid4()), principal_id="p2"))
    assert foreign == unknown == "LEASE_NOT_FOUND"
    forged = dataclasses.replace(iso.ctx, principal_id="p2")
    assert await err_code(kc.authorize_use(pool, forged, principal_id="p2", backend=iso.auth)) == "LEASE_NOT_FOUND"
    assert await kc.find_scope(pool, iso.w.namespace_id, iso.w.project_id, "no-such-purpose") is None
    assert await kc.find_scope(pool, iso.foreign_ns, iso.w.project_id, "pilot") is None
    assert before == [await count(pool, f"SELECT count(*) FROM {t}") for t in tables]


# ── one captured read context ───────────────────────────────────────────────────────────────────


async def test_R_CTX_01_begin_read_without_selection_refuses(pool, world):
    auth = SyntheticAuthority({"p1"})
    assert (
        await err_code(kc.begin_read(pool, world.scope_id, principal_id="p1", operation_id="op-n", backend=auth))
        == "NO_SELECTION"
    )


async def test_R_CTX_02_captured_generation_survives_promotion_and_new_operation_sees_new(pool):
    iso = await isolation_world(pool)
    w, auth = iso.w, iso.auth
    g3 = make_gen("g3", {"doc3": "# Three\n\nharbor lantern quartz zebra third generation text.\n"}, ["doc3"])
    await build_sealed(w, g3)
    await promote_simple(w, "g3", "op-g3", auth)  # promotion advances while a reader holds g1
    old = await kc.read_member_exact(pool, iso.ctx, principal_id="p1", member_id="doc", backend=auth)
    assert old["text"] == ELIG and iso.ctx.generation_id == "g1"
    assert (
        await err_code(kc.read_member_exact(pool, iso.ctx, principal_id="p1", member_id="doc3", backend=auth))
        == "NOT_FOUND"
    )
    fresh = await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-new", backend=auth)
    assert fresh.generation_id == "g3"
    assert (await kc.read_member_exact(pool, fresh, principal_id="p1", member_id="doc3", backend=auth))[
        "member_id"
    ] == "doc3"


async def test_R_CTX_03_promotion_in_the_middle_of_a_search_does_not_splice_generations(pool):
    iso = await isolation_world(pool)
    w, auth = iso.w, iso.auth
    g3 = make_gen("g3b", {"doc3": "# Three\n\nharbor lantern quartz zebra third generation text.\n"}, ["doc3"])
    await build_sealed(w, g3)
    calls = {"n": 0}

    async def gate():
        calls["n"] += 1
        if calls["n"] == 2:  # between candidate retrieval and the response/use commit
            await promote_simple(w, "g3b", "op-mid", auth)
        await kc.authorize_use(pool, iso.ctx, principal_id="p1", backend=auth)

    async def fake_embed(_):
        return embed(ELIG)

    async def fake_sparse(_):
        return [sparse(ELIG)]

    with patch.object(searcher, "embed_single", fake_embed), patch.object(searcher, "embed_sparse", fake_sparse):
        await gate()
        res = await searcher.search(
            "harbor lantern", w.namespace_id, limit=10, generation=iso.ctx.eligibility, pre_disclosure_check=gate
        )
    assert (
        await kc.get_selection(pool, w.scope_id)
    ).generation_id == "g3b"  # the promotion really happened mid-operation
    assert _ids(res.results) <= set(iso.elig_ids) and res.results  # ...but this operation stayed on the captured g1


async def test_R_CTX_04_lease_expiry_uses_database_time_and_refuses_even_with_cached_bytes(pool):
    w = await make_world(pool, lease_ttl_seconds=1)
    auth = SyntheticAuthority({"p1"})
    await build_sealed(w, make_gen("gl", {"doc": ELIG}, ["doc"]))
    await promote_simple(w, "gl", "op-gl", auth)
    ctx = await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-ttl", backend=auth)
    cached = await kc.read_member_exact(pool, ctx, principal_id="p1", member_id="doc", backend=auth)
    assert cached["text"] == ELIG
    await asyncio.sleep(1.4)
    assert await err_code(kc.authorize_use(pool, ctx, principal_id="p1", backend=auth)) == "LEASE_EXPIRED"
    assert (
        await err_code(kc.read_member_exact(pool, ctx, principal_id="p1", member_id="doc", backend=auth))
        == "LEASE_EXPIRED"
    )
    assert await err_code(kc.renew_lease(pool, ctx, principal_id="p1", backend=auth)) == "LEASE_EXPIRED"  # no revival
    assert (
        await err_code(kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-ttl", backend=auth))
        == "LEASE_EXPIRED"
    )


async def test_R_CTX_05_renewal_extends_a_valid_lease(pool):
    w = await make_world(pool, lease_ttl_seconds=2)
    auth = SyntheticAuthority({"p1"})
    await build_sealed(w, make_gen("gr", {"doc": ELIG}, ["doc"]))
    await promote_simple(w, "gr", "op-gr", auth)
    ctx = await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-rn", backend=auth)
    await asyncio.sleep(1.2)
    await kc.renew_lease(pool, ctx, principal_id="p1", backend=auth)
    await asyncio.sleep(1.2)  # past the ORIGINAL expiry, inside the renewed one
    assert (await kc.authorize_use(pool, ctx, principal_id="p1", backend=auth)).witness


async def test_R_CTX_06_released_lease_and_active_lease_limit(pool):
    w = await make_world(pool, max_active_leases=2)
    auth = SyntheticAuthority({"p1"})
    await build_sealed(w, make_gen("gm", {"doc": ELIG}, ["doc"]))
    await promote_simple(w, "gm", "op-gm", auth)
    c1 = await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-l1", backend=auth)
    await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-l2", backend=auth)
    assert (
        await err_code(kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-l3", backend=auth))
        == "LIMIT_EXCEEDED"
    )
    await kc.release_lease(pool, c1, principal_id="p1")
    assert await err_code(kc.authorize_use(pool, c1, principal_id="p1", backend=auth)) == "LEASE_NOT_FOUND"
    await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-l3", backend=auth)


# ── AC5-08 revocation timeline with the (synthetic) backend ordering witness ────────────────────


async def test_AC5_08_revocation_between_context_and_byte_read_blocks_the_read_and_cached_bodies(pool):
    iso = await isolation_world(pool)
    first = await kc.read_member_exact(pool, iso.ctx, principal_id="p1", member_id="doc", backend=iso.auth)
    iso.auth.revoke("p1")
    assert (
        await err_code(kc.read_member_exact(pool, iso.ctx, principal_id="p1", member_id="doc", backend=iso.auth))
        == "AUTHORITY_REVOKED"
    )
    assert (
        await err_code(kc.read_chunk(pool, iso.ctx, principal_id="p1", chunk_id=iso.elig_ids[0], backend=iso.auth))
        == "AUTHORITY_REVOKED"
    )
    assert first["text"] == ELIG  # bytes already disclosed cannot be recalled; nothing new is disclosed
    assert (
        await err_code(
            kc.begin_read(pool, iso.w.scope_id, principal_id="p1", operation_id="op-after", backend=iso.auth)
        )
        == "AUTHORITY_REVOKED"
    )


async def test_AC5_08_revocation_before_external_model_disclosure_stops_before_rerank(pool):
    iso = await isolation_world(pool)
    state = {"rerank_called": False, "checks": 0}

    async def fake_rerank(query, candidates, top_k):
        state["rerank_called"] = True
        return candidates

    async def fake_embed(_):
        return embed(ELIG)

    async def fake_sparse(_):
        return [sparse(ELIG)]

    async def gate():
        state["checks"] += 1
        await kc.authorize_use(pool, iso.ctx, principal_id="p1", backend=iso.auth)

    async def revoke_at_second_check(purpose):
        if len(iso.auth.calls) == 2:  # the pre-rerank (pre-disclosure) check
            iso.auth.revoke("p1")

    iso.auth.calls.clear()
    iso.auth.on_check = revoke_at_second_check
    with (
        patch.object(settings, "rerank_enabled", True),
        patch.object(searcher, "rerank", fake_rerank),
        patch.object(searcher, "embed_single", fake_embed),
        patch.object(searcher, "embed_sparse", fake_sparse),
    ):
        await gate()
        with pytest.raises(kc.KcGenerationError) as ei:
            await searcher.search(
                "harbor", iso.w.namespace_id, limit=5, generation=iso.ctx.eligibility, pre_disclosure_check=gate
            )
    assert ei.value.code == "AUTHORITY_REVOKED" and state["rerank_called"] is False


async def test_AC5_08_revocation_before_response_commit_prevents_the_response(pool):
    iso = await isolation_world(pool)

    async def fake_embed(_):
        return embed(ELIG)

    async def fake_sparse(_):
        return [sparse(ELIG)]

    async def gate():
        await kc.authorize_use(pool, iso.ctx, principal_id="p1", backend=iso.auth)

    iso.auth.calls.clear()

    async def revoke_at_final(purpose):
        if len(iso.auth.calls) == 2:  # rerank disabled -> the second check IS the response/use commit point
            iso.auth.revoke("p1")

    iso.auth.on_check = revoke_at_final
    with patch.object(searcher, "embed_single", fake_embed), patch.object(searcher, "embed_sparse", fake_sparse):
        await gate()
        with pytest.raises(kc.KcGenerationError) as ei:
            await searcher.search(
                "harbor", iso.w.namespace_id, limit=5, generation=iso.ctx.eligibility, pre_disclosure_check=gate
            )
    assert ei.value.code == "AUTHORITY_REVOKED"


async def test_AC5_08_authority_outage_is_unavailable_never_a_stale_allow(pool):
    iso = await isolation_world(pool)
    iso.auth.outage = True
    assert (
        await err_code(kc.read_member_exact(pool, iso.ctx, principal_id="p1", member_id="doc", backend=iso.auth))
        == "CURRENT_AUTHORITY_UNAVAILABLE"
    )
    kc.configure_authority_backend(None)  # default backend, nothing injected
    assert await err_code(kc.authorize_use(pool, iso.ctx, principal_id="p1")) == "CURRENT_AUTHORITY_UNAVAILABLE"


async def test_AC5_08_backend_ordering_witness_is_recorded_on_the_lease_and_operation(pool):
    iso = await isolation_world(pool)
    assert iso.ctx.authority["backend"] == "synthetic-test-authority" and iso.ctx.authority["witness"].startswith(
        "synthetic-ordering:"
    )
    op = await pool.fetchval(
        "SELECT authority_ref FROM kc_selection_operations WHERE scope_id=$1::uuid AND operation_id='op-g1'",
        iso.w.scope_id,
    )
    assert json.loads(op)["witness"].startswith("synthetic-ordering:")


# ── T5: retention / GC / lock order ─────────────────────────────────────────────────────────────


async def _two_generations(pool, ttl=300):
    w = await make_world(pool, lease_ttl_seconds=ttl)
    auth = SyntheticAuthority({"p1"})
    await build_sealed(w, make_gen("ga", {"doc": ELIG}, ["doc"]))
    await build_sealed(w, make_gen("gb", {"doc": ELIG + "more\n"}, ["doc"]))
    await promote_simple(w, "ga", "op-ga", auth)
    return w, auth


async def test_R_GC_01_selected_generation_and_valid_leases_retain_their_membership(pool):
    w, auth = await _two_generations(pool)
    assert await err_code(kc.mark_gc_pending(pool, w.scope_id, "ga")) == "GC_REFUSED"  # selected
    ctx = await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-r1", backend=auth)
    await promote_simple(w, "gb", "op-gb", auth)  # ga is now unselected but still leased
    assert await err_code(kc.mark_gc_pending(pool, w.scope_id, "ga")) == "GC_REFUSED"
    assert (await kc.read_member_exact(pool, ctx, principal_id="p1", member_id="doc", backend=auth))["text"] == ELIG
    await kc.release_lease(pool, ctx, principal_id="p1")
    assert await kc.mark_gc_pending(pool, w.scope_id, "ga") == "gc_pending"


async def test_R_GC_02_destructive_gc_is_disabled_and_nothing_is_deleted(pool):
    w, auth = await _two_generations(pool)
    await promote_simple(w, "gb", "op-gb", auth)
    await kc.mark_gc_pending(pool, w.scope_id, "ga")
    tables = ("chunks", "evidence_documents", "sparse_vectors", "kc_generations", "kc_generation_members")
    before = [await count(pool, f"SELECT count(*) FROM {t}") for t in tables]
    assert kc.DESTRUCTIVE_GC_ENABLED is False
    assert await err_code(kc.collect_generation(pool, w.scope_id, "ga")) == "GC_DISABLED"
    assert before == [await count(pool, f"SELECT count(*) FROM {t}") for t in tables]


async def test_R_GC_03_gc_pending_generation_cannot_be_selected_or_leased(pool):
    w, auth = await _two_generations(pool)
    await promote_simple(w, "gb", "op-gb", auth)
    await kc.mark_gc_pending(pool, w.scope_id, "ga")
    sel = await kc.get_selection(pool, w.scope_id)
    code = await err_code(
        kc.promote(
            pool,
            w.scope_id,
            operation_id="op-back",
            candidate_generation_id="ga",
            expected_incarnation=sel.incarnation_id,
            expected_revision=sel.revision,
            expected_generation=sel.generation_id,
            principal_id="p1",
            backend=auth,
        )
    )
    assert code == "GENERATION_NOT_SERVABLE"


async def test_R_GC_04_renewal_first_then_gc_refuses(pool):
    w, auth = await _two_generations(pool)
    ctx = await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-rf", backend=auth)
    await promote_simple(w, "gb", "op-gb", auth)
    await kc.renew_lease(pool, ctx, principal_id="p1", backend=auth)
    assert await err_code(kc.mark_gc_pending(pool, w.scope_id, "ga")) == "GC_REFUSED"
    assert (await kc.authorize_use(pool, ctx, principal_id="p1", backend=auth)).witness


async def test_R_GC_05_gc_first_then_renewal_of_an_expired_lease_cannot_revive_it(pool):
    w, auth = await _two_generations(pool, ttl=1)
    ctx = await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id="op-gf", backend=auth)
    await promote_simple(w, "gb", "op-gb", auth)
    await asyncio.sleep(1.4)  # lease expired by DB time
    assert await kc.mark_gc_pending(pool, w.scope_id, "ga") == "gc_pending"
    assert await err_code(kc.renew_lease(pool, ctx, principal_id="p1", backend=auth)) == "LEASE_EXPIRED"
    assert await err_code(kc.authorize_use(pool, ctx, principal_id="p1", backend=auth)) == "LEASE_EXPIRED"


async def test_R_GC_06_renew_and_gc_race_never_both_succeed(pool):
    both = 0
    for i in range(12):
        w, auth = await _two_generations(pool)
        ctx = await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id=f"op-race{i}", backend=auth)
        await promote_simple(w, "gb", f"op-gb{i}", auth)
        gate = await pool.acquire()
        tx = gate.transaction()
        await tx.start()
        await gate.execute(
            "SELECT 1 FROM kc_generations WHERE scope_id=$1::uuid AND generation_id='ga' FOR UPDATE", w.scope_id
        )
        t_renew = asyncio.create_task(kc.renew_lease(pool, ctx, principal_id="p1", backend=auth))
        t_gc = asyncio.create_task(kc.mark_gc_pending(pool, w.scope_id, "ga"))
        await asyncio.sleep(0.3)  # both are parked on the generation row
        await tx.commit()
        await pool.release(gate)
        r_renew, r_gc = await asyncio.gather(t_renew, t_gc, return_exceptions=True)
        ok_renew, ok_gc = not isinstance(r_renew, Exception), not isinstance(r_gc, Exception)
        assert not isinstance(r_renew, kc.KcGenerationError) or r_renew.code in {
            "GENERATION_NOT_SERVABLE",
            "LEASE_NOT_FOUND",
        }
        assert not isinstance(r_gc, kc.KcGenerationError) or r_gc.code == "GC_REFUSED"
        both += ok_renew and ok_gc
        state = await pool.fetchval(
            "SELECT state FROM kc_generations WHERE scope_id=$1::uuid AND generation_id='ga'", w.scope_id
        )
        # invariant: a lease that was renewed successfully always protects its generation
        assert not (ok_renew and state == "gc_pending")
    assert both == 0


async def test_R_GC_07_mixed_workload_follows_one_lock_order_without_deadlocks(pool):
    w, auth = await _two_generations(pool)
    dead_before = await pool.fetchval("SELECT deadlocks FROM pg_stat_database WHERE datname=current_database()")
    ctxs = [
        await kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id=f"op-m{i}", backend=auth)
        for i in range(6)
    ]
    cands = []
    for i in range(4):
        g = make_gen(f"gx{i}", {"doc": ELIG + f"variant {i}\n"}, ["doc"])
        await kc.create_candidate(pool, w.scope_id, g.manifest, admission_ref="a", profile_ref="p")
        cands.append(g)

    async def seal(g):
        await stage_all(w, g)
        return await kc.seal_generation(pool, w.scope_id, g.generation_id, oracle_closure={"doc"})

    async def prom(i):
        for _ in range(3):
            try:
                await promote_simple(w, "gb", f"op-mix-{i}-{uuid.uuid4().hex[:6]}", auth)
                return
            except kc.KcGenerationError as e:
                if e.code != "SELECTION_CONFLICT":
                    raise

    ops = (
        [seal(g) for g in cands]
        + [prom(i) for i in range(4)]
        + [kc.renew_lease(pool, c, principal_id="p1", backend=auth) for c in ctxs]
        + [kc.begin_read(pool, w.scope_id, principal_id="p1", operation_id=f"op-nb{i}", backend=auth) for i in range(4)]
        + [kc.mark_gc_pending(pool, w.scope_id, "ga") for _ in range(2)]
    )
    results = await asyncio.gather(*ops, return_exceptions=True)
    bad = [
        r
        for r in results
        if isinstance(r, Exception)
        and not (
            isinstance(r, kc.KcGenerationError)
            and r.code in {"GC_REFUSED", "SELECTION_CONFLICT", "GENERATION_NOT_SERVABLE"}
        )
    ]
    assert not bad, bad
    assert not [r for r in results if isinstance(r, kc.KcGenerationError) and r.code == "LOCK_FAILURE"]
    assert await pool.fetchval("SELECT deadlocks FROM pg_stat_database WHERE datname=current_database()") == dead_before


# ── plan evidence ───────────────────────────────────────────────────────────────────────────────


async def test_R_PLAN_01_actual_plan_reads_the_materialized_eligible_relation_in_every_arm(pool):
    import os
    from contextlib import asynccontextmanager
    from pathlib import Path

    iso = await isolation_world(pool)
    recorded: dict[str, str] = {}
    real = repository.acquire_search_connection

    @asynccontextmanager
    async def spy(p):
        async with real(p) as conn:

            class Proxy:
                def __getattr__(self, name):
                    return getattr(conn, name)

                async def fetch(self, sql, *args):
                    plan = await conn.fetch("EXPLAIN (COSTS OFF) " + sql, *args)
                    recorded["plan"] = "\n".join(r[0] for r in plan)
                    recorded["plan_cache_mode"] = await conn.fetchval("SHOW plan_cache_mode")
                    return await conn.fetch(sql, *args)

            yield Proxy()

    with patch.object(repository, "acquire_search_connection", spy):
        await repository.hybrid_search(
            embed(CANARY),
            "canaryonlytoken",
            iso.w.namespace_id,
            limit=5,
            query_sparse=sparse(CANARY),
            generation=iso.ctx.eligibility,
        )
    import re

    plan = re.sub(r"'\[[-0-9.e, ]+\]'::vector", "'[<1024-dim synthetic query vector>]'::vector", recorded["plan"])
    assert recorded["plan_cache_mode"] == "force_custom_plan"  # the existing custom-plan setting is preserved
    assert "CTE eligible" in plan
    assert plan.count("CTE Scan on eligible") == 3  # semantic + full-text + sparse arms read the eligible relation
    assert plan.count("on chunks c") >= 1  # the base table is touched by the eligible CTE and the final id join only
    out = os.environ.get("KC_EVIDENCE_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "plan-hybrid-3way-eligible.txt").write_text(plan + "\n")


async def test_R_CTX_07_exact_fetch_rehashes_stored_bytes_at_read_time(pool):
    iso = await isolation_world(pool)
    b = await kc.get_member_binding(pool, iso.w.scope_id, "g1", "doc")
    # a faulty writer got past the backstop and altered the sealed body after sealing
    async with pool.acquire() as c, c.transaction():
        await c.execute("SET LOCAL session_replication_role = replica")
        await c.execute(
            "UPDATE evidence_documents SET raw_content = raw_content || 'x' WHERE namespace_id=$1 AND source_path=$2",
            b.namespace_id,
            b.source_path,
        )
    code = await err_code(kc.read_member_exact(pool, iso.ctx, principal_id="p1", member_id="doc", backend=iso.auth))
    assert code == "CONTENT_MISMATCH"  # never repaired from another generation, never returned
