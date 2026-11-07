"""KB-CONS-01-N: seal (AC5-01..06), CAS selection / ABA / lost-ack (AC5-09/10/11), authority refusal (SYNTHETIC)."""

from __future__ import annotations

import asyncio
import unicodedata

import asyncpg
import pytest

from scrutator.db import connection
from scrutator.db import kc_generations as kc
from tests.kc_generations.helpers import (
    SyntheticAuthority,
    build_sealed,
    count,
    make_gen,
    promote_simple,
    stage_all,
    stage_one,
)
from tests.kc_generations.test_records_and_staging import EDGES, TEXTS, _gen, err_code, tamper

# ── AC5-01 complete generation / exact readback ─────────────────────────────────────────────────


async def test_AC5_01_complete_generation_seal_report_has_exact_bytes_and_hashes(pool, world):
    gen = _gen("g-ok")
    report = await build_sealed(world, gen)
    assert report["closure"] == sorted(TEXTS) and len(report["members"]) == 3
    by = {m["member_id"]: m for m in report["members"]}
    alpha = TEXTS["root-a"].encode("utf-8")
    assert len(alpha) > len(TEXTS["root-a"])  # multibyte: byte count != character count
    assert by["root-a"]["bytes"] == len(alpha)
    assert by["shared"]["chunk_count"] >= 1 and report["total_bytes"] == sum(len(t.encode()) for t in TEXTS.values())
    state = await pool.fetchval(
        "SELECT state FROM kc_generations WHERE scope_id=$1::uuid AND generation_id='g-ok'", world.scope_id
    )
    assert state == "sealed"


# ── AC5-02 missing member ───────────────────────────────────────────────────────────────────────


async def test_AC5_02_missing_member_refused_before_selection(pool, world):
    gen = _gen("g-miss")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    await stage_all(world, gen, skip={"shared"})
    with pytest.raises(kc.KcGenerationError) as ei:
        await kc.seal_generation(pool, world.scope_id, "g-miss", oracle_closure=TEXTS.keys())
    assert ei.value.code == "SEAL_REFUSED" and ei.value.detail["reason"] == "MISSING_MEMBER"
    auth = SyntheticAuthority({"p1"})
    assert await err_code(promote_simple(world, "g-miss", "op-miss", auth)) == "GENERATION_NOT_SERVABLE"
    assert (await kc.get_selection(pool, world.scope_id)).revision == 0


async def test_AC5_02_missing_body_row_detected_by_fresh_readback_at_seal(pool, world):
    gen = _gen("g-miss2")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    await stage_all(world, gen)
    b = await kc.get_member_binding(pool, world.scope_id, "g-miss2", "shared")
    await tamper(
        pool, "DELETE FROM evidence_documents WHERE namespace_id=$1 AND source_path=$2", b.namespace_id, b.source_path
    )
    with pytest.raises(kc.KcGenerationError) as ei:
        await kc.seal_generation(pool, world.scope_id, "g-miss2", oracle_closure=TEXTS.keys())
    assert ei.value.detail["reason"] == "MISSING_BODY"  # metadata alone does not pass


# ── AC5-03 stale digest / revision ──────────────────────────────────────────────────────────────


async def test_AC5_03_old_valid_bytes_do_not_satisfy_a_new_selected_member(pool, world):
    old = _gen("g-old")
    await build_sealed(world, old)
    new_texts = {**TEXTS, "shared": "# Shared\n\nShared dependency, REVISION TWO text.\n"}
    new = make_gen("g-new", new_texts, ["root-a", "root-b"], EDGES, revisions={"shared": "2"})
    await kc.create_candidate(pool, world.scope_id, new.manifest, admission_ref="a", profile_ref="p")
    b = await kc.get_member_binding(pool, world.scope_id, "g-new", "shared")
    stale = TEXTS["shared"]  # independently valid (hash-consistent) OLD bytes
    chunks = kc.prepare_member_chunks(b.namespace, b.source_path, stale)
    code = await err_code(
        kc.stage_member(
            pool,
            world.scope_id,
            "g-new",
            "shared",
            body=stale.encode(),
            chunks=chunks,
            embeddings=[[0.0] * 1024] * len(chunks),
            sparse_weights=[{}] * len(chunks),
            producer="p",
        )
    )
    assert code == "MEMBER_MISMATCH"


async def test_AC5_03_mismatched_chunk_producer_stamp_refused_at_stage(pool, world):
    gen = _gen("g-prod")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    b = await kc.get_member_binding(pool, world.scope_id, "g-prod", "shared")
    chunks = kc.prepare_member_chunks(b.namespace, b.source_path, TEXTS["shared"])
    chunks[0] = {**chunks[0], "content_hash": "0" * 64}  # a valid-looking stamp that does not match the chunk bytes
    code = await err_code(
        kc.stage_member(
            pool,
            world.scope_id,
            "g-prod",
            "shared",
            body=TEXTS["shared"].encode(),
            chunks=chunks,
            embeddings=[[0.0] * 1024] * len(chunks),
            sparse_weights=[{}] * len(chunks),
            producer="p",
        )
    )
    assert code == "MEMBER_INVALID"


# ── AC5-04 tampered bytes ───────────────────────────────────────────────────────────────────────


async def test_AC5_04_mutated_raw_body_with_unchanged_stamp_rejected_at_seal(pool, world):
    gen = _gen("g-tamp")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    await stage_all(world, gen)
    b = await kc.get_member_binding(pool, world.scope_id, "g-tamp", "root-b")
    await tamper(
        pool,
        "UPDATE evidence_documents SET raw_content=replace(raw_content,'Beta','Bata') WHERE namespace_id=$1 AND source_path=$2",  # noqa: E501
        b.namespace_id,
        b.source_path,
    )
    with pytest.raises(kc.KcGenerationError) as ei:
        await kc.seal_generation(pool, world.scope_id, "g-tamp", oracle_closure=TEXTS.keys())
    assert ei.value.detail["reason"] == "BODY"


async def test_AC5_04_mutated_chunk_body_with_still_valid_stamp_rejected_at_seal(pool, world):
    gen = _gen("g-tamp2")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    await stage_all(world, gen)
    b = await kc.get_member_binding(pool, world.scope_id, "g-tamp2", "shared")
    # content changed; the stored content_hash column (stamp) and the whole-document stamp stay "valid"
    await tamper(
        pool,
        "UPDATE chunks SET content=content || ' tampered' WHERE namespace_id=$1 AND source_path=$2 AND chunk_index=0",
        b.namespace_id,
        b.source_path,
    )
    with pytest.raises(kc.KcGenerationError) as ei:
        await kc.seal_generation(pool, world.scope_id, "g-tamp2", oracle_closure=TEXTS.keys())
    assert ei.value.detail["reason"] == "CHUNK_BYTES"


async def test_AC5_04_chunk_set_digest_catches_a_consistent_chunk_swap(pool, world):
    gen = _gen("g-tamp3")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    await stage_all(world, gen)
    b = await kc.get_member_binding(pool, world.scope_id, "g-tamp3", "shared")
    # a faulty writer rewrites content AND its stamp consistently: per-chunk hashes agree, set digest does not
    await tamper(
        pool,
        "UPDATE chunks SET content='swapped', content_hash=encode(sha256('swapped'::bytea),'hex') WHERE namespace_id=$1 AND source_path=$2 AND chunk_index=0",  # noqa: E501
        b.namespace_id,
        b.source_path,
    )
    with pytest.raises(kc.KcGenerationError) as ei:
        await kc.seal_generation(pool, world.scope_id, "g-tamp3", oracle_closure=TEXTS.keys())
    assert ei.value.detail["reason"] == "CHUNK_SET_DIGEST"


@pytest.mark.parametrize(
    "variant",
    [
        lambda t: unicodedata.normalize("NFD", t),  # canonically equivalent Unicode, different bytes
        lambda t: t.replace("\n", "\r\n"),  # newline normalisation
        lambda t: t + " ",
    ],
    ids=["nfd", "crlf", "trailing-space"],
)
async def test_AC5_04_no_normalisation_hides_a_difference(pool, world, variant):
    gen = _gen("g-norm" + str(id(variant) % 9999))
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    b = await kc.get_member_binding(pool, world.scope_id, gen.generation_id, "root-a")
    changed = variant(TEXTS["root-a"])
    assert changed != TEXTS["root-a"]
    chunks = kc.prepare_member_chunks(b.namespace, b.source_path, changed)
    code = await err_code(
        kc.stage_member(
            pool,
            world.scope_id,
            gen.generation_id,
            "root-a",
            body=changed.encode(),
            chunks=chunks,
            embeddings=[[0.0] * 1024] * len(chunks),
            sparse_weights=[{}] * len(chunks),
            producer="p",
        )
    )
    assert code == "MEMBER_MISMATCH"


# ── AC5-05 wrong / withdrawn admission, mixed revisions ─────────────────────────────────────────


async def test_AC5_05_mixed_member_revisions_across_generations_cannot_be_combined(pool, world):
    g1 = make_gen("g-r1", {"only": "# One\n\nrevision one\n"}, ["only"], revisions={"only": "1"})
    g2 = make_gen("g-r2", {"only": "# One\n\nrevision two\n"}, ["only"], revisions={"only": "2"})
    await build_sealed(world, g1)
    await kc.create_candidate(pool, world.scope_id, g2.manifest, admission_ref="a", profile_ref="p")
    b2 = await kc.get_member_binding(pool, world.scope_id, "g-r2", "only")
    ch = kc.prepare_member_chunks(b2.namespace, b2.source_path, "# One\n\nrevision one\n")
    code = await err_code(
        kc.stage_member(
            pool,
            world.scope_id,
            "g-r2",
            "only",
            body=b"# One\n\nrevision one\n",
            chunks=ch,
            embeddings=[[0.0] * 1024] * len(ch),
            sparse_weights=[{}] * len(ch),
            producer="p",
        )
    )
    assert code == "MEMBER_MISMATCH"  # rev-1 bytes cannot stand in for rev 2 of the selected manifest


async def test_AC5_05_withdrawn_admission_refused_at_promotion_and_at_current_use(pool, world):
    await build_sealed(world, _gen("g-adm"))
    auth = SyntheticAuthority({"p1"})
    auth.revoke("p1")
    assert await err_code(promote_simple(world, "g-adm", "op-adm", auth)) == "AUTHORITY_REVOKED"
    assert (await kc.get_selection(pool, world.scope_id)).revision == 0


# ── AC5-06 closure vs independent oracle ────────────────────────────────────────────────────────


async def test_AC5_06_closure_must_equal_the_independent_oracle(pool, world):
    gen = _gen("g-orc")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    await stage_all(world, gen)
    with pytest.raises(kc.KcGenerationError) as ei:
        await kc.seal_generation(pool, world.scope_id, "g-orc", oracle_closure={"root-a", "root-b"})  # drops shared
    assert ei.value.detail["reason"] == "ORACLE"
    assert (
        await pool.fetchval(
            "SELECT state FROM kc_generations WHERE scope_id=$1::uuid AND generation_id='g-orc'", world.scope_id
        )
        == "staging"
    )


async def test_AC5_06_assembler_that_omits_a_required_edge_is_refused(pool, world):
    # the assembler's own manifest drops the shared dependency of root-b: members != required closure
    texts = {"root-a": TEXTS["root-a"], "root-b": TEXTS["root-b"], "shared": TEXTS["shared"]}
    gen = make_gen("g-edge", texts, ["root-a", "root-b"], [("root-a", "shared")])  # shared still reachable via root-a
    await build_sealed(world, gen)  # shared is retained once, via the shared reference
    texts2 = {
        "root-a": TEXTS["root-a"],
        "root-b": TEXTS["root-b"],
        "shared": TEXTS["shared"],
        "orphan": "# O\n\norphan\n",
    }
    gen2 = make_gen("g-orph", texts2, ["root-a", "root-b"], EDGES)
    await kc.create_candidate(pool, world.scope_id, gen2.manifest, admission_ref="a", profile_ref="p")
    await stage_all(world, gen2)
    with pytest.raises(kc.KcGenerationError) as ei:
        await kc.seal_generation(pool, world.scope_id, "g-orph", oracle_closure=texts2.keys())
    assert ei.value.detail["reason"] == "CLOSURE"  # an unreferenced member is not required closure


async def test_AC5_06_shared_dependency_retained_once_and_cycles_terminate(pool, world):
    gen = make_gen("g-cyc", {"a": "# A\n\na\n", "b": "# B\n\nb\n"}, ["a"], [("a", "b"), ("b", "a")])
    report = await build_sealed(world, gen)
    assert report["closure"] == ["a", "b"]


# ── selection: one native row, CAS, operations in the same transaction ──────────────────────────


async def test_R_SEL_01_promote_commits_selection_and_operation_in_one_transaction(pool, world):
    await build_sealed(world, _gen("g-p1"))
    auth = SyntheticAuthority({"p1"})
    r = await promote_simple(world, "g-p1", "op-1", auth)
    assert (r.outcome, r.committed.revision, r.is_current) == ("committed", 1, True)
    sel = await kc.get_selection(pool, world.scope_id)
    assert (sel.revision, sel.generation_id) == (1, "g-p1") and sel.incarnation_id == r.committed.incarnation_id
    op = await pool.fetchrow(
        "SELECT * FROM kc_selection_operations WHERE scope_id=$1::uuid AND operation_id='op-1'", world.scope_id
    )
    assert op["committed_revision"] == 1 and op["candidate_generation_id"] == "g-p1"
    assert r.authority["witness"].startswith("synthetic-ordering:")  # the backend's ordering witness is recorded


async def test_R_SEL_02_pointer_only_update_without_operation_row_cannot_commit(pool, world):
    await build_sealed(world, _gen("g-ptr"))
    sel = await kc.get_selection(pool, world.scope_id)
    manifest_sha = await pool.fetchval(
        "SELECT manifest_sha256 FROM kc_generations WHERE scope_id=$1::uuid AND generation_id='g-ptr'", world.scope_id
    )
    with pytest.raises(asyncpg.PostgresError) as ei:
        async with pool.acquire() as c:
            await c.execute(
                "UPDATE kc_selections SET generation_id='g-ptr', manifest_sha256=$2, revision=revision+1 WHERE scope_id=$1::uuid",  # noqa: E501
                world.scope_id,
                manifest_sha,
            )
    assert "KC_SELECTION_OPERATION_MISSING" in str(ei.value)
    assert (await kc.get_selection(pool, world.scope_id)) == sel


async def test_R_SEL_03_revision_cannot_reset_skip_and_incarnation_cannot_be_overwritten(pool, world):
    await build_sealed(world, _gen("g-rv"))
    await promote_simple(world, "g-rv", "op-rv", SyntheticAuthority({"p1"}))
    for sql in (
        "UPDATE kc_selections SET revision=0 WHERE scope_id=$1::uuid",
        "UPDATE kc_selections SET revision=revision+2 WHERE scope_id=$1::uuid",
        "UPDATE kc_selections SET incarnation_id=gen_random_uuid() WHERE scope_id=$1::uuid",
        "DELETE FROM kc_selections WHERE scope_id=$1::uuid",
    ):
        with pytest.raises(asyncpg.PostgresError):
            async with pool.acquire() as c:
                await c.execute(sql, world.scope_id)


async def test_R_SEL_04_unsealed_candidate_cannot_be_selected(pool, world):
    gen = _gen("g-uns")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    assert (
        await err_code(promote_simple(world, "g-uns", "op-uns", SyntheticAuthority({"p1"})))
        == "GENERATION_NOT_SERVABLE"
    )


async def test_R_SEL_05_wrong_prior_generation_with_right_revision_conflicts(pool, world):
    await build_sealed(world, _gen("g-pg1"))
    await build_sealed(world, make_gen("g-pg2", {"x": "# X\n\nx\n"}, ["x"]))
    auth = SyntheticAuthority({"p1"})
    await promote_simple(world, "g-pg1", "op-pg1", auth)
    sel = await kc.get_selection(pool, world.scope_id)
    code = await err_code(
        kc.promote(
            pool,
            world.scope_id,
            operation_id="op-pg2",
            candidate_generation_id="g-pg2",
            expected_incarnation=sel.incarnation_id,
            expected_revision=sel.revision,
            expected_generation="g-WRONG",
            principal_id="p1",
            backend=auth,
        )
    )
    assert code == "SELECTION_CONFLICT"
    assert (await kc.get_selection(pool, world.scope_id)).revision == 1


async def test_R_SEL_06_stale_token_conflicts_and_never_silently_retries(pool, world):
    await build_sealed(world, _gen("g-st1"))
    await build_sealed(world, make_gen("g-st2", {"x": "# X\n\nx\n"}, ["x"]))
    auth = SyntheticAuthority({"p1"})
    sel0 = await kc.get_selection(pool, world.scope_id)
    await promote_simple(world, "g-st1", "op-st1", auth)
    with pytest.raises(kc.KcGenerationError) as ei:
        await kc.promote(
            pool,
            world.scope_id,
            operation_id="op-st2",
            candidate_generation_id="g-st2",
            expected_incarnation=sel0.incarnation_id,
            expected_revision=0,
            expected_generation=None,
            principal_id="p1",
            backend=auth,
        )
    assert ei.value.code == "SELECTION_CONFLICT" and ei.value.detail["current"]["revision"] == "1"


# ── AC5-09 concurrent CAS race + ABA + incarnation ──────────────────────────────────────────────


async def _blocked_backends(pool, n: int, timeout=10.0) -> None:
    for _ in range(int(timeout / 0.05)):
        waiting = await count(
            pool,
            "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock' AND state='active'",  # noqa: E501
        )
        if waiting >= n:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("contenders never both blocked on the selection row")


async def test_AC5_09_two_real_connections_one_cas_winner(pool, world):
    await build_sealed(world, _gen("g-ra"))
    await build_sealed(world, make_gen("g-rb", {"x": "# X\n\nx\n"}, ["x"]))
    auth = SyntheticAuthority({"p1"})
    sel = await kc.get_selection(pool, world.scope_id)

    def contender(gid, op):
        return kc.promote(
            pool,
            world.scope_id,
            operation_id=op,
            candidate_generation_id=gid,
            expected_incarnation=sel.incarnation_id,
            expected_revision=sel.revision,
            expected_generation=None,
            principal_id="p1",
            backend=auth,
        )

    holder = await pool.acquire()
    tx = holder.transaction()
    await tx.start()
    await holder.execute("SELECT 1 FROM kc_selections WHERE scope_id=$1::uuid FOR UPDATE", world.scope_id)
    t1 = asyncio.create_task(contender("g-ra", "op-ra"))
    t2 = asyncio.create_task(contender("g-rb", "op-rb"))
    await _blocked_backends(pool, 2)  # barrier: both contenders are in flight from the SAME expected token
    await tx.commit()
    await pool.release(holder)
    results = await asyncio.gather(t1, t2, return_exceptions=True)
    wins = [r for r in results if isinstance(r, kc.PromotionResult)]
    losers = [r for r in results if isinstance(r, kc.KcGenerationError)]
    assert len(wins) == 1 and len(losers) == 1 and losers[0].code == "SELECTION_CONFLICT"
    assert (await kc.get_selection(pool, world.scope_id)).revision == 1
    assert (
        await count(pool, "SELECT count(*) FROM kc_selection_operations WHERE scope_id=$1::uuid", world.scope_id) == 1
    )


async def test_AC5_09_obsolete_token_fails_after_A_to_B_to_A(pool, world):
    await build_sealed(world, _gen("g-A"))
    await build_sealed(world, make_gen("g-B", {"x": "# X\n\nx\n"}, ["x"]))
    auth = SyntheticAuthority({"p1"})
    await promote_simple(world, "g-A", "op-1", auth)
    tokenA1 = await kc.get_selection(pool, world.scope_id)  # (rev 1, A)
    await promote_simple(world, "g-B", "op-2", auth)
    await promote_simple(world, "g-A", "op-3", auth)
    cur = await kc.get_selection(pool, world.scope_id)
    assert cur.generation_id == "g-A" and cur.revision == 3  # value is back to A, revision is not
    with pytest.raises(kc.KcGenerationError) as ei:  # the first A token must NOT succeed again
        await kc.promote(
            pool,
            world.scope_id,
            operation_id="op-4",
            candidate_generation_id="g-B",
            expected_incarnation=tokenA1.incarnation_id,
            expected_revision=tokenA1.revision,
            expected_generation=tokenA1.generation_id,
            principal_id="p1",
            backend=auth,
        )
    assert ei.value.code == "SELECTION_CONFLICT"


async def test_AC5_09_scope_incarnation_mismatch_refuses(pool, world):
    await build_sealed(world, _gen("g-inc"))
    auth = SyntheticAuthority({"p1"})
    sel = await kc.get_selection(pool, world.scope_id)
    other = "00000000-0000-4000-8000-000000000001"
    with pytest.raises(kc.KcGenerationError) as ei:
        await kc.promote(
            pool,
            world.scope_id,
            operation_id="op-inc",
            candidate_generation_id="g-inc",
            expected_incarnation=other,
            expected_revision=sel.revision,
            expected_generation=None,
            principal_id="p1",
            backend=auth,
        )
    assert ei.value.code == "SELECTION_CONFLICT"


def test_R_SEL_07_revision_overflow_and_noncanonical_tokens_refused():
    for bad in ("9223372036854775808", "-1", "01", "1.0", "", True, 1.5, None, 2**63):
        with pytest.raises(kc.KcGenerationError) as ei:
            kc.parse_revision(bad)
        assert ei.value.code == "TOKEN_INVALID"
    assert kc.parse_revision("9223372036854775807") == 2**63 - 1 and kc.parse_revision(7) == 7


# ── lost ack / operation outcome ────────────────────────────────────────────────────────────────


async def test_AC5_11_lost_ack_retry_with_same_operation_reconciles_exactly_once(pool, world):
    await build_sealed(world, _gen("g-ack"))
    auth = SyntheticAuthority({"p1"})
    sel = await kc.get_selection(pool, world.scope_id)

    def call():
        return kc.promote(
            pool,
            world.scope_id,
            operation_id="op-ack",
            candidate_generation_id="g-ack",
            expected_incarnation=sel.incarnation_id,
            expected_revision=sel.revision,
            expected_generation=None,
            principal_id="p1",
            backend=auth,
        )

    first = await call()  # the client "never received" this result
    again = await call()  # same operation id + exact arguments
    assert (first.outcome, again.outcome) == ("committed", "replayed")
    assert again.committed == first.committed and again.is_current
    assert (await kc.get_selection(pool, world.scope_id)).revision == 1  # exactly one logical acceptance
    assert (
        await count(pool, "SELECT count(*) FROM kc_selection_operations WHERE scope_id=$1::uuid", world.scope_id) == 1
    )


async def test_AC5_11_replayed_old_operation_is_not_presented_as_current_after_a_later_promotion(pool, world):
    await build_sealed(world, _gen("g-h1"))
    await build_sealed(world, make_gen("g-h2", {"x": "# X\n\nx\n"}, ["x"]))
    auth = SyntheticAuthority({"p1"})
    sel = await kc.get_selection(pool, world.scope_id)
    kwargs = dict(
        operation_id="op-h1",
        candidate_generation_id="g-h1",
        expected_incarnation=sel.incarnation_id,
        expected_revision=sel.revision,
        expected_generation=None,
        principal_id="p1",
        backend=auth,
    )
    await kc.promote(pool, world.scope_id, **kwargs)
    await promote_simple(world, "g-h2", "op-h2", auth)
    replay = await kc.promote(pool, world.scope_id, **kwargs)
    assert replay.outcome == "replayed" and replay.is_current is False
    assert replay.committed.generation_id == "g-h1" and replay.current.generation_id == "g-h2"


async def test_AC5_11_same_operation_id_with_different_arguments_refuses(pool, world):
    await build_sealed(world, _gen("g-d1"))
    await build_sealed(world, make_gen("g-d2", {"x": "# X\n\nx\n"}, ["x"]))
    auth = SyntheticAuthority({"p1"})
    sel = await kc.get_selection(pool, world.scope_id)
    base = dict(
        operation_id="op-dd",
        expected_incarnation=sel.incarnation_id,
        expected_revision=sel.revision,
        expected_generation=None,
        principal_id="p1",
        backend=auth,
    )
    await kc.promote(pool, world.scope_id, candidate_generation_id="g-d1", **base)
    with pytest.raises(kc.KcGenerationError) as ei:
        await kc.promote(pool, world.scope_id, candidate_generation_id="g-d2", **base)
    assert ei.value.code == "OPERATION_CONFLICT"
    assert (await kc.get_selection(pool, world.scope_id)).generation_id == "g-d1"


async def test_AC5_11_two_simultaneous_identical_requests_create_one_selection_commit(pool, world):
    await build_sealed(world, _gen("g-sim"))
    auth = SyntheticAuthority({"p1"})
    sel = await kc.get_selection(pool, world.scope_id)

    def call():
        return kc.promote(
            pool,
            world.scope_id,
            operation_id="op-sim",
            candidate_generation_id="g-sim",
            expected_incarnation=sel.incarnation_id,
            expected_revision=sel.revision,
            expected_generation=None,
            principal_id="p1",
            backend=auth,
        )

    rs = await asyncio.gather(call(), call(), call())
    assert sorted(r.outcome for r in rs) == ["committed", "replayed", "replayed"]
    assert (await kc.get_selection(pool, world.scope_id)).revision == 1


async def test_AC5_11_read_only_validation_writes_nothing(pool, world):
    gen = _gen("g-ro")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    await stage_all(world, gen, skip={"shared"})
    tables = (
        "chunks",
        "evidence_documents",
        "sparse_vectors",
        "kc_generations",
        "kc_generation_members",
        "kc_selections",
        "kc_selection_operations",
        "kc_read_leases",
    )

    async def snapshot():
        return [await count(pool, f"SELECT count(*) FROM {t}") for t in tables]

    before = await snapshot()
    kc.parse_manifest(gen.manifest, max_members=64, max_manifest_bytes=10**6, max_total_bytes=10**9)  # pure validator
    with pytest.raises(kc.KcGenerationError):
        await kc.seal_generation(pool, world.scope_id, "g-ro", oracle_closure=TEXTS.keys())  # refused seal
    assert await snapshot() == before
    assert (
        await pool.fetchval(
            "SELECT state FROM kc_generations WHERE scope_id=$1::uuid AND generation_id='g-ro'", world.scope_id
        )
        == "staging"
    )


# ── T4: authority ───────────────────────────────────────────────────────────────────────────────


async def test_R_AUTH_01_with_no_real_authority_promotion_refuses_and_changes_nothing(pool, world):
    await build_sealed(world, _gen("g-na"))
    kc.configure_authority_backend(None)  # the shipped default
    code = await err_code(promote_simple(world, "g-na", "op-na", None))
    assert code == "CURRENT_AUTHORITY_UNAVAILABLE"
    assert (await kc.get_selection(pool, world.scope_id)).revision == 0
    assert (
        await count(pool, "SELECT count(*) FROM kc_selection_operations WHERE scope_id=$1::uuid", world.scope_id) == 0
    )


async def test_R_AUTH_02_authority_outage_and_malformed_observations_are_unavailable_not_allow(pool, world):
    await build_sealed(world, _gen("g-out"))
    auth = SyntheticAuthority({"p1"})
    auth.outage = True
    assert await err_code(promote_simple(world, "g-out", "op-o1", auth)) == "CURRENT_AUTHORITY_UNAVAILABLE"

    class NoWitness:
        name = "x"

        async def check(self, conn, *, principal_id, scope_id, purpose):
            return kc.AuthorityObservation("x", principal_id, scope_id, purpose, "", "")

    assert await err_code(promote_simple(world, "g-out", "op-o2", NoWitness())) == "CURRENT_AUTHORITY_UNAVAILABLE"

    class Hang:
        name = "h"

        async def check(self, conn, *, principal_id, scope_id, purpose):
            await asyncio.sleep(60)

    old = kc.AUTHORITY_TIMEOUT_S
    kc.AUTHORITY_TIMEOUT_S = 0.2
    try:
        assert await err_code(promote_simple(world, "g-out", "op-o3", Hang())) == "CURRENT_AUTHORITY_UNAVAILABLE"
    finally:
        kc.AUTHORITY_TIMEOUT_S = old
    assert (await kc.get_selection(pool, world.scope_id)).revision == 0


async def test_R_AUTH_03_revocation_before_selection_prevents_promotion(pool, world):
    await build_sealed(world, _gen("g-rv1"))
    auth = SyntheticAuthority({"p1"})
    auth.revoke("p1")
    assert await err_code(promote_simple(world, "g-rv1", "op-rv1", auth)) == "AUTHORITY_REVOKED"


# ── AC5-10 interrupt / restart ──────────────────────────────────────────────────────────────────


async def test_AC5_10_interrupted_staging_never_becomes_visible_and_prior_selection_survives(pool, world, kc_dsn):
    await build_sealed(world, _gen("g-prev"))
    auth = SyntheticAuthority({"p1"})
    await promote_simple(world, "g-prev", "op-prev", auth)
    nxt = make_gen("g-half", {"a": "# A\n\nnew a\n", "b": "# B\n\nnew b\n"}, ["a", "b"])
    await kc.create_candidate(pool, world.scope_id, nxt.manifest, admission_ref="a", profile_ref="p")
    await stage_one(world, "g-half", "a", nxt.texts["a"])  # "crash" after a subset of members
    assert (
        await err_code(kc.seal_generation(pool, world.scope_id, "g-half", oracle_closure={"a", "b"})) == "SEAL_REFUSED"
    )
    assert await err_code(promote_simple(world, "g-half", "op-half", auth)) == "GENERATION_NOT_SERVABLE"
    sel = await kc.get_selection(pool, world.scope_id)
    assert (sel.generation_id, sel.revision) == ("g-prev", 1)
    # restart: a brand-new pool observes one complete selection and the preserved previous bytes
    p2 = await asyncpg.create_pool(kc_dsn, min_size=1, max_size=2, init=connection._init_connection)
    rc = await kc.begin_read(p2, world.scope_id, principal_id="p1", operation_id="op-read-after-restart", backend=auth)
    assert rc.generation_id == "g-prev"
    got = await kc.read_member_exact(p2, rc, principal_id="p1", member_id="shared", backend=auth)
    assert got["text"] == TEXTS["shared"]
    await p2.close()


async def test_AC5_10_backend_killed_before_cas_commit_preserves_prior_selection(pool, world):
    await build_sealed(world, _gen("g-k1"))
    await build_sealed(world, make_gen("g-k2", {"x": "# X\n\nx\n"}, ["x"]))
    auth = SyntheticAuthority({"p1"})
    await promote_simple(world, "g-k1", "op-k1", auth)
    sel = await kc.get_selection(pool, world.scope_id)
    holder = await pool.acquire()
    tx = holder.transaction()
    await tx.start()
    await holder.execute("SELECT 1 FROM kc_selections WHERE scope_id=$1::uuid FOR UPDATE", world.scope_id)
    task = asyncio.create_task(
        kc.promote(
            pool,
            world.scope_id,
            operation_id="op-k2",
            candidate_generation_id="g-k2",
            expected_incarnation=sel.incarnation_id,
            expected_revision=sel.revision,
            expected_generation=sel.generation_id,
            principal_id="p1",
            backend=auth,
        )
    )
    await _blocked_backends(pool, 1)
    pid = await pool.fetchval(
        "SELECT pid FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock' AND state='active' LIMIT 1"  # noqa: E501
    )
    await pool.execute("SELECT pg_terminate_backend($1)", pid)  # the promoting backend dies before it can commit
    with pytest.raises((asyncpg.PostgresError, asyncpg.InterfaceError, ConnectionError)):
        await task
    await tx.rollback()
    await pool.release(holder)
    after = await kc.get_selection(pool, world.scope_id)
    assert (after.generation_id, after.revision) == ("g-k1", 1)
    assert (
        await count(
            pool,
            "SELECT count(*) FROM kc_selection_operations WHERE scope_id=$1::uuid AND operation_id='op-k2'",
            world.scope_id,
        )
        == 0
    )


async def test_AC5_10_seal_interrupted_leaves_generation_staging_and_unselectable(pool, world):
    gen = _gen("g-sk")
    await kc.create_candidate(pool, world.scope_id, gen.manifest, admission_ref="a", profile_ref="p")
    await stage_all(world, gen)
    holder = await pool.acquire()
    tx = holder.transaction()
    await tx.start()
    await holder.execute(
        "SELECT 1 FROM kc_generations WHERE scope_id=$1::uuid AND generation_id='g-sk' FOR UPDATE", world.scope_id
    )
    task = asyncio.create_task(kc.seal_generation(pool, world.scope_id, "g-sk", oracle_closure=TEXTS.keys()))
    await _blocked_backends(pool, 1)
    pid = await pool.fetchval(
        "SELECT pid FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock' AND state='active' LIMIT 1"  # noqa: E501
    )
    await pool.execute("SELECT pg_terminate_backend($1)", pid)
    with pytest.raises((asyncpg.PostgresError, asyncpg.InterfaceError, ConnectionError)):
        await task
    await tx.rollback()
    await pool.release(holder)
    row = await pool.fetchrow(
        "SELECT state, seal_report FROM kc_generations WHERE scope_id=$1::uuid AND generation_id='g-sk'", world.scope_id
    )
    assert row["state"] == "staging" and row["seal_report"] is None
    # a fresh seal now succeeds from the durable staged state
    report = await kc.seal_generation(pool, world.scope_id, "g-sk", oracle_closure=TEXTS.keys())
    assert report["generation_id"] == "g-sk"


async def test_R_SEL_08_revision_skip_is_refused_even_when_an_operation_row_for_it_exists(pool, world):
    await build_sealed(world, _gen("g-skip"))
    await promote_simple(world, "g-skip", "op-skip1", SyntheticAuthority({"p1"}))
    sel = await kc.get_selection(pool, world.scope_id)
    msha = await pool.fetchval(
        "SELECT manifest_sha256 FROM kc_generations WHERE scope_id=$1::uuid AND generation_id='g-skip'", world.scope_id
    )
    with pytest.raises(asyncpg.PostgresError) as ei:
        async with pool.acquire() as c, c.transaction():
            # the commit-time operation check is satisfied on purpose: only the monotonic-revision guard can refuse
            await c.execute(
                "INSERT INTO kc_selection_operations (scope_id, operation_id, request_sha256, candidate_generation_id, "
                "candidate_manifest_sha256, expected_incarnation, expected_revision, committed_incarnation, "
                "committed_revision, principal_id, authority_ref) VALUES ($1::uuid,'op-forged',repeat('a',64),'g-skip',$2,"  # noqa: E501
                "$3::uuid,1,$3::uuid,3,'p1','{}'::jsonb)",
                world.scope_id,
                msha,
                sel.incarnation_id,
            )
            await c.execute("UPDATE kc_selections SET revision=3 WHERE scope_id=$1::uuid", world.scope_id)
    assert "exactly one" in str(ei.value)
    assert (await kc.get_selection(pool, world.scope_id)).revision == 1
