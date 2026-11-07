"""KB-CONS-01-M AC5 MODEL-ONLY cases (synthetic bytes, in-memory, no DB/network/deployed instance).

Each case: (a) the safe model passes, (b) where a faulty variant exists, the *checker* must
REJECT it and name the first divergent invariant. Results of (b) are recorded in FAULTY_RESULTS
for REPORT.md. A PASS here is a model result only, never a live AC5 result.
"""

from __future__ import annotations

import threading

import pytest

from . import generation_model as gm
from .generation_model import (
    Authz,
    Denied,
    InjectedFailure,
    Manifest,
    NotServable,
    Store,
    ingest_and_promote,
    make_member,
    sha256,
)

NS, PROJ, READER, SRC1, SRC2 = "synthetic-ns", "synthetic-proj", "reader-A", "pin-0001", "pin-0002"

OLD = {"doc-a": b"OLD-A synthetic bytes", "doc-b": b"OLD-B synthetic bytes", "doc-c": b"OLD-C synthetic bytes"}
NEW = {"doc-a": b"NEW-A synthetic bytes!", "doc-b": b"NEW-B synthetic bytes!", "doc-c": b"NEW-C synthetic bytes!"}

FAULTY_RESULTS: dict[str, dict] = {}


def manifest(
    gen: str, payload: dict[str, bytes], src: str = SRC1, project: str = PROJ, ns: str = NS, deps=None
) -> Manifest:
    deps = deps or {}
    return Manifest(
        gen,
        src,
        project,
        ns,
        tuple(make_member(k, v, revision=gen, deps=deps.get(k, ())) for k, v in payload.items()),
    )


def build(cls=Store, *, with_g1: bool = True):
    authz = Authz()
    authz.grant(READER, NS)
    store = cls(authz, NS, {SRC1, SRC2})
    g1 = manifest("g1", OLD)
    if with_g1:
        ingest_and_promote(store, g1, OLD, READER, PROJ)
    return store, g1


def record(case: str, variant: str, verdict: gm.Verdict) -> None:
    FAULTY_RESULTS[f"{case}/{variant}"] = {
        "rejected": not verdict.ok,
        "invariant": verdict.invariant,
        "detail": verdict.detail,
    }
    assert not verdict.ok, f"{case}: checker accepted faulty variant {variant}"


def stage_and_try(store: Store, m: Manifest, payload, project=PROJ):
    try:
        ingest_and_promote(store, m, payload, READER, project)
        return None
    except NotServable as e:
        return e


# --- case 1 -------------------------------------------------------------------------------
def test_c01_complete_generation_and_byte_readback():
    store, g1 = build()
    g2 = manifest("g2", NEW)
    ingest_and_promote(store, g2, NEW, READER, PROJ)
    assert store.ptr == "g2"
    for k, v in NEW.items():
        assert store.read_member("g2", k, READER) == v
    assert gm.check_pointer_is_complete_generation(store).ok
    assert gm.check_previous_generation_intact(store, g1, OLD).ok  # g1 bytes still retained
    assert gm.check_view_has_only_accepted(store, READER).ok


# --- cases 2,3,4,5,6,7: validation refusals; faulty = validation-free promoter --------------
def _refusal(case, m, payload, boundary, project=PROJ, mutate=None):
    store, g1 = build()
    if mutate:
        mutate(store)
    err = stage_and_try(store, m, payload, project)
    assert err is not None and err.boundary == boundary, err
    assert store.ptr == "g1"  # previous complete pointer unchanged
    assert gm.check_previous_generation_intact(store, g1, OLD).ok
    assert gm.check_view_has_only_accepted(store, READER).ok
    # faulty: promotes without validating -> pointer names an unverifiable generation
    fstore, _ = build(gm.NoValidationStore)
    if mutate:
        mutate(fstore)
    stage_and_try(fstore, m, payload, project)
    record(
        case,
        "NoValidationStore",
        gm.check_pointer_is_complete_generation(fstore)
        if boundary in ("missing-member", "digest-mismatch", "length-mismatch")
        else _scope_verdict(fstore, m, project),
    )


def _scope_verdict(store: Store, m: Manifest, project: str) -> gm.Verdict:
    """I7: pointer must not name a generation from a non-accepted source / wrong project / missing dep."""
    if store.ptr == m.generation_id:
        if m.namespace != store.namespace:
            return gm.Verdict(False, "I7:pointer-namespace-is-own", m.namespace)
        if m.source_pin not in store.accepted_sources:
            return gm.Verdict(False, "I7:pointer-source-is-accepted-current", m.source_pin)
        if project != m.project:
            return gm.Verdict(False, "I7:pointer-project-matches-request", f"{project}!={m.project}")
        ids = {x.artifact_id for x in m.members}
        for mem in m.members:
            for d in mem.deps:
                if d not in ids:
                    return gm.Verdict(False, "I7:dependency-closure-complete", f"{mem.artifact_id}->{d}")
    return gm.Verdict(True)


def test_c02_one_member_missing():
    partial = {k: v for k, v in NEW.items() if k != "doc-b"}
    m = manifest("g2", NEW)
    _refusal("c02", m, partial, "missing-member")


def test_c03_stale_digest():
    # stored bytes are a previous (stale) revision; manifest pins the new digest
    stale = dict(NEW, **{"doc-b": b"OLD-B synthetic bytes"})
    m = manifest("g2", NEW)
    _refusal("c03", m, stale, "length-mismatch")  # stale content differs in length from pinned
    # same-length stale variant -> digest boundary
    stale2 = dict(NEW, **{"doc-b": b"STALE-B synthetic bytes"[: len(NEW["doc-b"])].ljust(len(NEW["doc-b"]), b"!")})
    _refusal("c03b", m, stale2, "digest-mismatch")


def test_c04_tampered_bytes_same_length():
    tampered = dict(NEW, **{"doc-c": NEW["doc-c"][:-1] + b"?"})
    assert len(tampered["doc-c"]) == len(NEW["doc-c"])
    _refusal("c04", manifest("g2", NEW), tampered, "digest-mismatch")


def test_c05_non_current_or_foreign_source():
    m = manifest("g2", NEW, src="pin-unaccepted")
    _refusal("c05", m, NEW, "non-current-source")
    foreign = manifest("g2", NEW, ns="other-ns")
    _refusal("c05b", foreign, NEW, "foreign-namespace")


def test_c06_missing_dependency():
    m = manifest("g2", NEW, deps={"doc-a": ("doc-zzz",)})
    _refusal("c06", m, NEW, "missing-dependency")


def test_c07_wrong_project():
    _refusal("c07", manifest("g2", NEW), NEW, "wrong-project", project="other-proj")


# --- case 8: revoked reader ----------------------------------------------------------------
def test_c08_revoked_reader_denied_even_with_cached_bytes():
    store, g1 = build()
    assert store.accepted_view(READER)
    store.authz.revoke(READER, NS)
    with pytest.raises(Denied):
        store.accepted_view(READER)
    with pytest.raises(Denied):
        store.read_member("g1", "doc-a", READER)
    assert all(store.blobs[store.storage_key(g1, m)] for m in g1.members)  # bytes retained, not served
    assert gm.check_revoked_denied(store, READER).ok
    # previous cached generation after a NEW promotion, then revocation
    store.authz.grant(READER, NS)
    g2 = manifest("g2", NEW)
    ingest_and_promote(store, g2, NEW, READER, PROJ)
    store.authz.revoke(READER, NS)
    with pytest.raises(Denied):
        store.read_member("g1", "doc-a", READER)  # previous generation, still retained, still denied
    # re-grant is the only way back (current authorization, not digest/identity)
    store.authz.grant(READER, NS)
    assert store.read_member("g1", "doc-a", READER) == OLD["doc-a"]
    # faulty: serves cached complete generation without re-check
    f, _ = build(gm.CachedViewStore)
    f.authz.revoke(READER, NS)
    record("c08", "CachedViewStore", gm.check_revoked_denied(f, READER))


# --- case 9: CAS race ----------------------------------------------------------------------
def _race(cls):
    store, _ = build(cls)
    ga, gb = manifest("ga", NEW), manifest("gb", {k: v + b"-B" for k, v in NEW.items()})
    pa, pb = NEW, {k: v + b"-B" for k, v in NEW.items()}
    exp = (store.ptr, store.version)  # both competitors observed the same current pointer
    for m, p in ((ga, pa), (gb, pb)):
        store.stage(m, p)
        store.validate(m, READER, PROJ)
    results: list = []
    for g in ("ga", "gb"):  # deterministic interleaving: both validated before either promotes
        try:
            store.promote(g, *exp)
            results.append(g)
        except NotServable as e:
            results.append(e)
    return store, results


def test_c09_competing_promotion_cas_single_winner():
    store, results = _race(Store)
    assert gm.check_single_winner(store, results, ["ga", "gb"]).ok
    assert store.ptr == "ga" and isinstance(results[1], NotServable) and results[1].boundary == "cas-conflict"
    # full generation retained & never mixed
    assert gm.check_pointer_is_complete_generation(store).ok
    fstore, fres = _race(gm.LastWriterWinsStore)
    record("c09", "LastWriterWinsStore", gm.check_single_winner(fstore, fres, ["ga", "gb"]))


def test_c09b_threaded_race_has_exactly_one_winner():
    for _ in range(50):
        store, _ = build()
        exp = (store.ptr, store.version)
        gens = [manifest(f"t{i}", {k: v + bytes([65 + i]) for k, v in NEW.items()}) for i in range(4)]
        for i, m in enumerate(gens):
            store.stage(m, {k: v + bytes([65 + i]) for k, v in NEW.items()})
            store.validate(m, READER, PROJ)  # R3: selection requires a validation witness
        out: list = [None] * 4
        bar = threading.Barrier(4)

        def run(i):
            bar.wait()
            try:
                store.promote(f"t{i}", *exp)
                out[i] = f"t{i}"
            except NotServable as e:
                out[i] = e

        ts = [threading.Thread(target=run, args=(i,)) for i in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert gm.check_single_winner(store, out, [f"t{i}" for i in range(4)]).ok


# --- case 10: interrupted / rollback (77307111 counterexample) -------------------------------
def _interrupted(cls):
    store, g1 = build(cls)
    g2 = manifest("g2", NEW)
    store.fail_after_writes = 1  # first new member written, then crash
    with pytest.raises(InjectedFailure):
        ingest_and_promote(store, g2, NEW, READER, PROJ)
    return store, g1


def test_c10_interrupted_after_first_new_member_previous_generation_intact():
    store, g1 = _interrupted(Store)
    assert store.ptr == "g1"  # pointer unchanged
    assert gm.check_previous_generation_intact(store, g1, OLD).ok
    for k, v in OLD.items():  # every previous member reads at its OLD digest
        assert sha256(store.read_member("g1", k, READER)) == sha256(v)
    assert gm.check_view_has_only_accepted(store, READER).ok  # half-written g2 not visible
    # one new member really was written (the fault was injected where intended)
    assert sum(1 for k in store.blobs if k[1].startswith("g2/")) == 1
    # faulty in-place variant: first new member overwrote the old one
    fstore, fg1 = _interrupted(gm.InPlaceStore)
    assert fstore.ptr == "g1"  # pointer alone is unchanged ...
    record("c10", "InPlaceStore", gm.check_previous_generation_intact(fstore, fg1, OLD))  # ... bytes are not
    # and the half-written candidate leaks into the accepted view
    lstore, _ = _interrupted(gm.LeakyViewStore)
    record("c10", "LeakyViewStore", gm.check_view_has_only_accepted(lstore, READER))


def test_c10b_recovery_after_interruption_retry_completes():
    store, g1 = _interrupted(Store)
    store.fail_after_writes = None
    g2 = manifest("g2", NEW)
    ingest_and_promote(store, g2, NEW, READER, PROJ)
    assert store.ptr == "g2" and gm.check_pointer_is_complete_generation(store).ok
    assert gm.check_previous_generation_intact(store, g1, OLD).ok


# --- case 11: repeated idempotent check ------------------------------------------------------
def test_c11_repeated_check_is_idempotent_and_read_only():
    store, g1 = build()
    g2 = manifest("g2", NEW)
    store.stage(g2, NEW)
    snap = (dict(store.blobs), store.ptr, store.version)
    for _ in range(3):
        store.validate(g2, READER, PROJ)
    assert (dict(store.blobs), store.ptr, store.version) == snap  # validation is read-only
    # repeated stage of identical bytes is a no-op at the byte level
    store.stage(g2, NEW)
    assert store.blobs == snap[0]
    ingest_and_promote(store, manifest("g3", NEW), NEW, READER, PROJ)
    v = store.version
    # retry of an already-won promotion against a stale expectation is refused, not duplicated
    with pytest.raises(NotServable) as e:
        store.promote("g3", "g1", v - 1)
    assert e.value.boundary == "cas-conflict" and store.version == v


# --- case 12: no cross-namespace reads/writes ------------------------------------------------
class _CrossNsStore(Store):
    def _write(self, key, data, op="write"):
        super()._write(key, data, op)
        self.ns_log.append(("write", "shared-ns"))  # unsafe: mirrors write into another namespace


def test_c12_no_cross_namespace_reads_or_writes():
    store, g1 = build()
    ingest_and_promote(store, manifest("g2", NEW), NEW, READER, PROJ)
    store.accepted_view(READER)
    assert gm.check_namespaces(store).ok and store.ns_log
    assert {n for _, n in store.ns_log} == {NS}
    f, _ = build(_CrossNsStore)
    record("c12", "CrossNsStore", gm.check_namespaces(f))
    # foreign manifest is refused before any read of the foreign namespace
    s2, _ = build()
    n = len(s2.ns_log)
    assert stage_and_try(s2, manifest("gx", NEW, ns="other-ns"), NEW).boundary == "foreign-namespace"
    assert all(x[1] == NS for x in s2.ns_log[n:])


# --- AC4/AC2 explicit: candidate isolation -------------------------------------------------
def test_ac4_incomplete_candidate_never_visible_in_accepted_view():
    store, _ = build()
    g2 = manifest("g2", NEW)
    store.stage(g2, {"doc-a": NEW["doc-a"]})  # incomplete candidate is "searchable" in storage
    assert set(store.accepted_view(READER)) == set(OLD)
    lstore, _ = build(gm.LeakyViewStore)
    lstore.stage(g2, {"doc-a": NEW["doc-a"]})
    record("ac4", "LeakyViewStore", gm.check_view_has_only_accepted(lstore, READER))


# --- R1: immutable generation identity ---------------------------------------------------------
def _selected_g1_with_snapshot(cls=Store):
    store, g1 = build(cls)
    return store, g1, store.snapshot()


def test_r1a_same_id_different_manifest_refused_before_any_state_change():
    store, g1, snap = _selected_g1_with_snapshot()
    with pytest.raises(NotServable) as e:
        store.stage(manifest("g1", NEW), NEW)  # same ID "g1", different manifest + bytes
    assert e.value.boundary == "immutable-identity-conflict"
    assert gm.check_state_unchanged(store, snap).ok
    assert gm.check_selected_identity_immutable(store, g1, OLD).ok
    assert store.ptr == "g1" and store.read_member("g1", "doc-a", READER) == OLD["doc-a"]
    # mutant: pre-correction staging overwrites the selected generation's manifest and bytes
    f, fg1, _ = _selected_g1_with_snapshot(gm.OverwriteStore)
    f.stage(manifest("g1", NEW), NEW)
    assert f.ptr == "g1"  # pointer and version are untouched ...
    record("r1a", "OverwriteStore", gm.check_selected_identity_immutable(f, fg1, OLD))  # ... identity is not


def test_r1b_same_manifest_different_bytes_refused_before_any_state_change():
    store, g1, snap = _selected_g1_with_snapshot()
    with pytest.raises(NotServable) as e:
        store.stage(g1, NEW)  # g1's own manifest, different bytes for every member
    assert e.value.boundary == "immutable-identity-conflict"
    assert gm.check_state_unchanged(store, snap).ok
    assert gm.check_previous_generation_intact(store, g1, OLD).ok
    # one conflicting member among otherwise identical ones is still refused, nothing written
    mixed = dict(OLD, **{"doc-c": b"OLD-C synthetic BYTES"})
    with pytest.raises(NotServable):
        store.stage(g1, mixed)
    assert gm.check_state_unchanged(store, snap).ok
    f, fg1, _ = _selected_g1_with_snapshot(gm.OverwriteStore)
    f.stage(fg1, NEW)
    record("r1b", "OverwriteStore", gm.check_selected_identity_immutable(f, fg1, OLD))


def test_r1c_interrupted_same_id_rewrite_changes_nothing():
    store, g1, snap = _selected_g1_with_snapshot()
    store.fail_after_writes = 1  # would crash after the first write, if a write were attempted
    with pytest.raises(NotServable) as e:  # refused BEFORE the injected failure can matter
        store.stage(manifest("g1", NEW), NEW)
    assert e.value.boundary == "immutable-identity-conflict"
    assert gm.check_state_unchanged(store, snap).ok
    assert gm.check_selected_identity_immutable(store, g1, OLD).ok
    f, fg1, _ = _selected_g1_with_snapshot(gm.OverwriteStore)
    f.fail_after_writes = 1
    with pytest.raises(InjectedFailure):
        f.stage(manifest("g1", NEW), NEW)
    assert f.ptr == "g1" and f.version == 1
    record("r1c", "OverwriteStore", gm.check_selected_identity_immutable(f, fg1, OLD))


def test_r1d_conflict_on_staged_candidate_is_prechecked_for_all_members():
    store, _ = build()
    store.stage(manifest("g2", NEW), {"doc-c": NEW["doc-c"]})  # partial candidate g2
    snap = store.snapshot()
    other = dict(NEW, **{"doc-c": b"DIFFERENT-C synthetic"})
    with pytest.raises(NotServable):  # doc-a/doc-b are missing and would be written first
        store.stage(manifest("g2", other), other)
    # manifest also differs (digest of doc-c) -> refused at manifest level; use same manifest instead
    m2 = manifest("g2", NEW)
    with pytest.raises(NotServable) as e:
        store.stage(m2, dict(NEW, **{"doc-c": b"DIFFERENT-C synthetic"}))
    assert e.value.boundary == "immutable-identity-conflict"
    assert gm.check_state_unchanged(store, snap).ok  # doc-a and doc-b were NOT written
    f, _ = build(gm.CheckAsYouGoStore)
    f.stage(manifest("g2", NEW), {"doc-c": NEW["doc-c"]})
    fsnap = f.snapshot()
    with pytest.raises(NotServable):
        f.stage(m2, dict(NEW, **{"doc-c": b"DIFFERENT-C synthetic"}))
    record("r1d", "CheckAsYouGoStore", gm.check_state_unchanged(f, fsnap))


def test_r1f_same_id_different_manifest_without_byte_collision_refused_alone():
    """The manifest-level guard must refuse on its own: here every payload byte that exists under
    the stored keys is identical, so the byte pre-check cannot be what refuses the call."""
    store, g1, snap = _selected_g1_with_snapshot()
    extra = dict(OLD, **{"doc-d": b"OLD-D synthetic bytes"})  # extra member, no stored blob under its key
    dropped = {k: v for k, v in OLD.items() if k != "doc-c"}  # dropped member
    redeps = manifest("g1", OLD, deps={"doc-a": ("doc-b",)})  # changed deps, identical bytes
    cases = [(manifest("g1", extra), extra), (manifest("g1", dropped), dropped), (redeps, OLD)]
    for m, payload in cases:
        assert m != g1
        with pytest.raises(NotServable) as e:
            store.stage(m, payload)
        assert e.value.boundary == "immutable-identity-conflict"
        assert "manifest g1" in str(e.value)  # refused at the manifest guard, not the byte guard
        assert gm.check_state_unchanged(store, snap).ok
        assert gm.check_selected_identity_immutable(store, g1, OLD).ok
    # mutant: manifest guard removed (byte check kept) lets the manifest be replaced
    f, fg1, _ = _selected_g1_with_snapshot(gm.NoManifestGuardStore)
    f.stage(manifest("g1", extra), extra)
    assert f.manifests["g1"] != fg1
    record("r1f", "NoManifestGuardStore", gm.check_state_unchanged(f, snap))


def test_r3_validate_and_promote_guards_exercised_alone():
    """Direct calls so each guard is the only thing that can refuse (found by the mutation sweep)."""
    store, _ = build()
    # validate: foreign namespace is refused on its own (stage() is not in the path)
    with pytest.raises(NotServable) as e:
        store.validate(manifest("gx", NEW, ns="other-ns"), READER, PROJ)
    assert e.value.boundary == "foreign-namespace"
    # validate: authorization is checked now, on an otherwise fully valid staged candidate
    g2 = manifest("g2", NEW)
    store.stage(g2, NEW)
    store.authz.revoke(READER, NS)
    with pytest.raises(NotServable) as e:
        store.validate(g2, READER, PROJ)
    assert e.value.boundary == "authorization"
    store.authz.grant(READER, NS)
    # validate reads every member through the scoped reader and logs it (namespace audit trail)
    n = len(store.ns_log)
    store.validate(g2, READER, PROJ)
    assert store.ns_log[n:] == [("read", NS)] * len(g2.members)
    # promote: the promoted generation leaves the candidate set and is recorded as ever selected
    assert "g2" in store.candidates and "g2" not in store.ever_selected
    store.promote("g2", store.ptr, store.version)
    assert "g2" not in store.candidates and "g2" in store.ever_selected


def test_r1e_exact_retry_only_observes_or_fills_missing():
    store, g1, snap = _selected_g1_with_snapshot()
    store.stage(g1, OLD)  # exact retry of the selected generation: no-op
    assert gm.check_state_unchanged(store, snap).ok  # also not re-added to candidates
    # partial candidate, then exact retry fills only the missing members
    g2 = manifest("g2", NEW)
    store.stage(g2, {"doc-a": NEW["doc-a"]})
    before = dict(store.blobs)
    store.stage(g2, NEW)
    assert store.blobs[store.storage_key(g2, g2.members[0])] == before[store.storage_key(g2, g2.members[0])]
    assert {k for k in store.blobs if k[1].startswith("g2/")} == {store.storage_key(g2, m) for m in g2.members}
    assert sum(1 for k in store.ns_log if k[0] == "write") == 3 + 3  # g1 x3 + g2 x3: one write per member, no rewrite
    ingest_and_promote(store, g2, NEW, READER, PROJ)
    assert store.ptr == "g2" and gm.check_pointer_is_complete_generation(store).ok


# --- R2: bounded ABA trace, pointer-only mutant ----------------------------------------------------
def _aba(cls):
    store, _ = build(cls)  # g1 selected, version 1
    obs = (store.ptr, store.version)  # competitor A observed (g1, 1) and stages ga meanwhile
    ga = manifest("ga", {k: v + b"-A" for k, v in NEW.items()})
    store.stage(ga, {k: v + b"-A" for k, v in NEW.items()})
    store.validate(ga, READER, PROJ)
    g2 = manifest("g2", NEW)
    ingest_and_promote(store, g2, NEW, READER, PROJ)  # B: g1 -> g2 (version 2)
    store.promote("g1", "g2", 2)  # C: g2 -> g1 again (version 3): pointer VALUE is back to g1
    assert (store.ptr, store.version) == ("g1", 3)
    try:
        store.promote("ga", *obs)  # A acts on its stale observation (g1, 1)
        outcome: str | Exception = "ga"
    except NotServable as e:
        outcome = e
    return store, outcome


def test_r2a_aba_stale_observation_refused_by_version():
    store, outcome = _aba(Store)
    assert gm.check_stale_observation_refused(outcome).ok
    assert isinstance(outcome, NotServable) and outcome.boundary == "cas-conflict"
    assert (store.ptr, store.version) == ("g1", 3)  # ga never became the pointer
    f, fout = _aba(gm.PointerOnlyCasStore)
    assert f.ptr == "ga"  # the mutant silently accepts the stale observation
    record("r2a", "PointerOnlyCasStore", gm.check_stale_observation_refused(fout))


# --- R2: served read of a staged generation ---------------------------------------------------------
def test_r2b_served_read_refuses_staged_generation_diagnostic_read_is_labelled():
    store, _ = build()
    store.stage(manifest("g2", NEW), NEW)  # staged, validated or not, never selected
    assert gm.check_staged_not_served(store, "g2", "doc-a", READER).ok
    with pytest.raises(NotServable) as e:
        store.read_member("g2", "doc-a", READER)
    assert e.value.boundary == "not-selected-generation"
    # DIAGNOSTIC-ONLY path exists, is explicit by name, and is NOT the served path
    assert store.diagnostic_read_member("g2", "doc-a", READER) == NEW["doc-a"]
    store.authz.revoke(READER, NS)
    with pytest.raises(Denied):
        store.diagnostic_read_member("g2", "doc-a", READER)
    f, _ = build(gm.UnguardedReadStore)
    f.stage(manifest("g2", NEW), NEW)
    record("r2b", "UnguardedReadStore", gm.check_staged_not_served(f, "g2", "doc-a", READER))


# --- R2: transitive dependency chain (measured subcase) ----------------------------------------------
def test_r2c_transitive_dependency_chain_gap_refused():
    chain = {"doc-a": ("doc-b",), "doc-b": ("doc-c",)}
    partial = {k: v for k, v in NEW.items() if k != "doc-c"}
    m = Manifest(
        "g2",
        SRC1,
        PROJ,
        NS,
        tuple(make_member(k, v, revision="g2", deps=chain.get(k, ())) for k, v in NEW.items() if k != "doc-c"),
    )
    # doc-a -> doc-b -> doc-c with doc-c absent from the manifest: the gap is two hops from the root
    store, _ = build()
    err = stage_and_try(store, m, partial)
    assert err is not None and err.boundary == "missing-dependency" and "doc-b->doc-c" in str(err)
    assert store.ptr == "g1"
    f, _ = build(gm.RootOnlyDepStore)
    stage_and_try(f, m, partial)
    record("r2c", "RootOnlyDepStore", _scope_verdict(f, m, PROJ))


# --- R3 (round 4): direct promote requires an immutable validation witness --------------------------
def _direct_promote(store: Store, gen: str, ptr=None, ver=None):
    ptr = store.ptr if ptr is None else ptr
    ver = store.version if ver is None else ver
    try:
        store.promote(gen, ptr, ver)
        return gen
    except NotServable as e:
        return e


def test_r4a_direct_promote_of_missing_generation_refused():
    store, _ = build()
    snap = store.snapshot()
    out = _direct_promote(store, "not-registered")
    assert isinstance(out, NotServable) and out.boundary == "unregistered-generation"
    assert gm.check_promote_refused_unchanged(store, snap, out).ok
    assert (store.ptr, store.version) == ("g1", 1)
    f, _ = build(gm.NoValidationFenceStore)
    fsnap = f.snapshot()
    record(
        "r4a",
        "NoValidationFenceStore",
        gm.check_promote_refused_unchanged(f, fsnap, _direct_promote(f, "not-registered")),
    )
    assert f.ptr == "not-registered"  # the mutant really moved the pointer to a missing manifest


def test_r4b_direct_promote_of_incomplete_staged_generation_refused():
    store, _ = build()
    g2 = manifest("g2", NEW)
    store.stage(g2, {"doc-a": NEW["doc-a"]})  # partial: doc-b, doc-c never staged
    snap = store.snapshot()
    out = _direct_promote(store, "g2")
    assert isinstance(out, NotServable) and out.boundary == "unvalidated-candidate"
    assert gm.check_promote_refused_unchanged(store, snap, out).ok
    # validate cannot rescue it: still incomplete, so no witness is ever issued
    with pytest.raises(NotServable) as e:
        store.validate(g2, READER, PROJ)
    assert e.value.boundary == "missing-member" and "g2" not in store.witnesses
    snap = store.snapshot()  # validate logs its reads; compare promote against the post-validate state
    assert gm.check_promote_refused_unchanged(store, snap, _direct_promote(store, "g2")).ok
    f, _ = build(gm.NoValidationFenceStore)
    f.stage(g2, {"doc-a": NEW["doc-a"]})
    fsnap = f.snapshot()
    record("r4b", "NoValidationFenceStore", gm.check_promote_refused_unchanged(f, fsnap, _direct_promote(f, "g2")))
    assert f.ptr == "g2"
    assert not gm.check_pointer_is_complete_generation(f).ok  # pointer now names a partial candidate


def test_r4c_staged_complete_but_unvalidated_refused_then_ready_promotes():
    store, _ = build()
    g2 = manifest("g2", NEW)
    store.stage(g2, NEW)  # complete bytes, but nobody validated
    snap = store.snapshot()
    out = _direct_promote(store, "g2")
    assert isinstance(out, NotServable) and out.boundary == "unvalidated-candidate"
    assert gm.check_promote_refused_unchanged(store, snap, out).ok
    store.validate(g2, READER, PROJ)  # issues the witness
    w = store.witnesses["g2"]
    assert (w.generation_id, w.manifest, w.project, w.namespace, w.source_pin) == ("g2", g2, PROJ, NS, SRC1)
    # positive: ready candidate promotes directly, CAS version still advances by exactly one
    store.promote("g2", "g1", 1)
    assert (store.ptr, store.version) == ("g2", 2)
    assert gm.check_pointer_is_complete_generation(store).ok
    assert store.accepted_view(READER) == NEW


def test_r4d_witness_is_bound_to_manifest_and_actual_bytes():
    # (a) witness of a different manifest/bytes under the same generation_id is not accepted
    store, _ = build()
    g2 = manifest("g2", NEW)
    store.stage(g2, NEW)
    other, _ = build()
    alt = {k: v + b"-ALT" for k, v in NEW.items()}
    g2_alt = manifest("g2", alt)
    other.stage(g2_alt, alt)
    other.validate(g2_alt, READER, PROJ)
    store.witnesses["g2"] = other.witnesses["g2"]  # forged/transplanted witness, same generation_id
    snap = store.snapshot()
    out = _direct_promote(store, "g2")
    assert isinstance(out, NotServable) and out.boundary == "witness-manifest-mismatch"
    assert gm.check_promote_refused_unchanged(store, snap, out).ok
    # (b) same manifest, but the stored bytes no longer match what was validated
    store2, _ = build()
    store2.stage(g2, NEW)
    store2.validate(g2, READER, PROJ)
    key = store2.storage_key(g2, g2.members[1])
    store2.blobs[key] = b"X" * len(store2.blobs[key])  # same length, different bytes after validation
    snap2 = store2.snapshot()
    out2 = _direct_promote(store2, "g2")
    assert isinstance(out2, NotServable) and out2.boundary == "witness-bytes-mismatch"
    assert gm.check_promote_refused_unchanged(store2, snap2, out2).ok
    # (c) validate itself refuses a manifest that differs from the registered one: no witness issued
    store3, _ = build()
    store3.stage(g2, NEW)
    with pytest.raises(NotServable) as e:
        store3.validate(g2_alt, READER, PROJ)
    assert e.value.boundary == "immutable-identity-conflict" and "g2" not in store3.witnesses
    # (d) a witness bound to another project/namespace/source pin is not accepted either
    for forged in (
        gm.AdmissionWitness("g2", g2, "other-proj", NS, SRC1, store2.witnesses["g2"].bytes_digest),
        gm.AdmissionWitness("g2", g2, PROJ, "other-ns", SRC1, store2.witnesses["g2"].bytes_digest),
        gm.AdmissionWitness("g2", g2, PROJ, NS, SRC2, store2.witnesses["g2"].bytes_digest),
    ):
        store4, _ = build()
        store4.stage(g2, NEW)
        store4.witnesses["g2"] = forged
        out4 = _direct_promote(store4, "g2")
        assert isinstance(out4, NotServable) and out4.boundary in {"witness-binding-mismatch"}
        assert (store4.ptr, store4.version) == ("g1", 1)
    f, _ = build(gm.NoValidationFenceStore)
    f.stage(g2, NEW)
    f.witnesses["g2"] = other.witnesses["g2"]
    record(
        "r4d", "NoValidationFenceStore", gm.check_promote_refused_unchanged(f, f.snapshot(), _direct_promote(f, "g2"))
    )


def test_r4e_fence_is_separate_from_current_use_authorization_and_cas():
    store, _ = build()
    g2 = manifest("g2", NEW)
    store.stage(g2, NEW)
    store.validate(g2, READER, PROJ)
    store.authz.revoke(READER, NS)
    store.promote("g2", "g1", 1)  # selection does not consult grants...
    with pytest.raises(Denied):
        store.accepted_view(READER)  # ...every use does
    # a witnessed candidate with a stale observation is still refused by CAS (ABA semantics intact)
    store2, _ = build()
    store2.stage(g2, NEW)
    store2.validate(g2, READER, PROJ)
    out = _direct_promote(store2, "g2", "g1", 0)
    assert isinstance(out, NotServable) and out.boundary == "cas-conflict" and store2.ptr == "g1"
    # a promoted-then-retried candidate keeps its witness; stale retry is refused by CAS, not duplicated
    store2.promote("g2", "g1", 1)
    out2 = _direct_promote(store2, "g2", "g1", 1)
    assert isinstance(out2, NotServable) and out2.boundary == "cas-conflict" and store2.version == 2


def test_r4f_fence_corner_guards_exercised_alone():
    # (a) validate of an UNREGISTERED manifest whose bytes happen to exist issues no witness
    store, _ = build()
    g9 = manifest("g9", NEW)
    for mem in g9.members:
        store.blobs[store.storage_key(g9, mem)] = NEW[mem.artifact_id]  # bypasses stage on purpose
    store.validate(g9, READER, PROJ)
    assert "g9" not in store.witnesses
    out = _direct_promote(store, "g9")
    assert isinstance(out, NotServable) and out.boundary == "unregistered-generation"
    # (b) a registered foreign-namespace manifest with a self-consistent witness is still refused
    store2, _ = build()
    gx = manifest("gx", NEW, ns="other-ns")
    store2.manifests["gx"] = gx
    store2.witnesses["gx"] = store2._witness_for(gx)
    out2 = _direct_promote(store2, "gx")
    assert isinstance(out2, NotServable) and out2.boundary == "foreign-namespace"
    assert (store2.ptr, store2.version) == ("g1", 1)
    # (c) the fence runs BEFORE the CAS comparison: an unvalidated candidate with a stale observation
    # is reported as unvalidated, not as a CAS conflict
    store3, _ = build()
    store3.stage(manifest("g2", NEW), NEW)
    out3 = _direct_promote(store3, "g2", "g1", 0)
    assert isinstance(out3, NotServable) and out3.boundary == "unvalidated-candidate"


def _validated_g2(cls=Store):
    store, _ = build(cls)
    g2 = manifest("g2", NEW)
    store.stage(g2, NEW)
    store.validate(g2, READER, PROJ)
    return store, g2


def test_r5a_withdrawn_source_refused_at_promote_nothing_changes():
    # reviewer's trace: validate g2 with SRC1, withdraw SRC1, promote with the correct observation
    store, g2 = _validated_g2()
    store.withdraw_source(SRC1)
    snap = store.snapshot()
    out = _direct_promote(store, "g2", "g1", 1)
    assert isinstance(out, NotServable) and out.boundary == "non-current-source"
    assert gm.check_promote_refused_unchanged(store, snap, out).ok
    assert (store.ptr, store.version) == ("g1", 1)
    assert store.witnesses["g2"] is snap[1]["g2"] and store.blobs == snap[2]  # witnesses + member bytes
    assert gm.check_pointer_is_complete_generation(store).ok
    # mutant: the round-4 model moves the pointer to a generation whose source is no longer accepted
    f, _ = _validated_g2(gm.NoSourceCurrencyFenceStore)
    f.withdraw_source(SRC1)
    fsnap = f.snapshot()
    record(
        "r5a",
        "NoSourceCurrencyFenceStore",
        gm.check_promote_refused_unchanged(f, fsnap, _direct_promote(f, "g2", "g1", 1)),
    )
    assert (f.ptr, f.version) == ("g2", 2)


def test_r5b_currency_is_reevaluated_each_attempt_reacceptance_needs_no_fresh_validate():
    # Rule chosen: live check per promote attempt; the witness is not revoked by a temporary withdrawal.
    store, _ = _validated_g2()
    store.withdraw_source(SRC1)
    assert _direct_promote(store, "g2", "g1", 1).boundary == "non-current-source"
    store.accept_source(SRC1)  # re-accepted: same pin, same bytes, same witness
    assert store.witnesses["g2"] is not None
    store.withdraw_source(SRC1)  # not cached: withdrawn again -> refused again
    assert _direct_promote(store, "g2", "g1", 1).boundary == "non-current-source"
    store.accept_source(SRC1)
    store.promote("g2", "g1", 1)
    assert (store.ptr, store.version) == ("g2", 2)
    # and a withdrawal AFTER selection does not unselect; every use is governed by its own checks
    store.withdraw_source(SRC1)
    assert store.ptr == "g2"


def test_r5c_positive_promote_with_current_source_and_other_pin_withdrawn():
    store, _ = _validated_g2()
    store.withdraw_source(SRC2)  # an unrelated pin leaving the set does not matter
    store.promote("g2", "g1", 1)
    assert (store.ptr, store.version) == ("g2", 2)
    assert gm.check_pointer_is_complete_generation(store).ok


def test_r5d_currency_is_separate_from_witness_cas_and_use_grant():
    # stale CAS + withdrawn source: CAS reported first (currency is the last check before the change)
    store, _ = _validated_g2()
    store.withdraw_source(SRC1)
    assert _direct_promote(store, "g2", "g1", 0).boundary == "cas-conflict"
    # unvalidated + withdrawn source: the admission witness is reported, not currency
    s2, _ = build()
    s2.stage(manifest("g2", NEW), NEW)
    s2.withdraw_source(SRC1)
    assert _direct_promote(s2, "g2", "g1", 1).boundary == "unvalidated-candidate"
    # grant revoked but source current: selection still works (grant is a use-time check)
    s3, _ = _validated_g2()
    s3.authz.revoke(READER, NS)
    s3.promote("g2", "g1", 1)
    with pytest.raises(Denied):
        s3.accepted_view(READER)


def test_r5e_validate_source_guard_exercised_alone():
    # validate's own current-source check must refuse BEFORE a witness is issued; the promote-time currency
    # fence must not be able to mask its removal (surfaced by the round-5 sweep: mutant survived without this)
    store, _ = build()
    m = manifest("g2", NEW, src="pin-unaccepted")
    store.stage(m, NEW)
    snap = store.snapshot()
    with pytest.raises(NotServable) as ei:
        store.validate(m, READER, PROJ)
    assert ei.value.boundary == "non-current-source"
    assert "g2" not in store.witnesses
    assert gm.check_state_unchanged(store, snap).ok


# --- round 6: source currency at the promotion linearization point ----------------------------------------
class _Probe:
    """Deterministic interleaving seams. ``switch_hook`` runs inside promote after every fence, right before the
    pointer write; ``mutation_hook`` runs inside a source mutation while it holds (or bypasses) the lock, before
    it is applied."""

    switch_hook = None
    mutation_hook = None
    fence_hook = None

    def _source_currency_fence(self, gen_id):
        super()._source_currency_fence(gen_id)
        if self.fence_hook:  # runs right after the currency check passed, before the revision compare
            self.fence_hook()

    def _before_switch(self):
        if self.switch_hook:
            self.switch_hook()

    def _in_source_mutation(self):
        if self.mutation_hook:
            self.mutation_hook()


def _probed(cls):
    return type(f"Probed{cls.__name__}", (_Probe, cls), {})


def _validated_g2_probed(cls):
    store, _ = _validated_g2(_probed(cls))
    return store


def _withdraw_during_switch(store, pin=SRC1):
    """Hook body: start a competing withdrawal while promote sits between its last fence and the pointer write.
    Records whether the withdrawal could COMPLETE inside that window (it must not: it needs the same lock)."""
    box = {}
    t = threading.Thread(target=lambda: store.withdraw_source(pin))

    def hook():
        t.start()
        t.join(timeout=0.4)
        box["completed_inside_window"] = not t.is_alive()

    store.switch_hook = hook
    box["thread"] = t
    return box


def test_r6a_withdrawal_inside_the_switch_window_cannot_complete_before_the_switch():
    store = _validated_g2_probed(Store)
    box = _withdraw_during_switch(store)
    store.promote("g2", "g1", 1)  # selection linearizes first: the competing withdrawal waited for the lock
    box["thread"].join(timeout=5)
    assert box["completed_inside_window"] is False
    assert not box["thread"].is_alive()
    assert (store.ptr, store.version) == ("g2", 2)  # the switch stays valid
    assert SRC1 not in store.accepted_sources  # the withdrawal completed strictly AFTER the switch
    assert gm.check_pointer_is_complete_generation(store).ok
    # later behaviour as in round 5: the withdrawal after selection does not unselect, grants stay use-time,
    # but nothing from the withdrawn source can be selected or validated any more
    assert store.accepted_view(READER) == NEW
    g3 = manifest("g3", NEW, src=SRC1)
    store.stage(g3, NEW)
    with pytest.raises(NotServable) as ei:
        store.validate(g3, READER, PROJ)
    assert ei.value.boundary == "non-current-source"
    # mutant (i): withdrawal bypasses the lock -> it completes inside the window, between check and switch
    f = _validated_g2_probed(gm.UnlockedNoRevisionStore)
    fbox = _withdraw_during_switch(f)
    f.promote("g2", "g1", 1)
    fbox["thread"].join(timeout=5)
    ok = not fbox["completed_inside_window"] and not (f.ptr == "g2" and SRC1 not in f.accepted_sources)
    record(
        "r6a",
        "UnlockedNoRevisionStore",
        gm.Verdict(
            ok,
            "I12:source-mutation-serialized-with-selection",
            f"ptr={f.ptr} src1_accepted={SRC1 in f.accepted_sources}",
        ),
    )
    assert (f.ptr, f.version) == ("g2", 2) and SRC1 not in f.accepted_sources  # the defect really happened


def _withdrawal_holding_the_lock_first(store, pin=SRC1, op="withdraw_source"):
    """A source mutation (default: withdrawal) enters its critical section (lock held, not yet applied); promote is
    then started and must queue behind it. Returns the outcome box."""
    inside, release = threading.Event(), threading.Event()
    store.mutation_hook = lambda: (inside.set(), release.wait(timeout=5))
    w = threading.Thread(target=lambda: getattr(store, op)(pin))
    w.start()
    assert inside.wait(timeout=5)
    out: dict = {}

    def run_promote():
        try:
            store.promote("g2", "g1", 1)
            out["result"] = "promoted"
        except NotServable as e:
            out["result"] = e

    p = threading.Thread(target=run_promote)
    p.start()
    p.join(timeout=0.4)
    out["promote_blocked"] = p.is_alive()
    release.set()
    w.join(timeout=5)
    p.join(timeout=5)
    return out


def test_r6b_promote_queued_behind_a_withdrawal_sees_it_and_refuses_unchanged():
    store = _validated_g2_probed(Store)
    snap = store.snapshot()
    out = _withdrawal_holding_the_lock_first(store)
    assert out["promote_blocked"] is True  # selection waits for the source mutation: one lock
    assert isinstance(out["result"], NotServable) and out["result"].boundary == "non-current-source"
    assert gm.check_promote_refused_unchanged(store, snap, out["result"]).ok  # ptr, version, witnesses, bytes, logs
    assert (store.ptr, store.version) == ("g1", 1)
    assert store.witnesses["g2"] is snap[1]["g2"] and store.blobs == snap[2]
    # mutant (i): withdrawal bypasses the lock -> promote does not queue behind it
    f = _validated_g2_probed(gm.UnlockedSourceMutationStore)
    fout = _withdrawal_holding_the_lock_first(f)
    record(
        "r6b",
        "UnlockedSourceMutationStore",
        gm.Verdict(
            fout["promote_blocked"] and isinstance(fout["result"], NotServable),
            "I12:source-mutation-serialized-with-selection",
            f"promote_blocked={fout['promote_blocked']} result={fout['result']!r}",
        ),
    )
    # mutant (ii): currency read from a snapshot taken before the lock was acquired -> stale "current" at the switch
    s2 = _validated_g2_probed(gm.SnapshotBeforeLockStore)
    s2snap = s2.snapshot()
    s2out = _withdrawal_holding_the_lock_first(s2)
    record(
        "r6b",
        "SnapshotBeforeLockStore",
        gm.check_promote_refused_unchanged(s2, s2snap, s2out["result"]),
    )
    assert (s2.ptr, s2.version) == ("g2", 2)  # the stale snapshot really moved the pointer


def test_r6b2_promote_queued_behind_a_re_acceptance_sees_it_and_succeeds():
    # the acceptance side of the registry is serialized with selection as well (not only withdrawal)
    store = _validated_g2_probed(Store)
    store.withdraw_source(SRC1)
    out = _withdrawal_holding_the_lock_first(store, op="accept_source")
    assert out["promote_blocked"] is True
    assert out["result"] == "promoted" and (store.ptr, store.version) == ("g2", 2)
    f = _validated_g2_probed(gm.UnlockedSourceMutationStore)
    f.withdraw_source(SRC1)
    fout = _withdrawal_holding_the_lock_first(f, op="accept_source")
    record(
        "r6b2",
        "UnlockedSourceMutationStore",
        gm.Verdict(
            fout["promote_blocked"] and fout["result"] == "promoted",
            "I12:source-mutation-serialized-with-selection",
            f"promote_blocked={fout['promote_blocked']} result={fout['result']!r}",
        ),
    )


def test_r6c2_revision_is_read_before_the_currency_check_not_after():
    # a lock-ignoring writer lands right after the currency check passed: the revision recorded BEFORE the check
    # must differ from the one at the switch
    store = _validated_g2_probed(gm.UnlockedSourceMutationStore)
    store.fence_hook = lambda: store.withdraw_source(SRC1)
    snap = store.snapshot()
    with pytest.raises(NotServable) as ei:
        store.promote("g2", "g1", 1)
    assert ei.value.boundary == "source-changed-during-selection"
    assert store.snapshot() == snap
    assert (store.ptr, store.version) == ("g1", 1)


def test_r6c_revision_compare_refuses_a_lock_skipping_writer_that_bumps_the_revision_before_the_compare():
    # a source writer that skips the lock but still bumps the revision before the compare: the switch refuses.
    # Narrow scope: the guarantee is for lock-taking writers; the compare and the pointer assignment are separate
    # steps, so a bypassing writer landing after the compare is NOT_MEASURED (not claimed, not tested).
    store = _validated_g2_probed(gm.UnlockedSourceMutationStore)
    snap = store.snapshot()
    box = _withdraw_during_switch(store)
    with pytest.raises(NotServable) as ei:
        store.promote("g2", "g1", 1)
    box["thread"].join(timeout=5)
    assert box["completed_inside_window"] is True  # the bypassing writer did land in the window...
    assert ei.value.boundary == "source-changed-during-selection"  # ...and the revision compare caught it
    assert (store.ptr, store.version) == ("g1", 1)
    assert store.witnesses["g2"] is snap[1]["g2"] and store.blobs == snap[2]
    assert gm.check_pointer_is_complete_generation(store).ok
    # mutant (iii): revision not compared -> the pointer advances to the withdrawn source
    f = _validated_g2_probed(gm.UnlockedNoRevisionStore)
    fsnap = f.snapshot()
    fbox = _withdraw_during_switch(f)
    try:
        f.promote("g2", "g1", 1)
        fout: str | Exception = "g2"
    except NotServable as e:
        fout = e
    fbox["thread"].join(timeout=5)
    record("r6c", "UnlockedNoRevisionStore", gm.check_promote_refused_unchanged(f, fsnap, fout))
    assert (f.ptr, f.version) == ("g2", 2)


def test_r6d_positive_and_independent_checks_unchanged_by_the_lock():
    # positive promote, no concurrent mutation: revision unchanged by promote, switch happens
    store = _validated_g2_probed(Store)
    rev = store.source_revision
    store.promote("g2", "g1", 1)
    assert (store.ptr, store.version, store.source_revision) == ("g2", 2, rev)
    # acceptance/withdrawal are the only revision movers
    store.withdraw_source(SRC2)
    store.accept_source(SRC2)
    assert store.source_revision == rev + 2
    # round-5 re-acceptance rule still holds through the Store operations (no fresh validate needed)
    s2 = _validated_g2_probed(Store)
    s2.withdraw_source(SRC1)
    assert _direct_promote(s2, "g2", "g1", 1).boundary == "non-current-source"
    s2.accept_source(SRC1)
    s2.promote("g2", "g1", 1)
    assert (s2.ptr, s2.version) == ("g2", 2)
    # witness, CAS and use-time grant stay separate: stale CAS first, grant revocation does not block selection
    s3 = _validated_g2_probed(Store)
    s3.withdraw_source(SRC1)
    assert _direct_promote(s3, "g2", "g1", 0).boundary == "cas-conflict"
    s4 = _validated_g2_probed(Store)
    s4.authz.revoke(READER, NS)
    s4.promote("g2", "g1", 1)
    with pytest.raises(Denied):
        s4.accepted_view(READER)


def test_r6e_source_set_is_not_mutable_from_outside():
    store, _ = build()
    with pytest.raises(AttributeError):
        store.accepted_sources.discard(SRC1)  # frozenset snapshot
    with pytest.raises(AttributeError):
        store.accepted_sources = {SRC1}  # no setter
    seed = {SRC1}
    s2 = Store(Authz(), NS, seed)
    seed.add("pin-injected")  # the caller's set is copied, not shared
    assert s2.accepted_sources == frozenset({SRC1})


def test_checker_is_not_vacuous_every_faulty_variant_rejected():
    """Meta: the faulty variants recorded above must all be rejected (checker disagrees with them)."""
    # Population is filled by the case tests (order-independent: re-run the minimal set here).
    for fn in (
        test_c08_revoked_reader_denied_even_with_cached_bytes,
        test_c09_competing_promotion_cas_single_winner,
        test_c10_interrupted_after_first_new_member_previous_generation_intact,
        test_c12_no_cross_namespace_reads_or_writes,
        test_ac4_incomplete_candidate_never_visible_in_accepted_view,
        test_c04_tampered_bytes_same_length,
        test_c05_non_current_or_foreign_source,
        test_c06_missing_dependency,
        test_c07_wrong_project,
        test_c02_one_member_missing,
        test_c03_stale_digest,
        test_r1a_same_id_different_manifest_refused_before_any_state_change,
        test_r1b_same_manifest_different_bytes_refused_before_any_state_change,
        test_r1c_interrupted_same_id_rewrite_changes_nothing,
        test_r1d_conflict_on_staged_candidate_is_prechecked_for_all_members,
        test_r1f_same_id_different_manifest_without_byte_collision_refused_alone,
        test_r2a_aba_stale_observation_refused_by_version,
        test_r2b_served_read_refuses_staged_generation_diagnostic_read_is_labelled,
        test_r2c_transitive_dependency_chain_gap_refused,
        test_r4a_direct_promote_of_missing_generation_refused,
        test_r4b_direct_promote_of_incomplete_staged_generation_refused,
        test_r4d_witness_is_bound_to_manifest_and_actual_bytes,
        test_r5a_withdrawn_source_refused_at_promote_nothing_changes,
        test_r6a_withdrawal_inside_the_switch_window_cannot_complete_before_the_switch,
        test_r6b_promote_queued_behind_a_withdrawal_sees_it_and_refuses_unchanged,
        test_r6b2_promote_queued_behind_a_re_acceptance_sees_it_and_succeeds,
        test_r6c_revision_compare_refuses_a_lock_skipping_writer_that_bumps_the_revision_before_the_compare,
    ):
        fn()
    assert FAULTY_RESULTS and all(v["rejected"] for v in FAULTY_RESULTS.values())
