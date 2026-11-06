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
    ):
        fn()
    assert FAULTY_RESULTS and all(v["rejected"] for v in FAULTY_RESULTS.values())
