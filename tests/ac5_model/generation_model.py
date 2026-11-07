"""KB-CONS-01-M: MODEL-ONLY, SYNTHETIC executable model of AC5 generation staging and selection.

This is NOT Scrutator code, touches no database, no network, no deployed instance and no real
corpus. It models, in memory, the properties AC5 requires of a private-consumer generation
switch, and ships *faulty variants* so that the checker (``check_*``) is shown to REJECT them.

Model assumptions are listed in REPORT.md (section "Model assumptions"); nothing here proves a
real DB transaction, crash recovery, real authorization or a live AC5 result.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

# ----------------------------------------------------------------------------- identities


def sha256(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class Member:
    """One manifest member. ``artifact_id`` is the semantic id; storage identity is separate."""

    artifact_id: str
    revision: str
    representation: str
    length: int
    digest: str
    deps: tuple[str, ...] = ()  # artifact_ids that must be members of the same generation


@dataclass(frozen=True)
class Manifest:
    generation_id: str
    source_pin: str  # accepted source/edition commit
    project: str
    namespace: str
    members: tuple[Member, ...]

    def member(self, artifact_id: str) -> Member | None:
        return next((m for m in self.members if m.artifact_id == artifact_id), None)


@dataclass(frozen=True)
class AdmissionWitness:
    """Immutable candidate-complete validation witness, issued ONLY by a successful ``validate``.

    Bound to (generation_id, manifest identity, project, namespace, source pin, actual member bytes).
    It carries no authorization: current-use grants are re-checked at every use, never frozen here.
    """

    generation_id: str
    manifest: Manifest
    project: str
    namespace: str
    source_pin: str
    bytes_digest: str  # digest over the (artifact_id, sha256) pairs of the bytes actually read


def make_member(artifact_id: str, data: bytes, revision: str = "r1", deps: tuple[str, ...] = ()) -> Member:
    return Member(artifact_id, revision, "utf8-text", len(data), sha256(data), deps)


# ----------------------------------------------------------------------------- authorization


class Authz:
    """Current-grant oracle. Re-consulted at every fetch/serve/use; never cached by the store."""

    def __init__(self) -> None:
        self._grants: set[tuple[str, str]] = set()  # (reader, namespace)

    def grant(self, reader: str, namespace: str) -> None:
        self._grants.add((reader, namespace))

    def revoke(self, reader: str, namespace: str) -> None:
        self._grants.discard((reader, namespace))

    def allowed(self, reader: str, namespace: str) -> bool:
        return (reader, namespace) in self._grants


class Denied(Exception):
    pass


class NotServable(Exception):
    def __init__(self, boundary: str, detail: str = "") -> None:
        super().__init__(f"{boundary}: {detail}")
        self.boundary = boundary


# ----------------------------------------------------------------------------- store


class InjectedFailure(Exception):
    pass


class Store:
    """Namespaced blob store + generation registry + current pointer.

    ``storage_key`` for a staged member is generation-qualified (immutable staging). The unsafe
    subclass ``InPlaceStore`` instead keys by (namespace, artifact_id) like reusing source paths.
    A read/write log records every namespace touched (cross-namespace check).
    """

    def __init__(self, authz: Authz, namespace: str, accepted_sources: set[str]) -> None:
        self.authz = authz
        self.namespace = namespace
        # Source pins considered current/accepted. PRIVATE: the set is copied, never exposed mutably, and changed only
        # by ``accept_source`` / ``withdraw_source``, which take the SAME lock as ``promote`` and bump
        # ``source_revision``. (Round 6: a plain mutable set could change between the currency check and the switch.)
        self._accepted_sources: set[str] = set(accepted_sources)
        self.source_revision = 0  # bumped on every source acceptance/withdrawal (committed under the lock)
        self.blobs: dict[tuple[str, str], bytes] = {}
        self.manifests: dict[str, Manifest] = {}
        self.candidates: set[str] = set()
        self.ever_selected: set[str] = set()  # generations that were ever the selected pointer
        self.witnesses: dict[str, AdmissionWitness] = {}  # generation_id -> issued-by-validate witness
        self.ptr: str | None = None  # accepted-generation pointer
        self.version = 0  # pointer version for CAS
        self._lock = threading.Lock()
        self.ns_log: list[tuple[str, str]] = []  # (op, namespace)
        self.fail_after_writes: int | None = None  # fault injection
        self._writes = 0

    # --- source registry (serialized with selection) -----------------------------------------
    @property
    def accepted_sources(self) -> frozenset[str]:
        """Read-only snapshot of the accepted source pins (a frozenset: cannot be mutated from outside)."""
        return frozenset(self._accepted_sources)

    def accept_source(self, pin: str) -> None:
        """(Re-)accept a source pin. Same lock as ``promote``; bumps ``source_revision``."""
        with self._lock:
            self._in_source_mutation()
            self._accepted_sources.add(pin)
            self.source_revision += 1

    def withdraw_source(self, pin: str) -> None:
        """Withdraw a source pin. Same lock as ``promote``: it either completes strictly before a selection
        (the selection then refuses) or strictly after the switch (the switch stays valid). Bumps ``source_revision``."""
        with self._lock:
            self._in_source_mutation()
            self._accepted_sources.discard(pin)
            self.source_revision += 1

    def _in_source_mutation(self) -> None:
        """Test seam (no-op): runs while the mutation holds the lock, before it is applied."""

    def _before_switch(self) -> None:
        """Test seam (no-op): runs inside ``promote`` after every fence, immediately before the pointer write."""

    # --- addressing -----------------------------------------------------------------
    def storage_key(self, m: Manifest, mem: Member) -> tuple[str, str]:
        return (m.namespace, f"{m.generation_id}/{mem.artifact_id}@{mem.revision}")

    # --- staging ---------------------------------------------------------------------
    def _write(self, key: tuple[str, str], data: bytes, op: str = "write") -> None:
        if self.fail_after_writes is not None and self._writes >= self.fail_after_writes:
            raise InjectedFailure(f"injected failure before write #{self._writes + 1}")
        self.ns_log.append((op, key[0]))
        self.blobs[key] = data
        self._writes += 1

    def stage(self, m: Manifest, payload: dict[str, bytes]) -> None:
        """Stage candidate members. Never touches the pointer. Partial failure leaves old intact."""
        if m.namespace != self.namespace:  # refuse BEFORE any write/read of a foreign namespace
            raise NotServable("foreign-namespace", m.namespace)
        self._stage_unchecked(m, payload)

    def snapshot(self) -> tuple:
        """Full observable state, used by controls to prove a refused call changed nothing."""
        return (
            dict(self.manifests),
            dict(self.witnesses),
            dict(self.blobs),
            set(self.candidates),
            set(self.ever_selected),
            self.ptr,
            self.version,
            list(self.ns_log),
        )

    def _stage_unchecked(self, m: Manifest, payload: dict[str, bytes]) -> None:
        """Generation ID, manifest and member bytes are IMMUTABLE once first written.

        R1: every identity conflict is detected BEFORE any state change (manifest registration,
        candidate set, blob write, fault-injection counter). An exact retry may only observe
        identical bytes or fill members that are still missing.
        """
        known = self.manifests.get(m.generation_id)
        if known is not None and known != m:
            raise NotServable("immutable-identity-conflict", f"manifest {m.generation_id}")
        for mem in m.members:
            if mem.artifact_id in payload:
                have = self.blobs.get(self.storage_key(m, mem))
                if have is not None and have != payload[mem.artifact_id]:
                    raise NotServable("immutable-identity-conflict", f"bytes {m.generation_id}/{mem.artifact_id}")
        self.manifests[m.generation_id] = m
        if m.generation_id not in self.ever_selected:
            self.candidates.add(m.generation_id)
        self._writes = 0
        for mem in m.members:
            if mem.artifact_id in payload:
                key = self.storage_key(m, mem)
                if key in self.blobs:  # identical bytes (checked above): observe, do not rewrite
                    continue
                self._write(key, payload[mem.artifact_id])

    # --- closure validation ------------------------------------------------------------
    def validate(self, m: Manifest, reader: str, project: str) -> None:
        """Validate ALL members by reading actual stored bytes through the current scoped reader.
        Raises NotServable naming the first failing boundary."""
        if m.namespace != self.namespace:
            raise NotServable("foreign-namespace", m.namespace)
        registered = self.manifests.get(m.generation_id)
        if registered is not None and registered != m:
            raise NotServable("immutable-identity-conflict", f"validate manifest {m.generation_id}")
        if project != m.project:
            raise NotServable("wrong-project", f"{project}!={m.project}")
        with self._lock:  # read the registry coherently with source mutation (validate never moves the pointer)
            current = m.source_pin in self._accepted_sources
        if not current:
            raise NotServable("non-current-source", m.source_pin)
        if not self.authz.allowed(reader, m.namespace):
            raise NotServable("authorization", reader)
        self._check_dependencies(m)
        for mem in m.members:
            data = self.blobs.get(self.storage_key(m, mem))
            self.ns_log.append(("read", m.namespace))
            if data is None:
                raise NotServable("missing-member", mem.artifact_id)
            if len(data) != mem.length:
                raise NotServable("length-mismatch", mem.artifact_id)
            if sha256(data) != mem.digest:
                raise NotServable("digest-mismatch", mem.artifact_id)
        if registered is not None:  # only a REGISTERED, fully verified candidate earns a witness
            self.witnesses[m.generation_id] = self._witness_for(m)

    def _check_dependencies(self, m: Manifest) -> None:
        ids = {x.artifact_id for x in m.members}
        for mem in m.members:
            for d in mem.deps:
                if d not in ids:
                    raise NotServable("missing-dependency", f"{mem.artifact_id}->{d}")

    # --- selection ----------------------------------------------------------------------
    def _witness_for(self, m: Manifest) -> AdmissionWitness:
        parts = []
        for mem in m.members:
            data = self.blobs.get(self.storage_key(m, mem))
            parts.append(f"{mem.artifact_id}={sha256(data) if data is not None else 'ABSENT'}")
        return AdmissionWitness(
            m.generation_id, m, m.project, m.namespace, m.source_pin, sha256("\n".join(parts).encode())
        )

    def _admission_fence(self, gen_id: str) -> None:
        """Selection precondition, evaluated BEFORE any pointer/version change: the generation must be
        registered, hold an issued witness for exactly the registered manifest, and the stored member
        bytes must still be the bytes the witness covers. Current-use authorization is NOT checked here
        (re-checked at every use in accepted_view/read_member)."""
        m = self.manifests.get(gen_id)
        if m is None:
            raise NotServable("unregistered-generation", gen_id)
        w = self.witnesses.get(gen_id)
        if w is None:
            raise NotServable("unvalidated-candidate", gen_id)
        if w.generation_id != gen_id or w.manifest != m:
            raise NotServable("witness-manifest-mismatch", gen_id)
        if (w.project, w.namespace, w.source_pin) != (m.project, m.namespace, m.source_pin):
            raise NotServable("witness-binding-mismatch", gen_id)
        if w.namespace != self.namespace:
            raise NotServable("foreign-namespace", w.namespace)
        if w.bytes_digest != self._witness_for(m).bytes_digest:
            raise NotServable("witness-bytes-mismatch", gen_id)

    def _source_currency_fence(self, gen_id: str) -> None:
        """Selection precondition, evaluated under the selection lock IMMEDIATELY before the pointer/version
        change: the source pin of the registered manifest must be accepted/current NOW. Distinct from
        (a) the admission witness (generation identity, bytes; it also owns the unregistered case, so an
        unregistered id never reaches this check), (b) the CAS comparison, and (c) current reader grant /
        use-time revocation, which stays at use. Fail-closed: a pin not in ``accepted_sources`` is refused.
        Rule chosen (not 'frozen set'): currency is re-evaluated against the live set at every promote
        attempt; a later re-acceptance of the same pin needs NO fresh validate (the witness binds the
        pin and bytes, both unchanged), but it is never cached: removing the pin again refuses again."""
        m = self.manifests.get(gen_id)
        if m is not None and m.source_pin not in self._accepted_sources:  # caller holds self._lock
            raise NotServable("non-current-source", m.source_pin)

    def _source_revision_fence(self, revision_at_check: int) -> None:
        """Last selection precondition, immediately before the pointer/version change and under the same lock:
        the source registry revision must still be the one the currency check saw. SCOPE OF THE GUARANTEE: the
        model proves the property only for writers that take ``_lock`` (every native source mutation and the
        selection must run under the shared lock/transaction; that is a mandatory precondition, not an option).
        For a writer that skips the lock but still bumps ``source_revision`` this compare refuses a bump that
        lands before it runs (tested, conservative: an accept+withdraw pair in between also refuses); it is a
        best-effort tripwire, NOT an atomic primitive, because the compare and the pointer assignment are two
        separate steps and a lock-bypassing writer can land after the compare. Defence against writers that
        bypass the lock/transaction is NOT_MEASURED. Separate from the witness, CAS and use-time grants."""
        if self.source_revision != revision_at_check:
            raise NotServable("source-changed-during-selection", f"rev {revision_at_check}->{self.source_revision}")

    def promote(self, gen_id: str, expected_ptr: str | None, expected_version: int) -> None:
        """Native-CAS model: switch only if the candidate is witnessed AND pointer+version still equal
        expectation AND its source is still current. The fences run first so a refused call changes
        nothing. Source mutation takes the same lock (round 6), so for cooperating writers (all that take ``_lock``) currency holds at the switch itself. Non-cooperating writers that bypass the lock are NOT_MEASURED. The native service must implement the same rule in the real selection transaction, with every source mutation inside it."""
        with self._lock:
            self._admission_fence(gen_id)
            if self.ptr != expected_ptr or self.version != expected_version:
                raise NotServable("cas-conflict", f"ptr={self.ptr} v={self.version}")
            revision_at_check = self.source_revision
            self._source_currency_fence(gen_id)
            self._before_switch()
            self._source_revision_fence(revision_at_check)
            self.ptr = gen_id
            self.version += 1
            self.ever_selected.add(gen_id)
            self.candidates.discard(gen_id)

    # --- accepted view (search/context) ---------------------------------------------------
    def accepted_view(self, reader: str) -> dict[str, bytes]:
        """What search/context may expose: ONLY members of the selected complete manifest, with
        the grant re-checked now (a cached previous generation is not served after revocation)."""
        if self.ptr is None:
            return {}
        m = self.manifests[self.ptr]
        if not self.authz.allowed(reader, m.namespace):
            raise Denied(reader)
        out = {}
        for mem in m.members:
            self.ns_log.append(("read", m.namespace))
            out[mem.artifact_id] = self.blobs[self.storage_key(m, mem)]
        return out

    def read_member(self, gen_id: str, artifact_id: str, reader: str) -> bytes:
        """SERVED read: only a generation that was selected at some point may be read through it.
        A staged/candidate generation is refused (``not-selected-generation``)."""
        m = self.manifests[gen_id]
        if not self.authz.allowed(reader, m.namespace):
            raise Denied(reader)
        if gen_id not in self.ever_selected:
            raise NotServable("not-selected-generation", gen_id)
        return self._raw_member(m, artifact_id)

    def diagnostic_read_member(self, gen_id: str, artifact_id: str, reader: str) -> bytes:
        """DIAGNOSTIC-ONLY, never to be exposed by a served API: reads any staged generation
        (used by validation-style inspection and by tests). Still authorization-checked."""
        m = self.manifests[gen_id]
        if not self.authz.allowed(reader, m.namespace):
            raise Denied(reader)
        return self._raw_member(m, artifact_id)

    def _raw_member(self, m: Manifest, artifact_id: str) -> bytes:
        mem = m.member(artifact_id)
        assert mem is not None
        self.ns_log.append(("read", m.namespace))
        return self.blobs[self.storage_key(m, mem)]


# ----------------------------------------------------------------------------- unsafe variants


class LeakyViewStore(Store):
    """UNSAFE: accepted view exposes any searchable chunk, including incomplete candidates."""

    def accepted_view(self, reader: str) -> dict[str, bytes]:
        out = super().accepted_view(reader) if self.ptr else {}
        for g in self.candidates:
            m = self.manifests[g]
            for mem in m.members:
                data = self.blobs.get(self.storage_key(m, mem))
                if data is not None:
                    out[f"{g}:{mem.artifact_id}"] = data
        return out


class CachedViewStore(Store):
    """UNSAFE: serves the previous complete generation from cache without re-checking the grant."""

    def accepted_view(self, reader: str) -> dict[str, bytes]:
        if self.ptr is None:
            return {}
        m = self.manifests[self.ptr]
        return {mem.artifact_id: self.blobs[self.storage_key(m, mem)] for mem in m.members}


class LastWriterWinsStore(Store):
    """UNSAFE: promotion ignores the expected pointer; last writer silently replaces."""

    def promote(self, gen_id: str, expected_ptr: str | None, expected_version: int) -> None:
        with self._lock:
            self.ptr = gen_id
            self.version += 1
            self.candidates.discard(gen_id)


class OverwriteStore(Store):
    """UNSAFE (R1): the pre-correction staging. Replaces the manifest under an existing generation
    ID and overwrites populated keys, so a selected generation can change before validation."""

    def _stage_unchecked(self, m: Manifest, payload: dict[str, bytes]) -> None:
        self.manifests[m.generation_id] = m
        self.candidates.add(m.generation_id)
        self._writes = 0
        for mem in m.members:
            if mem.artifact_id in payload:
                self._write(self.storage_key(m, mem), payload[mem.artifact_id])


class NoManifestGuardStore(Store):
    """UNSAFE (R1): byte pre-check kept, manifest-level identity guard removed. A same-ID manifest
    whose payload does not collide with stored bytes silently replaces the registered manifest."""

    def _stage_unchecked(self, m: Manifest, payload: dict[str, bytes]) -> None:
        for mem in m.members:
            if mem.artifact_id in payload:
                have = self.blobs.get(self.storage_key(m, mem))
                if have is not None and have != payload[mem.artifact_id]:
                    raise NotServable("immutable-identity-conflict", f"bytes {m.generation_id}/{mem.artifact_id}")
        self.manifests[m.generation_id] = m
        if m.generation_id not in self.ever_selected:
            self.candidates.add(m.generation_id)
        self._writes = 0
        for mem in m.members:
            if mem.artifact_id in payload:
                key = self.storage_key(m, mem)
                if key in self.blobs:
                    continue
                self._write(key, payload[mem.artifact_id])


class InPlaceStore(OverwriteStore):
    """UNSAFE (no immutability guard): reuses the active generation's source paths (77307111 counterexample)."""

    def storage_key(self, m: Manifest, mem: Member) -> tuple[str, str]:
        return (m.namespace, f"{mem.artifact_id}")  # no generation qualifier


class CheckAsYouGoStore(Store):
    """UNSAFE (R1): writes missing members first and only then trips over a conflicting one, so a
    refused call still leaves partial state behind (no pre-check of ALL members)."""

    def _stage_unchecked(self, m: Manifest, payload: dict[str, bytes]) -> None:
        known = self.manifests.get(m.generation_id)
        if known is not None and known != m:
            raise NotServable("immutable-identity-conflict", f"manifest {m.generation_id}")
        self.manifests[m.generation_id] = m
        self._writes = 0
        for mem in m.members:
            if mem.artifact_id in payload:
                key = self.storage_key(m, mem)
                have = self.blobs.get(key)
                if have is not None and have != payload[mem.artifact_id]:
                    raise NotServable("immutable-identity-conflict", f"bytes {mem.artifact_id}")
                if have is None:
                    self._write(key, payload[mem.artifact_id])


class NoValidationFenceStore(Store):
    """UNSAFE (R3): public promote checks only pointer+version (the pre-correction behaviour); it
    selects unregistered, partial, unvalidated or witness-mismatched candidates."""

    def _admission_fence(self, gen_id: str) -> None:
        return None


class NoSourceCurrencyFenceStore(Store):
    """UNSAFE (round 5): the round-4 model. Witness and CAS are checked but source currency is not
    re-checked at promote, so a generation whose pin was withdrawn after validate can still be selected."""

    def _source_currency_fence(self, gen_id: str) -> None:
        return None


class UnlockedSourceMutationStore(Store):
    """UNSAFE (round 6, mutant i): source withdrawal/acceptance bypasses the selection lock (it still bumps
    the revision). A withdrawal can land between the currency check and the switch. Models the hazard only; the
    model's guarantee does not extend to writers that bypass the lock (NOT_MEASURED)."""

    def accept_source(self, pin: str) -> None:
        self._in_source_mutation()
        self._accepted_sources.add(pin)
        self.source_revision += 1

    def withdraw_source(self, pin: str) -> None:
        self._in_source_mutation()
        self._accepted_sources.discard(pin)
        self.source_revision += 1


class SnapshotBeforeLockStore(Store):
    """UNSAFE (round 6, mutant ii): the currency fence reads a snapshot of the accepted sources taken BEFORE the
    selection lock is acquired, so a mutation that completes while promote waits for the lock is not seen."""

    def promote(self, gen_id: str, expected_ptr: str | None, expected_version: int) -> None:
        snapshot = frozenset(self._accepted_sources)
        with self._lock:
            self._admission_fence(gen_id)
            if self.ptr != expected_ptr or self.version != expected_version:
                raise NotServable("cas-conflict", f"ptr={self.ptr} v={self.version}")
            m = self.manifests.get(gen_id)
            if m is not None and m.source_pin not in snapshot:
                raise NotServable("non-current-source", m.source_pin)
            revision_at_check = self.source_revision
            self._before_switch()
            self._source_revision_fence(revision_at_check)
            self.ptr = gen_id
            self.version += 1
            self.ever_selected.add(gen_id)
            self.candidates.discard(gen_id)


class NoRevisionFenceStore(Store):
    """UNSAFE (round 6, mutant iii): the source revision is never compared at the switch."""

    def _source_revision_fence(self, revision_at_check: int) -> None:
        return None


class UnlockedNoRevisionStore(UnlockedSourceMutationStore, NoRevisionFenceStore):
    """UNSAFE (round 6): lock bypass AND no revision compare (the full round-5 behaviour under concurrency)."""


class PointerOnlyCasStore(Store):
    """UNSAFE (R2): CAS compares the pointer VALUE only and ignores the version, so an A->B->A
    sequence lets a stale observation win."""

    def promote(self, gen_id: str, expected_ptr: str | None, expected_version: int) -> None:
        with self._lock:
            if self.ptr != expected_ptr:
                raise NotServable("cas-conflict", f"ptr={self.ptr}")
            self.ptr = gen_id
            self.version += 1
            self.ever_selected.add(gen_id)
            self.candidates.discard(gen_id)


class UnguardedReadStore(Store):
    """UNSAFE (R2): the served read_member exposes any staged generation."""

    def read_member(self, gen_id: str, artifact_id: str, reader: str) -> bytes:
        return self.diagnostic_read_member(gen_id, artifact_id, reader)


class RootOnlyDepStore(Store):
    """UNSAFE (R2 transitive): validates dependencies of the first (root) member only."""

    def _check_dependencies(self, m: Manifest) -> None:
        ids = {x.artifact_id for x in m.members}
        for d in m.members[0].deps if m.members else ():
            if d not in ids:
                raise NotServable("missing-dependency", f"{m.members[0].artifact_id}->{d}")


class NoValidationStore(Store):
    """UNSAFE: promotes without validating closure/digests (trusts metadata)."""

    def stage(self, m: Manifest, payload: dict[str, bytes]) -> None:
        self._stage_unchecked(m, payload)

    def validate(self, m: Manifest, reader: str, project: str) -> None:
        return None

    def _admission_fence(self, gen_id: str) -> None:  # no validation => no witness => no fence
        return None

    def _source_currency_fence(self, gen_id: str) -> None:  # "no validation at all" includes no source check
        return None

    def _source_revision_fence(self, revision_at_check: int) -> None:
        return None


# ----------------------------------------------------------------------------- consumer flow


def ingest_and_promote(store: Store, m: Manifest, payload: dict[str, bytes], reader: str, project: str) -> None:
    """Stage -> validate all -> CAS promote against the pointer observed BEFORE staging."""
    exp_ptr, exp_ver = store.ptr, store.version
    store.stage(m, payload)
    store.validate(m, reader, project)
    store.promote(m.generation_id, exp_ptr, exp_ver)


# ----------------------------------------------------------------------------- checkers


@dataclass
class Verdict:
    ok: bool
    invariant: str = ""
    detail: str = ""


def check_previous_generation_intact(store: Store, old: Manifest, old_payload: dict[str, bytes]) -> Verdict:
    """I1: every previously accepted member still reads at its old exact digest."""
    for mem in old.members:
        data = store.blobs.get(store.storage_key(old, mem))
        if data is None or sha256(data) != mem.digest:
            return Verdict(False, "I1:previous-accepted-member-readable-at-old-digest", mem.artifact_id)
    return Verdict(True)


def check_pointer_is_complete_generation(store: Store) -> Verdict:
    """I2: the pointer, if set, names a generation whose every member verifies."""
    if store.ptr is None:
        return Verdict(True)
    m = store.manifests[store.ptr]
    for mem in m.members:
        data = store.blobs.get(store.storage_key(m, mem))
        if data is None or sha256(data) != mem.digest:
            return Verdict(False, "I2:pointer-names-complete-verified-generation", mem.artifact_id)
    return Verdict(True)


def check_view_has_only_accepted(store: Store, reader: str) -> Verdict:
    """I3: accepted view contains only members of the selected manifest."""
    try:
        view = store.accepted_view(reader)
    except Denied:
        return Verdict(True)
    allowed = {x.artifact_id for x in store.manifests[store.ptr].members} if store.ptr else set()
    extra = set(view) - allowed
    if extra:
        return Verdict(False, "I3:candidate-isolated-from-accepted-view", ",".join(sorted(extra)))
    return Verdict(True)


def check_revoked_denied(store: Store, reader: str) -> Verdict:
    """I4: after revocation, the accepted view denies even though bytes remain."""
    try:
        store.accepted_view(reader)
    except Denied:
        return Verdict(True)
    return Verdict(False, "I4:revoked-reader-denied-at-use", reader)


def check_single_winner(store: Store, results: list[str | Exception], gens: list[str]) -> Verdict:
    """I5: exactly one competing promotion wins; pointer equals the winner."""
    winners = [g for g, r in zip(gens, results, strict=False) if not isinstance(r, Exception)]
    if len(winners) != 1:
        return Verdict(False, "I5:cas-exactly-one-winner", f"winners={winners}")
    if store.ptr != winners[0]:
        return Verdict(False, "I5:pointer-equals-winner", f"ptr={store.ptr} winner={winners[0]}")
    return Verdict(True)


def check_namespaces(store: Store) -> Verdict:
    """I6: no operation touched any namespace other than the model's own."""
    bad = [x for x in store.ns_log if x[1] != store.namespace]
    if bad:
        return Verdict(False, "I6:no-cross-namespace-io", repr(bad[0]))
    return Verdict(True)


def check_selected_identity_immutable(store: Store, sel: Manifest, payload: dict[str, bytes]) -> Verdict:
    """I8: the manifest registered under a selected generation ID and every member's bytes are
    exactly what was selected (identity is immutable after selection)."""
    if store.manifests.get(sel.generation_id) != sel:
        return Verdict(False, "I8:selected-manifest-immutable", sel.generation_id)
    for mem in sel.members:
        if store.blobs.get(store.storage_key(sel, mem)) != payload[mem.artifact_id]:
            return Verdict(False, "I8:selected-member-bytes-immutable", mem.artifact_id)
    return Verdict(True)


def check_state_unchanged(store: Store, snap: tuple) -> Verdict:
    """I8b: a refused call changed nothing observable (no partial write, no registration)."""
    if store.snapshot() != snap:
        return Verdict(False, "I8b:refused-call-leaves-no-state-change")
    return Verdict(True)


def check_stale_observation_refused(outcome: str | Exception) -> Verdict:
    """I9: a promotion based on a stale (ptr, version) observation after A->B->A is refused."""
    if isinstance(outcome, Exception):
        return Verdict(True)
    return Verdict(False, "I9:aba-stale-version-promotion-refused", f"promoted {outcome}")


def check_staged_not_served(store: Store, gen_id: str, artifact_id: str, reader: str) -> Verdict:
    """I10: the served read path never returns bytes of a never-selected (staged) generation."""
    try:
        store.read_member(gen_id, artifact_id, reader)
    except (NotServable, Denied):
        return Verdict(True)
    return Verdict(False, "I10:staged-generation-not-served", f"{gen_id}/{artifact_id}")


def check_promote_refused_unchanged(store: Store, snap: tuple, outcome: str | Exception) -> Verdict:
    """I11: a direct promote of a missing / partial / unvalidated / mismatched candidate is refused
    and changes no pointer, version, candidate set or registry state."""
    if not isinstance(outcome, Exception):
        return Verdict(False, "I11:direct-promote-requires-validation-witness", f"promoted {outcome}")
    if store.snapshot() != snap:
        return Verdict(False, "I11:refused-promote-leaves-no-state-change")
    return Verdict(True)
