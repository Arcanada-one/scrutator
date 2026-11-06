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
        self.accepted_sources = accepted_sources  # source pins considered current/accepted
        self.blobs: dict[tuple[str, str], bytes] = {}
        self.manifests: dict[str, Manifest] = {}
        self.candidates: set[str] = set()
        self.ever_selected: set[str] = set()  # generations that were ever the selected pointer
        self.ptr: str | None = None  # accepted-generation pointer
        self.version = 0  # pointer version for CAS
        self._lock = threading.Lock()
        self.ns_log: list[tuple[str, str]] = []  # (op, namespace)
        self.fail_after_writes: int | None = None  # fault injection
        self._writes = 0

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
        if project != m.project:
            raise NotServable("wrong-project", f"{project}!={m.project}")
        if m.source_pin not in self.accepted_sources:
            raise NotServable("non-current-source", m.source_pin)
        if not self.authz.allowed(reader, m.namespace):
            raise NotServable("authorization", reader)
        ids = {x.artifact_id for x in m.members}
        for mem in m.members:
            for d in mem.deps:
                if d not in ids:
                    raise NotServable("missing-dependency", f"{mem.artifact_id}->{d}")
        for mem in m.members:
            data = self.blobs.get(self.storage_key(m, mem))
            self.ns_log.append(("read", m.namespace))
            if data is None:
                raise NotServable("missing-member", mem.artifact_id)
            if len(data) != mem.length:
                raise NotServable("length-mismatch", mem.artifact_id)
            if sha256(data) != mem.digest:
                raise NotServable("digest-mismatch", mem.artifact_id)

    # --- selection ----------------------------------------------------------------------
    def promote(self, gen_id: str, expected_ptr: str | None, expected_version: int) -> None:
        """Native-CAS model: switch only if pointer+version still equal expectation."""
        with self._lock:
            if self.ptr != expected_ptr or self.version != expected_version:
                raise NotServable("cas-conflict", f"ptr={self.ptr} v={self.version}")
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

    def validate(self, m: Manifest, reader: str, project: str) -> None:
        trimmed = Manifest(
            m.generation_id,
            m.source_pin,
            m.project,
            m.namespace,
            tuple(
                x if i == 0 else Member(x.artifact_id, x.revision, x.representation, x.length, x.digest)
                for i, x in enumerate(m.members)
            ),
        )
        super().validate(trimmed, reader, project)


class NoValidationStore(Store):
    """UNSAFE: promotes without validating closure/digests (trusts metadata)."""

    def stage(self, m: Manifest, payload: dict[str, bytes]) -> None:
        self._stage_unchecked(m, payload)

    def validate(self, m: Manifest, reader: str, project: str) -> None:
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
