# KB-CONS-01-M — AC5 generation-closure model: REPORT

**Label: MODEL-ONLY, SYNTHETIC.** In-memory model with synthetic member IDs and bytes. No database, no network, no deployed Scrutator, no kc-corpus profile/grants, no producer code, no shared/public namespace was touched. A PASS here is a model result, **not** a live AC5 result, not an acceptance of KB-CONS-01, not a readiness claim.

## Case table (AC1)

| Case | Result |
|---|---|
| C01 complete generation + byte readback | PASS |
| C02 one member missing | PASS |
| C03 stale digest | PASS |
| C04 tampered bytes | PASS |
| C05 non-current / foreign source | PASS |
| C06 missing dependency | PASS |
| C07 wrong project | PASS |
| C08 revoked reader (cached bytes remain) | PASS |
| C09 concurrent promotion / CAS race (deterministic) | PASS |
| C09b CAS race, 4 threads x 50 rounds | PASS |
| C10 interrupted after first new member (77307111 counterexample) | PASS |
| C10b retry/recovery after interruption | PASS |
| C11 repeated idempotent check | PASS |
| C12 no cross-namespace reads/writes | PASS |
| R1a same ID / different manifest refused, no state change | PASS |
| R1b same manifest / different bytes refused, no state change | PASS |
| R1c interrupted same-ID rewrite changes nothing | PASS |
| R1d conflict pre-checked for all members (no partial write) | PASS |
| R1e exact retry observes identical bytes / fills missing only | PASS |
| R2a bounded ABA trace (g1->g2->g1) refused by version | PASS |
| R2b served read refuses staged generation; diagnostic read labelled | PASS |
| R2c transitive dependency chain gap refused | PASS |
| R4a direct promote of a missing generation refused, no state change | PASS |
| R4b direct promote of an incomplete staged generation refused, no state change | PASS |
| R4c complete-but-unvalidated refused; validated ready candidate promotes | PASS |
| R4d witness bound to manifest / project / namespace / pin / actual bytes | PASS |
| R4e fence separate from use-time authorization; CAS/ABA intact | PASS |
| R5a withdrawn source refused at promote, nothing changes | PASS |
| R5b currency re-evaluated per attempt; re-acceptance needs no fresh validate | PASS |
| R5c positive promote with a current source | PASS |
| R5d currency separate from witness, CAS and use-time grant | PASS |
| R5e validate current-source guard alone (no witness issued) | PASS |
| R4f fence corner guards (unregistered witness, foreign ns, fence-before-CAS order) | PASS |
| AC4 incomplete candidate absent from accepted view | PASS |
| META checker rejects every faulty variant | PASS |

## Subcase table: MEASURED vs NOT_MEASURED (R2)

MEASURED = a model test above passed AND a deliberately broken variant is rejected by its checker. NOT_MEASURED = no test exists in this PR; it must not be read as pass or fail.

| Subcase | Status | Rejected mutant / reason NOT_MEASURED |
|---|---|---|
| Immutable generation ID: same ID, different manifest | PASS | OverwriteStore |
| Immutable member bytes: same manifest, different bytes | PASS | OverwriteStore |
| Interrupted same-ID rewrite | PASS | OverwriteStore |
| Conflict detected before any write (no partial state) | PASS | CheckAsYouGoStore |
| Exact retry: observe identical / fill missing only | PASS | (positive control; R1a-d mutants cover the negative side) |
| Bounded ABA g1->g2->g1, stale (ptr, version) observation | PASS | PointerOnlyCasStore |
| Served read of a staged generation | PASS | UnguardedReadStore |
| Transitive dependency chain (a->b->c, c absent) | PASS | RootOnlyDepStore |
| Direct promote needs an immutable validation witness (missing / partial / unvalidated / foreign witness) | PASS | NoValidationFenceStore |
| Source pin still current at promote (re-checked under the selection lock, separate from witness / CAS / use grant) | PASS | NoSourceCurrencyFenceStore |
| Pre-ranking search isolation (candidate chunks excluded BEFORE ranking, not filtered after) | NOT_MEASURED | model has no ranker; only the post-selection accepted view (AC4) is modelled |
| Effect-time external revocation (grant revoked between check and effect) | NOT_MEASURED | revocation is checked at call time only; no check-then-use window modelled |
| Durable lost-ack reconciliation (promotion committed, ack lost, caller retries) | NOT_MEASURED | only a stale-expectation retry is refused (C11); recognising one's own committed win is not modelled |
| Search top-k isolation (candidate chunks cannot displace accepted chunks in the top-k) | NOT_MEASURED | no ranker / top-k in the model |
| Revocation between authorization and disclosure (check-then-disclose window) | NOT_MEASURED | authorization and read are one synchronous step in the model |
| External-authority outage / fence (Auth Arcana unreachable, fail-closed) | NOT_MEASURED | the authorization oracle never fails in the model |
| Complete versioned closure (dependency pins a specific revision, not just an artifact id) | NOT_MEASURED | deps are artifact ids only; revision-pinned deps not modelled |
| Cyclic dependency closure | NOT_MEASURED | not tested |
| Multibyte / non-ASCII member content (byte length vs character length) | NOT_MEASURED | all fixtures are ASCII |
| Shared member across generations (same artifact+revision in two generations) | NOT_MEASURED | revision is generation-qualified in all fixtures |

Direct reads: `Store.read_member` is the SERVED path and refuses any never-selected (staged) generation (`not-selected-generation`); `Store.diagnostic_read_member` can read a staged generation and is **diagnostic-only**, never to be exposed by a served API (R2b).

Extra rows not exercised by the model (never read as PASS):

| Item | Result |
|---|---|
| Real Postgres transaction / advisory-lock behaviour of `replace_source_chunks_atomic` | NOT_MEASURED |
| Crash recovery of a real process / DB (only an injected exception is modelled) | NOT_MEASURED |
| Real Auth Arcana / `principal_namespace_grants` revocation latency | NOT_MEASURED |
| Live AC5 on an accepted pilot generation (needs KCLIVE-07 accepted source, KCLIVE-08 assembler) | NOT_MEASURED |

## Faulty variants rejected by the checker (AC2, AC3, AC4)

Each faulty variant was run through the same checkers; `rejected=True` means the checker disagreed with it.

| Case/variant | Rejected | First divergent invariant | Detail |
|---|---|---|---|
| ac4/LeakyViewStore | True | `I3:candidate-isolated-from-accepted-view` | g2:doc-a |
| c02/NoValidationStore | True | `I2:pointer-names-complete-verified-generation` | doc-b |
| c03/NoValidationStore | True | `I2:pointer-names-complete-verified-generation` | doc-b |
| c03b/NoValidationStore | True | `I2:pointer-names-complete-verified-generation` | doc-b |
| c04/NoValidationStore | True | `I2:pointer-names-complete-verified-generation` | doc-c |
| c05/NoValidationStore | True | `I7:pointer-source-is-accepted-current` | pin-unaccepted |
| c05b/NoValidationStore | True | `I7:pointer-namespace-is-own` | other-ns |
| c06/NoValidationStore | True | `I7:dependency-closure-complete` | doc-a->doc-zzz |
| c07/NoValidationStore | True | `I7:pointer-project-matches-request` | other-proj!=synthetic-proj |
| c08/CachedViewStore | True | `I4:revoked-reader-denied-at-use` | reader-A |
| c09/LastWriterWinsStore | True | `I5:cas-exactly-one-winner` | winners=['ga', 'gb'] |
| c10/InPlaceStore | True | `I1:previous-accepted-member-readable-at-old-digest` | doc-a |
| c10/LeakyViewStore | True | `I3:candidate-isolated-from-accepted-view` | g2:doc-a |
| c12/CrossNsStore | True | `I6:no-cross-namespace-io` | ('write', 'shared-ns') |
| r1a/OverwriteStore | True | `I8:selected-manifest-immutable` | g1 |
| r1b/OverwriteStore | True | `I8:selected-member-bytes-immutable` | doc-a |
| r1c/OverwriteStore | True | `I8:selected-manifest-immutable` | g1 |
| r1d/CheckAsYouGoStore | True | `I8b:refused-call-leaves-no-state-change` |  |
| r1f/NoManifestGuardStore | True | `I8b:refused-call-leaves-no-state-change` |  |
| r2a/PointerOnlyCasStore | True | `I9:aba-stale-version-promotion-refused` | promoted ga |
| r2b/UnguardedReadStore | True | `I10:staged-generation-not-served` | g2/doc-a |
| r2c/RootOnlyDepStore | True | `I7:dependency-closure-complete` | doc-b->doc-c |
| r4a/NoValidationFenceStore | True | `I11:direct-promote-requires-validation-witness` | promoted not-registered |
| r4b/NoValidationFenceStore | True | `I11:direct-promote-requires-validation-witness` | promoted g2 |
| r4d/NoValidationFenceStore | True | `I11:direct-promote-requires-validation-witness` | promoted g2 |
| r5a/NoSourceCurrencyFenceStore | True | `I11:direct-promote-requires-validation-witness` | promoted g2 |
| r6a/UnlockedNoRevisionStore | True | `I12:source-mutation-serialized-with-selection` | ptr=g2 src1_accepted=False |
| r6b/SnapshotBeforeLockStore | True | `I11:direct-promote-requires-validation-witness` | promoted promoted |
| r6b/UnlockedSourceMutationStore | True | `I12:source-mutation-serialized-with-selection` | promote_blocked=False result='promoted' |
| r6b2/UnlockedSourceMutationStore | True | `I12:source-mutation-serialized-with-selection` | promote_blocked=False result=NotServable('non-current-source: pin-0001') |
| r6c/UnlockedNoRevisionStore | True | `I11:direct-promote-requires-validation-witness` | promoted g2 |

Mapping: `InPlaceStore` = overwrite-in-place of the active generation's source paths (AC2, 77307111 counterexample; `I1` previous accepted member no longer reads at its old digest). `LastWriterWinsStore` = promotion ignoring the expected pointer (AC3; `I5`). `LeakyViewStore` = candidate chunks visible in the accepted view (AC4; `I3`). `CachedViewStore` = old complete bytes served after revocation (AC4; `I4`). `NoValidationStore` = metadata-only promotion (`I2`/`I7`). `CrossNsStore` = write mirrored to another namespace (`I6`).

## Provenance

- Scrutator base: `origin/main` = `1b697e97c6842ea92a86de0cc8933c18bd24cb14` (required: at or after 0cb546a3). Branch head at run time: `8e63683c8304e1b247037463d4bb5d5838aa98d5` (working tree may carry uncommitted report).
- Source sha256: `generation_model.py` = `d41f7c0f877fb1766876a97269308d13e72d4621f477dcdf72271b87d2f25216`
- Fixture/test sha256: `test_generation_closure_model.py` = `4169f281e81e8ab3a54f0e9bea73cc4b331c279927d35ae4e578ffd853d6e5df`; `conftest.py` = `49cb4acdbd89b7f10d9cc7089acc738fd0a21172422578d5bd6e7ca53b440225`; `run_and_report.py` = `6b07992ab3a690c2c6e8b71122f77e1054a6c0415717f2d5a87ae59c74a0d36a`
- Fixtures are inline synthetic byte strings (`OLD-*`, `NEW-*`); no external fixture file exists.
- Command: `cd scrutator && PYTHONPATH=src python -m pytest tests/ac5_model -q -p no:cacheprovider` (via `python tests/ac5_model/run_and_report.py REPORT.md`)
- Exit code: **0**
- Environment: Python 3.12.3, Linux-6.8.0-147-generic-x86_64-with-glibc2.39, host `arcana-devs`; pytest summary: `44 passed in 4.06s`

## Model assumptions

1. Storage is an in-memory dict keyed by (namespace, key); immutable staging = generation-qualified key `gen/artifact@revision`. The real mapping to Scrutator `source_path` / `evidence_documents` rows is not implemented.
2. The generation pointer is a single in-memory register guarded by a `threading.Lock` with a version counter; this stands in for a native atomic CAS that **does not exist in Scrutator today** (see below).
3. Authorization is a set of (reader, namespace) grants consulted on every read; no JWT, token lifetime, cache TTL or propagation delay is modelled. Revocation is instantaneous.
4. Failure is an injected Python exception after N writes; no process kill, partial page write, WAL, or restart is modelled.
5. The competing-promotion case interleaves deterministically (both validate, then both promote) plus a 4-thread barrier stress; it does not explore all schedules.
6. Digest domain is SHA-256 over raw bytes + length. Document-ID, revision, receipt and source-hash domains are reduced to (artifact_id, revision, source_pin); their distinctness is modelled only by separate fields.
7. Dependency closure is a finite same-manifest check; a transitive chain is exercised (R2c); cyclic closure is not modelled.
8b. Immutable identity (R1) is enforced by the model's staging step; the real storage layer must provide the same guarantee (unique/immutable key per generation member). Not implemented here.
8. Old-generation retention/cleanup is out of scope; the model never deletes a generation.
9. (R3, round 4) Selection is fenced: `Store.promote` first runs `_admission_fence`, which requires a registered generation, an `AdmissionWitness` issued only by a successful `Store.validate` of exactly the registered manifest, bound to (generation_id, manifest, project, namespace, source pin, digest of the member bytes actually read), and re-derives the bytes digest from the stored bytes at promote time. The fence runs before the CAS comparison and before any pointer/version change. It carries no authorization; grants are re-checked at every use (`accepted_view`, `read_member`). The witness is a model object: a native implementation would need an equivalent durable, transactionally bound admission record. 
10. (round 5) Source currency is a separate selection check: `Store._source_currency_fence` runs under the same lock, after the admission fence and the CAS comparison and immediately before the pointer/version change, and refuses a registered generation whose source pin is not in `accepted_sources` NOW. Rule chosen: the live set is read at every promote attempt (it is NOT modelled as frozen for the transaction); a later re-acceptance of the same pin needs no fresh validate (witness binds the pin and bytes, unchanged) and nothing is cached; a withdrawal after selection does not unselect. Grants stay use-time. The native service must implement the same rule in the real selection transaction.
11. (round 6) Source currency is atomic at the promotion linearization point. `accepted_sources` is no longer an externally mutable set: it is a private copy exposed as a read-only frozenset, changed only by `Store.accept_source` / `Store.withdraw_source`, which take the SAME lock as `promote` and bump `source_revision`. `promote` reads the revision, runs the currency fence, and `_source_revision_fence` compares the revision again immediately before the pointer/version change (a best-effort tripwire, not an atomic primitive: the compare and the pointer assignment are separate steps). So, for writers that take the shared lock, a withdrawal either completes strictly before the selection (the selection refuses, nothing changes) or strictly after the switch (the switch stays valid; later validate/promote of that source refuse; no unselect). Adversarial interleavings are pinned with two deterministic seams (`_before_switch`, `_in_source_mutation`) and threads. SCOPE: the guarantee holds only for cooperating writers, i.e. writers that take the shared lock/transaction; that lock/transaction is mandatory for ALL native source mutations (accept, withdraw) and for selection (promote). Defence against a non-cooperating writer that bypasses the lock is NOT_MEASURED: the tests pin only that a lock-skipping writer that still bumps the revision before the compare is refused (mutant i/iii), not an atomic fence-and-swap, and a bypassing writer landing after the compare is not modelled. This is a model contract with an in-process lock, not a native race measurement. The native service must implement the same rule in the real selection transaction (current-source compare and pointer swap inside one database transaction, every source mutation inside it); native/database behaviour is NOT_MEASURED here.

## What this model does NOT prove

- That Scrutator's `index_documents()/_persist_prepared()` / `replace_source_chunks_atomic()` behave as the safe model; the unsafe `InPlaceStore` is a model of the hazard named in 77307111, not a measurement of the real code.
- Real DB transaction semantics, isolation level, lock behaviour, or crash recovery.
- Real current authorization (Auth Arcana, `principal_namespace_grants`, revocation propagation).
- Any live AC5 behaviour, deployment, accepted pilot generation, or the real consumer fetch path (`fetcher.py`, `content_exact`).
- Cross-namespace isolation of the real service (only the model's own I/O log is checked).

## Native generation-switch mechanism in Scrutator today

**None found** at `0cb546a3`. Search of `src/` (including `db/migrations/001..006`) finds no generation/manifest/pointer table or compare-and-swap selection; `replace_source_chunks_atomic()` replaces one `(namespace_id, source_path)` under an advisory lock per source, and `index_documents()` persists documents independently with mixed results. Existing `CAS` hits are NULL-only entity repair tools unrelated to source generations. **Blocker for the real implementation:** a native atomic CAS generation-switch (pointer table + versioned compare-and-set against the expected current pointer, plus generation-qualified storage identities so staged members never reuse the active generation's `source_path`) must be designed and added; the validator alone cannot prevent a writer overwriting the previous generation.
