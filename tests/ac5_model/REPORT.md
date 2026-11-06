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
| AC4 incomplete candidate absent from accepted view | PASS |
| META checker rejects every faulty variant | PASS |

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

Mapping: `InPlaceStore` = overwrite-in-place of the active generation's source paths (AC2, 77307111 counterexample; `I1` previous accepted member no longer reads at its old digest). `LastWriterWinsStore` = promotion ignoring the expected pointer (AC3; `I5`). `LeakyViewStore` = candidate chunks visible in the accepted view (AC4; `I3`). `CachedViewStore` = old complete bytes served after revocation (AC4; `I4`). `NoValidationStore` = metadata-only promotion (`I2`/`I7`). `CrossNsStore` = write mirrored to another namespace (`I6`).

## Provenance

- Scrutator base: `origin/main` = `0cb546a395008b501fd47c6b1037bcae8e7e664f` (required: at or after 0cb546a3). Branch head at run time: `0cb546a395008b501fd47c6b1037bcae8e7e664f` (working tree may carry uncommitted report).
- Source sha256: `generation_model.py` = `45ed0fb1900ae40121edb3a9ffb7343c206d115157b75b9a9b023e11d351b598`
- Fixture/test sha256: `test_generation_closure_model.py` = `ed51fca721f3811c724213d78e674df5769ff0078eb395338a072a6248ab063e`; `conftest.py` = `b98481ab527def4b3a199f6e262f744477f155d46d34195e37b236d496b4033d`; `run_and_report.py` = `8b6d2c1ae222144a5769defc3cabe20125e30dbc45dacc9ce64c137008da5174`
- Fixtures are inline synthetic byte strings (`OLD-*`, `NEW-*`); no external fixture file exists.
- Command: `cd scrutator && PYTHONPATH=src python -m pytest tests/ac5_model -q -p no:cacheprovider` (via `python tests/ac5_model/run_and_report.py REPORT.md`)
- Exit code: **0**
- Environment: Python 3.12.3, Linux-6.8.0-147-generic-x86_64-with-glibc2.39, host `arcana-devs`; pytest summary: `16 passed in 0.85s`

## Model assumptions

1. Storage is an in-memory dict keyed by (namespace, key); immutable staging = generation-qualified key `gen/artifact@revision`. The real mapping to Scrutator `source_path` / `evidence_documents` rows is not implemented.
2. The generation pointer is a single in-memory register guarded by a `threading.Lock` with a version counter; this stands in for a native atomic CAS that **does not exist in Scrutator today** (see below).
3. Authorization is a set of (reader, namespace) grants consulted on every read; no JWT, token lifetime, cache TTL or propagation delay is modelled. Revocation is instantaneous.
4. Failure is an injected Python exception after N writes; no process kill, partial page write, WAL, or restart is modelled.
5. The competing-promotion case interleaves deterministically (both validate, then both promote) plus a 4-thread barrier stress; it does not explore all schedules.
6. Digest domain is SHA-256 over raw bytes + length. Document-ID, revision, receipt and source-hash domains are reduced to (artifact_id, revision, source_pin); their distinctness is modelled only by separate fields.
7. Dependency closure is a finite same-manifest check; transitive/cyclic closure is not modelled.
8. Old-generation retention/cleanup is out of scope; the model never deletes a generation.

## What this model does NOT prove

- That Scrutator's `index_documents()/_persist_prepared()` / `replace_source_chunks_atomic()` behave as the safe model; the unsafe `InPlaceStore` is a model of the hazard named in 77307111, not a measurement of the real code.
- Real DB transaction semantics, isolation level, lock behaviour, or crash recovery.
- Real current authorization (Auth Arcana, `principal_namespace_grants`, revocation propagation).
- Any live AC5 behaviour, deployment, accepted pilot generation, or the real consumer fetch path (`fetcher.py`, `content_exact`).
- Cross-namespace isolation of the real service (only the model's own I/O log is checked).

## Native generation-switch mechanism in Scrutator today

**None found** at `0cb546a3`. Search of `src/` (including `db/migrations/001..006`) finds no generation/manifest/pointer table or compare-and-swap selection; `replace_source_chunks_atomic()` replaces one `(namespace_id, source_path)` under an advisory lock per source, and `index_documents()` persists documents independently with mixed results. Existing `CAS` hits are NULL-only entity repair tools unrelated to source generations. **Blocker for the real implementation:** a native atomic CAS generation-switch (pointer table + versioned compare-and-set against the expected current pointer, plus generation-qualified storage identities so staged members never reuse the active generation's `source_path`) must be designed and added; the validator alone cannot prevent a writer overwriting the previous generation.
