# Private generation selection (KB-CONS-01-N)

Reference for the native generation-selection primitive. It is a **synthetic-fixture slice**: it can be exercised on
owned synthetic data, it does not start or accept the private consumer, and it ships with **no real private-use
authority** (every use refuses until an owner configures one).

## Where it lives

| Piece | Location |
| --- | --- |
| Installer (the real one) | `src/scrutator/db/schema.sql` — block between `KC_GENERATION_SCHEMA_BEGIN/END` |
| Review snapshot (byte-identical block) | `src/scrutator/db/migrations/007_kc_generations.sql` |
| Primitive | `src/scrutator/db/kc_generations.py` |
| Eligibility in ranking arms | `db/repository.py` (`hybrid_search`, `search_with_filters`, `generation=`) |
| Private read routes | `src/scrutator/kc_router.py` (`/v1/kc/read/*`) |
| Legacy bypass closure | `auth/dependency.py` (`require_tenant_context` hides private namespaces) |

## Records

`kc_corpus_scopes` (registered scope + finite limits) → `kc_generations` (immutable manifest bytes, state) →
`kc_generation_members` (exact expected bytes/digest, server-generated opaque native path) → `kc_selections` (one row per
scope: `incarnation_id`, `revision BIGINT`, `generation_id`, `manifest_sha256`) → `kc_selection_operations`
(committed outcome, same transaction) → `kc_read_leases` (retention, never an authorization grant).

## Operations

1. `create_candidate` — immutable manifest + expected member rows; same id with different bytes is `GENERATION_ID_CONFLICT`
   before any change.
2. `stage_member` — populate one member under the staging writer GUC; identical retry is a no-op; a populated member is never
   replaced; U+0000 / non-UTF-8 bodies are refused.
3. `seal_generation` — exact member-set equality, closure equal to the independent oracle, fresh full-byte readback.
4. `promote` — one short transaction: selection row lock, operation lookup, sealed-candidate lock, current authority,
   conditional `UPDATE … WHERE incarnation AND revision AND prior generation`, operation row. Zero rows is
   `SELECTION_CONFLICT`. Same operation id + same arguments replays the historical outcome with a separate current
   observation (`is_current`).
5. `begin_read` / `authorize_use` / `renew_lease` — one captured selection + lease; every use re-checks lease state (database
   clock) and current authority.
6. `mark_gc_pending` — non-destructive half of GC. **Destructive collection is disabled** (`collect_generation` raises
   `GC_DISABLED`).

Lock order: `kc_selections` → `kc_generations` → member/source/chunk rows. Nothing takes a selection lock after a generation
lock; staging takes only the generation (share) and member (update) rows.

## Refusal codes

`CURRENT_AUTHORITY_UNAVAILABLE`, `AUTHORITY_REVOKED`, `SELECTION_CONFLICT`, `OPERATION_CONFLICT`,
`GENERATION_ID_CONFLICT`, `GENERATION_NOT_SERVABLE`, `LEASE_EXPIRED`, `LEASE_NOT_FOUND`, `LIMIT_EXCEEDED`,
`UNSUPPORTED_PRIVATE_OPERATION`, `MEMBER_MISMATCH`, `MEMBER_CONFLICT`, `NUL_IN_BODY`, `LOCK_FAILURE`, `GC_DISABLED`.

## Known limits

* The database triggers are a fail-closed backstop, not a privilege boundary; separated write roles are not measured.
* The synthetic manifest profile (`KCSyntheticManifest/v1`) is not the producer's reviewed manifest; the real producer
  validator is not wired in.
* Entity graph / LTM views are refused (`UNSUPPORTED_PRIVATE_OPERATION`) until a version-bound reconstruction exists.
* Tests need a disposable PostgreSQL with pgvector: `tests/kc_generations/run_with_temp_pg.sh` (see its header). Without
  `KC_TEST_PG_ADMIN_DSN` they skip and are **not measured**.
