# Operate the LTM Reflect Layer

The Reflect layer (R in TEMPR) derives **meta-facts** — concise summaries,
contradictions, and derived relations — from groups of related chunks. It
closes the feature-parity gap with Hindsight and supports the LTM benchmark.

## Status

- Code merged in Scrutator 0.3.0.
- Migration `003_reflect.sql` adds `meta_facts` and `reflect_runs`.
- Recall integration is **disabled by default** (`SCRUTATOR_LTM_RECALL_INCLUDE_META_FACTS=false`)
  until a reviewed corpus run confirms it.

## Endpoints

### `POST /v1/ltm/reflect`

Trigger one reflect run.

```bash
curl -X POST "$SCRUTATOR_BASE_URL/v1/ltm/reflect" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"namespace": "datarim-kb", "max_chunks": 50, "dry_run": false}'
```

Body fields (all optional):
- `namespace` (default `"arcanada"`)
- `since` — ISO-8601 timestamp; only chunks indexed at or after this time
- `max_chunks` — overrides `SCRUTATOR_LTM_REFLECT_MAX_CHUNKS_PER_RUN`
- `dry_run` — if `true`, returns preview without writing to DB

Response:
```json
{
  "summary": {
    "run_id": "...",
    "status": "done|aborted|failed",
    "chunks_scanned": 41,
    "meta_facts_created": 7,
    "cost_usd": 0.0,
    "req_count": 5,
    "abort_reason": null,
    "duration_ms": 28453.21
  },
  "preview": null
}
```

### `GET /v1/ltm/meta_facts`

Listing for debug / inspection.

```bash
curl "$SCRUTATOR_BASE_URL/v1/ltm/meta_facts?namespace=datarim-kb&fact_type=summary&limit=20"
```

## Configuration (env, prefix `SCRUTATOR_`)

| Key | Default | Purpose |
|-----|---------|---------|
| `LTM_REFLECT_ENABLED` | `true` | Master kill-switch — `false` → 503 |
| `LTM_REFLECT_MAX_CHUNKS_PER_RUN` | `50` | Cap per run |
| `LTM_REFLECT_MAX_META_FACTS_PER_CHUNK` | `5` | Cap per group |
| `LTM_REFLECT_BUDGET_USD` | `0.01` | Hard $ cap (run aborts) |
| `LTM_REFLECT_BUDGET_REQ_COUNT` | `100` | Hard request cap |
| `LTM_REFLECT_MAX_DEPTH` | `1` | DB-level + code invariant |
| `LTM_RECALL_INCLUDE_META_FACTS` | `false` | Recall verification gate |
| `LTM_RECALL_META_FACT_SCORE_FACTOR` | `0.7` | Score penalty on meta-facts |
| `LTM_REFLECT_GROUPING` | `cosine` | Grouping primitive — `entity` or `cosine` |
| `LTM_REFLECT_COSINE_THRESHOLD` | `0.85` | Cosine edge threshold for union-find clustering |

## Cosine Grouping

The default grouping primitive replaces single-entity-name joins with
content-based clustering on dense BGE-M3 embeddings.

**Algorithm** (`scrutator.ltm.grouping.cluster_by_cosine`):

1. `SELECT id, content, embedding_dense FROM chunks WHERE embedding_dense IS NOT NULL ORDER BY id`.
2. Build `sims = V @ V.T` (n × n cosine matrix; assumes unit-norm BGE-M3 vectors).
3. Union-find over edges with `sims[i,j] ≥ threshold` (default `0.85`).
4. Emit groups of size ≥ 2; singletons filtered.

**Determinism:** stable cluster roots require a stable input order such as
`ORDER BY chunk_id`. NumPy is declared as `numpy>=1.26`, not exactly pinned, so
record and reinstall the exact resolved NumPy version when reproducing results
across environments.

**Resource bound:** O(n²) memory + time. Capped at
`LTM_REFLECT_MAX_CHUNKS_PER_RUN=50` (≈ 200 KB / <2 ms per run). DoS-safe.

**Schema contract — `meta_facts.entity_ids` MAY be empty.** Cosine-grouped
meta-facts have `entity_ids = []` because cluster membership is not anchored to
a specific entity. Downstream consumers MUST handle the empty case (do NOT
filter via `WHERE entity_ids @> '{X}'`; query by `source_chunk_ids` instead).

**Trust boundary:** clustering trusts pre-stored embeddings. Adversarial
embeddings inserted via the ingest path could induce mega-clusters; embedding
dimension validation (1024) at INSERT remains the boundary control.

**Fallback to entity grouping:** set `SCRUTATOR_LTM_REFLECT_GROUPING=entity`.

## Safety invariants

- **Depth=1** enforced by Pydantic validator + DB CHECK constraint.
  Reflect-of-reflect is rejected at every layer.
- **Provenance** — each meta-fact stores `source_chunk_ids UUID[]` (≥1 entry).
- **Namespace isolation** — `ReflectJob` runs against one `namespace_id`.
- **Budget caps** — `ReflectBudgetExceeded` aborts mid-run with
  `status=aborted` and `abort_reason` recorded in `reflect_runs`.

## Rollback

| Layer | Command |
|-------|---------|
| Recall | `SCRUTATOR_LTM_RECALL_INCLUDE_META_FACTS=false` (default off) |
| Grouping | `SCRUTATOR_LTM_REFLECT_GROUPING=entity` (revert to the entity path) |
| Threshold tightening | `SCRUTATOR_LTM_REFLECT_COSINE_THRESHOLD=0.95` (reduce cluster size) |
| Reflect | `SCRUTATOR_LTM_REFLECT_ENABLED=false` → 503 |
| Schema | `DROP TABLE meta_facts CASCADE; DROP TABLE reflect_runs CASCADE;` |
| Code | `git revert <range>` and redeploy 0.2.0 container |

All four are non-destructive to the underlying chunk state.

## Pilot run

```bash
curl -X POST "$SCRUTATOR_BASE_URL/v1/ltm/reflect" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"namespace":"datarim-kb","max_chunks":50,"dry_run":false}' | jq .
```

Expected: `status=done`, `meta_facts_created>=1`, `cost_usd=0.0`, duration <5min.

## Production TEMPR backfill

`tools/backfill_ltm_temper.py` — one-shot, idempotent/resumable backfill of
entity/edge/temporal-event extraction for chunks that predate the TEMPR
pipeline (zero rows in `entities` pointing back at them via `source_chunk_id`).
Same hard-gate convention as `tools/backfill_sections.py` (see
the [navigation reference](../reference/navigation.md)): defaults to `--dry-run` (report candidate count, zero
LLM calls, zero writes); `--live` performs real extraction (billed LLM calls
via `settings.ltm_mc_url`/`ltm_connector`/`ltm_model`) and real upserts.
**Operator-run only — not invoked by CI, by tests, or by any task automation.**

```bash
python tools/backfill_ltm_temper.py --namespace arcanada
python tools/backfill_ltm_temper.py --namespace arcanada --live --limit 200
```

Idempotent: `upsert_entity`/`upsert_entity_edge`/`upsert_entity_event` are
`ON CONFLICT` upserts repair `source_chunk_id` on every
re-run), so a crashed or `--limit`-batched run can simply be re-invoked — only
chunks still missing entities are re-selected. Per-chunk extraction failures
are logged and skipped, not fatal to the run.

## Periodic runner

`python -m scrutator.ltm.reflect_runner` runs one bounded incremental reflect
pass from inside the Scrutator runtime. It uses the same LLM settings and budget
caps as `POST /v1/ltm/reflect`, persists a UTC cursor, and advances that cursor
only after a non-dry-run summary with `status="done"`.

```bash
python -m scrutator.ltm.reflect_runner \
  --namespace wiki \
  --state-file /var/lib/scrutator/ltm-reflect/cursor.json \
  --max-chunks 50 \
  --dry-run
```

Production wrapper files:

- `deploy/ltm-reflect-run.sh` — host-side `docker exec` wrapper with a
  state-directory lock.
- `deploy/ltm-reflect.service` — hardened oneshot service.
- `deploy/ltm-reflect.timer` — hourly timer.

The cursor directory is a fail-closed bind mount shared by the host wrapper and
the Scrutator container. Provision it before `docker compose up` and reject
symlinks; Compose will not create it implicitly:

```bash
install -d -o root -g root -m 0700 /var/lib/scrutator/ltm-reflect
```

`docker inspect` must show that exact host path mounted read-write at the same
container path. The cursor is not durable if it exists only in the container
overlay, so do not create the readiness marker until persistence across a
controlled container recreation has been verified.

The service has `ConditionPathExists=/var/lib/scrutator/ltm-reflect-ready`.
Create that marker only after a bounded safety review, an observed dry-run, a
successful supervised backlog drain for the configured namespace, and cursor
persistence across container recreation. Namespace isolation means the `wiki`
reflect runner does not authorize or trigger the operator-gated `arcanada`
full-corpus backfill. Before the marker exists,
installing or enabling the timer remains inert.
