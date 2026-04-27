# LTM Reflect Layer (LTM-0013)

The Reflect layer (R in TEMPR) derives **meta-facts** — concise summaries,
contradictions, and derived relations — from groups of related chunks. It
closes the feature-parity gap with Hindsight and unblocks the LTM benchmark
(LTM-0009).

## Status

- Code merged in Scrutator 0.3.0.
- Migration `003_reflect.sql` adds `meta_facts` and `reflect_runs`.
- Recall integration is **disabled by default** (`SCRUTATOR_LTM_RECALL_INCLUDE_META_FACTS=false`)
  until verified on the 41-chunk LTM-0012 corpus.

## Endpoints

### `POST /v1/ltm/reflect`

Trigger one reflect run.

```bash
curl -X POST http://arcana-db:8310/v1/ltm/reflect \
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
curl "http://arcana-db:8310/v1/ltm/meta_facts?namespace=datarim-kb&fact_type=summary&limit=20"
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
| Reflect | `SCRUTATOR_LTM_REFLECT_ENABLED=false` → 503 |
| Schema | `DROP TABLE meta_facts CASCADE; DROP TABLE reflect_runs CASCADE;` |
| Code | `git revert <range>` and redeploy 0.2.0 container |

All four are non-destructive to LTM-0012 state.

## Pilot run (Step 11)

```bash
curl -X POST http://arcana-db:8310/v1/ltm/reflect \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"namespace":"datarim-kb","max_chunks":50,"dry_run":false}' | jq .
```

Expected: `status=done`, `meta_facts_created>=1`, `cost_usd=0.0`, duration <5min.

## Factor sweep (Step 12)

For each factor in `{0.5, 0.7, 0.9, 1.0}`:

```bash
SCRUTATOR_LTM_RECALL_META_FACT_SCORE_FACTOR=$f \
SCRUTATOR_LTM_RECALL_INCLUDE_META_FACTS=true \
pnpm run bench:scrutator -- --with-meta-facts \
  --score-factor=$f \
  --report=benchmark/reports/v5/scrutator/$DATE.with-meta-facts-factor-$f.json
```

Decision rule:
`chosen = argmax_{factor in {0.5, 0.7, 0.9}} mean_recall@5(factor)`
subject to `by_class.factual(chosen) >= baseline - 2pp`.

Factor=1.0 is diagnostic only (excluded from default candidates due to
hallucination amplification risk).
