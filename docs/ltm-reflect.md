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
| `LTM_REFLECT_GROUPING` | `cosine` | Grouping primitive — `entity` (LTM-0013) or `cosine` (LTM-0018) |
| `LTM_REFLECT_COSINE_THRESHOLD` | `0.85` | Cosine edge threshold for union-find clustering |

## A2 Cosine Grouping (LTM-0018)

Default grouping primitive since LTM-0018. Replaces single-entity-name JOIN of
LTM-0013 with content-based clustering on dense BGE-M3 embeddings.

**Algorithm** (`scrutator.ltm.grouping.cluster_by_cosine`):

1. `SELECT id, content, embedding_dense FROM chunks WHERE embedding_dense IS NOT NULL ORDER BY id`.
2. Build `sims = V @ V.T` (n × n cosine matrix; assumes unit-norm BGE-M3 vectors).
3. Union-find over edges with `sims[i,j] ≥ threshold` (default `0.85`).
4. Emit groups of size ≥ 2; singletons filtered.

**Determinism:** stable cluster roots when caller passes `ORDER BY chunk_id` and
numpy version is pinned (`requirements.txt`).

**Resource bound:** O(n²) memory + time. Capped at
`LTM_REFLECT_MAX_CHUNKS_PER_RUN=50` (≈ 200 KB / <2 ms per run). DoS-safe.

**Schema contract — `meta_facts.entity_ids` MAY be empty.** Cosine-grouped
meta-facts have `entity_ids = []` because cluster membership is not anchored to
a specific entity. Downstream consumers MUST handle the empty case (do NOT
filter via `WHERE entity_ids @> '{X}'`; query by `source_chunk_ids` instead).

**Trust boundary:** clustering trusts pre-stored embeddings. Adversarial
embeddings inserted via the ingest path could induce mega-clusters; embedding
dimension validation (1024) at INSERT remains the boundary control.

**Fallback to LTM-0013 entity grouping:** set `SCRUTATOR_LTM_REFLECT_GROUPING=entity`.

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
| Grouping (LTM-0018) | `SCRUTATOR_LTM_REFLECT_GROUPING=entity` (revert to LTM-0013 entity-path) |
| Threshold tightening | `SCRUTATOR_LTM_REFLECT_COSINE_THRESHOLD=0.95` (collapse to A1-floor behaviour) |
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
