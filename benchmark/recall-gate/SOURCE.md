# Recall Gate — Harness Provenance

## Harness invocation strategy

The gate uses **Option A (path invocation)** — the harness is invoked by absolute path on the arcana-db self-hosted runner, pointing to the Disk Arcana-synced `~/arcanada` tree.

**CI env var:** `HARNESS_PATH=/home/ci-runner/arcanada/Projects/Long Term Memory/benchmark/scripts/ltm-bench-query.py`

The arcana-db runner has the Mac-synced Arcanada KB tree available at `~/arcanada/` (pull follower via Disk Arcana, per ADR-0001). The harness and its query JSONL files (`benchmark/queries/*.jsonl`) are already present there. No vendoring is needed.

## Vendoring fallback (Option A unavailable)

If the runner is sandboxed to the Scrutator repo checkout only and cannot read `~/arcanada/`:

1. Copy the harness and query files into `benchmark/recall-gate/vendor/`:
   - `vendor/ltm-bench-query.py` — from `Projects/Long Term Memory/benchmark/scripts/ltm-bench-query.py`
   - `vendor/queries/factual.jsonl`
   - `vendor/queries/multi-hop.jsonl`
   - `vendor/queries/temporal.jsonl`
2. Update `HARNESS_PATH` in `recall-regression.yml` to `benchmark/recall-gate/vendor/ltm-bench-query.py`
3. Update this file with the source commit SHA and copy date.

**Vendored copy provenance (fill in if vendoring is activated):**
- Source repo: `arcanada` (root KB git repo)
- Source path: `Projects/Long Term Memory/benchmark/scripts/ltm-bench-query.py`
- Source commit: _TBD_
- Copied at: _TBD_

## Gate mode: with-entities (expand_entities=true)

The gate runs the harness with `--expand-entities` and compares against a `with-entities` baseline.

**Why with-entities?**  The production `/v1/ltm/recall` endpoint has `expand_entities: bool = True`
as its default in `RecallRequest` (see `src/scrutator/ltm/models.py`).  Any agent calling the
endpoint without explicitly passing `expand_entities=false` receives entity-enriched results.
The gate must defend the retrieval path that real users actually experience.  A `no-entities`
baseline would guard a non-default opt-out mode and let regressions in the production path go
undetected.

Mode consistency rule: `baseline.json` `mode` field, harness invocation flag (`--expand-entities`),
and the report filename suffix (`.with-entities.json`) must always agree.  A mismatch causes a
born-red gate — observed all-zeros against a non-zero baseline every build for the wrong reason.

## Baseline seeding note

`baseline.json` was seeded from `reports/v4/scrutator/2026-04-26.datarim-kb.with-entities.json`
(36-query run, expand_entities=true, real measured recall).  The live endpoint returned HTTP 500
on 2026-06-21 (PostgreSQL connectivity issue, not a Scrutator code regression).  Recalibrate via
`--update-baseline` after the first clean CI run on the arcana-db runner (requires live endpoint).
