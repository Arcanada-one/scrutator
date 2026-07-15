# Recall Gate — Harness Provenance

## Harness invocation strategy

The gate uses the documented vendoring fallback because the arcana-kb
self-hosted runner no longer has the historical `~/arcanada` follower path.
The harness and its three query files are committed beneath
`benchmark/recall-gate/vendor/`, making the gate checkout-local.

**CI env var:** `HARNESS_PATH=benchmark/recall-gate/vendor/ltm-bench-query.py`

Update the snapshot deliberately when the canonical harness changes; do not
silently fall back to a host path.

## Vendoring fallback (Option A unavailable)

If the runner is sandboxed to the Scrutator repo checkout only and cannot read `~/arcanada/`:

1. Copy the harness and query files into `benchmark/recall-gate/vendor/`:
   - `vendor/ltm-bench-query.py` — from `Projects/Long Term Memory/benchmark/scripts/ltm-bench-query.py`
   - `vendor/queries/factual.jsonl`
   - `vendor/queries/multi-hop.jsonl`
   - `vendor/queries/temporal.jsonl`
2. Update `HARNESS_PATH` in `recall-regression.yml` to `benchmark/recall-gate/vendor/ltm-bench-query.py`
3. Update this file with the source commit SHA and copy date.

**Vendored copy provenance:**
- Source repo: `arcanada` (root KB git repo)
- Source path: `Projects/Long Term Memory/benchmark/scripts/ltm-bench-query.py`
- Query source path: `Projects/Long Term Memory/benchmark/queries/`
- Source commit: `270a3c0843d5c41d2a61feedadc5477d56bd4f45`
- Copied at: `2026-07-15`
- Harness adaptation: only `QUERIES_DIR` and report directories were made
  relative to the vendored script directory; benchmark logic is unchanged.
- Vendored harness SHA-256: `a9c0e304435b25b1d90d3bc31f791735cf72ed9658c0353ad4356d160d2cee5c`
- Factual SHA-256: `66ebbec22459763f6337d87503bbd35a913d10c2c1481d36b242fab13fc20767`
- Multi-hop SHA-256: `136af39e509a18658380473350929a313019003afdf717d67b1dec078f5f595f`
- Temporal SHA-256: `14206a5707bb12afd30aee16d2b42ecd8e98ea7a4e28a701e24df2848535a032`

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
