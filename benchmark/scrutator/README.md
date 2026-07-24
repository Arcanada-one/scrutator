# benchmark/scrutator/ — Scrutator `/v1/search` recall harness

**Task:** SRCH-0015. **Governs:** golden-set versioning + a multi-model recall/MRR/nDCG harness
for Scrutator's hybrid search endpoint (`/v1/search`), with a runtime liveness pre-flight so a
dead corpus path reads as `SKIPPED (stale)` instead of a silent recall miss.

## Why a third (fourth) lineage, not a merge into an existing one

This repo has three other benchmark trees. This one is deliberately separate from all of them —
here is why, so nobody re-merges them by accident later:

- **`benchmark/scripts/` (TypeScript, LTM ingest/embedding harness — `bench-runner.ts`,
  `metrics.ts`, `queries.ts`).** Different language, different target: measures the *Long Term
  Memory* ingest/embedding pipeline, not Scrutator's `/v1/search` endpoint. `benchmark/recall-gate/`
  wraps this harness's Python sibling (`ltm-bench-query.py`) as a CI gate for the
  `/v1/ltm/recall` path — see below.
- **`benchmark/recall-gate/`** — the standing CI gate (SRCH-0030 lineage) that runs
  `ltm-bench-query.py` against the `datarim-kb` 36-query set over `/v1/ltm/recall` and compares
  per-class recall@5 against a committed baseline. It is a **consumer of a different harness**
  (the TS/LTM one), targets a **different endpoint** (`/v1/ltm/recall`, not `/v1/search`), and
  measures a different corpus (`datarim-kb`, not `arcanada`). This directory's harness produces
  an independent signal; it does not extend or replace `recall-gate`.
- **`benchmark/ml-vs-llm/`** — the ML-vs-LLM extraction-task benchmark suite (SRCH-0018):
  NER, relation extraction, classification, and its own end-to-end recall bench, comparing
  small ML models against LLM baselines on *extraction* tasks. It has its own golden sets
  (`golden/golden_manual_30_*.json`) and its own corpus sample (`golden/corpus_100.json`).
  This directory's harness is retrieval-recall-only (recall@k, MRR, nDCG@k against
  `/v1/search`) and does not touch NER/RE/classification; SRCH-0014 Phase 2 is expected to
  extend `ml-vs-llm/` for those tasks, reusing this directory's golden-set governance pattern,
  not its code.
- **`/opt/ml-benchmark/`** — the pre-SRCH-0015 status quo: ad hoc scripts that ran once on a
  PROD host and were never committed (`bench_reranking.py`, `bench_e2e_recall.py`, named in the
  original SRCH-0015 brief as reusable assets, were unrecoverable from either checkout when this
  task started). This directory exists specifically so that failure mode stops recurring:
  everything here is git-tracked inside `arcanada-workspace`.

If you're about to add a fifth Scrutator/LTM benchmark tree, don't — extend the correct one of
these four instead, or explain in this file why a fifth is warranted.

## What's here

| Path | Purpose |
|---|---|
| `harness.py` | Multi-model dispatch (BGE-M3 hybrid via `/v1/search`, BGE-Reranker, one LLM baseline via Model Connector), liveness pre-flight, infra-fail/threshold-fail exit-code split. |
| `rerank_gate.py` | SRCH-0031 paired OFF/ON experiment runner. Requires two loopback endpoints, validates `rrf` versus `colbert_rerank` citations, freezes corpus fingerprints, checks repeated ordered results, reports per-class gain/loss transitions, and fails closed on invalid evidence. |
| `live/granted_context_app.py` | Loopback-only wrapper around the deployed FastAPI app. It resolves the existing benchmark principal through live namespace grants; it bypasses JWT transport only because no reader secret is copied to the benchmark host. |
| `live/run_rerank_gate.sh` | Exact-image live runner. Starts isolated OFF/ON listeners with lifespan disabled, mutation credentials blanked, bounded DB pools, strict cleanup, and production-container identity checks. |
| `measure.py` | The original SRCH-0031 throwaway script, migrated verbatim (Precondition P1) as the pre-`harness.py` reference implementation. Kept until `harness.py`'s V-AC-06 reproduction check has run live and passed; not extended further. |
| `golden/golden-arcanada-v0.jsonl` | The 33-row v0 golden set (15 factual / 8 multi-hop / 10 temporal), migrated byte-identical from `measure.py`'s original home. |
| `golden/review-log.md` | Sidecar promotion log: candidate → two-LLM-agreement filter → human verification → `gold`. |
| `golden-arcanada-v0.results-summary.json` | The SRCH-0031 baseline measurement (`measure.py` output) — overall recall@5 = 0.909. `harness.py` must reproduce this within ±0.03 (V-AC-06) before it's trusted as `measure.py`'s replacement. |
| `SRCH-0031-recall-report.md` | The original baseline report and golden-set methodology writeup, migrated verbatim. |
| `CONSUMERS.md` | Downstream-consumer map (verified / verified-elsewhere / unverified). |
| `tools/` | Golden-set growth tooling: candidate generation + two-LLM-agreement disagreement filter. Code only — human verification of any batch is a separate, non-automatable step (see `tools/README.md`). |
| `tests/test_harness.py` | Unit tests: golden-row loading, `corpus_pinned_at` parsing, liveness fixture, recall/MRR/nDCG math, infra-fail-vs-threshold-fail exit codes. |

## Golden-set versions

- `golden-arcanada-v0.jsonl` — the original 33-row SRCH-0031 seed. Ships with this task.
- `v0.1` / `v0.2` / `v1` — the planned 3-batch growth path (~25-40 candidate docs per batch,
  target 100-150 rows total). **Not created by this task** — see § Golden-set growth status
  below; growth requires human (operator) verification per row, which this task does not
  fabricate.

## Golden-set growth status (human-annotation gate)

The golden-set growth path (v0 → v0.1 → v0.2 → v1, per the plan's Fork A) requires a human
(operator) verification pass on every candidate row before it can be promoted to `gold`. This
task builds the candidate-generation and two-LLM-agreement-filter *tooling* (`tools/`) but does
**not** run it to produce v0.1/v0.2/v1 data, and does **not** fabricate labels. Running that
tooling and completing the human-review sitting is left for the operator — see
`tools/README.md` for the exact invocation once that's authorized.

## Running the harness

```bash
python benchmark/scrutator/harness.py \
    --golden benchmark/scrutator/golden/golden-arcanada-v0.jsonl \
    --models bge-m3 \
    --namespace arcanada
```

See `harness.py --help` for the full flag surface (multi-model dispatch, dry-run smoke flags,
exit codes). Running this against live PROD Scrutator at any real volume, or wiring the CI
workflow's schedule trigger live, is operator-gated — see `benchmark/scrutator/tests/` for the
smoke-tested paths that don't require that sign-off.

## Running the SRCH-0031 rerank gate

`rerank_enabled` is a process-global setting; a request body field named `rerank` is ignored.
The gate therefore compares two isolated processes and never restarts the production container.
On Arcana-KB, stage this directory in a private temporary path and run:

```bash
benchmark/scrutator/live/run_rerank_gate.sh \
    scrutator-deploy:<DEPLOYED_40_HEX_COMMIT> \
    <PRIVATE_BENCHMARK_SOURCE_DIR> \
    <PRIVATE_OUTPUT_DIR>
```

The default `deployed` scope requires the candidate tag to equal the current production
container tag. To test an exact, not-yet-deployed fix, set
`SCRUTATOR_BENCHMARK_SCOPE=candidate`; a green candidate run reports
`CANDIDATE_ELIGIBLE`, never `ELIGIBLE_TO_FLIP`.

Exit codes are `0` for scope-appropriate eligibility, `1` for a valid terminal `KEEP_OFF`,
and `2` for invalid or incomplete evidence. Status `0` or `1` is accepted only when a
validated `summary.json` exists. A `200` response is not enough to prove the treatment:
every OFF result must carry `citation.score_kind="rrf"` and every ON result must carry
`citation.score_kind="colbert_rerank"`. The runner also invalidates corpus drift, repeated-order
instability, observed score ties, missing results, and ColBERT soft-fallback logs.

The current gate is deliberately conservative: no class may lose a hit, at least one paired
miss must become a hit, the historical fixed-set floors must hold, multi-hop all-gold retrieval
must not regress, and ON p95 must remain within 5 seconds. This is a decision on the fixed
33-row `arcanada` set, not a claim of statistically general search superiority.
