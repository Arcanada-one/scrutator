# Downstream Consumers of `benchmark/scrutator/`

**Task:** SRCH-0015 (D-REQ-06). Corrected per `/dr-plan`'s § 1 re-check against `wt-kb-scrutator`
and `wt-srch-recall` — the PRD's original map was built from a search of `/home/dev/arcanada`
only and understated how many of these references actually exist elsewhere. See § Verification
below for the corrected count this file's verification command relies on.

| Consumer | Relationship to this harness | Status |
|---|---|---|
| SRCH-0014 (ML vs LLM research, has its own PRD) | SRCH-0015 is SRCH-0014's Subtask 1; Phase 2's extraction-task benchmark (NER/fact-triples/classification, in `benchmark/ml-vs-llm/`) reuses this task's golden-set governance pattern once it exists | **Verified** — `datarim/tasks/SRCH-0014-task-description.md` and `PRD-SRCH-0014-ml-vs-llm.md` both exist and reference SRCH-0015 by id. |
| SRCH-0018 (E2E recall pipeline, in progress) | Sibling, not a consumer of this harness — depends on SRCH-0016, and its own reranking-comparison results are an input candidate for this harness's `bge-reranker` baseline once that dispatch path is wired (see README's Scope decision), not the reverse | **Verified** — `PRD-SRCH-0018-e2e-pipeline.md` names SRCH-0016 as its dependency, not SRCH-0015. |
| SRCH-0023 (multi-tenant isolation, has its own PRD) | Orthogonal — this harness is a correctly-namespaced caller (`namespace: arcanada`), unaffected by SRCH-0023's Approach A2+B1+B2 either direction | **Verified** — confirmed against `PRD-SRCH-0023.md`. |
| SRCH-0031 (baseline recall measurement — this task's direct predecessor; produced the v0 golden set + `measure.py` this harness migrates and extends) | This harness productionizes SRCH-0031's artifacts | **Verified-elsewhere** — no `datarim/tasks/SRCH-0031-*` file or `datarim/backlog.md` line exists in this checkout (`/home/dev/arcanada`), but `wt-kb-scrutator/datarim/backlog.md:892` and `wt-srch-recall/datarim/backlog.md:891` both carry the identical line: "Per-class recall@5 measurement on `/v1/search` path with rerank ON... Gated on PostgreSQL restoration (SRCH-0030 infra dependency, HTTP 503 as of 2026-06-22)." Filed as a backlog item there, not (yet) an initialized task in any of the 3 checkouts — a cross-worktree drift, not a phantom reference. |
| LTM-0023 (quantization recall PILOT) | Anticipated consumer of the golden set for quantized-embedding recall comparison, once filed | **Verified-elsewhere** — `wt-kb-scrutator/datarim/backlog.md:523` / `wt-srch-recall/datarim/backlog.md:523`: "Recall-measurement PILOT — measure recall@k delta of 4-bit/2-bit vector quantization vs float32 pgvector baseline... via existing LTM benchmark harness... Gates any future quantization adoption (R4 in PRD-SRCH-0027)." Sourced from `discovered-during-SRCH-0027`. Same cross-worktree-drift pattern as SRCH-0031, above. |
| "SRCH-0030" (recall gate — would consume this harness's pass/fail exit code as its CI gate signal) | Would gate on D-REQ-05's exit-code contract | **Confirmed unverified** — no own backlog line or task file in any of the 3 checkouts (`/home/dev/arcanada`, `wt-kb-scrutator`, `wt-srch-recall`); exists only as a dependency mention inside SRCH-0031's backlog text (above). Note: this id is also independently in use in this checkout for the *different*, already-shipped `/v1/ltm/recall` per-class regression gate (`benchmark/recall-gate/`, `.github/workflows/recall-regression.yml`) — that gate is unrelated to this harness (different endpoint, different corpus) and should not be confused with the still-unfiled "SRCH-0030 recall gate" concept named in the original SRCH-0015 operator brief. |
| Companion index-freshness task (working id `SRCH-0036` per the plan's Fork B collision correction — re-verify at actual spawn time per `dr-init-id-collision-window`) | Not this harness's scope — fixes the root cause (live index drift) this harness's liveness pre-flight only works around | Not filed by this task (operator decision, per PRD § Next Steps). |

## Verification (V-AC-05)

Exactly one row above (the "SRCH-0030" row) carries the not-found status this file's
verification grep matches on — SRCH-0031 and LTM-0023 are corrected to **verified-elsewhere**
(present in `wt-kb-scrutator`/`wt-srch-recall`, absent from this checkout) per the plan's § 1 /
§ 6 correction to the PRD's original ≥3 count. See the plan's Validation Checklist for the
exact check command.
