# Golden-set review log

Sidecar promotion log: candidate → two-LLM-agreement filter → human verification → `gold`.
One entry per batch, per the PRD's Fork 1 governance process.

## v0 (retroactive entry — SRCH-0031, 2026-07-10)

- **Authorship:** fully manual (single-author, single-reading-pass) — predates this task's
  two-LLM-agreement tooling. See `SRCH-0031-recall-report.md` § Golden-set methodology for
  the exact construction process (corpus discovery → existence verification → direct
  document reading → hand-written questions, non-circular by construction).
- **Review:** the "human verification" step and the "authorship" step were the same
  single-operator pass (no separate two-LLM filter existed yet at v0's construction time).
- **Rows promoted:** 33 (15 factual / 8 multi-hop / 10 temporal).
- **`corpus_pinned_at`:** backfilled to `2026-07-10` by this task (Phase 5 Step 1) — the
  rows were already human-verified against live documents on that date, per SRCH-0031's own
  provenance; this is not a re-verification, just recording the date they were last confirmed.

## v0.1 / v0.2 / v1 — not yet run

The 3-batch growth path (Fork A) requires the operator to run
`tools/generate_candidates.py` and complete a human-review sitting per batch — see
`tools/README.md`. **Not performed by this task** (human-annotation gate). When a batch is
run, append an entry here in this format:

```
## v0.N (batch N — <date>)

- Candidate source documents: <list or count>
- Candidates generated (pass 1): <n>
- Survived two-LLM-agreement filter: <n>
- Promoted to gold after human review: <n>
- Rejected at human review (and why, briefly): <n>
- `corpus_pinned_at`: <date>
```
