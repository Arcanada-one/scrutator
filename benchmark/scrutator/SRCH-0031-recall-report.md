# SRCH-0031 — Baseline Recall Measurement (Scrutator hybrid search, namespace `arcanada`)

**Date:** 2026-07-10
**Endpoint:** `POST http://100.70.137.104:8310/v1/search`
**Namespace:** `arcanada`
**Golden set:** `golden-arcanada-v0.jsonl` (33 queries)

## Headline result

| Class | N | recall@1 | recall@5 | latency p50 (top-5 call) |
|---|---|---|---|---|
| factual | 15 | 0.600 | 0.933 | 1014 ms |
| multi-hop | 8 | 0.750 | 1.000 | 1175 ms |
| temporal | 10 | 0.600 | 0.800 | 997 ms |
| **overall** | **33** | **0.636** | **0.909** | **1024 ms** |

This is the **baseline hybrid RRF** result. It also **is** the current "rerank ON" result — see next section.

## Rerank ON ≡ Rerank OFF (re-confirmed)

`/v1/search` accepts a `rerank` field but drops it silently. Re-verified on 3 representative queries (one per class) with `include_content:true`:

- Top-level JSON keys: identical.
- `source_path` order across top-5: **identical**.
- `score` per result: **identical**.
- `content` per result: **identical**.
- Only field that differs between `rerank:true` and `rerank:false` calls: `search_time_ms` (server-side timing jitter, not a content signal).

**Conclusion: the ColBERT rerank stage (SRCH-0029) is not implemented in `/v1/search` today.** Whatever "rerank ON" means operationally right now is byte-identical to plain hybrid RRF. The recall numbers above are therefore valid for *both* "rerank on" and "rerank off" as currently deployed — there is exactly one search behavior in production, not two.

## Golden-set methodology (v0 seed, non-circular)

To avoid the circularity trap (using search output as its own ground truth), the golden set was built by:

1. **Corpus discovery, not corpus assumption.** Ran ~35 broad `/v1/search` queries across topic areas (Datarim, Rules of Robotics, Scrutator, Model Connector, Long Term Memory, Muneral, Arcanada Support, Disk Arcana, Arganize.me, Verdicus, archives) and collected the union of distinct `source_path`s returned (77 distinct paths, see `all_paths.txt`).
2. **Existence verification.** Checked every discovered `source_path` against the live filesystem at `/home/dev/arcanada` (read-only). **12 of 77 paths no longer exist** — e.g. `docs/about-scrutator.md`, `ltm-smoke-test.md`, all of `Projects/Verdicus/datarim/*` (moved into Verdicus's own `code/` repo), most of `Projects/Datarim/datarim/{archive,progress,projectbrief,reflection,docs/evolution-log}.md`. This confirms **the index is a stale snapshot** (git history on surviving files dates to ~2026-05-11) that has drifted from the current KB tree. Golden questions were built **exclusively from paths confirmed to still exist**, so every gold label is checkable against today's actual file content — not against what the (possibly-stale) index claims.
3. **Direct reading, not search-derived labels.** For each candidate document, the file was read in full (or materially) via the `Read` tool. A natural-phrasing question was then hand-written from what was actually in the document — never lifted verbatim from a heading, and never checked against what `/v1/search` itself would return. The gold `source_path`(s) come from "I read this file and it answers this question," full stop.
4. **Class design:**
   - **factual** (15): single fact from one document (IP addresses, license names, dates-as-facts-not-temporal-reasoning, config values).
   - **multi-hop** (8): question requires combining a fact from two different documents to fully answer (e.g. Mem0→Cognee swap reasoning spans `research/shortlist.md` + `benchmark/mem0-gate/verdict.md`). Per the task's scoring rule, a hit counts if **any** gold path lands in top-k — this is standard recall@k convention, but note it does **not** verify that *both* hops were actually retrieved together; it's a lower bar than true multi-hop completeness.
   - **temporal** (10): question turns on a specific date/timestamp recorded in the document ("Last Updated," archive completion date, task-creation date).
5. Coworker delegation for bulk-reading these ~20 files was attempted first (per global CLAUDE.md mandate for ≥3-file reads) but **failed closed** — the `code` and `datarim` coworker profiles in this environment are marked `recommended_provider: none` (`unknown provider 'none'` error), i.e. deliberately routed elsewhere and not available here. Fell back to native `Read` per the documented fallback rule ("if the chosen provider fails, fall back to native Read for that turn only").

### Honest caveats

- **v0 seed size (33 queries), single author, single reading pass.** This is not a statistically powered benchmark — it is the first non-circular golden set for this corpus, sized to unblock SRCH-0031/SRCH-0015, not to be cited as an authoritative recall figure. Confidence intervals on n=8–15 per class are wide.
- **Coverage skew.** Because I could only build questions from documents I could verify still exist, coverage over-represents Long Term Memory (8 files touched) and Rules of Robotics (3 files) relative to other corpus areas (e.g. nothing from Verdicus, since its `datarim/` tree is gone from disk; nothing from Consilium beyond a credentials file I deliberately skipped).
- **Index staleness is itself a finding, not just a caveat.** 12/77 sampled source_paths are dead links today. Any production consumer trusting `/v1/search` results to point at live files will hit ~15% dead-path rate on this sample (not a golden-set metric, just an observation from the discovery pass — a full dead-link audit is out of scope here).
- **Two genuine misses worth reading, not statistical noise:**
  - **F15** (LTM-0002 scoring weights, "recall@5 overall 25%") — 0/5 miss. The document (`benchmark/README.md`) never surfaced; four different `Long Term Memory` docs about scoring/criteria did instead — looks like a genuine near-duplicate-content confusion within the LTM corpus.
  - **T1/T9** (exact "Last Updated"/"Created" date lookups on `datarim/techContext.md` and `Projects/Datarim/datarim/tasks.md`) — both missed at top-5. Dates-as-metadata (a line like `**Last Updated:** 2026-04-12`) appear weakly weighted by the current hybrid ranker versus body-text terms; other date-grounded queries (T3–T8, T10) hit fine when the date co-occurs with distinctive prose, not just a frontmatter-style line.
- **No ColBERT signal was measured** — by definition, since it's not implemented. These numbers characterize dense+sparse hybrid RRF only.

## Artifacts

- Golden set: `/tmp/claude-1002/-home-dev-wt-kb-scrutator/1ae8b277-4a92-43b8-bf03-a8a625b17026/scratchpad/srch-0031-measure/golden-arcanada-v0.jsonl`
- Measurement script: `/tmp/claude-1002/-home-dev-wt-kb-scrutator/1ae8b277-4a92-43b8-bf03-a8a625b17026/scratchpad/srch-0031-measure/measure.py`
- Per-query detail (returned top-5 paths, hit/miss, latency): `/tmp/claude-1002/-home-dev-wt-kb-scrutator/1ae8b277-4a92-43b8-bf03-a8a625b17026/scratchpad/srch-0031-measure/results-detail.json`
- Summary JSON: `/tmp/claude-1002/-home-dev-wt-kb-scrutator/1ae8b277-4a92-43b8-bf03-a8a625b17026/scratchpad/srch-0031-measure/results-summary.json`
- Corpus discovery raw output: `/tmp/claude-1002/-home-dev-wt-kb-scrutator/1ae8b277-4a92-43b8-bf03-a8a625b17026/scratchpad/srch-0031-measure/all_paths.txt`
