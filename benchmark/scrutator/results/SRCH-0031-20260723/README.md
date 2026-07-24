# SRCH-0031 live rerank gate — 2026-07-23

Decision: **KEEP OFF**. The per-class recall gate did not become eligible to
run to completion because the treatment failed two prerequisite checks against
the live index.

## Attempt 1 — deployed image

- Image: `scrutator-deploy:19d7db0a12046fbb07e2f94f81c28a91c72bb08c`
- Result: invalid evidence. The ON sibling returned `score_kind=rrf`.
- Root cause: the live Embedding API accepts at most 16 ColBERT inputs, while
  recall@5 widens the rerank pool to 20. The API returned HTTP 400 and Scrutator
  correctly soft-failed to RRF.
- Independent live probes returned HTTP 200 for batch sizes 1, 2, and 5, and:
  `{"detail":"batch size 20 exceeds max 16"}` / the equivalent for 30.

## Attempt 2 — paged candidate image

- Image: `scrutator-deploy:822360f8983c4b13a21cd698cafdf6c2f18f9696`
- Change: ColBERT transport requests are paged at 16 without changing candidate
  order, MaxSim scoring, or the two sibling configurations.
- Result: invalid evidence. The first ON `/v1/search` request exceeded the
  45-second evidence timeout.
- The Embedding API journal recorded the first real document page as:
  `ColBERT: 16 texts in 39.708s`. That single page alone is 7.9 times the
  predeclared 5-second ON p95 budget; the remaining four documents had not
  completed before the client timeout.

Both attempts used two loopback-only siblings, the real `kb-observer` namespace
grant, lifespan disabled, immutable image IDs, and the same golden corpus hash.
The runner removed both siblings on exit. The production Scrutator identity was
not changed by the benchmark and its post-run health response is preserved with
attempt 2.

No recall number is reported: neither attempt produced valid treatment
responses for the full 33-query factual/multi-hop/temporal set. A partial or RRF
fallback result would misstate ColBERT recall.
