-- SRCH-0039 (Mechanism C): Exact whole-document source store for the LARGE evidence corpus.
--
-- Doc snapshot of the block added to `schema.sql` (this repo has NO migration runner; `schema.sql`
-- via `apply_schema()` is the idempotent source of truth — migrations/ are doc snapshots kept in
-- sync for review/history).
--
-- Consilium-ratified Mechanism C (creative-SRCH-0039-storage-fork.md): an isolated Postgres table,
-- SEPARATE from `source_documents`, so the skills always-exact / 256 KB-cap / fail-closed-409
-- policy stays cleanly isolated from evidence's flag-gated, larger, gracefully-degrading policy.
--
-- Why an isolated PG table (not filesystem CAS) at current scale:
--   * Correctness — the `raw_content` upsert lands INSIDE `replace_source_chunks_atomic()`, so the
--     row and its chunks commit or roll back together (crash-consistent). `raw_content` is the SAME
--     `full_content` string that `content_hash` is computed over, so
--     `sha256(raw_content) == content_hash` holds by construction (never recomputed at read).
--   * Security — the byte-fetch and the namespace authz check are the SAME read
--     (`WHERE doc_id = $1 AND namespace_id = ANY($2::int[])`); no split authz surface.
--   * Reliability — captured in the existing PG backup as one consistent unit.
-- Documented scale-trigger to revisit → filesystem CAS: ~1-2 orders more evidence, multi-MB median
-- docs, or a base-backup/WAL ceiling. Because the table is isolated and `FetchResponse` is frozen,
-- that extraction is a contained, consumer-invisible backfill migration.
--
-- Policy divergence from skills (`source_documents`):
--   * Skills: `content_exact=True` ALWAYS; a missing row → fail-closed 409.
--   * Evidence: flag-gated (`SCRUTATOR_EVIDENCE_EXACT_BYTES`, default-off). Flag ON + row present →
--     exact bytes; row ABSENT → graceful degradation to reassembly (`content_exact=False`), NOT 409
--     — the huge existing corpus is backfilled gradually, so row-absence is an expected transient
--     state, not an integrity failure. No 256 KB per-document cap (the evidence corpus is large).
--
-- Idempotent — safe to re-apply. Keyed `(namespace_id, source_path)` to match how chunks resolve a
-- source generation. `doc_id` mirrors `chunks.metadata->'section'->>'doc_id'` for the fetch-by-id
-- read path. No cross-namespace dedup (sidesteps a content-hash confirmation oracle); IF a future
-- contract ever exposes `content_hash` as a request key, re-key `(namespace_id, content_hash)`.

CREATE TABLE IF NOT EXISTS evidence_documents (
    namespace_id INT REFERENCES namespaces(id) ON DELETE CASCADE NOT NULL,
    source_path TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (namespace_id, source_path)
);

-- Fetch-by-id read path resolves the blob by the namespace-scoped doc_id.
-- Plain btree (NOT gin) — doc_id is a short 16-char hash; raw_content is never indexed.
CREATE INDEX IF NOT EXISTS idx_evidence_documents_doc_id
    ON evidence_documents(namespace_id, doc_id);
