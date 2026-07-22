-- SRCH-0038 1b: Exact whole-document source store for the skills namespace.
--
-- The 1a mechanism stamped the exact pre-chunk bytes (`doc_raw_content`) into
-- `chunks.metadata`, which carries `idx_chunks_metadata … USING gin(metadata)`
-- (default `jsonb_ops`). `jsonb_ops` indexes every scalar value as its own entry
-- with a hard ~2704-byte (⅓-page) ceiling, so a real multi-KB skill body raised
-- `ERROR: index row size … exceeds maximum 2704` at INSERT and failed the whole
-- `replace_source_chunks_atomic` transaction. Real skills are 3–50 KB → essentially
-- every real skill was un-indexable.
--
-- This isolated table holds the exact bytes OUT of any GIN index (it is an
-- integrity/blob store, never searched), so the entry-size ceiling never applies.
-- The upsert runs INSIDE `replace_source_chunks_atomic` (same transaction) so the
-- blob and its chunks are crash-consistent. `raw_content` is the SAME `full_content`
-- string that `content_hash` is computed over (compute_doc_content_hash), so
-- `sha256(raw_content) == content_hash` holds by construction.
--
-- Idempotent — safe to re-apply. Keyed `(namespace_id, source_path)` to match how
-- chunks resolve a source generation (mirrors entity_sources / structured_graph_sources).
-- `doc_id` mirrors `chunks.metadata->'section'->>'doc_id'` for the fetch-by-id read path.

CREATE TABLE IF NOT EXISTS source_documents (
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
CREATE INDEX IF NOT EXISTS idx_source_documents_doc_id
    ON source_documents(namespace_id, doc_id);
