-- LTM-0013: Reflect layer (R in TEMPR)
-- Adds meta_facts table (LLM-derived summaries with provenance)
-- and reflect_runs table (per-run audit trail).
-- Idempotent — safe to re-apply.

CREATE TABLE IF NOT EXISTS reflect_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id INT REFERENCES namespaces(id) NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    model_used TEXT NOT NULL,
    chunks_scanned INT DEFAULT 0,
    meta_facts_created INT DEFAULT 0,
    cost_usd NUMERIC(10, 6) DEFAULT 0,
    req_count INT DEFAULT 0,
    abort_reason TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_reflect_runs_namespace ON reflect_runs(namespace_id);
CREATE INDEX IF NOT EXISTS idx_reflect_runs_status ON reflect_runs(status);

CREATE TABLE IF NOT EXISTS meta_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id INT REFERENCES namespaces(id) NOT NULL,
    fact_type TEXT NOT NULL,
    content TEXT NOT NULL,
    source_chunk_ids UUID[] NOT NULL,
    entity_ids UUID[] DEFAULT '{}',
    depth INT NOT NULL DEFAULT 1 CHECK (depth >= 1 AND depth <= 1),
    derived_at TIMESTAMPTZ DEFAULT NOW(),
    model_used TEXT NOT NULL,
    reflect_run_id UUID REFERENCES reflect_runs(id) ON DELETE CASCADE,
    embedding_dense vector(1024),
    properties JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_meta_facts_namespace ON meta_facts(namespace_id);
CREATE INDEX IF NOT EXISTS idx_meta_facts_run ON meta_facts(reflect_run_id);
CREATE INDEX IF NOT EXISTS idx_meta_facts_chunks ON meta_facts USING gin(source_chunk_ids);
CREATE INDEX IF NOT EXISTS idx_meta_facts_entities ON meta_facts USING gin(entity_ids);
CREATE INDEX IF NOT EXISTS idx_meta_facts_dense
    ON meta_facts USING hnsw (embedding_dense vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
