-- Scrutator database schema (SRCH-0004)
-- Requires: CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS namespaces (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    namespace_id INT REFERENCES namespaces(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    UNIQUE(namespace_id, name)
);

CREATE TABLE IF NOT EXISTS streams (
    id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id INT REFERENCES namespaces(id) NOT NULL,
    project_id INT REFERENCES projects(id),
    stream_id INT REFERENCES streams(id),
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    chunk_index INT NOT NULL,
    parent_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_dense vector(1024),
    textsearch_ru tsvector GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED,
    textsearch_en tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata JSONB DEFAULT '{}',
    token_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    indexed_at TIMESTAMPTZ,
    UNIQUE(namespace_id, source_path, chunk_index)
);

-- SRCH-0048: source identity is tenant-scoped. Older deployments created a
-- global uniqueness constraint which allowed one namespace to replace another.
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_source_path_chunk_index_key;
CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_namespace_source_chunk
    ON chunks(namespace_id, source_path, chunk_index);

-- HNSW for cosine similarity (dense vectors)
CREATE INDEX IF NOT EXISTS idx_chunks_dense ON chunks
    USING hnsw (embedding_dense vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- GIN for full-text search (dual language)
CREATE INDEX IF NOT EXISTS idx_chunks_fts_ru ON chunks USING gin(textsearch_ru);
CREATE INDEX IF NOT EXISTS idx_chunks_fts_en ON chunks USING gin(textsearch_en);

-- Lookup indexes
CREATE INDEX IF NOT EXISTS idx_chunks_namespace ON chunks(namespace_id);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING gin(metadata);

CREATE TABLE IF NOT EXISTS sparse_vectors (
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE PRIMARY KEY,
    token_weights JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id SERIAL PRIMARY KEY,
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    target_chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    created_by TEXT DEFAULT 'dreamer',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_chunk_id, target_chunk_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_chunk_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_chunk_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges(edge_type);

-- LTM: Named entities (knowledge graph nodes)
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id INT REFERENCES namespaces(id) NOT NULL,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    description TEXT,
    properties JSONB DEFAULT '{}',
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(namespace_id, name, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_entities_namespace ON entities(namespace_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities USING gin(to_tsvector('simple', name));
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

-- LTM: Entity-to-entity relationships
CREATE TABLE IF NOT EXISTS entity_edges (
    id SERIAL PRIMARY KEY,
    source_entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_entity_id, target_entity_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_entity_edges_source ON entity_edges(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_edges_target ON entity_edges(target_entity_id);

-- LTM-0025: per-source ownership for deterministic structured graph convergence.
CREATE TABLE IF NOT EXISTS entity_sources (
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE NOT NULL,
    namespace_id INT REFERENCES namespaces(id) ON DELETE CASCADE NOT NULL,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(entity_id, source_path)
);

CREATE INDEX IF NOT EXISTS idx_entity_sources_namespace_source
    ON entity_sources(namespace_id, source_path);
CREATE INDEX IF NOT EXISTS idx_entity_sources_source
    ON entity_sources(source_path);

CREATE TABLE IF NOT EXISTS entity_edge_sources (
    edge_id INT REFERENCES entity_edges(id) ON DELETE CASCADE NOT NULL,
    namespace_id INT REFERENCES namespaces(id) ON DELETE CASCADE NOT NULL,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(edge_id, source_path)
);

CREATE INDEX IF NOT EXISTS idx_entity_edge_sources_namespace_source
    ON entity_edge_sources(namespace_id, source_path);
CREATE INDEX IF NOT EXISTS idx_entity_edge_sources_source
    ON entity_edge_sources(source_path);

CREATE TABLE IF NOT EXISTS structured_graph_sources (
    namespace_id INT REFERENCES namespaces(id) ON DELETE CASCADE NOT NULL,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace_id, source_path)
);

CREATE INDEX IF NOT EXISTS idx_structured_graph_sources_source
    ON structured_graph_sources(source_path);

-- LTM: Pipeline job state (resume on failure)
CREATE TABLE IF NOT EXISTS ltm_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id INT REFERENCES namespaces(id) NOT NULL,
    source_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    current_step TEXT,
    total_chunks INT DEFAULT 0,
    processed_chunks INT DEFAULT 0,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- LTM Temporal layer (LTM-0012) — see migrations/002_temporal.sql
CREATE EXTENSION IF NOT EXISTS btree_gist;

CREATE TABLE IF NOT EXISTS entity_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id INT REFERENCES namespaces(id) NOT NULL,
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    when_t TIMESTAMPTZ,
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ,
    description TEXT,
    properties JSONB DEFAULT '{}',
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    superseded_by UUID REFERENCES entity_events(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (namespace_id, entity_id, event_type, when_t)
);

CREATE INDEX IF NOT EXISTS idx_events_entity ON entity_events(entity_id);
CREATE INDEX IF NOT EXISTS idx_events_when ON entity_events(when_t);
CREATE INDEX IF NOT EXISTS idx_events_namespace ON entity_events(namespace_id);
CREATE INDEX IF NOT EXISTS idx_events_valid_period
    ON entity_events USING gist (tstzrange(valid_from, valid_to, '[)'));
CREATE INDEX IF NOT EXISTS idx_events_active
    ON entity_events(entity_id) WHERE valid_to IS NULL;

-- LTM Reflect layer (LTM-0013) — see migrations/003_reflect.sql

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

-- SRCH-0023: local FK-only cache fallback for principal -> allowed-namespace resolution.
-- Source of truth is Auth Arcana ReBAC (OpenFGA); this table is a degraded RBAC facade used
-- only when the live OpenFGA endpoint is unreachable/unconfirmed. Seeded out-of-band by an
-- operator-run grant script (scripts/grant_namespace.py) — never written by request-path code.
CREATE TABLE IF NOT EXISTS principal_namespace_grants (
    id SERIAL PRIMARY KEY,
    principal_id TEXT NOT NULL,
    namespace_id INT REFERENCES namespaces(id) ON DELETE CASCADE NOT NULL,
    granted_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(principal_id, namespace_id)
);

CREATE INDEX IF NOT EXISTS idx_principal_grants_principal ON principal_namespace_grants(principal_id);

-- SRCH-0038 1b: Exact whole-document source store for the skills namespace — see
-- migrations/004_source_documents.sql. Holds the exact pre-chunk bytes OUT of any GIN
-- index (the 1a `chunks.metadata` GIN seam hit the ~2704-byte jsonb_ops entry ceiling on
-- real multi-KB skills). Upserted inside `replace_source_chunks_atomic` (same transaction →
-- crash-consistent); `raw_content` is the exact `full_content` that `content_hash` hashes,
-- so `sha256(raw_content) == content_hash` by construction. NOT GIN-indexed.
CREATE TABLE IF NOT EXISTS source_documents (
    namespace_id INT REFERENCES namespaces(id) ON DELETE CASCADE NOT NULL,
    source_path TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (namespace_id, source_path)
);

CREATE INDEX IF NOT EXISTS idx_source_documents_doc_id
    ON source_documents(namespace_id, doc_id);

-- SRCH-0039 (Mechanism C): Exact whole-document source store for the LARGE evidence corpus — see
-- migrations/005_evidence_documents.sql. A SEPARATE table from `source_documents` so the skills
-- always-exact / 256 KB-cap / fail-closed-409 policy stays isolated from evidence's flag-gated
-- (`evidence_exact_bytes`), larger (NO 256 KB cap), gracefully-degrading policy. Upserted inside
-- `replace_source_chunks_atomic` (same transaction → crash-consistent); `raw_content` is the exact
-- `full_content` that `content_hash` hashes, so `sha256(raw_content) == content_hash` by
-- construction. NOT GIN-indexed. Byte-fetch and the namespace authz check are the SAME read.
CREATE TABLE IF NOT EXISTS evidence_documents (
    namespace_id INT REFERENCES namespaces(id) ON DELETE CASCADE NOT NULL,
    source_path TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (namespace_id, source_path)
);

CREATE INDEX IF NOT EXISTS idx_evidence_documents_doc_id
    ON evidence_documents(namespace_id, doc_id);
