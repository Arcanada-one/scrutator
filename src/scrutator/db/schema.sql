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
    UNIQUE(source_path, chunk_index)
);

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
