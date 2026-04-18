# Scrutator Architecture

> Full PRD: see `datarim/prd/PRD-SRCH-0001-scrutator.md` in the Arcanada knowledge base.

## Overview

Scrutator is the unified retrieval engine for the Arcanada ecosystem. It provides semantic search, full-text search, and hybrid ranking across all knowledge sources.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        arcana-db Server                             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              Scrutator API (FastAPI :8310)               │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ │       │
│  │  │ /v1/chunk│ │/v1/index │ │/v1/search│ │/v1/dream/ │ │       │
│  │  │          │ │          │ │          │ │  analyze  │ │       │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘ │       │
│  │       │             │            │              │       │       │
│  │  ┌────▼─────────────▼────────────▼──────────────▼─────┐ │       │
│  │  │              Core Engine                           │ │       │
│  │  │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │ │       │
│  │  │  │ Chunker  │ │Embedder  │ │  Hybrid Searcher   │ │ │       │
│  │  │  │(adaptive)│ │(BGE-M3)  │ │(dense+FTS+RRF)     │ │ │       │
│  │  │  └──────────┘ └──────────┘ └────────────────────┘ │ │       │
│  │  └───────────────────────┬────────────────────────────┘ │       │
│  └──────────────────────────┼──────────────────────────────┘       │
│                             │                                       │
│  ┌──────────────────────────▼──────────────────────────────┐       │
│  │          PostgreSQL (pgvector + FTS)                     │       │
│  │  ┌────────────┐ ┌──────────────┐ ┌──────────────────┐  │       │
│  │  │ namespaces │ │    chunks    │ │   graph_edges    │  │       │
│  │  │ projects   │ │  + vectors   │ │  (cross-ref)     │  │       │
│  │  │ streams    │ │  + tsvector  │ │                  │  │       │
│  │  └────────────┘ └──────────────┘ └──────────────────┘  │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌──────────────────────┐                                          │
│  │ Embedding API (:8300)│  ← BAAI/bge-m3 (dense+sparse+ColBERT)   │
│  └──────────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector store | pgvector (PostgreSQL) | Zero new infra, HNSW handles 1M+ vectors |
| Full-text search | PostgreSQL FTS (tsvector) | Dual-language (ru+en), hybrid → ~84% precision |
| Hybrid ranking | RRF (k=60) | Proven pattern: dense+FTS → fuse, pull 20 → return 10 |
| Chunking | Adaptive multi-strategy | MD header-based primary, semantic fallback, parent-child |
| Dreaming | Single Dreamer + pluggable analyzers | Scrutator = search API, Dreamer = orchestrator |
| Namespace | Hierarchical (namespace → project → stream) | Cross-namespace edges via Dreamer |

## Multi-Namespace Model

```
Universe: Veritas Arcana
├── Namespace: arcanada (default)
│   ├── Project: scrutator
│   │   ├── Stream: embedding-engine
│   │   ├── Stream: chunking
│   │   └── Stream: search-pipeline
│   ├── Project: datarim
│   ├── Project: dreamer
│   └── ...
├── Namespace: personal
│   ├── Project: notes
│   └── Project: research
└── Cross-Namespace Graph (Dreamer-managed)
```

## Hybrid Search (RRF)

```sql
WITH semantic AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
    FROM chunks WHERE namespace_id = $2
    LIMIT 20
),
fulltext AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(textsearch, $3) DESC) AS rank
    FROM chunks WHERE textsearch @@ $3 AND namespace_id = $2
    LIMIT 20
)
SELECT COALESCE(s.id, f.id),
       COALESCE(1.0/(60+s.rank), 0) + COALESCE(1.0/(60+f.rank), 0) AS score
FROM semantic s FULL OUTER JOIN fulltext f ON s.id = f.id
ORDER BY score DESC LIMIT 10;
```

## API Endpoints (planned)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/chunk` | POST | Chunk a document (adaptive strategy) |
| `/v1/index` | POST | Index chunks (embed + store) |
| `/v1/search` | POST | Hybrid search with RRF |
| `/v1/dream/analyze` | POST | Dreaming analysis |
| `/v1/namespaces` | GET/POST | Namespace management |
| `/v1/stats` | GET | Index statistics |
