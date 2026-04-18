# Scrutator

> **One human life matters**

**Scrutator** (Latin: *scrutator* — "one who thoroughly investigates, searches, gets to the essence") — the foundational Knowledge Retrieval & Meaning Engine for the [Arcanada Ecosystem](https://arcanada.one).

## Etymology

The name comes from the Latin *scrutator* — "one who thoroughly investigates." The root *scrutari* means "to search through, examine" (originally — literally rummaging through rags, *scruta* — "junk, rags"), later acquiring the figurative meaning of careful investigation.

- **Direct meaning** — "investigator," "seeker," a system that "combs through" data
- **Connotation of thoroughness** — meticulousness, the ability to separate the important from the noise
- **English connection** — scrutiny/scrutinize (close examination, verification, audit)
- **Historical trace** — in medieval tradition, a *scrutator* was an official responsible for verifying votes (in papal elections), a metaphor for a trusted arbiter

## Architecture

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
         ▲              ▲              ▲              ▲
         │ MCP          │ REST         │ REST         │ REST
    Claude Code    Agent Dreamer   Munera Workers  Personal Asst
```

## Key Components

| Component | Description |
|-----------|-------------|
| **Embedding Engine** | BGE-M3: dense (1024d) + sparse + ColBERT, fp16 optimization |
| **Chunking Engine** | Adaptive multi-strategy: MD header-based, semantic fallback, parent-child |
| **Hybrid Search** | Dense vectors + PostgreSQL FTS → Reciprocal Rank Fusion (RRF) |
| **Dreaming Module** | Periodic knowledge systematization via Agent Dreamer integration |
| **Multi-Namespace** | Hierarchical: namespace → project → stream, cross-namespace graph edges |

## Tech Stack

- **Python 3.12** + FastAPI + uvicorn
- **BAAI/bge-m3** — multilingual embeddings (100+ languages, RU↔EN: 0.887 similarity)
- **PostgreSQL** + pgvector (HNSW) + FTS (tsvector)
- **RRF** — Reciprocal Rank Fusion (k=60) for hybrid ranking

## Quick Start

```bash
# Clone
git clone https://github.com/Arcanada-one/scrutator.git
cd scrutator

# Install dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
```

> **Note:** Full server deployment requires access to the Arcanada Tailscale mesh and arcana-db PostgreSQL instance.

## Project Status

Scrutator is in early development. See the [architecture docs](docs/architecture.md) for the full design.

**Roadmap:**
- [x] SRCH-0001: PRD + Architecture
- [x] SRCH-0008: GitHub Repo Setup
- [ ] SRCH-0002: Embedding Enhancement (fp16 + sparse)
- [ ] SRCH-0003: Chunking Engine
- [ ] SRCH-0004: Search & Retrieval Pipeline
- [ ] SRCH-0005: Dreaming Module
- [ ] SRCH-0006: LTM Integration
- [ ] SRCH-0007: Website & Blog

## Part of the Arcanada Ecosystem

Scrutator is the search foundation for the entire [Arcanada](https://arcanada.one) ecosystem. Without quality retrieval, no agent can effectively work with accumulated knowledge.

## License

[MIT](LICENSE)
