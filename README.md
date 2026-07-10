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

## Recall@k Regression Gate

CI gate that runs the LTM-0009 benchmark harness against live Scrutator and fails the build when per-class recall@5 drops below a committed baseline.

| Class | Baseline (recall@5) | Regression threshold |
|-------|--------------------|--------------------|
| factual | 0.50 | 0.05 |
| multi-hop | 0.4545 | 0.05 |
| temporal | 0.6667 | 0.07 |

Thresholds are per-class (factual / multi-hop / temporal independently — not averaged). Temporal has a slightly looser delta because it is the known-weak, higher-variance class.

**Exit codes:** `0` = all classes pass; `1` = recall regression detected; `2` = transport/infrastructure error (network flake — does NOT count as a recall regression).

**Run manually:**
```bash
# Gate against an existing report:
python benchmark/recall-gate/recall_gate.py --report <path-to-report.json>

# Run the harness and gate in one step (requires arcana-db Tailscale access):
python benchmark/recall-gate/recall_gate.py --run --harness <path-to-ltm-bench-query.py>

# Refresh baseline after an intentional recall change (requires review):
python benchmark/recall-gate/recall_gate.py --report <path> --update-baseline
```

**Runner requirement:** the CI job runs on `[self-hosted, linux, arcana-db, docker]` (co-located with Scrutator — reaches `:8310` on localhost). GitHub-hosted runners cannot reach the Tailscale-only endpoint and are billing-blocked org-wide.

**Baseline recalibration:** after an intentional change that improves recall, run `--update-baseline` on the arcana-db runner with a fresh report, review the diff in the PR, then merge. The baseline seeded in `baseline.json` was captured from the 2026-04-26 `with-entities` run (36 queries); recalibrate from a `no-entities` run on first clean CI pass.

## Index Freshness Detection (SRCH-0036)

`scrutator.tools.index_freshness` compares the `source_path`s currently indexed for a namespace against the current corpus (filesystem scan or an ingest manifest), and reports:

- **STALE** — indexed but no longer present in the corpus (deleted/moved on disk). This is the drift that dragged the recall gate below baseline in SRCH-0031 (12/77 `datarim-kb` sources had moved).
- **MISSING** — present in the corpus but never indexed.

The tool is **read-only**: it enumerates and reports, and can emit a dry-run re-index **plan** (`--plan`) describing the delete/re-ingest actions a future run would take. It never executes those actions — actually deleting stale chunks or re-ingesting missing sources against a live namespace is a separate, hard-gated operator step.

```bash
# Report-only (default), scanning a filesystem corpus root:
python -m scrutator.tools.index_freshness --namespace arcanada --corpus-root /path/to/kb

# Same, but also emit a dry-run re-index plan (still not executed):
python -m scrutator.tools.index_freshness --namespace arcanada --corpus-root /path/to/kb --plan

# Using an ingest manifest instead of a filesystem scan, writing the JSON report to a file:
python -m scrutator.tools.index_freshness --namespace arcanada --manifest ingest-manifest.json --output report.json

# CI use — exit 1 if anything is stale or missing:
python -m scrutator.tools.index_freshness --namespace arcanada --corpus-root /path/to/kb --fail-on-stale
```

By default it reads `SCRUTATOR_DATABASE_URL` via `scrutator.config.settings` (override with `--database-url`). `--probe-url` optionally does a read-only `GET /health` check before detection.

## Quick Start

```bash
# Clone
git clone https://github.com/Arcanada-one/scrutator.git
cd scrutator

# Install dependencies
pip install -r requirements-dev.txt

# Run tests (including recall gate unit tests)
pytest tests/ benchmark/recall-gate/tests/ -v

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
