# Scrutator

> **One human life matters**

**Scrutator** (Latin: *scrutator* — "one who thoroughly investigates, searches, gets to the essence") — the foundational Knowledge Retrieval & Meaning Engine for the [Arcanada Ecosystem](https://arcanada.ai).

## Etymology

The name comes from the Latin *scrutator* — "one who thoroughly investigates." The root *scrutari* means "to search through, examine" (originally — literally rummaging through rags, *scruta* — "junk, rags"), later acquiring the figurative meaning of careful investigation.

- **Direct meaning** — "investigator," "seeker," a system that "combs through" data
- **Connotation of thoroughness** — meticulousness, the ability to separate the important from the noise
- **English connection** — scrutiny/scrutinize (close examination, verification, audit)
- **Historical trace** — in medieval tradition, a *scrutator* was an official responsible for verifying votes (in papal elections), a metaphor for a trusted arbiter

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Arcana-KB                                  │
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

> Two additional read-only GET routes on the same `:8310` service —
> `/v1/navigate/outline` and `/v1/navigate/section` — omitted from the diagram above
> for space. See [§ API — Navigation](#api--navigation) below.

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

The recall gate runs the committed benchmark harness against live Scrutator and fails when per-class recall@5 drops below the reviewed baseline.

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

# Run the harness and gate in one step (requires Arcana-KB Tailscale access):
python benchmark/recall-gate/recall_gate.py --run --harness <path-to-ltm-bench-query.py>

# Refresh baseline after an intentional recall change (requires review):
python benchmark/recall-gate/recall_gate.py --report <path> --update-baseline
```

**Runner requirement:** the CI job is co-located with Scrutator on Arcana-KB and reaches `:8310` on localhost. The workflow currently selects the host through its compatibility label `[self-hosted, linux, arcana-db, docker]`; keep that label until the runner registration is renamed. GitHub-hosted runners cannot reach the Tailscale-only endpoint.

**Baseline recalibration:** after an intentional recall improvement, run `--update-baseline` on the Arcana-KB runner with a fresh report, review the diff in the PR, then merge. Never lower the baseline to hide a regression.

## Index Freshness Detection

`scrutator.tools.index_freshness` compares the `source_path`s currently indexed for a namespace against the current corpus (filesystem scan or an ingest manifest), and reports:

- **STALE** — indexed but no longer present in the corpus (deleted or moved on disk).
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
## API — Navigation

The hierarchical-navigation layer adds index-time
section normalization (`chunker/splitters.py`'s `slugify` + `normalize_heading_path`), two
read-only endpoints, and an opt-in `group_by` on `/v1/search`. See
[`documentation/reference/navigation.md`](documentation/reference/navigation.md) for the full reference.

### `GET /v1/navigate/outline`

Returns the hierarchical table-of-contents tree for a `(namespace, source_path)`, assembled at
query time from the chunks' normalized `section` metadata.

```
GET /v1/navigate/outline?namespace=arcanada&source_path=notes/example.md&max_nodes=2000
→ 200 OutlineResponse { source_path, namespace, doc_id, total_chunks, outline: [...] }
→ 404 unknown namespace/source_path; 422 if total_chunks exceeds max_nodes (default 2000, hard
  ceiling 10000) — the response fails loudly rather than silently truncating the tree.
```

### `GET /v1/navigate/section`

Returns a chunk's section context: ancestors (breadcrumb), self, siblings, and children.

```
GET /v1/navigate/section?chunk_id=<uuid>
→ 200 SectionContext { chunk_id, doc_id, section_key, ancestors, self, siblings, children }
→ 422 chunk_id is not a valid UUID; 404 chunk not found.
```

### `group_by` on `POST /v1/search` (opt-in)

`group_by: "document" | "section" | null` (default `null`) folds fused RRF hits into groups
post-fusion, in-memory — the underlying RRF query and ranking are unchanged. Omitting `group_by`
leaves `/v1/search` byte-identical to the flat-results behaviour.

```jsonc
// SearchRequest, additive field:
"group_by": "document"   // or "section", or omit/null for today's flat results

// SearchResponse.results elements become (when group_by is set):
{ "group_key": "...", "doc_id": "...", "score": 0.05,
  "representative": { /* a SearchResult */ }, "member_chunk_ids": ["..."], "member_count": 3 }
```

**Backfill.** Both endpoints degrade gracefully for chunks indexed before section metadata was introduced (no `section`
metadata yet): they fall back to a single flat root section rather than erroring. Run
`python tools/backfill_sections.py --namespace <ns>` (dry-run by default; pass `--live` to write)
to populate `section` for existing chunks — idempotent, safe to re-run, zero embedding calls.

## API — Exact fetch-by-id

`POST /v1/fetch` returns a whole document (or a bounded range) by opaque id, with an
ingest-bound integrity hash — the exact/version-pinned counterpart to fuzzy `/v1/search`.
Namespace authorization is identical to `/v1/search` (`Depends(require_tenant_context)`); an
unknown or cross-namespace id answers `404` with no existence oracle.

```jsonc
// Request
{
  "by": "document_id" | "source_id" | "chunk_id",   // opaque ids only (S3)
  "id": "0123456789abcdef",                          // 16-hex doc id, or a UUID for chunk_id
  "range": "full"                                    // or {"parent_of_chunk": "<uuid>"}
                                                     // or {"offset_start": N, "offset_end": M}
  , "include": ["content", "provenance"]
}
// Response (FetchResponse) — every field but `content` is server-derived
{ "source_id", "path", "content", "content_len_tokens", "content_hash",
  "index_snapshot_id", "indexed_at", "embedding_model_id", "namespace",
  "trust_class": "skill" | "evidence", "chunk_manifest": [...], "stale": false }
```

- **Selectors.** `document_id` and `source_id` are aliases for the same opaque document id
  (`compute_doc_id`); `chunk_id` is a chunk UUID. No selector accepts a filesystem path —
  path-like / malformed ids are rejected at request validation (`422`) before any DB access.
- **`content_hash`.** The **whole-document** SHA-256, prefixed `sha256:`, **stamped at
  ingest** into `metadata.section.doc_content_hash` and **read** at fetch — never recomputed over
  the response. The `/v1/search` hit's `content_hash`/`source_id` (additive fields) equal the
  fetch values, so `search → fetch by source_id` is a closed, hash-verified roundtrip.
- **`range`.** `full` reassembles all chunks in `chunk_index` order; `parent_of_chunk`
  returns a chunk's whole parent doc; `offset_start/offset_end` slices the reassembled content
  (offsets are reassembly-relative in the current API). An offset slice never
  re-hashes: `content_hash` stays the whole-doc ingest hash.
- **`trust_class` is a non-authorizing hint.** `"skill"` (namespace ==
  `SCRUTATOR_SKILLS_NAMESPACE`) vs `"evidence"`. It does **not** authorize execution — the
  execution gate is the consumer's config-pinned BLAKE3 digest, a deliberately distinct concern
  from this SHA-256 fetch-integrity hash. Scrutator remains untrusted transport.
- **`stale`** is currently `false` because the service does not read the live source during fetch.

**Backfill.** Chunks indexed before document hashes were introduced lack `doc_content_hash`; fetch returns their
`content_hash` as `""` (never a recomputed value) until re-index or an offline, idempotent
`python scripts/backfill_doc_content_hash.py` (dry-run with `--dry-run`) binds the hash once
from the stored chunk concatenation — an ingest-equivalent bind, not a response-time recompute
so the response path never invents a hash.

## Quick Start

```bash
# Clone
git clone https://github.com/Arcanada-one/scrutator.git
cd scrutator

# Install dependencies
pip install -r requirements-dev.txt

# Run the service and benchmark test suites
PYTHONPATH=src pytest tests/ -v
PYTHONPATH=src:benchmark/scrutator pytest benchmark/scrutator/tests/ -v

# Lint
ruff check src/ tests/ benchmark/scrutator/
ruff format --check src/ tests/ benchmark/scrutator/harness.py benchmark/scrutator/tests/
```

> **Note:** Full server deployment requires access to the Arcanada Tailscale mesh and the Arcana-KB PostgreSQL instance.

## Project Status

Scrutator 0.3.0 is deployed on Arcana-KB as a Tailscale-only service. The main branch deploys through GitHub Actions, and the deployment workflow verifies the reviewed SHA before running the transactional update and health check. See the [architecture explanation](documentation/explanation/architecture.md) and [API reference](documentation/reference/api.md).

**Roadmap:**
- [x] Hybrid dense, sparse, and full-text retrieval
- [x] Namespace-scoped indexing, navigation, exact fetch, graph, memory, and LTM routes
- [x] Transactional main-branch deployment with rollback and health verification
- [ ] Public product website and tutorials

## Part of the Arcanada Ecosystem

Scrutator is the search foundation for the entire [Arcanada](https://arcanada.ai) ecosystem. Without quality retrieval, no agent can effectively work with accumulated knowledge.

## License

[MIT](LICENSE)
