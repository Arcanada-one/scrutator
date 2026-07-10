# Hierarchical Navigation (SRCH-0021)

> Reference (Diátaxis). For the design rationale, see `datarim/prd/PRD-SRCH-0021.md` and
> `datarim/plans/SRCH-0021-plan.md` in the Arcanada knowledge base.

## Section metadata

At index time, the chunker normalizes each markdown chunk's header stack (`heading_hierarchy`,
already stored as `["# Doc", "## Section"]`-style strings) into a `section` object written
alongside it in `chunks.metadata` (additive — no DDL, no new column):

```jsonc
{
  "heading_hierarchy": ["# Doc", "## Section", "### Sub"],  // unchanged, back-compat
  "section": {
    "doc_id": "a1b2c3d4e5f6a7b8",           // sha256(f"{namespace}|{source_path}")[:16]
    "heading_path": ["Doc", "Section", "Sub"],
    "depth": 3,
    "anchor": "sub",
    "anchor_path": ["doc", "section", "sub"],
    "section_key": "doc/section/sub",
    "schema_version": 1
  }
}
```

`section_key` is stable within a `doc_id` and is the grouping/deep-link key used by both
navigation endpoints and `group_by`. Non-markdown chunks (code, plain text) get `section: null`
explicitly — there is no document structure to normalize.

`slugify()`, `SECTION_SCHEMA_VERSION`, and `normalize_heading_path()` live in
`src/scrutator/chunker/splitters.py` — the single source of truth for slug/section derivation,
shared by the indexer write path and `tools/backfill_sections.py`.

## Endpoints

### `GET /v1/navigate/outline`

| Param | Required | Default | Notes |
|---|---|---|---|
| `namespace` | yes | — | must resolve via `GET /v1/namespaces` (read-only — never auto-creates) |
| `source_path` | yes | — | |
| `max_nodes` | no | 2000 | hard-capped server-side at 10000 regardless of caller input |

Response (`OutlineResponse`): `source_path`, `namespace`, `doc_id`, `total_chunks`, and a nested
`outline` tree of `OutlineNode { title, anchor, depth, section_key, chunk_ids, children }`.

- `404` — unknown `namespace` or `source_path`.
- `422` — `total_chunks` exceeds `max_nodes` (checked *before* tree assembly — a truncated tree
  would silently hide children, which is worse than a typed error).

### `GET /v1/navigate/section`

| Param | Required | Notes |
|---|---|---|
| `chunk_id` | yes | must be a UUID |

Response (`SectionContext`): `chunk_id`, `doc_id`, `section_key`, `ancestors` (breadcrumb path),
`self` (the target section + its `chunk_ids`), `siblings` (same parent, same depth), `children`
(one level deeper).

- `422` — `chunk_id` is not a valid UUID.
- `404` — chunk does not exist.

### `group_by` on `POST /v1/search`

Optional field on `SearchRequest`: `group_by: "document" | "section" | null` (default `null`).
When set, the already-fused RRF results are folded post-fusion, in Python, into
`GroupedSearchResult { group_key, doc_id, score, representative, member_chunk_ids, member_count }`
— one entry per distinct `doc_id` (or `section_key`), `score` = the max member score, ordered by
first appearance in the fused RRF order. **The RRF query and ranking are never touched** — folding
happens strictly after `hybrid_search()` returns. Omitting `group_by` returns the exact
`SearchResponse` shape Scrutator has always returned (byte-identical, enforced by a committed
snapshot test).

## Un-backfilled documents (pre-SRCH-0021 data)

Chunks indexed before SRCH-0021 have no `section` key yet. Both navigation endpoints and
`group_by` degrade gracefully rather than erroring:

- **Outline**: all of a document's un-backfilled chunks fold into a single flat root node
  (`section_key = "root"`), ordered by `chunk_index`.
- **Section context**: an un-backfilled chunk resolves to that same flat root — no ancestors, no
  siblings/children beyond other un-backfilled chunks of the same document.
- **`group_by`**: falls back to grouping by `source_path` when `section` is absent.

This is what makes it safe to ship this code before the backfill runs.

## Backfill

`tools/backfill_sections.py` derives `section` for existing chunks from their already-stored
`heading_hierarchy` (or the flat-root fallback when it's empty), with **zero embedding calls**.

```bash
# Report the candidate count — no writes (default):
python tools/backfill_sections.py --namespace arcanada

# Perform the live UPDATE (operator-run — HARD-GATED, see the plan's rollout sequence):
python tools/backfill_sections.py --namespace arcanada --live
```

Idempotent: `metadata = metadata || '{"section": ...}'::jsonb` is an additive merge, and a row
whose `section.schema_version` already matches `SECTION_SCHEMA_VERSION` is excluded by the
backfill's own `WHERE` clause — re-running against an already-backfilled namespace updates 0 rows.

**Rollback** (data): the backfill only adds a `section` key. To remove it:

```sql
UPDATE chunks SET metadata = metadata - 'section' WHERE metadata ? 'section';
```
