# `relationship-graph` namespace: provenance passthrough is already sufficient (PROPOSAL note)

**Status:** PROPOSAL, `PLANNED_NOT_INGESTED` — nothing in this repository is indexed, deployed, or
granted by this change. **Serves:** `AUP-GRAPH-009` (`arcanada-universal-program`,
`governance/design/GRAPH-009-graph-kb-namespace.md`), portion `AUP-GRAPH-009:polyglot4`.

## What this note is

`arcanada-universal-program` wants to publish its `RelationshipGraph/v1` (code/contract/route/
document/work-item dependency graph, with a commit and a `deterministic|inferred|observed` label
on every edge) into this Scrutator instance's KB, one small Markdown page per graph node, with a
`aup_provenance` YAML frontmatter block per page (schema, source repo, source commit, graph
digest, node id, and a count of deterministic/inferred/observed edges touching that node).

Its own design note assumed two possible gaps on this side and stated them as "not independently
verified" (analogy from a different KB lane, `AUP-DAT-011`, one commit prior):

1. Whether the generic Markdown frontmatter this repository already extracts on ingest
   (`chunker/metadata.py::extract_frontmatter`) actually survives, unmodified, all the way to a
   `/v1/search` hit's `metadata` field.
2. Whether a new namespace needs a `SCRUTATOR_FEEDER_NAMESPACES`-shaped allowlist entry added in
   this repository's own source.

`tests/test_relationship_graph_provenance_passthrough.py` (added by this PR) answers both, using
this repository's own real code, no DB, no network, no route added:

1. **Yes, it survives, byte for byte.** `chunk_document()` → `_chunk_dicts()` →
   `SearchResult.metadata["frontmatter"]` carries the whole `aup_provenance` block unmodified —
   this is the exact mechanism `AUP-DAT-011`'s history-datarim lane already depends on, now pinned
   by a regression test for this specific (deeper, JSON-graph-derived) provenance shape. **No
   change to `IndexRequest`, the chunker, or the indexer is needed for provenance to reach a
   search hit.**
2. **No such allowlist exists here.** `IndexRequest.namespace` is a caller-supplied `str` (see
   `db/models.py`); read access is gated per-principal by the ReBAC/OpenFGA tuple system (or its
   local `principal_namespace_grants` fallback) described in `auth.dependencies.yaml` and
   `auth/dependency.py`, not by a static list in source. Declaring `relationship-graph` as a valid
   namespace is a **grant**, made at activation time for whichever principal runs the KB-feeder
   ingest and whichever principals should read it back — not a code change in this repository.

## What this means for the design note's activation checklist

Its checklist item 3 ("`relationship-graph` appended to `SCRUTATOR_FEEDER_NAMESPACES` on the
Scrutator side") named a symbol that does not exist in this codebase; the real remaining
activation work is entirely the reader-grant step already named in that note's item 5, plus the
projector tool (item 1) and the taxonomy route in `kb-feeder` (item 2, already proposed as a
separate, inert PR against `Arcanada-one/arcanada-workspace`). This repository requires **no
source change** to receive and return `aup_provenance`-bearing pages once those two land.

## Non-goals of this PR

- No namespace is created, granted, or indexed.
- No route, config default, or allowlist is changed.
- No production Scrutator instance (`arcana-kb`) is touched — this PR only adds a test and this
  note, run and read locally in a disposable clone.
