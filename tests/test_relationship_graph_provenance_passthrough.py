"""AUP-GRAPH-009 (`polyglot4`, arcanada-universal-program) — PROPOSAL, not deployed.

Verifies that a `relationship-graph` KB page (one Markdown page per RelationshipGraph/v1 node,
as sketched in that program's `governance/design/GRAPH-009-graph-kb-namespace.md` design note)
carries its `aup_provenance` YAML frontmatter block — schema/source_repo/source_commit/
graph_digest/node_id/node_type/edge_provenance_summary — unmodified all the way through
Scrutator's real chunk → persisted-metadata pipeline, with NO changes to `IndexRequest`,
the chunker, or the indexer: this is the existing generic Markdown-frontmatter passthrough
(`chunk.metadata.frontmatter` -> `_chunk_dicts()`'s `metadata.frontmatter` -> `SearchResult.metadata`),
the same mechanism `AUP-DAT-011`'s history-datarim namespace already relies on.

This test exercises real Scrutator code (`chunk_document`, `_chunk_dicts`, `SearchResult`) with
no DB, no network, and no namespace/route change — it does not index, deploy, or grant anything.
It exists to turn an assumption (stated as `Not verified this life` in the design note it serves)
into a checked, regression-guarded fact before any activation work is proposed.
"""

from __future__ import annotations

from scrutator.chunker import chunk_document
from scrutator.db.models import SearchResult
from scrutator.search.indexer import _chunk_dicts

_AUP_PROVENANCE = {
    "schema": "AupGraphProvenance/v1",
    "source_repo": "Arcanada-one/arcanada-universal-program",
    "source_commit": "805ab497f854e65f3699e2ab760c888eb454cb68",
    "graph_file": "receipts/graph/verifier-out/polyglot4-kbquery0/program.graph.json",
    "graph_digest": "sha256:71cc2cff5420f178156c1d393a5044ee5833c2cb515e4e5d53294dfda8aba571",
    "built_at_utc": "2026-09-05T20:29:58Z",
    "node_id": "code_unit:tools/graph/build_graph.py",
    "node_type": "code_unit",
    "edge_provenance_summary": {"deterministic": 3, "inferred": 0, "observed": 0},
}

_PAGE = f"""---
aup_provenance:
  schema: {_AUP_PROVENANCE['schema']}
  source_repo: {_AUP_PROVENANCE['source_repo']}
  source_commit: {_AUP_PROVENANCE['source_commit']}
  graph_file: {_AUP_PROVENANCE['graph_file']}
  graph_digest: {_AUP_PROVENANCE['graph_digest']}
  built_at_utc: "{_AUP_PROVENANCE['built_at_utc']}"
  node_id: "{_AUP_PROVENANCE['node_id']}"
  node_type: {_AUP_PROVENANCE['node_type']}
  edge_provenance_summary:
    deterministic: 3
    inferred: 0
    observed: 0
---
# code_unit: tools/graph/build_graph.py

## Edges in (what depends on this node), with provenance
- `imports` (deterministic) <- `code_unit:tools/graph/impact.py`
"""

_SOURCE_PATH = "kb-graph-projection/arcanada-universal-program/code_unit/tools/graph/build_graph.py.md"


def test_frontmatter_survives_chunking():
    result = chunk_document(_PAGE, source_path=_SOURCE_PATH, source_type="markdown")
    assert result.chunks, "the page must produce at least one chunk"
    fm = result.chunks[0].metadata.frontmatter
    assert fm["aup_provenance"]["source_commit"] == _AUP_PROVENANCE["source_commit"]
    assert fm["aup_provenance"]["graph_digest"] == _AUP_PROVENANCE["graph_digest"]
    assert fm["aup_provenance"]["node_id"] == _AUP_PROVENANCE["node_id"]
    assert fm["aup_provenance"]["edge_provenance_summary"] == {"deterministic": 3, "inferred": 0, "observed": 0}


def test_frontmatter_survives_persisted_metadata_shape():
    result = chunk_document(_PAGE, source_path=_SOURCE_PATH, source_type="markdown")
    rows = _chunk_dicts(result, namespace="relationship-graph", source_path=_SOURCE_PATH, full_content=_PAGE)
    assert rows, "at least one persisted chunk row"
    stored_fm = rows[0]["metadata"]["frontmatter"]
    assert stored_fm["aup_provenance"] == _AUP_PROVENANCE


def test_frontmatter_survives_into_search_result_model():
    """Proves the shape a `/v1/search` hit would actually return: `SearchResult.metadata` is
    populated verbatim from the persisted row's `metadata` dict (see `search/searcher.py`,
    `metadata=r.get("metadata", {})`) -- no server-side code reads or interprets `aup_provenance`
    (this PR adds none), it only has to survive being carried, which it already does."""
    result = chunk_document(_PAGE, source_path=_SOURCE_PATH, source_type="markdown")
    rows = _chunk_dicts(result, namespace="relationship-graph", source_path=_SOURCE_PATH, full_content=_PAGE)
    row = rows[0]
    hit = SearchResult(
        chunk_id=row["id"],
        content=row["content"],
        source_path=row["source_path"],
        source_type=row["source_type"],
        chunk_index=row["chunk_index"],
        score=1.0,
        namespace="relationship-graph",
        project="arcanada-universal-program",
        metadata=row["metadata"],
    )
    assert hit.metadata["frontmatter"]["aup_provenance"]["source_commit"] == _AUP_PROVENANCE["source_commit"]
    assert hit.namespace == "relationship-graph"


def test_namespace_field_is_a_free_string_not_a_static_allowlist():
    """Corrects an assumption in the design note this test serves: there is no
    `SCRUTATOR_FEEDER_NAMESPACES`-shaped constant inside this repository to edit (verified by
    reading `db/models.py::IndexRequest.namespace: str = "arcanada"` -- namespace is caller-supplied
    and gated by the ReBAC/`principal_namespace_grants` reader-grant mechanism in
    `auth/dependency.py` and `auth.dependencies.yaml`, not by a code-level allowlist). Declaring
    `relationship-graph` as a namespace therefore needs a reader grant at activation time, not a
    source change here -- this test only pins the field's type so a future change can't silently
    narrow it to a closed enum without this test telling the author why that would break this PR's
    premise."""
    from scrutator.db.models import IndexRequest

    req = IndexRequest(content="x", source_path="p.md", namespace="relationship-graph")
    assert req.namespace == "relationship-graph"
