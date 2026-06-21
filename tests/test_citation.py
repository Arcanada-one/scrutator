"""Tests for SRCH-0029 M1: typed Citation payload.

Covers:
- Citation model shape + schema_version frozen contract
- score_kind Literal values
- SearchResult gains citation: Citation | None field
- JSON round-trip
- Backward compat: SearchResult without citation still valid
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from scrutator.db.models import Citation, SearchResult


class TestCitationShape:
    """V-AC-1 — Citation shape frozen (schema_version==1, field names + types)."""

    def test_schema_version_is_1(self):
        c = Citation(
            chunk_id="abc",
            source_path="docs/readme.md",
            source_type="md",
            chunk_index=0,
            relevance_score=0.032,
            score_kind="rrf",
        )
        assert c.schema_version == 1

    def test_required_fields_present(self):
        c = Citation(
            chunk_id="x",
            source_path="a/b.md",
            source_type="markdown",
            chunk_index=3,
            relevance_score=0.05,
            score_kind="rrf",
        )
        assert c.chunk_id == "x"
        assert c.source_path == "a/b.md"
        assert c.source_type == "markdown"
        assert c.chunk_index == 3
        assert c.relevance_score == 0.05
        assert c.score_kind == "rrf"

    def test_heading_hierarchy_defaults_empty(self):
        c = Citation(
            chunk_id="x",
            source_path="a.md",
            source_type="md",
            chunk_index=0,
            relevance_score=0.01,
            score_kind="rrf",
        )
        assert c.heading_hierarchy == []

    def test_heading_hierarchy_can_be_set(self):
        c = Citation(
            chunk_id="x",
            source_path="a.md",
            source_type="md",
            chunk_index=0,
            relevance_score=0.01,
            score_kind="rrf",
            heading_hierarchy=["Section A", "Sub B"],
        )
        assert c.heading_hierarchy == ["Section A", "Sub B"]

    def test_score_kind_rrf(self):
        c = Citation(
            chunk_id="x",
            source_path="a.md",
            source_type="md",
            chunk_index=0,
            relevance_score=0.032,
            score_kind="rrf",
        )
        assert c.score_kind == "rrf"

    def test_score_kind_colbert_rerank(self):
        c = Citation(
            chunk_id="x",
            source_path="a.md",
            source_type="md",
            chunk_index=0,
            relevance_score=4.5,
            score_kind="colbert_rerank",
        )
        assert c.score_kind == "colbert_rerank"

    def test_score_kind_rejects_invalid(self):
        """Only 'rrf' and 'colbert_rerank' are valid score_kind values."""
        with pytest.raises(ValidationError):
            Citation(
                chunk_id="x",
                source_path="a.md",
                source_type="md",
                chunk_index=0,
                relevance_score=0.01,
                score_kind="invalid_kind",
            )

    def test_frozen_field_names(self):
        """Contract: these exact field names MUST exist on Citation (ARCA-0180 consumer)."""
        c = Citation(
            chunk_id="c1",
            source_path="p.md",
            source_type="md",
            chunk_index=1,
            relevance_score=0.1,
            score_kind="rrf",
        )
        d = c.model_dump()
        expected_keys = {
            "schema_version",
            "chunk_id",
            "source_path",
            "source_type",
            "chunk_index",
            "heading_hierarchy",
            "relevance_score",
            "score_kind",
        }
        actual_keys = set(d.keys())
        assert expected_keys == actual_keys, f"Citation shape changed (ARCA-0180 contract broken). Got: {actual_keys}"


class TestSearchResultCitationField:
    """SearchResult gains citation: Citation | None = None (additive, back-compat)."""

    def test_search_result_without_citation_valid(self):
        """Back-compat: existing SearchResult creation still works with citation=None."""
        r = SearchResult(
            chunk_id="abc",
            content="some content",
            source_path="wiki/test.md",
            source_type="markdown",
            chunk_index=0,
            score=0.032,
            namespace="arcanada",
        )
        assert r.citation is None

    def test_search_result_with_citation_populated(self):
        c = Citation(
            chunk_id="abc",
            source_path="wiki/test.md",
            source_type="markdown",
            chunk_index=0,
            relevance_score=0.032,
            score_kind="rrf",
        )
        r = SearchResult(
            chunk_id="abc",
            content="some content",
            source_path="wiki/test.md",
            source_type="markdown",
            chunk_index=0,
            score=0.032,
            namespace="arcanada",
            citation=c,
        )
        assert r.citation is not None
        assert r.citation.chunk_id == "abc"
        assert r.citation.score_kind == "rrf"
        assert r.citation.schema_version == 1

    def test_search_result_json_round_trip_with_citation(self):
        c = Citation(
            chunk_id="c1",
            source_path="docs/arch.md",
            source_type="md",
            chunk_index=2,
            relevance_score=0.047,
            score_kind="rrf",
            heading_hierarchy=["Architecture", "Design"],
        )
        r = SearchResult(
            chunk_id="c1",
            content="hello",
            source_path="docs/arch.md",
            source_type="md",
            chunk_index=2,
            score=0.047,
            namespace="arcanada",
            citation=c,
        )
        serialized = r.model_dump_json()
        parsed = json.loads(serialized)
        assert parsed["citation"]["schema_version"] == 1
        assert parsed["citation"]["score_kind"] == "rrf"
        assert parsed["citation"]["heading_hierarchy"] == ["Architecture", "Design"]

    def test_search_result_json_round_trip_without_citation(self):
        """None citation serializes to null and round-trips correctly."""
        r = SearchResult(
            chunk_id="c1",
            content="hello",
            source_path="docs/arch.md",
            source_type="md",
            chunk_index=0,
            score=0.01,
            namespace="arcanada",
        )
        serialized = r.model_dump_json()
        parsed = json.loads(serialized)
        assert parsed["citation"] is None

    def test_colbert_rerank_citation_score_kind(self):
        """When rerank is ON, citation.score_kind must be 'colbert_rerank'."""
        c = Citation(
            chunk_id="x",
            source_path="a.md",
            source_type="md",
            chunk_index=0,
            relevance_score=6.7,  # ColBERT MaxSim scale (larger than RRF ~0.05)
            score_kind="colbert_rerank",
        )
        r = SearchResult(
            chunk_id="x",
            source_path="a.md",
            source_type="md",
            chunk_index=0,
            score=6.7,
            namespace="ns",
            citation=c,
        )
        assert r.citation.score_kind == "colbert_rerank"
        assert r.citation.relevance_score == pytest.approx(6.7)
