"""Tests for LTM ingest and recall pipelines."""

from unittest.mock import AsyncMock, patch

import pytest

from scrutator.ltm.models import Entity, RecallResult
from scrutator.ltm.pipeline import IngestPipeline, RecallPipeline


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.extract_json = AsyncMock()
    return llm


@pytest.fixture
def ingest(mock_llm):
    return IngestPipeline(llm=mock_llm, namespace="test", namespace_id=1)


@pytest.fixture
def recall(mock_llm):
    return RecallPipeline(llm=mock_llm, namespace="test", namespace_id=1)


class TestExtractEntities:
    @pytest.mark.asyncio
    async def test_valid_entities(self, ingest, mock_llm):
        mock_llm.extract_json.return_value = [
            {"name": "Scrutator", "type": "project", "description": "Search engine"},
            {"name": "PostgreSQL", "type": "technology", "description": "Database"},
        ]
        entities = await ingest.extract_entities("Scrutator uses PostgreSQL")
        assert len(entities) == 2
        assert entities[0].name == "Scrutator"
        assert entities[0].entity_type == "project"

    @pytest.mark.asyncio
    async def test_llm_returns_garbage(self, ingest, mock_llm):
        mock_llm.extract_json.return_value = {"raw": "No entities here"}
        entities = await ingest.extract_entities("Some text")
        assert entities == []

    @pytest.mark.asyncio
    async def test_malformed_entities_filtered(self, ingest, mock_llm):
        mock_llm.extract_json.return_value = [
            {"name": "Good", "type": "concept"},
            {"name": "", "type": "concept"},  # invalid: empty name
            {"bad_key": "value"},  # invalid: missing name
        ]
        entities = await ingest.extract_entities("text")
        assert len(entities) == 1
        assert entities[0].name == "Good"

    @pytest.mark.asyncio
    async def test_max_entities_enforced(self, ingest, mock_llm):
        ingest.max_entities_per_chunk = 2
        mock_llm.extract_json.return_value = [{"name": f"E{i}", "type": "concept"} for i in range(10)]
        entities = await ingest.extract_entities("text")
        assert len(entities) == 2


class TestExtractEdges:
    @pytest.mark.asyncio
    async def test_valid_edges(self, ingest, mock_llm):
        mock_llm.extract_json.return_value = [
            {"source": "Scrutator", "target": "PostgreSQL", "relation": "uses"},
        ]
        entities = [
            Entity(name="Scrutator", entity_type="project"),
            Entity(name="PostgreSQL", entity_type="technology"),
        ]
        edges = await ingest.extract_edges("Scrutator uses PostgreSQL", entities)
        assert len(edges) == 1
        assert edges[0].source == "Scrutator"
        assert edges[0].relation == "uses"

    @pytest.mark.asyncio
    async def test_edges_from_garbage(self, ingest, mock_llm):
        mock_llm.extract_json.return_value = {"raw": "nothing"}
        edges = await ingest.extract_edges("text", [])
        assert edges == []

    @pytest.mark.asyncio
    async def test_unknown_entity_in_edge_filtered(self, ingest, mock_llm):
        mock_llm.extract_json.return_value = [
            {"source": "A", "target": "UNKNOWN", "relation": "knows"},
        ]
        entities = [Entity(name="A", entity_type="person")]
        edges = await ingest.extract_edges("text", entities)
        assert len(edges) == 0  # UNKNOWN not in entities list


class TestRecallPipeline:
    @pytest.mark.asyncio
    async def test_enrich_results_with_entities(self, recall):
        with patch("scrutator.ltm.pipeline.repository") as mock_repo:
            mock_repo.get_entities_for_chunks = AsyncMock(
                return_value={
                    "c1": [{"name": "Scrutator", "entity_type": "project", "description": None, "properties": {}}],
                }
            )
            mock_repo.get_entity_edges_for_chunks = AsyncMock(
                return_value={
                    "c1": [{"source_name": "Scrutator", "target_name": "PG", "relation": "uses", "weight": 1.0}],
                }
            )

            search_results = [
                {
                    "chunk_id": "c1",
                    "content": "text",
                    "source_path": "p.md",
                    "score": 0.9,
                    "namespace": "test",
                    "project": None,
                    "metadata": {},
                },
            ]
            enriched = await recall.enrich_with_entities(search_results)
            assert len(enriched) == 1
            assert len(enriched[0].entities) == 1
            assert enriched[0].entities[0].name == "Scrutator"

    @pytest.mark.asyncio
    async def test_enrich_empty_results(self, recall):
        with patch("scrutator.ltm.pipeline.repository") as mock_repo:
            mock_repo.get_entities_for_chunks = AsyncMock(return_value={})
            mock_repo.get_entity_edges_for_chunks = AsyncMock(return_value={})

            enriched = await recall.enrich_with_entities([])
            assert enriched == []


class TestDedup:
    @pytest.mark.asyncio
    async def test_dedup_merges_aliases(self, ingest, mock_llm):
        mock_llm.extract_json.return_value = [
            {"canonical": "PostgreSQL", "aliases": ["Postgres", "PG"]},
        ]
        entity_names = ["PostgreSQL", "Postgres", "PG", "Scrutator"]
        groups = await ingest.dedup_entities(entity_names)
        assert len(groups) == 1
        assert groups[0]["canonical"] == "PostgreSQL"
        assert "Postgres" in groups[0]["aliases"]

    @pytest.mark.asyncio
    async def test_dedup_garbage_returns_empty(self, ingest, mock_llm):
        mock_llm.extract_json.return_value = {"raw": "nothing"}
        groups = await ingest.dedup_entities(["A", "B"])
        assert groups == []

    @pytest.mark.asyncio
    async def test_dedup_skipped_when_few_entities(self, ingest, mock_llm):
        groups = await ingest.dedup_entities(["Only_one"])
        assert groups == []
        mock_llm.extract_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_dedup_malformed_groups_filtered(self, ingest, mock_llm):
        mock_llm.extract_json.return_value = [
            {"canonical": "Good", "aliases": ["G"]},
            {"no_canonical": "bad"},  # missing canonical
            {"canonical": "", "aliases": []},  # empty canonical
        ]
        groups = await ingest.dedup_entities(["Good", "G", "X"])
        assert len(groups) == 1
        assert groups[0]["canonical"] == "Good"


class TestRerank:
    @pytest.mark.asyncio
    async def test_rerank_reorders(self, recall, mock_llm):
        mock_llm.extract_json.return_value = ["c2", "c1"]
        results = [
            RecallResult(chunk_id="c1", content="a", source_path="a.md", score=0.9, namespace="test"),
            RecallResult(chunk_id="c2", content="b", source_path="b.md", score=0.8, namespace="test"),
        ]
        reranked = await recall.rerank(query="test query", results=results)
        assert reranked[0].chunk_id == "c2"
        assert reranked[1].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_rerank_garbage_preserves_order(self, recall, mock_llm):
        mock_llm.extract_json.return_value = {"raw": "nothing"}
        results = [
            RecallResult(chunk_id="c1", content="a", source_path="a.md", score=0.9, namespace="test"),
            RecallResult(chunk_id="c2", content="b", source_path="b.md", score=0.8, namespace="test"),
        ]
        reranked = await recall.rerank(query="test query", results=results)
        assert reranked[0].chunk_id == "c1"  # original order preserved

    @pytest.mark.asyncio
    async def test_rerank_skipped_when_single_result(self, recall, mock_llm):
        results = [
            RecallResult(chunk_id="c1", content="a", source_path="a.md", score=0.9, namespace="test"),
        ]
        reranked = await recall.rerank(query="q", results=results)
        assert len(reranked) == 1
        mock_llm.extract_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_rerank_unknown_ids_ignored(self, recall, mock_llm):
        mock_llm.extract_json.return_value = ["unknown_id", "c1", "c2"]
        results = [
            RecallResult(chunk_id="c1", content="a", source_path="a.md", score=0.9, namespace="test"),
            RecallResult(chunk_id="c2", content="b", source_path="b.md", score=0.8, namespace="test"),
        ]
        reranked = await recall.rerank(query="q", results=results)
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "c1"  # unknown id skipped, c1 first


class TestEnrichWithMetaFacts:
    """LTM-0013 — meta-fact augmentation in recall."""

    @pytest.mark.asyncio
    async def test_returns_results_unchanged_when_flag_off(self, recall):
        from scrutator.config import settings

        original = settings.ltm_recall_include_meta_facts
        settings.ltm_recall_include_meta_facts = False
        try:
            results = [
                RecallResult(chunk_id="c1", content="a", source_path="a.md", score=0.9, namespace="test"),
            ]
            out = await recall.enrich_with_meta_facts(results, query_embedding=[0.0])
            assert out is results  # short-circuit, same list
        finally:
            settings.ltm_recall_include_meta_facts = original

    @pytest.mark.asyncio
    async def test_no_embedding_short_circuits(self, recall):
        from scrutator.config import settings

        settings.ltm_recall_include_meta_facts = True
        try:
            results = [
                RecallResult(chunk_id="c1", content="a", source_path="a.md", score=0.9, namespace="test"),
            ]
            out = await recall.enrich_with_meta_facts(results, query_embedding=None)
            assert len(out) == 1
            assert all(not r.metadata.get("meta_fact") for r in out)
        finally:
            settings.ltm_recall_include_meta_facts = False

    @pytest.mark.asyncio
    async def test_appends_meta_facts_with_score_factor(self, recall):
        from scrutator.config import settings

        settings.ltm_recall_include_meta_facts = True
        try:
            with patch("scrutator.ltm.pipeline.repository") as mock_repo:
                mock_repo.search_meta_facts = AsyncMock(
                    return_value=[
                        {
                            "id": "mf-1",
                            "fact_type": "summary",
                            "content": "synthetic meta",
                            "score": 0.8,
                            "source_chunk_ids": ["c1"],
                            "reflect_run_id": "r-1",
                            "model_used": "openrouter/gemini-2.5-flash",
                        }
                    ]
                )
                results = [
                    RecallResult(chunk_id="c1", content="a", source_path="a.md", score=0.9, namespace="test"),
                ]
                out = await recall.enrich_with_meta_facts(results, query_embedding=[0.0] * 1024, score_factor=0.5)
                assert len(out) == 2
                assert out[1].chunk_id == "meta:mf-1"
                assert out[1].metadata["meta_fact"] is True
                assert out[1].metadata["fact_type"] == "summary"
                assert out[1].score == pytest.approx(0.8 * 0.5)
        finally:
            settings.ltm_recall_include_meta_facts = False

    @pytest.mark.asyncio
    async def test_synthetic_chunk_id_preserved_through_rerank(self, recall, mock_llm):
        # Rerank receives meta:mf-1 alongside c1 — must preserve the prefix
        mock_llm.extract_json.return_value = ["meta:mf-1", "c1"]
        results = [
            RecallResult(chunk_id="c1", content="a", source_path="a.md", score=0.9, namespace="test"),
            RecallResult(
                chunk_id="meta:mf-1",
                content="synthetic",
                source_path="meta_fact/summary",
                score=0.4,
                namespace="test",
                metadata={"meta_fact": True},
            ),
        ]
        reranked = await recall.rerank(query="q", results=results)
        assert reranked[0].chunk_id == "meta:mf-1"
        assert reranked[0].metadata.get("meta_fact") is True
