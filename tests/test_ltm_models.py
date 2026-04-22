"""Tests for LTM pydantic models."""

import pytest
from pydantic import ValidationError

from scrutator.ltm.models import (
    Entity,
    EntityEdge,
    IngestRequest,
    IngestResponse,
    JobStatus,
    LtmJob,
    RecallRequest,
    RecallResponse,
    RecallResult,
)


class TestIngestRequest:
    def test_valid_request(self):
        req = IngestRequest(content="Hello world", source_path="test.md")
        assert req.namespace == "arcanada"
        assert req.project is None

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError, match="content must not be empty"):
            IngestRequest(content="  ", source_path="test.md")

    def test_empty_source_path_rejected(self):
        with pytest.raises(ValidationError, match="source_path must not be empty"):
            IngestRequest(content="text", source_path="")

    def test_content_max_length(self):
        long = "x" * 500_001
        with pytest.raises(ValidationError, match="exceeds"):
            IngestRequest(content=long, source_path="test.md")

    def test_custom_namespace(self):
        req = IngestRequest(content="text", source_path="p.md", namespace="custom", project="proj")
        assert req.namespace == "custom"
        assert req.project == "proj"


class TestIngestResponse:
    def test_creation(self):
        resp = IngestResponse(job_id="abc-123", status=JobStatus.PENDING)
        assert resp.job_id == "abc-123"
        assert resp.status == JobStatus.PENDING


class TestEntity:
    def test_valid(self):
        e = Entity(name="Scrutator", entity_type="project", description="Search engine")
        assert e.name == "Scrutator"
        assert e.properties == {}

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError, match="name must not be empty"):
            Entity(name="", entity_type="project")

    def test_empty_type_rejected(self):
        with pytest.raises(ValidationError, match="entity_type must not be empty"):
            Entity(name="X", entity_type="  ")

    def test_with_properties(self):
        e = Entity(name="X", entity_type="concept", properties={"url": "http://x.com"})
        assert e.properties["url"] == "http://x.com"


class TestEntityEdge:
    def test_valid(self):
        edge = EntityEdge(source="Scrutator", target="Arcanada", relation="part_of")
        assert edge.weight == 1.0

    def test_empty_relation_rejected(self):
        with pytest.raises(ValidationError, match="relation must not be empty"):
            EntityEdge(source="A", target="B", relation="")


class TestLtmJob:
    def test_defaults(self):
        job = LtmJob(id="j1", namespace="arcanada", source_path="test.md")
        assert job.status == JobStatus.PENDING
        assert job.total_chunks == 0
        assert job.processed_chunks == 0
        assert job.error is None

    def test_all_statuses(self):
        for s in JobStatus:
            job = LtmJob(id="j1", namespace="ns", source_path="p.md", status=s)
            assert job.status == s


class TestRecallRequest:
    def test_valid(self):
        req = RecallRequest(query="What is Scrutator?")
        assert req.limit == 10
        assert req.expand_entities is True

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError, match="query must not be empty"):
            RecallRequest(query="  ")

    def test_limit_capped(self):
        req = RecallRequest(query="test", limit=100)
        assert req.limit == 50

    def test_limit_min(self):
        with pytest.raises(ValidationError, match="limit must be >= 1"):
            RecallRequest(query="test", limit=0)


class TestRecallResult:
    def test_with_entities(self):
        r = RecallResult(
            chunk_id="c1",
            content="text",
            source_path="p.md",
            score=0.85,
            namespace="arcanada",
            entities=[Entity(name="X", entity_type="concept")],
            relations=[EntityEdge(source="X", target="Y", relation="related")],
        )
        assert len(r.entities) == 1
        assert len(r.relations) == 1


class TestRecallResponse:
    def test_creation(self):
        resp = RecallResponse(
            results=[],
            total=0,
            query="test",
            search_time_ms=12.5,
        )
        assert resp.total == 0
