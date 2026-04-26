"""Tests for LTM pydantic models."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from scrutator.ltm.models import (
    Entity,
    EntityEdge,
    EntityEvent,
    EventType,
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


class TestRecallRequestTemporal:
    def test_defaults_backward_compatible(self):
        req = RecallRequest(query="x")
        assert req.as_of is None
        assert req.time_range is None
        assert req.temporal_boost == 0.3

    def test_as_of_accepts_iso(self):
        req = RecallRequest(query="x", as_of="2026-04-26T10:00:00+00:00")
        assert req.as_of == datetime(2026, 4, 26, 10, 0, tzinfo=UTC)

    def test_time_range_ordered(self):
        t1 = datetime(2026, 4, 1, tzinfo=UTC)
        t2 = datetime(2026, 4, 30, tzinfo=UTC)
        req = RecallRequest(query="x", time_range=(t1, t2))
        assert req.time_range == (t1, t2)

    def test_time_range_inverted_rejected(self):
        t1 = datetime(2026, 4, 30, tzinfo=UTC)
        t2 = datetime(2026, 4, 1, tzinfo=UTC)
        with pytest.raises(ValidationError, match="start must be before end"):
            RecallRequest(query="x", time_range=(t1, t2))

    def test_boost_out_of_range(self):
        with pytest.raises(ValidationError, match=r"\[0\.0, 1\.0\]"):
            RecallRequest(query="x", temporal_boost=1.5)
        with pytest.raises(ValidationError, match=r"\[0\.0, 1\.0\]"):
            RecallRequest(query="x", temporal_boost=-0.1)


class TestEntityEvent:
    def test_minimal_with_when_t(self):
        e = EntityEvent(
            entity_name="TUNE-0003",
            event_type=EventType.ARCHIVED,
            when_t=datetime(2026, 4, 16, tzinfo=UTC),
        )
        assert e.entity_name == "TUNE-0003"
        assert e.valid_from is None

    def test_minimal_with_valid_from(self):
        e = EntityEvent(
            entity_name="X",
            event_type="released",
            valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert e.when_t is None

    def test_no_timestamp_rejected(self):
        with pytest.raises(ValidationError, match="at least one of when_t"):
            EntityEvent(entity_name="X", event_type="archived")

    def test_valid_period_inverted_rejected(self):
        with pytest.raises(ValidationError, match="valid_from must be before valid_to"):
            EntityEvent(
                entity_name="X",
                event_type="archived",
                valid_from=datetime(2026, 4, 1, tzinfo=UTC),
                valid_to=datetime(2026, 3, 1, tzinfo=UTC),
            )

    def test_empty_entity_name_rejected(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            EntityEvent(
                entity_name="  ",
                event_type="archived",
                when_t=datetime(2026, 4, 16, tzinfo=UTC),
            )

    def test_description_truncated(self):
        long_desc = "x" * 600
        e = EntityEvent(
            entity_name="X",
            event_type="archived",
            when_t=datetime(2026, 4, 16, tzinfo=UTC),
            description=long_desc,
        )
        assert len(e.description) == 500

    def test_event_type_freeform(self):
        # Allow extras (LLM may emit unknown types) — pipeline filters
        e = EntityEvent(
            entity_name="X",
            event_type="custom_unknown",
            when_t=datetime(2026, 4, 16, tzinfo=UTC),
        )
        assert e.event_type == "custom_unknown"

    def test_zero_length_valid_period_rejected(self):
        t = datetime(2026, 4, 16, tzinfo=UTC)
        with pytest.raises(ValidationError, match="valid_from must be before valid_to"):
            EntityEvent(entity_name="X", event_type="archived", valid_from=t, valid_to=t)

    def test_microsecond_valid_period_ok(self):
        t1 = datetime(2026, 4, 16, tzinfo=UTC)
        t2 = t1 + timedelta(microseconds=1)
        e = EntityEvent(entity_name="X", event_type="archived", valid_from=t1, valid_to=t2)
        assert e.valid_to > e.valid_from
