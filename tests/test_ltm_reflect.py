"""Tests for LTM-0013 Reflect layer (ReflectBudget + ReflectJob)."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from scrutator.config import settings
from scrutator.ltm.reflect import (
    ReflectBudget,
    ReflectBudgetExceeded,
    ReflectJob,
)


@pytest.fixture
def entity_grouping(monkeypatch):
    """Pin LTM-0013 entity-grouping path for legacy ReflectJob tests."""
    monkeypatch.setattr(settings, "ltm_reflect_grouping", "entity")
    yield


# ---- ReflectBudget -----------------------------------------------------------


class TestReflectBudget:
    def test_check_passes_when_below_caps(self):
        b = ReflectBudget(max_usd=0.01, max_req=10)
        b.check()  # no raise

    def test_check_raises_at_usd_cap(self):
        b = ReflectBudget(max_usd=0.001, max_req=100)
        b.charge(0.001)
        with pytest.raises(ReflectBudgetExceeded, match="USD cap"):
            b.check()

    def test_check_raises_at_req_cap(self):
        b = ReflectBudget(max_usd=10.0, max_req=2)
        b.charge(0.0)
        b.charge(0.0)
        with pytest.raises(ReflectBudgetExceeded, match="req cap"):
            b.check()

    def test_charge_accumulates(self):
        b = ReflectBudget(max_usd=10.0, max_req=10)
        b.charge(0.001)
        b.charge(0.002)
        assert b.spent_usd == pytest.approx(0.003)
        assert b.req_count == 2


# ---- ReflectJob --------------------------------------------------------------


@pytest.fixture
def llm():
    m = AsyncMock()
    m.extract_json = AsyncMock()
    m.model = "openrouter/gemini-2.5-flash"
    return m


@pytest.fixture
def budget():
    return ReflectBudget(max_usd=1.0, max_req=100)


@pytest.fixture
def job(llm, budget):
    return ReflectJob(
        llm=llm,
        namespace="arcanada",
        namespace_id=1,
        budget=budget,
        max_meta_facts_per_group=3,
    )


def _patch_repo(**overrides):
    """Build (context-manager, mocks-dict) for reflect.repository.

    `patch.multiple` does not yield a useful object by default — we expose the
    AsyncMocks directly so tests can assert on them.
    """
    mocks = {
        "create_reflect_run": AsyncMock(return_value="run-1"),
        "finalize_reflect_run": AsyncMock(return_value=None),
        "fetch_chunks_for_reflect": AsyncMock(return_value={}),
        "insert_meta_fact": AsyncMock(return_value="meta-1"),
    }
    mocks.update(overrides)
    return patch.multiple("scrutator.ltm.reflect.repository", **mocks), mocks


@pytest.mark.usefixtures("entity_grouping")
class TestReflectJob:
    async def test_empty_namespace_returns_zero_facts(self, job):
        ctx, _mocks = _patch_repo()
        with ctx:
            summary, facts = await job.run()
        assert facts == []
        assert summary.status == "done"
        assert summary.meta_facts_created == 0
        assert summary.chunks_scanned == 0

    async def test_single_chunk_group_skipped(self, job, llm):
        groups = {"Alpha": [{"chunk_id": "c1", "content": "alone"}]}
        ctx, _m = _patch_repo(fetch_chunks_for_reflect=AsyncMock(return_value=groups))
        with ctx:
            summary, facts = await job.run()
        assert facts == []
        assert summary.chunks_scanned == 1
        llm.extract_json.assert_not_called()

    async def test_happy_path_persists_one_fact(self, job, llm):
        groups = {
            "Scrutator": [
                {"chunk_id": "11111111-1111-1111-1111-111111111111", "content": "A"},
                {"chunk_id": "22222222-2222-2222-2222-222222222222", "content": "B"},
            ]
        }
        llm.extract_json.return_value = [
            {
                "fact_type": "summary",
                "content": "Scrutator combines hybrid search and entity graph.",
                "source_chunk_indexes": [0, 1],
            }
        ]
        ctx, mocks = _patch_repo(fetch_chunks_for_reflect=AsyncMock(return_value=groups))
        with (
            ctx,
            patch(
                "scrutator.ltm.reflect._embed_for_meta_fact",
                AsyncMock(return_value=[0.0] * 1024),
            ),
        ):
            summary, facts = await job.run()
        assert summary.meta_facts_created == 1
        assert facts[0].content.startswith("Scrutator combines")
        assert facts[0].depth == 1
        assert facts[0].id == "meta-1"
        mocks["insert_meta_fact"].assert_awaited_once()

    async def test_dry_run_does_not_persist(self, job, llm):
        groups = {
            "X": [
                {"chunk_id": "11111111-1111-1111-1111-111111111111", "content": "a"},
                {"chunk_id": "22222222-2222-2222-2222-222222222222", "content": "b"},
            ]
        }
        llm.extract_json.return_value = [
            {"fact_type": "summary", "content": "X is a thing.", "source_chunk_indexes": [0, 1]}
        ]
        ctx, mocks = _patch_repo(fetch_chunks_for_reflect=AsyncMock(return_value=groups))
        with ctx:
            summary, facts = await job.run(dry_run=True)
        assert len(facts) == 1
        assert facts[0].id is None
        mocks["insert_meta_fact"].assert_not_awaited()
        assert summary.meta_facts_created == 1

    async def test_malformed_llm_output_skipped(self, job, llm):
        groups = {
            "X": [
                {"chunk_id": "11111111-1111-1111-1111-111111111111", "content": "a"},
                {"chunk_id": "22222222-2222-2222-2222-222222222222", "content": "b"},
            ]
        }
        llm.extract_json.return_value = {"raw": "not a list"}
        ctx, _m = _patch_repo(fetch_chunks_for_reflect=AsyncMock(return_value=groups))
        with ctx:
            summary, facts = await job.run()
        assert facts == []
        assert summary.status == "done"
        assert summary.req_count == 1  # LLM still charged

    async def test_invalid_fact_type_dropped(self, job, llm):
        groups = {
            "X": [
                {"chunk_id": "11111111-1111-1111-1111-111111111111", "content": "a"},
                {"chunk_id": "22222222-2222-2222-2222-222222222222", "content": "b"},
            ]
        }
        llm.extract_json.return_value = [
            {"fact_type": "nonsense", "content": "ignored", "source_chunk_indexes": [0]},
            {"fact_type": "summary", "content": "kept", "source_chunk_indexes": [0, 1]},
        ]
        ctx, _m = _patch_repo(fetch_chunks_for_reflect=AsyncMock(return_value=groups))
        with (
            ctx,
            patch(
                "scrutator.ltm.reflect._embed_for_meta_fact",
                AsyncMock(return_value=None),
            ),
        ):
            _, facts = await job.run()
        assert len(facts) == 1
        assert facts[0].content == "kept"

    async def test_budget_abort_sets_aborted_status(self, llm):
        budget = ReflectBudget(max_usd=0.001, max_req=10)
        budget.charge(0.001)  # already over USD cap
        job = ReflectJob(llm=llm, namespace="ns", namespace_id=1, budget=budget)
        groups = {
            "A": [{"chunk_id": "c1", "content": "x"}, {"chunk_id": "c2", "content": "y"}],
            "B": [{"chunk_id": "c3", "content": "x"}, {"chunk_id": "c4", "content": "y"}],
        }
        ctx, _m = _patch_repo(fetch_chunks_for_reflect=AsyncMock(return_value=groups))
        with ctx:
            summary, facts = await job.run()
        assert summary.status == "aborted"
        assert summary.abort_reason and "USD cap" in summary.abort_reason
        llm.extract_json.assert_not_called()

    async def test_chunk_without_entity_link_silently_skipped(self, job, llm):
        # Repository's fetch_chunks_for_reflect already excludes entity-less chunks
        # via INNER JOIN entities. ReflectJob then sees zero groups → no exception, no facts.
        ctx, _m = _patch_repo(fetch_chunks_for_reflect=AsyncMock(return_value={}))
        with ctx:
            summary, facts = await job.run()
        assert summary.chunks_scanned == 0
        assert summary.meta_facts_created == 0
        assert summary.status == "done"
        llm.extract_json.assert_not_called()

    async def test_depth_two_attempt_rejected(self, job):
        # Direct instantiation must reject depth != 1 even at the model layer
        from pydantic import ValidationError

        from scrutator.ltm.models import FactType, MetaFact

        with pytest.raises(ValidationError, match="depth must equal 1"):
            MetaFact(
                namespace="ns",
                fact_type=FactType.SUMMARY,
                content="x",
                source_chunk_ids=["00000000-0000-0000-0000-000000000001"],
                depth=2,
                model_used="m",
            )

    async def test_llm_exception_continues_run(self, job, llm):
        groups = {
            "X": [
                {"chunk_id": "11111111-1111-1111-1111-111111111111", "content": "a"},
                {"chunk_id": "22222222-2222-2222-2222-222222222222", "content": "b"},
            ]
        }
        llm.extract_json.side_effect = RuntimeError("MC down")
        ctx, _m = _patch_repo(fetch_chunks_for_reflect=AsyncMock(return_value=groups))
        with ctx:
            summary, facts = await job.run()
        assert facts == []
        assert summary.status == "done"  # per-group failure ≠ run failure
        assert summary.req_count == 1  # still charged the failed call


class TestReflectCosineGrouping:
    """LTM-0018 — cosine-grouping branch + integration."""

    async def test_cosine_grouping_branch_calls_cosine_fetch(self, job, llm, monkeypatch):
        """Default LTM-0018 path: grouping=cosine routes to fetch_chunks_for_reflect_cosine."""
        monkeypatch.setattr(settings, "ltm_reflect_grouping", "cosine")
        monkeypatch.setattr(settings, "ltm_reflect_cosine_threshold", 0.85)
        cosine_mock = AsyncMock(return_value={})
        ctx, _m = _patch_repo(fetch_chunks_for_reflect_cosine=cosine_mock)
        with ctx:
            summary, facts = await job.run(max_chunks=30)
        cosine_mock.assert_awaited_once()
        kwargs = cosine_mock.await_args.kwargs
        assert kwargs["threshold"] == 0.85
        assert kwargs["limit"] == 30
        assert summary.status == "done"
        assert facts == []

    async def test_reflect_run_cosine_grouping_30_chunks(self, job, llm, monkeypatch):
        monkeypatch.setattr(settings, "ltm_reflect_grouping", "cosine")
        monkeypatch.setattr(settings, "ltm_reflect_cosine_threshold", 0.85)

        # Synthesise 30 unit-norm vectors in 5 tight clusters of 6 each.
        rng = np.random.default_rng(42)
        cluster_centers = rng.normal(size=(5, 16)).astype(np.float32)
        cluster_centers /= np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        rows: list[np.ndarray] = []
        for c in cluster_centers:
            for _ in range(6):
                noise = rng.normal(scale=0.05, size=16).astype(np.float32)
                v = c + noise
                v /= np.linalg.norm(v)
                rows.append(v)
        vectors = np.stack(rows)

        from scrutator.ltm.grouping import cluster_by_cosine

        index_groups = cluster_by_cosine(vectors, 0.85)
        assert len(index_groups) >= 5  # tight synthetic clusters → at least 5 groups

        # Build the dict that fetch_chunks_for_reflect_cosine would return.
        chunk_uuid = lambda i: f"00000000-0000-0000-0000-{i:012d}"  # noqa: E731
        groups: dict[str, list[dict]] = {
            f"cluster_{root}": [
                {"chunk_id": chunk_uuid(i), "content": f"chunk-{i}", "entity_id": None} for i in indices
            ]
            for root, indices in index_groups.items()
        }

        # Mock LLM: 3 facts per group covering the first two source indexes.
        async def fake_extract(_user, system=None):  # noqa: ARG001
            return [
                {
                    "fact_type": "summary",
                    "content": f"Synthetic meta-fact #{n}",
                    "source_chunk_indexes": [0, 1],
                }
                for n in range(3)
            ]

        llm.extract_json.side_effect = fake_extract

        ctx, mocks = _patch_repo(
            fetch_chunks_for_reflect_cosine=AsyncMock(return_value=groups),
        )
        with (
            ctx,
            patch(
                "scrutator.ltm.reflect._embed_for_meta_fact",
                AsyncMock(return_value=[0.0] * 1024),
            ),
        ):
            summary, facts = await job.run(max_chunks=30)

        assert summary.status == "done"
        assert summary.meta_facts_created >= 10  # AC-2 mirror
        # Cosine path produces empty entity_ids — schema contract.
        assert all(fact.entity_ids == [] for fact in facts)
        # Source chunk IDs preserved from synthesised groups.
        assert all(fact.source_chunk_ids for fact in facts)
        mocks["insert_meta_fact"].assert_awaited()
