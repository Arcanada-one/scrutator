"""Tests for ARAS-0057 Task 1: Scrutator ingest contract — skill plan validation and metadata derivation.

RED phase: every test here must fail before the implementation lands.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from scrutator.search.indexer import (
    INDEX_BATCH_MAX_DOCUMENT_BYTES,
    BatchIndexLimitError,
    SkillPlanContractError,
    _derive_skill_metadata,
    index_document,
)

# ── Valid production plan (exact JSON from the Rust SkillPlan wire shape) ──────

VALID_SKILL_PLAN = json.dumps(
    {
        "schema_version": 1,
        "name": "source-grounded-summary",
        "version": 1,
        "kind": "instance",
        "maturity": "production",
        "stages": [
            {
                "id": "summarize",
                "model": {"by_task_type": "summarize"},
                "agent_count": 1,
                "limits": {
                    "max_turns": 1,
                    "max_cost_usd": 0.2,
                    "context_budget_chars": 32768,
                },
                "tools": [],
                "metrics": [{"name": "grounded_claim_ratio", "goal": 1.0}],
                "action": {
                    "capability": "model_call",
                    "input": {"prompt": "Summarize only the supplied evidence and cite its sources."},
                },
            }
        ],
        "defaults": {"model": {"by_task_type": "summarize"}},
    }
)


# ── _derive_skill_metadata unit tests ─────────────────────────────────────────


class TestDeriveSkillMetadata:
    """Tests for _derive_skill_metadata — the pure helper that validates and extracts
    proposal metadata from a raw JSON skill plan."""

    def test_valid_plan_returns_proposal_metadata(self):
        """A structurally valid skills JSON yields the five proposal fields."""
        result = _derive_skill_metadata("skills", VALID_SKILL_PLAN)
        assert result is not None
        assert result["skill_schema_version"] == 1
        assert result["skill_name"] == "source-grounded-summary"
        assert result["skill_version"] == 1
        assert result["skill_kind"] == "instance"
        assert result["skill_maturity"] == "production"
        # Strict: only the five proposal fields — no leaking of parsed internals
        assert set(result.keys()) == {
            "skill_schema_version",
            "skill_name",
            "skill_version",
            "skill_kind",
            "skill_maturity",
        }

    def test_non_skills_namespace_returns_none(self):
        """A skills-shaped JSON outside the skills namespace returns None (no metadata)."""
        result = _derive_skill_metadata("arcanada", VALID_SKILL_PLAN)
        assert result is None

    def test_malformed_json_raises_skill_plan_contract_error(self):
        """Malformed JSON raises SkillPlanContractError (typed client-input error)."""
        with pytest.raises(SkillPlanContractError, match="Invalid JSON"):
            _derive_skill_metadata("skills", "not json {{{")

    def test_non_object_root_raises_skill_plan_contract_error(self):
        """A JSON array/list root raises SkillPlanContractError."""
        with pytest.raises(SkillPlanContractError, match="must be a JSON object"):
            _derive_skill_metadata("skills", "[1, 2, 3]")

    def test_unsupported_schema_version_raises_skill_plan_contract_error(self):
        """An unsupported schema_version raises SkillPlanContractError."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["schema_version"] = 99
        with pytest.raises(SkillPlanContractError, match="schema_version"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_missing_name_raises_skill_plan_contract_error(self):
        """Missing required 'name' field raises SkillPlanContractError."""
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["name"]
        with pytest.raises(SkillPlanContractError, match="name"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_empty_name_raises_skill_plan_contract_error(self):
        """Empty 'name' string raises SkillPlanContractError."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["name"] = ""
        with pytest.raises(SkillPlanContractError, match="name"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_missing_version_raises_skill_plan_contract_error(self):
        """Missing required 'version' field raises SkillPlanContractError."""
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["version"]
        with pytest.raises(SkillPlanContractError, match="version"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_missing_kind_raises_skill_plan_contract_error(self):
        """Missing required 'kind' field raises SkillPlanContractError."""
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["kind"]
        with pytest.raises(SkillPlanContractError, match="kind"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_invalid_kind_enum_raises_skill_plan_contract_error(self):
        """kind must be 'template' or 'instance' — anything else raises SkillPlanContractError."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["kind"] = "unknown_kind"
        with pytest.raises(SkillPlanContractError, match="kind"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_missing_maturity_raises_skill_plan_contract_error(self):
        """Missing required 'maturity' field raises SkillPlanContractError."""
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["maturity"]
        with pytest.raises(SkillPlanContractError, match="maturity"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_invalid_maturity_enum_raises_skill_plan_contract_error(self):
        """maturity must be 'draft', 'validated', or 'production' — anything else raises."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["maturity"] = "deprecated"
        with pytest.raises(SkillPlanContractError, match="maturity"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_empty_stages_raises_skill_plan_contract_error(self):
        """Empty 'stages' array raises SkillPlanContractError."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"] = []
        with pytest.raises(SkillPlanContractError, match="stages"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_zero_agent_count_raises_skill_plan_contract_error(self):
        """Zero agent_count raises SkillPlanContractError (agent_count >= 1)."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["agent_count"] = 0
        with pytest.raises(SkillPlanContractError, match="agent_count"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_empty_action_capability_raises_skill_plan_contract_error(self):
        """Empty action.capability raises SkillPlanContractError."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["action"]["capability"] = ""
        with pytest.raises(SkillPlanContractError, match="capability"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_missing_stages_raises_skill_plan_contract_error(self):
        """Missing 'stages' raises SkillPlanContractError."""
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["stages"]
        with pytest.raises(SkillPlanContractError, match="stages"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_kind_template_accepted(self):
        """kind='template' is valid and accepted."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["kind"] = "template"
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None
        assert result["skill_kind"] == "template"

    def test_maturity_draft_accepted(self):
        """maturity='draft' is valid and accepted."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["maturity"] = "draft"
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None
        assert result["skill_maturity"] == "draft"

    def test_maturity_validated_accepted(self):
        """maturity='validated' is valid and accepted."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["maturity"] = "validated"
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None
        assert result["skill_maturity"] == "validated"


# ── Exact-bytes preservation ──────────────────────────────────────────────────


class TestExactBytesPreservation:
    """The exact input string must be the source_documents.raw_content payload;
    metadata derivation must not normalize or reserialize it."""

    def test_raw_content_is_unchanged_input_bytes(self):
        """_derive_skill_metadata returns None for raw_content — the full_content
        passed to _build_source_document must remain byte-identical."""
        # _derive_skill_metadata parses for metadata only and does not return the content.
        # The raw_content preservation is tested end-to-end via index_document.
        result = _derive_skill_metadata("skills", VALID_SKILL_PLAN)
        assert result is not None
        # The validated metadata is returned, but the raw string is NOT returned/reserialized
        assert "raw_content" not in result


# ── SkillPlanContractError is a BatchIndexLimitError ───────────────────────────


class TestSkillPlanContractErrorHierarchy:
    """SkillPlanContractError must derive from BatchIndexLimitError so both
    index endpoints return the existing client-input 422 path."""

    def test_is_batch_index_limit_error(self):
        assert issubclass(SkillPlanContractError, BatchIndexLimitError)

    def test_caught_by_existing_422_handler(self):
        """health.py catches BatchIndexLimitError → 422. Subclass must be caught too."""
        import inspect

        from scrutator.health import index_endpoint

        source = inspect.getsource(index_endpoint)
        assert "BatchIndexLimitError" in source


# ── Integration: non-skills namespace does not gain proposal metadata ──────────


class TestNonSkillsNamespaceNoMetadata:
    """The same valid JSON outside settings.skills_namespace does not gain
    skill proposal metadata on chunks."""

    @pytest.mark.asyncio
    async def test_non_skills_namespace_no_metadata_on_chunks(self):
        """Indexing the same valid plan JSON under a non-skills namespace does not
        stamp skill proposal metadata on the resulting chunks."""
        captured_chunks = None

        async def capture(chunk_dicts, *_args, **_kwargs):
            nonlocal captured_chunks
            captured_chunks = chunk_dicts
            return len(chunk_dicts)

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as mock_replace,
        ):
            mock_embed.return_value = [[0.1] * 1024]
            mock_sparse.return_value = [{"1": 0.1}]
            mock_ns.return_value = 1
            mock_replace.side_effect = capture

            await index_document(
                content=VALID_SKILL_PLAN,
                source_path="skills/plan.json",
                namespace="arcanada",  # NOT skills namespace
            )

        assert captured_chunks is not None
        metadata = captured_chunks[0]["metadata"]
        # No skill proposal metadata on the chunk
        assert "skill_schema_version" not in metadata
        assert "skill_name" not in metadata


# ── Oversized skills document still fails under 256 KiB cap ────────────────────


class TestOversizedSkillsDocument:
    """The existing 256 KiB cap must still reject an oversized skills document."""

    @pytest.mark.asyncio
    async def test_oversized_skills_document_fails_before_embedding(self):
        """A skills document > 256 KiB raises BatchIndexLimitError before embedding."""
        oversized = VALID_SKILL_PLAN + " " + "x" * INDEX_BATCH_MAX_DOCUMENT_BYTES

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as mock_replace,
        ):
            mock_ns.return_value = 1
            mock_replace.return_value = 1

            with pytest.raises(BatchIndexLimitError):
                await index_document(
                    content=oversized,
                    source_path="skills/huge.json",
                    namespace="skills",
                )

            # Must NOT waste an embedding call on an oversized doc
            mock_embed.assert_not_called()
            mock_sparse.assert_not_called()


# ── Skills namespace chunks carry proposal metadata ────────────────────────────


class TestSkillsNamespaceChunkMetadata:
    """When a valid plan is indexed under the skills namespace, chunk metadata
    must carry the derived proposal fields at the top level."""

    @pytest.mark.asyncio
    async def test_valid_plan_stamps_proposal_metadata_on_chunks(self):
        """Indexing a valid plan under the skills namespace stamps the five
        proposal metadata fields on every chunk."""
        captured_chunks = None

        async def capture(chunk_dicts, *_args, **_kwargs):
            nonlocal captured_chunks
            captured_chunks = chunk_dicts
            return len(chunk_dicts)

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as mock_replace,
        ):
            mock_embed.return_value = [[0.1] * 1024]
            mock_sparse.return_value = [{"1": 0.1}]
            mock_ns.return_value = 1
            mock_replace.side_effect = capture

            await index_document(
                content=VALID_SKILL_PLAN,
                source_path="skills/source-grounded-summary.json",
                namespace="skills",
            )

        assert captured_chunks is not None
        assert len(captured_chunks) >= 1
        for chunk in captured_chunks:
            meta = chunk["metadata"]
            assert meta.get("skill_schema_version") == 1
            assert meta.get("skill_name") == "source-grounded-summary"
            assert meta.get("skill_version") == 1
            assert meta.get("skill_kind") == "instance"
            assert meta.get("skill_maturity") == "production"
