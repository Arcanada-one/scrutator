"""Tests for ARAS-0057 Task 1: Scrutator ingest contract — skill plan validation,
metadata derivation, provenance stamping, and HTTP error propagation.
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
        assert result["schema_version"] == 1
        assert result["name"] == "source-grounded-summary"
        assert result["version"] == 1
        assert result["kind"] == "instance"
        assert result["maturity"] == "production"
        # Strict: only the five proposal fields — no leaking of parsed internals
        assert set(result.keys()) == {"schema_version", "name", "version", "kind", "maturity"}

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

    def test_empty_name_accepted(self):
        """Empty 'name' string is valid — Rust String allows empty."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["name"] = ""
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None
        assert result["name"] == ""

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

    def test_empty_action_capability_rejected(self):
        """Rust validate rejects empty/whitespace capability via trim().is_empty()."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["action"]["capability"] = ""
        with pytest.raises(SkillPlanContractError, match="capability"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_whitespace_action_capability_rejected(self):
        """Whitespace-only capability is rejected (trim().is_empty())."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["action"]["capability"] = "   "
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
        assert result["kind"] == "template"

    def test_maturity_draft_accepted(self):
        """maturity='draft' is valid and accepted."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["maturity"] = "draft"
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None
        assert result["maturity"] == "draft"

    def test_maturity_validated_accepted(self):
        """maturity='validated' is valid and accepted."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["maturity"] = "validated"
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None
        assert result["maturity"] == "validated"


# ── Hardened parity: reject documents Rust serde_json / SkillPlan cannot accept ──


class TestHardenedParityRejections:
    """These documents parse as valid Python JSON but MUST be rejected because
    Rust serde_json + SkillPlan deserialization would fail on them."""

    # ── Non-finite JSON constants (serde_json rejects NaN/Inf) ──────────

    def test_nan_in_max_cost_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["limits"]["max_cost_usd"] = float("nan")
        raw = json.dumps(plan, allow_nan=True)
        with pytest.raises(SkillPlanContractError, match="Non-finite"):
            _derive_skill_metadata("skills", raw)

    def test_infinity_in_max_cost_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["limits"]["max_cost_usd"] = float("inf")
        raw = json.dumps(plan, allow_nan=True)
        with pytest.raises(SkillPlanContractError, match="Non-finite"):
            _derive_skill_metadata("skills", raw)

    def test_neg_infinity_in_max_cost_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["limits"]["max_cost_usd"] = float("-inf")
        raw = json.dumps(plan, allow_nan=True)
        with pytest.raises(SkillPlanContractError, match="Non-finite"):
            _derive_skill_metadata("skills", raw)

    def test_nan_in_metric_goal_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["metrics"][0]["goal"] = float("nan")
        raw = json.dumps(plan, allow_nan=True)
        with pytest.raises(SkillPlanContractError, match="Non-finite"):
            _derive_skill_metadata("skills", raw)

    def test_literal_nan_token_rejected(self):
        raw = VALID_SKILL_PLAN.replace("0.2", "NaN")
        with pytest.raises(SkillPlanContractError):
            _derive_skill_metadata("skills", raw)

    def test_literal_infinity_token_rejected(self):
        raw = VALID_SKILL_PLAN.replace("0.2", "Infinity")
        with pytest.raises(SkillPlanContractError):
            _derive_skill_metadata("skills", raw)

    # ── schema_version as bool (True == 1 in Python) ────────────────────

    def test_schema_version_true_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["schema_version"] = True
        with pytest.raises(SkillPlanContractError, match="schema_version"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_schema_version_false_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["schema_version"] = False
        with pytest.raises(SkillPlanContractError, match="schema_version"):
            _derive_skill_metadata("skills", json.dumps(plan))

    # ── version / agent_count as bool ───────────────────────────────────

    def test_version_true_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["version"] = True
        with pytest.raises(SkillPlanContractError, match="version"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_agent_count_true_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["agent_count"] = True
        with pytest.raises(SkillPlanContractError, match="agent_count"):
            _derive_skill_metadata("skills", json.dumps(plan))

    # ── u32 / u64 overflow ──────────────────────────────────────────────

    def test_version_u32_overflow_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["version"] = 4_294_967_296  # u32 max + 1
        with pytest.raises(SkillPlanContractError, match="version"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_agent_count_u32_overflow_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["agent_count"] = 4_294_967_296
        with pytest.raises(SkillPlanContractError, match="agent_count"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_max_turns_u32_overflow_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["limits"]["max_turns"] = 4_294_967_296
        with pytest.raises(SkillPlanContractError, match="max_turns"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_context_budget_u64_overflow_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["limits"]["context_budget_chars"] = 18_446_744_073_709_551_616  # u64 max + 1
        with pytest.raises(SkillPlanContractError, match="context_budget_chars"):
            _derive_skill_metadata("skills", json.dumps(plan))

    # ── Missing / wrong defaults ────────────────────────────────────────

    def test_missing_defaults_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["defaults"]
        with pytest.raises(SkillPlanContractError, match="defaults"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_defaults_non_object_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["defaults"] = "not an object"
        with pytest.raises(SkillPlanContractError, match="defaults"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_defaults_model_empty_object_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["defaults"] = {"model": {}}
        with pytest.raises(SkillPlanContractError, match="variant key"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_defaults_model_empty_by_task_type_value_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["defaults"] = {"model": {"by_task_type": ""}}
        with pytest.raises(SkillPlanContractError, match="by_task_type"):
            _derive_skill_metadata("skills", json.dumps(plan))

    # ── Missing / wrong stage id ────────────────────────────────────────

    def test_missing_stage_id_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["stages"][0]["id"]
        with pytest.raises(SkillPlanContractError, match="stage\\[0\\].id"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_empty_stage_id_accepted(self):
        """Rust String allows empty stage id."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["id"] = ""
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    # ── Missing / wrong stage model ─────────────────────────────────────

    def test_missing_stage_model_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["stages"][0]["model"]
        with pytest.raises(SkillPlanContractError, match="stage\\[0\\].model"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_stage_model_non_object_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["model"] = "summarize"
        with pytest.raises(SkillPlanContractError, match="stage\\[0\\].model"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_stage_model_empty_object_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["model"] = {}
        with pytest.raises(SkillPlanContractError, match="variant key"):
            _derive_skill_metadata("skills", json.dumps(plan))

    # ── limits as non-object ────────────────────────────────────────────

    def test_limits_array_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["limits"] = [1, 2, 3]
        with pytest.raises(SkillPlanContractError, match="limits"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_limits_string_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["limits"] = "unlimited"
        with pytest.raises(SkillPlanContractError, match="limits"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_limits_null_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["limits"] = None
        with pytest.raises(SkillPlanContractError, match="limits"):
            _derive_skill_metadata("skills", json.dumps(plan))

    # ── Missing required limit fields ───────────────────────────────────

    def test_missing_max_turns_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["stages"][0]["limits"]["max_turns"]
        with pytest.raises(SkillPlanContractError, match="max_turns"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_missing_max_cost_usd_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["stages"][0]["limits"]["max_cost_usd"]
        with pytest.raises(SkillPlanContractError, match="max_cost_usd"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_missing_context_budget_chars_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["stages"][0]["limits"]["context_budget_chars"]
        with pytest.raises(SkillPlanContractError, match="context_budget_chars"):
            _derive_skill_metadata("skills", json.dumps(plan))

    # ── Missing tools / metrics arrays ──────────────────────────────────

    def test_missing_tools_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["stages"][0]["tools"]
        with pytest.raises(SkillPlanContractError, match="tools"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_tools_non_array_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["tools"] = "none"
        with pytest.raises(SkillPlanContractError, match="tools"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_tool_object_rejected(self):
        """Rust Vec<String> rejects object tools — each tool must be a plain string."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["tools"] = [{"name": "web_search"}]
        with pytest.raises(SkillPlanContractError, match="tools\\[0\\]"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_tool_empty_string_accepted(self):
        """Rust Vec<String> allows empty tool strings."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["tools"] = [""]
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    def test_string_tools_accepted(self):
        """Vec<String> — a list of non-empty strings is valid."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["tools"] = ["web_search", "file_read"]
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    def test_missing_metrics_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["stages"][0]["metrics"]
        with pytest.raises(SkillPlanContractError, match="metrics"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_metrics_non_array_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["metrics"] = 42
        with pytest.raises(SkillPlanContractError, match="metrics"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_metric_item_non_object_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["metrics"] = ["not_an_object"]
        with pytest.raises(SkillPlanContractError, match="metrics\\[0\\]"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_metric_missing_name_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["metrics"] = [{"goal": 0.5}]
        with pytest.raises(SkillPlanContractError, match="metrics\\[0\\].name"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_metric_missing_goal_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["metrics"] = [{"name": "metric"}]
        with pytest.raises(SkillPlanContractError, match="metrics\\[0\\].goal"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_metric_non_finite_goal_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["metrics"][0]["goal"] = float("nan")
        raw = json.dumps(plan, allow_nan=True)
        with pytest.raises(SkillPlanContractError, match="Non-finite"):
            _derive_skill_metadata("skills", raw)

    # ── ModelSpec externally-tagged enum ────────────────────────────────

    def test_model_literal_variant_accepted(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["model"] = {"literal": "gpt-5"}
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    def test_model_by_task_type_code_accepted(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["model"] = {"by_task_type": "code"}
        plan["defaults"]["model"] = {"by_task_type": "code"}
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    def test_model_by_task_type_default_accepted(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["model"] = {"by_task_type": "default"}
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    def test_model_by_task_type_unknown_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["model"] = {"by_task_type": "unknown_task"}
        with pytest.raises(SkillPlanContractError, match="by_task_type"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_model_two_variant_keys_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["model"] = {"literal": "x", "by_task_type": "summarize"}
        with pytest.raises(SkillPlanContractError, match="variant key"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_model_unknown_variant_key_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["model"] = {"unknown_key": "value"}
        with pytest.raises(SkillPlanContractError, match="literal"):
            _derive_skill_metadata("skills", json.dumps(plan))

    @pytest.mark.parametrize("location", ["stage", "defaults"])
    def test_model_extra_variant_key_rejected(self, location):
        plan = json.loads(VALID_SKILL_PLAN)
        target = plan["stages"][0] if location == "stage" else plan["defaults"]
        target["model"] = {"literal": "model", "unknown": 1}
        with pytest.raises(SkillPlanContractError, match="variant key"):
            _derive_skill_metadata("skills", json.dumps(plan))

    # ── action.input required ───────────────────────────────────────────

    def test_missing_action_input_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        del plan["stages"][0]["action"]["input"]
        with pytest.raises(SkillPlanContractError, match="action.input"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_action_input_null_accepted(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["action"]["input"] = None
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    # ── Duplicate key rejection ─────────────────────────────────────────

    def test_duplicate_top_level_key_rejected(self):
        """Duplicate keys in valid JSON syntax must be rejected (Python
        last-value-wins differs from serde)."""
        # Build valid JSON with duplicate "name" key at top level
        raw = '{"schema_version":1,"name":"first","name":"second",' + VALID_SKILL_PLAN[1:]
        with pytest.raises(SkillPlanContractError, match="Duplicate known key"):
            _derive_skill_metadata("skills", raw)

    def test_duplicate_nested_key_rejected(self):
        """Duplicate keys inside a nested object (stage limits) must be rejected."""
        raw = VALID_SKILL_PLAN.replace(
            '"max_turns": 1',
            '"max_turns": 1, "max_turns": 2',
        )
        with pytest.raises(SkillPlanContractError, match="Duplicate known key"):
            _derive_skill_metadata("skills", raw)

    # ── action.input collision probes: struct keys in Value context ────
    # Duplicate keys cannot be expressed as Python dict literals (last-value-
    # wins at construction), so these tests construct the raw JSON string with
    # targeted replacements that inject duplicate keys inside action.input.

    def _make_input_with_duplicate(self, key, val1, val2):
        """Return VALID_SKILL_PLAN with action.input containing a duplicate *key*."""
        plan = json.loads(VALID_SKILL_PLAN)
        sentinel = {"__SENTINEL__": True}
        plan["stages"][0]["action"]["input"] = sentinel
        raw = json.dumps(plan)
        dup_fragment = f"{json.dumps(key)}: {json.dumps(val1)}, {json.dumps(key)}: {json.dumps(val2)}"
        return raw.replace(json.dumps(sentinel), "{" + dup_fragment + "}")

    def test_action_input_duplicate_name_accepted(self):
        raw = self._make_input_with_duplicate("name", "a", "b")
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    def test_action_input_duplicate_id_accepted(self):
        raw = self._make_input_with_duplicate("id", "a", "b")
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    def test_action_input_contains_stages_key_accepted(self):
        plan = json.loads(VALID_SKILL_PLAN)
        sentinel = {"__S__": True}
        plan["stages"][0]["action"]["input"] = sentinel
        raw = json.dumps(plan)
        raw = raw.replace(
            json.dumps(sentinel),
            '{"stages":[],"name":"a","name":"b"}',
        )
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    def test_action_input_contains_agent_count_accepted(self):
        plan = json.loads(VALID_SKILL_PLAN)
        sentinel = {"__S__": True}
        plan["stages"][0]["action"]["input"] = sentinel
        raw = json.dumps(plan)
        raw = raw.replace(
            json.dumps(sentinel),
            '{"agent_count":1,"id":"a","id":"b"}',
        )
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    def test_action_input_contains_capability_accepted(self):
        plan = json.loads(VALID_SKILL_PLAN)
        sentinel = {"__S__": True}
        plan["stages"][0]["action"]["input"] = sentinel
        raw = json.dumps(plan)
        raw = raw.replace(
            json.dumps(sentinel),
            '{"capability":"x","input":"y","capability":"z"}',
        )
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    def test_action_input_with_goal_key_last_value_wins(self):
        plan = json.loads(VALID_SKILL_PLAN)
        sentinel = {"__S__": True}
        plan["stages"][0]["action"]["input"] = sentinel
        raw = json.dumps(plan)
        raw = raw.replace(
            json.dumps(sentinel),
            '{"goal":1.0,"goal":2.0}',
        )
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    # ── Stage-level duplicate field detection ──────────────────────────

    def test_duplicate_stage_id_rejected(self):
        """Duplicate id inside a stage object must be rejected."""
        plan = json.loads(VALID_SKILL_PLAN)
        raw = json.dumps(plan)
        raw = raw.replace('"id": "summarize"', '"id": "summarize", "id": "dup"')
        with pytest.raises(SkillPlanContractError, match="Duplicate known key"):
            _derive_skill_metadata("skills", raw)

    def test_duplicate_metric_name_in_list_accepted(self):
        """Duplicate metric names across different metrics — not duplicate keys."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["metrics"] = [
            {"name": "m", "goal": 1.0},
            {"name": "m", "goal": 2.0},
        ]
        raw = json.dumps(plan)
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    # ── Number boundary probes ─────────────────────────────────────────

    def test_huge_integers_in_action_input_accepted_as_f64(self):
        """serde_json::Value converts oversized integer lexemes to finite f64."""
        for n in (-9_223_372_036_854_775_809, 18_446_744_073_709_551_616):
            plan = json.loads(VALID_SKILL_PLAN)
            plan["stages"][0]["action"]["input"] = {"n": n}
            raw = json.dumps(plan)
            result = _derive_skill_metadata("skills", raw)
            assert result is not None

    def test_negative_cost_accepted(self):
        """Rust f64 allows negative max_cost_usd."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["limits"]["max_cost_usd"] = -5.0
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    # ── Negative zero (-0) for integer fields ──────────────────────────

    def test_neg_zero_version_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        raw = json.dumps(plan).replace('"version": 1', '"version": -0')
        with pytest.raises(SkillPlanContractError, match="version"):
            _derive_skill_metadata("skills", raw)

    def test_neg_zero_max_turns_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        raw = json.dumps(plan).replace('"max_turns": 1', '"max_turns": -0')
        with pytest.raises(SkillPlanContractError, match="max_turns"):
            _derive_skill_metadata("skills", raw)

    def test_neg_zero_in_action_input_accepted(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["action"]["input"] = {"n": 0}
        raw = json.dumps(plan).replace('"n": 0', '"n": -0')
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    # ── Lone surrogate rejection ───────────────────────────────────────

    def test_lone_surrogate_in_name_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["name"] = "a\ud800b"
        with pytest.raises(SkillPlanContractError, match="surrogate"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_lone_surrogate_in_action_input_string_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["action"]["input"] = {"k": "a\ud800b"}
        with pytest.raises(SkillPlanContractError, match="surrogate"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_lone_surrogate_in_object_key_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["action"]["input"] = {"a\ud800b": "value"}
        with pytest.raises(SkillPlanContractError, match="surrogate"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_lone_surrogate_in_unknown_field_accepted(self):
        """Unknown struct fields are ignored by Rust — surrogates allowed."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["__unknown_field__"] = "a\ud800b"
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    # ── Rust trim: U+001C not stripped ─────────────────────────────────

    def test_control_char_u001c_in_capability_accepted(self):
        """Rust trim() does not remove U+001C; Python strip() does."""
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["action"]["capability"] = ""
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    def test_unknown_field_key_lone_surrogate_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["a\ud800b"] = "value"
        with pytest.raises(SkillPlanContractError, match="surrogate"):
            _derive_skill_metadata("skills", json.dumps(plan))

    def test_nested_key_inside_ignored_unknown_value_accepts_lone_surrogate(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["__unknown__"] = {"a\ud800b": "value"}
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    def test_deep_unknown_field_is_ignored_without_recursion_limit(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["__unknown_deep__"] = "__DEEP__"
        nested = '{"x":' * 1100 + '"leaf"' + "}" * 1100
        raw = json.dumps(plan).replace('"__DEEP__"', nested)
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    def test_5000_digit_integer_in_unknown_field_is_ignored(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["__unknown_huge__"] = "__HUGE__"
        raw = json.dumps(plan).replace('"__HUGE__"', "1" + ("0" * 4999))
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    def test_malformed_json_inside_unknown_field_is_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["__unknown_bad__"] = "__BAD__"
        raw = json.dumps(plan).replace('"__BAD__"', '{"x":[1,]}')
        with pytest.raises(SkillPlanContractError, match="Invalid JSON"):
            _derive_skill_metadata("skills", raw)

    def test_valid_surrogate_pair_accepted(self):
        plan = json.loads(VALID_SKILL_PLAN)
        plan["name"] = "a\U00010000b"
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    @pytest.mark.parametrize(
        "setter",
        [
            lambda plan, value: plan["defaults"].update({"model": {"literal": value}}),
            lambda plan, value: plan["stages"][0].update({"model": {"literal": value}}),
            lambda plan, value: plan["stages"][0].update({"tools": [value]}),
            lambda plan, value: plan["stages"][0]["metrics"][0].update({"name": value}),
        ],
    )
    def test_lone_surrogate_in_every_known_string_context_rejected(self, setter):
        plan = json.loads(VALID_SKILL_PLAN)
        setter(plan, "a\ud800b")
        with pytest.raises(SkillPlanContractError, match="surrogate"):
            _derive_skill_metadata("skills", json.dumps(plan))

    # ── Recursion depth boundary (serde_json limit 128; wrapper ~5) ────

    def test_depth_123_accepted(self):
        value = "leaf"
        for _ in range(123):
            value = {"x": value}
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["action"]["input"] = value
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None

    def test_depth_124_rejected(self):
        value = "leaf"
        for _ in range(124):
            value = {"x": value}
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"][0]["action"]["input"] = value
        raw = json.dumps(plan)
        with pytest.raises(SkillPlanContractError, match="depth"):
            _derive_skill_metadata("skills", raw)

    # ── 1e400 differential ──────────────────────────────────────────

    def test_1e400_in_action_input_value_rejected(self):
        plan = json.loads(VALID_SKILL_PLAN)
        sentinel = {"__S__": True}
        plan["stages"][0]["action"]["input"] = sentinel
        raw = json.dumps(plan).replace(json.dumps(sentinel), '{"v":1e400}')
        with pytest.raises(SkillPlanContractError, match="non-finite"):
            _derive_skill_metadata("skills", raw)

    def test_1e400_in_unknown_field_value_accepted(self):
        plan = json.loads(VALID_SKILL_PLAN)
        sentinel = {"__S__": True}
        plan["__u__"] = sentinel
        raw = json.dumps(plan).replace(json.dumps(sentinel), "1e400")
        result = _derive_skill_metadata("skills", raw)
        assert result is not None

    @pytest.mark.parametrize("context", ["action.input", "typed f64"])
    def test_400_digit_integer_rejected_in_deserialized_number_contexts(self, context):
        huge = "1" + ("0" * 400)
        plan = json.loads(VALID_SKILL_PLAN)
        if context == "action.input":
            marker = {"__HUGE__": True}
            plan["stages"][0]["action"]["input"] = marker
            raw = json.dumps(plan).replace(json.dumps(marker), f'{{"n":{huge}}}')
        else:
            plan["stages"][0]["limits"]["max_cost_usd"] = "__HUGE__"
            raw = json.dumps(plan).replace('"__HUGE__"', huge)
        with pytest.raises(SkillPlanContractError, match="overflows f64|finite"):
            _derive_skill_metadata("skills", raw)

    def test_400_digit_integer_in_ignored_unknown_field_accepted(self):
        huge = "1" + ("0" * 400)
        plan = json.loads(VALID_SKILL_PLAN)
        plan["__unknown_huge__"] = "__HUGE__"
        raw = json.dumps(plan).replace('"__HUGE__"', huge)
        assert _derive_skill_metadata("skills", raw) is not None

    # ── Table-driven empty-string parity ───────────────────────────────

    @pytest.mark.parametrize(
        "field_path, setter",
        [
            ("name", lambda p, v: p.update({"name": v})),
            ("stage.id", lambda p, v: p["stages"][0].update({"id": v})),
            ("stage.model.literal", lambda p, v: p["stages"][0].update({"model": {"literal": v}})),
            ("stage.tools[0]", lambda p, v: p["stages"][0].update({"tools": [v]})),
            ("stage.metrics[0].name", lambda p, v: p["stages"][0]["metrics"][0].update({"name": v})),
        ],
        ids=lambda f: f,
    )
    def test_empty_string_accepted(self, field_path, setter):
        plan = json.loads(VALID_SKILL_PLAN)
        setter(plan, "")
        result = _derive_skill_metadata("skills", json.dumps(plan))
        assert result is not None, f"empty string rejected at {field_path}"


# ── Exact-bytes preservation ──────────────────────────────────────────────────


class TestExactBytesPreservation:
    """The exact input string must be the source_documents.raw_content payload;
    metadata derivation must not normalize or reserialize it."""

    def test_raw_content_is_unchanged_input_bytes(self):
        """_derive_skill_metadata returns proposal metadata only — the raw
        content is NOT returned or reserialized from this helper."""
        result = _derive_skill_metadata("skills", VALID_SKILL_PLAN)
        assert result is not None
        assert "raw_content" not in result

    @pytest.mark.asyncio
    async def test_source_document_raw_content_equals_exact_input(self):
        """The source_document payload passed to replace_source_chunks_atomic
        carries raw_content that is byte-identical to the input string."""
        captured_source_doc = None

        async def capture(chunk_dicts, *_args, source_document=None, **_kwargs):
            nonlocal captured_source_doc
            captured_source_doc = source_document
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
                namespace="skills",
            )

        assert captured_source_doc is not None, "source_document must be passed to persistence"
        assert captured_source_doc["raw_content"] == VALID_SKILL_PLAN
        assert (
            captured_source_doc["content_hash"]
            == "sha256:" + __import__("hashlib").sha256(VALID_SKILL_PLAN.encode()).hexdigest()
        )


# ── Caller metadata boundary ──────────────────────────────────────────────────


class TestCallerMetadataBoundary:
    """IndexRequest has no caller-controlled metadata field. Proposal metadata
    is derived server-side from the raw JSON only; no field in the document body
    can override the five server-derived values."""

    def test_index_request_has_no_metadata_field(self):
        from scrutator.db.models import IndexRequest

        fields = IndexRequest.model_fields
        assert "metadata" not in fields, "IndexRequest must not expose a caller-controlled metadata field"

    def test_conflicting_inline_fields_cannot_override_server_metadata(self):
        """Even if the raw JSON contains fields that collide with the derived
        metadata keys, the server-derived values take precedence because they
        are unpacked AFTER the chunk's own metadata dict."""
        plan = json.loads(VALID_SKILL_PLAN)
        # Inject conflicting fields into the raw document
        plan["name"] = "overridden-name"
        _derive_skill_metadata("skills", json.dumps(plan))
        # The validated name is whatever the document says — but the document
        # body CANNOT introduce extra proposal fields like "trust_class"
        plan["trust_class"] = "skill"
        result2 = _derive_skill_metadata("skills", json.dumps(plan))
        # Only the five proposal keys ever appear
        assert "trust_class" not in result2
        assert set(result2.keys()) == {"schema_version", "name", "version", "kind", "maturity"}


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


# ── HTTP endpoint: invalid skill returns typed 422 ────────────────────────────


class TestSkillIndexHttpEndpointErrors:
    """Both single and batch index endpoints must return 422 for invalid skills
    (NOT 503 persistence failures)."""

    @pytest.fixture(autouse=True)
    def _override_auth(self):
        from scrutator.auth.capabilities import NamespaceCapability, require_feeder_capability
        from scrutator.health import app

        app.dependency_overrides[require_feeder_capability] = lambda: NamespaceCapability(
            namespaces=frozenset({"skills", "arcanada"})
        )
        yield
        app.dependency_overrides.pop(require_feeder_capability, None)

    def test_single_index_invalid_skill_returns_422(self):
        from fastapi.testclient import TestClient

        from scrutator.health import app

        client = TestClient(app)
        resp = client.post(
            "/v1/index",
            json={
                "content": "not valid json",
                "source_path": "skills/bad.json",
                "namespace": "skills",
            },
        )
        assert resp.status_code == 422

    def test_batch_index_invalid_skill_returns_422(self):
        from fastapi.testclient import TestClient

        from scrutator.health import app

        client = TestClient(app)
        resp = client.post(
            "/v1/index/batch",
            json={
                "documents": [
                    {
                        "content": "not valid json",
                        "source_path": "skills/bad.json",
                        "namespace": "skills",
                    },
                ],
            },
        )
        assert resp.status_code == 422, (
            f"batch index with invalid skill must return 422, got {resp.status_code}: {resp.text[:200]}"
        )

    def test_batch_literal_lone_surrogate_returns_typed_422(self):
        from fastapi.testclient import TestClient

        from scrutator.health import app

        raw_body = (
            b'{"documents":[{"content":"\\ud800","source_path":"skills/literal-surrogate.json","namespace":"skills"}]}'
        )
        response = TestClient(app, raise_server_exceptions=False).post(
            "/v1/index/batch",
            content=raw_body,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422, response.text[:200]

    def test_batch_oversized_literal_surrogate_cannot_bypass_general_cap(self):
        from fastapi.testclient import TestClient

        from scrutator.health import app

        raw_body = (
            b'{"documents":[{"content":"'
            + (b"x" * (INDEX_BATCH_MAX_DOCUMENT_BYTES + 1))
            + b'\\ud800","source_path":"oversized-surrogate.txt","namespace":"arcanada"}]}'
        )
        response = TestClient(app, raise_server_exceptions=False).post(
            "/v1/index/batch",
            content=raw_body,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422, response.text[:200]

    @pytest.mark.parametrize("route", ["/v1/index", "/v1/index/batch"])
    @pytest.mark.parametrize("invalid_case", ["depth", "surrogate"])
    def test_hardened_skill_contract_failures_return_typed_422(self, route, invalid_case):
        from fastapi.testclient import TestClient

        from scrutator.health import app

        plan = json.loads(VALID_SKILL_PLAN)
        if invalid_case == "depth":
            value: object = "leaf"
            for _ in range(124):
                value = {"x": value}
            plan["stages"][0]["action"]["input"] = value
        else:
            plan["name"] = "a\ud800b"
        document = {
            "content": json.dumps(plan),
            "source_path": f"skills/{invalid_case}.json",
            "namespace": "skills",
        }
        payload = {"documents": [document]} if route.endswith("/batch") else document
        response = TestClient(app).post(route, json=payload)
        assert response.status_code == 422, response.text[:200]


# ── HTTP endpoint: oversized skill returns 422 before embedding ───────────────


class TestOversizedSkillsDocument:
    """The 256 KiB cap must reject an oversized structurally valid skill plan
    before embedding, chunking, or namespace persistence."""

    @pytest.fixture(autouse=True)
    def _override_auth(self):
        from scrutator.auth.capabilities import NamespaceCapability, require_feeder_capability
        from scrutator.health import app

        app.dependency_overrides[require_feeder_capability] = lambda: NamespaceCapability(
            namespaces=frozenset({"skills"})
        )
        yield
        app.dependency_overrides.pop(require_feeder_capability, None)

    def test_oversized_valid_json_returns_422_before_embedding(self):
        """Construct a structurally valid skill plan that exceeds 256 KiB
        (NOT by appending junk — by repeating an array element until the byte
        count crosses the cap), and assert the single-index endpoint returns 422
        without calling the embedder."""
        from fastapi.testclient import TestClient

        from scrutator.health import app

        # Build a valid plan with a single stage repeated until oversized
        single_stage = json.loads(VALID_SKILL_PLAN)["stages"][0]
        plan = json.loads(VALID_SKILL_PLAN)
        plan["stages"] = [single_stage]
        # Add copies until we cross the 256 KiB cap
        while len(json.dumps(plan).encode("utf-8")) < INDEX_BATCH_MAX_DOCUMENT_BYTES + 1024:
            plan["stages"].append(single_stage.copy())
        oversized = json.dumps(plan)
        assert len(oversized.encode("utf-8")) > INDEX_BATCH_MAX_DOCUMENT_BYTES

        with patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as mock_embed:
            client = TestClient(app)
            resp = client.post(
                "/v1/index",
                json={
                    "content": oversized,
                    "source_path": "skills/huge.json",
                    "namespace": "skills",
                },
            )
            assert resp.status_code == 422
            # Must NOT waste an embedding call on an oversized doc
            mock_embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_oversized_by_appended_junk_still_rejected(self):
        """A skills doc > 256 KiB also fails when bloated with appended junk
        (the cap rejects before JSON parsing)."""
        oversized = VALID_SKILL_PLAN + " " + "x" * INDEX_BATCH_MAX_DOCUMENT_BYTES

        with (
            patch("scrutator.search.indexer.embed_texts", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.indexer.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.indexer.upsert_namespace", new_callable=AsyncMock) as mock_ns,
            patch("scrutator.search.indexer.replace_source_chunks_atomic", new_callable=AsyncMock) as mock_replace,
        ):
            mock_ns.return_value = 1
            mock_replace.return_value = 1

            with pytest.raises(SkillPlanContractError):
                await index_document(
                    content=oversized,
                    source_path="skills/huge.json",
                    namespace="skills",
                )

            mock_embed.assert_not_called()
            mock_sparse.assert_not_called()


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
        assert "schema_version" not in metadata
        assert "maturity" not in metadata


# ── Skills namespace chunks carry proposal metadata + provenance ──────────────


class TestSkillsNamespaceChunkMetadata:
    """When a valid plan is indexed under the skills namespace, chunk metadata
    must carry the derived proposal fields and nonempty source provenance."""

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
            assert meta.get("schema_version") == 1
            assert meta.get("name") == "source-grounded-summary"
            assert meta.get("version") == 1
            assert meta.get("kind") == "instance"
            assert meta.get("maturity") == "production"

    @pytest.mark.asyncio
    async def test_skills_chunk_has_nonempty_source_provenance(self):
        """JSON skill plans have no markdown headings (chunker emits section=None),
        yet every chunk must carry nonempty doc_id and doc_content_hash in
        metadata.section so the search projection and fetch-by-doc_id work."""
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
                namespace="skills",
            )

        assert captured_chunks is not None
        assert len(captured_chunks) >= 1
        for chunk in captured_chunks:
            section = chunk["metadata"].get("section")
            assert section is not None, "skills chunk must have non-null section with doc_id and doc_content_hash"
            doc_id = section.get("doc_id")
            assert isinstance(doc_id, str) and len(doc_id) == 16, f"doc_id must be a 16-char hex string, got {doc_id!r}"
            doc_content_hash = section.get("doc_content_hash")
            assert isinstance(doc_content_hash, str) and doc_content_hash.startswith("sha256:"), (
                f"doc_content_hash must start with sha256:, got {doc_content_hash!r}"
            )
            # Verify the hash matches the content
            expected_hash = "sha256:" + __import__("hashlib").sha256(VALID_SKILL_PLAN.encode()).hexdigest()
            assert doc_content_hash == expected_hash
