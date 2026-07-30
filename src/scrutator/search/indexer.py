"""Index pipeline — chunk document, embed chunks, store in database."""

from __future__ import annotations

import hashlib
import json as _json
import logging
import math
from dataclasses import dataclass

from scrutator.chunker.engine import chunk_document
from scrutator.chunker.models import SectionMeta
from scrutator.chunker.splitters import compute_doc_id
from scrutator.config import settings
from scrutator.db.models import (
    INDEX_BATCH_MAX_DOCUMENT_BYTES,
    BatchIndexErrorCode,
    BatchIndexFailed,
    BatchIndexSucceeded,
    IndexRequest,
    IndexResponse,
)
from scrutator.db.repository import (
    replace_source_chunks_atomic,
    upsert_namespace,
    upsert_project,
)
from scrutator.search.embedder import embed_sparse, embed_texts
from scrutator.search.ingest_safety import scan_injection

logger = logging.getLogger(__name__)

INDEX_BATCH_MAX_CHUNKS = 256
INDEX_BATCH_MAX_TOKENS = 131_072
_DENSE_DIMENSIONS = 1024


class BatchIndexLimitError(ValueError):
    """Raised before embedding when a packed batch crosses a resource cap."""


class SkillPlanContractError(BatchIndexLimitError):
    """Raised when a skills-namespace document fails structural validation against the
    Rust ``SkillPlan`` wire shape before embedding/persistence. Derives from
    ``BatchIndexLimitError`` so both index endpoints return the existing 422 path."""


# ── ARAS-0057: skill plan validation and proposal-metadata derivation ──────────

_SKILL_KINDS = frozenset({"template", "instance"})
_SKILL_MATURITIES = frozenset({"draft", "validated", "production"})
# Rust TaskType enum — closed set, serde rejects unknown variants.
_SKILL_TASK_TYPES = frozenset({"code", "summarize", "default"})
# Rust u32 / u64 ranges — values outside these overflow the target serde types.
_U32_MAX = 4_294_967_295
_U64_MAX = 18_446_744_073_709_551_615


def _reject_non_finite_json(value: object) -> object:
    """Reject NaN, Infinity, and -Infinity during JSON parsing — these are valid
    Python ``json`` tokens that ``serde_json`` refuses by default."""
    raise ValueError(f"Non-finite JSON constant is not allowed: {value!r}")


def _is_finite_number(value: object) -> bool:
    """True for a ``serde_json``-compatible finite number (f64 or int that fits
    in f64 without overflow).  Rejects bool, NaN, infinities, and integers that
    would overflow ``f64``."""
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int):
        try:
            f = float(value)
        except OverflowError:
            return False
        return math.isfinite(f)
    return False


def _check_serde_number(value: object, label: str) -> None:
    """Reject numbers that ``serde_json`` cannot represent as finite f64."""
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SkillPlanContractError(f"{label}: non-finite float")
        return
    if isinstance(value, int):
        try:
            f = float(value)
        except OverflowError:
            raise SkillPlanContractError(f"{label}: integer overflows f64") from None
        if not math.isfinite(f):
            raise SkillPlanContractError(f"{label}: integer overflows f64") from None
        return


def _is_non_neg_u32(value: object) -> bool:
    """True when *value* is a strict integer (not bool, not ``-0``) in ``[0, _U32_MAX]``."""
    if isinstance(value, _NegZeroInt):
        return False
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _U32_MAX


def _is_pos_u32(value: object) -> bool:
    """True when *value* is a strict integer (not bool, not ``-0``) in ``[1, _U32_MAX]``."""
    if isinstance(value, _NegZeroInt):
        return False
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= _U32_MAX


def _is_non_neg_u64(value: object) -> bool:
    """True when *value* is a strict integer (not bool, not ``-0``) in ``[0, _U64_MAX]``."""
    if isinstance(value, _NegZeroInt):
        return False
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _U64_MAX


# Rust char::is_whitespace / str::trim Unicode White_Space set:
# U+0009..U+000D, U+0020, U+0085, U+00A0, U+1680,
# U+2000..U+200A, U+2028..U+2029, U+202F, U+205F, U+3000.
# Does NOT trim U+001C..U+001F, U+200B, U+200E/U+200F, or U+FEFF.
_RUST_WS: frozenset[str] = frozenset("\t\n\x0b\x0c\r                  　")

# serde_json default recursion limit.
_SERDE_DEPTH_LIMIT = 128


def _is_non_empty_string(value: object) -> bool:
    """True when *value* is a non-empty string after Rust-compatible whitespace trim."""
    if not isinstance(value, str):
        return False
    start = 0
    end = len(value)
    while start < end and value[start] in _RUST_WS:
        start += 1
    while end > start and value[end - 1] in _RUST_WS:
        end -= 1
    return start < end


class _NegZeroInt(int):
    """Marker for JSON ``-0`` (integer negative zero).  Python ``int('-0')``
    loses the sign and returns plain ``0``; Rust serde_json rejects ``-0`` when
    the target type is unsigned.  Wrapping the parsed value lets u32/u64
    validators reject it while the int behaves normally everywhere else."""

    __slots__ = ()


def _parse_int_strict(value: str) -> int:
    """Reject integer ``-0`` that Python would silently accept as ``0``."""
    if value == "-0":
        return _NegZeroInt(0)
    return int(value)


# Lone surrogates (U+D800-U+DFFF) — valid surrogate pairs are decoded by Python's
# json parser into the correct supplementary character; lone surrogates survive
# as raw codepoints and must be rejected in known String fields and object keys.
_SURROGATE_LO = 0xD800
_SURROGATE_HI = 0xDFFF


def _check_no_lone_surrogates(s: str, label: str) -> None:
    for ch in s:
        cp = ord(ch)
        if _SURROGATE_LO <= cp <= _SURROGATE_HI:
            raise SkillPlanContractError(f"Lone surrogate U+{cp:04X} in {label}")


class PairObject:
    """Lightweight ordered-pair holder returned by ``object_pairs_hook``.
    Preserves insertion order so the normalization pass can detect duplicate
    keys at known struct levels."""

    __slots__ = ("_pairs",)

    def __init__(self, pairs: list[tuple[str, object]]) -> None:
        self._pairs: tuple[tuple[str, object], ...] = tuple(pairs)

    def items(self):
        return self._pairs

    def get(self, key: str, default: object = None) -> object:
        for k, v in self._pairs:
            if k == key:
                return v
        return default


# Context-specific known-key sets for duplicate detection.
_KNOWN_ROOT = frozenset({"schema_version", "name", "version", "kind", "maturity", "stages", "defaults"})
_KNOWN_STAGE = frozenset({"id", "model", "agent_count", "limits", "tools", "metrics", "action"})
_KNOWN_LIMITS = frozenset({"max_turns", "max_cost_usd", "context_budget_chars"})
_KNOWN_METRIC = frozenset({"name", "goal"})
_KNOWN_ACTION = frozenset({"capability", "input"})
_KNOWN_DEFAULTS = frozenset({"model"})
_KNOWN_MODEL_ENUM = frozenset({"literal", "by_task_type"})
_KNOWN_STRING_FIELDS = frozenset(
    {
        ("root", "name"),
        ("stage", "id"),
        ("action", "capability"),
        ("model_enum", "literal"),
        ("metric", "name"),
    }
)
_EMPTY_KEYS: frozenset[str] = frozenset()


def _child_context(parent_ctx: str, key: str) -> tuple[str, frozenset[str]]:
    """Return ``(context_name, known_keys)`` for a child of *parent_ctx*
    accessed by *key*.  ``action.input`` and every ignored unknown-field
    subtree use the ``"value"`` context with an empty known-key set, so
    arbitrary JSON receives Rust-compatible last-value-wins behaviour."""
    if parent_ctx == "root":
        if key == "stages":
            return ("stage_list", _EMPTY_KEYS)
        if key == "defaults":
            return ("defaults", _KNOWN_DEFAULTS)
        return ("value", _EMPTY_KEYS)
    if parent_ctx == "stage_list":
        return ("stage", _KNOWN_STAGE)
    if parent_ctx == "stage":
        if key == "limits":
            return ("limits", _KNOWN_LIMITS)
        if key == "model":
            return ("model_enum", _KNOWN_MODEL_ENUM)
        if key == "metrics":
            return ("metric_list", _EMPTY_KEYS)
        if key == "action":
            return ("action", _KNOWN_ACTION)
        if key == "tools":
            return ("value", _EMPTY_KEYS)
        return ("value", _EMPTY_KEYS)
    if parent_ctx == "metric_list":
        return ("metric", _KNOWN_METRIC)
    if parent_ctx == "defaults":
        if key == "model":
            return ("model_enum", _KNOWN_MODEL_ENUM)
        return ("value", _EMPTY_KEYS)
    if parent_ctx == "action":
        if key == "input":
            return ("action_input_value", _EMPTY_KEYS)  # known path — check surrogates
        return ("value", _EMPTY_KEYS)
    if parent_ctx == "action_input_value":
        return ("action_input_value", _EMPTY_KEYS)  # propagate recursively
    # model_enum, limits, metric, value — children always arbitrary
    return ("value", _EMPTY_KEYS)


def _normalize_contextual(obj: object, context: str, known_keys: frozenset[str], depth: int = 0) -> object:
    """Walk a ``PairObject`` tree with explicit context awareness.
    Rejects duplicate known struct fields, non-finite floats in
    ``action_input_value`` and typed-f64 contexts, lone surrogates in keys
    and known String values, and excessive nesting depth (matching
    serde_json's recursion limit). Converts ``PairObject`` → ``dict``."""

    if depth >= _SERDE_DEPTH_LIMIT:
        raise SkillPlanContractError("Maximum recursion depth exceeded")

    # ── Lists: each item gets the item-level context ─────────────────
    if isinstance(obj, list):
        next_depth = depth + 1
        if context == "stage_list":
            return [_normalize_contextual(v, "stage", _KNOWN_STAGE, next_depth) for v in obj]
        if context == "metric_list":
            return [_normalize_contextual(v, "metric", _KNOWN_METRIC, next_depth) for v in obj]
        if context == "action_input_value":
            result = []
            for v in obj:
                item = _normalize_contextual(v, "action_input_value", _EMPTY_KEYS, next_depth)
                if isinstance(item, str):
                    _check_no_lone_surrogates(item, "action.input list element")
                result.append(item)
            return result
        return [_normalize_contextual(v, "value", _EMPTY_KEYS, next_depth) for v in obj]

    # ── Scalars: number + surrogate check ────────────────────────────
    if not isinstance(obj, PairObject):
        if context == "action_input_value":
            _check_serde_number(obj, "action.input")
            if isinstance(obj, str):
                _check_no_lone_surrogates(obj, "action.input string")
        # Preserve _NegZeroInt through normalization — typed validators
        # reject it; f64/action.input accept it like any other int.
        return obj

    # ── Structs: check duplicate known keys, surrogate-check keys ─────
    next_depth = depth + 1
    seen: set[str] = set()
    for key, _value in obj.items():
        _check_no_lone_surrogates(key, f"{context} object key")
        if key in known_keys:
            if key in seen:
                raise SkillPlanContractError(f"Duplicate known key in {context} struct: {key!r}")
            seen.add(key)

    result: dict[str, object] = {}
    for key, value in obj.items():
        child_ctx, child_keys = _child_context(context, key)
        normalized = _normalize_contextual(value, child_ctx, child_keys, next_depth)
        if isinstance(normalized, str) and (context, key) in _KNOWN_STRING_FIELDS:
            _check_no_lone_surrogates(normalized, f"{context}.{key}")
        if context == "stage" and key == "tools" and isinstance(normalized, list):
            for tool in normalized:
                if isinstance(tool, str):
                    _check_no_lone_surrogates(tool, "stage.tools[]")
        result[key] = normalized
    return result


def _validate_model_spec(label: str, model: object) -> None:
    """Require ``model`` to match the Rust ``ModelSpec`` externally-tagged enum:
    exactly one of ``{"literal": "<str>"}`` or
    ``{"by_task_type": "code"|"summarize"|"default"}``.
    Rust ``String`` allows empty, so an empty ``literal`` is valid."""
    if not isinstance(model, dict) or len(model) != 1:
        raise SkillPlanContractError(f"{label} must be an externally-tagged object with exactly one variant key")
    if "literal" in model:
        if not isinstance(model["literal"], str):
            raise SkillPlanContractError(f"{label}.literal must be a string")
    elif "by_task_type" in model:
        tt = model["by_task_type"]
        if tt not in _SKILL_TASK_TYPES:
            raise SkillPlanContractError(f"{label}.by_task_type must be one of {sorted(_SKILL_TASK_TYPES)}, got {tt!r}")
    else:
        raise SkillPlanContractError(f"{label} must be {{literal: string}} or {{by_task_type: code|summarize|default}}")


def _derive_skill_metadata(namespace: str, full_content: str) -> dict[str, object] | None:
    """Validate a skills-namespace document against the Rust ``SkillPlan`` wire
    shape and return only the five proposal-metadata fields, or ``None`` for a
    non-skills namespace.

    Raises ``SkillPlanContractError`` (a ``BatchIndexLimitError`` subclass, so
    the endpoints return 422) on any structural violation. The raw
    ``full_content`` bytes are never reserialized or returned from this helper.
    """
    if namespace != settings.skills_namespace:
        return None

    # 256 KiB exact-source cap — reject BEFORE JSON parsing, embedding, chunking,
    # or namespace persistence (the single-index path has no per-document cap upstream,
    # and the batch path's cap is a different check layer).
    try:
        content_bytes = full_content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SkillPlanContractError(f"Invalid Unicode in skills document: {exc}") from exc
    if len(content_bytes) > INDEX_BATCH_MAX_DOCUMENT_BYTES:
        raise SkillPlanContractError(f"skills document exceeds {INDEX_BATCH_MAX_DOCUMENT_BYTES}-byte exact-source cap")

    # Parse JSON, rejecting Python-only non-finite constants (NaN, Inf, -Inf)
    # that serde_json refuses. The raw string is never reserialized.
    try:
        plan_raw = _json.loads(
            full_content,
            parse_constant=_reject_non_finite_json,
            parse_int=_parse_int_strict,
            object_pairs_hook=PairObject,
        )
    except RecursionError as exc:
        raise SkillPlanContractError("JSON nesting depth exceeds parser limit") from exc
    except UnicodeEncodeError as exc:
        raise SkillPlanContractError(f"Invalid Unicode in skills document: {exc}") from exc
    except (ValueError, TypeError) as exc:
        cause = str(exc)
        raise SkillPlanContractError(f"Invalid JSON in skills document: {cause}") from exc

    if not isinstance(plan_raw, PairObject):
        raise SkillPlanContractError("Skills document root must be a JSON object")

    # Normalize PairObject → dict with context-aware duplicate detection,
    # lone-surrogate rejection, and NegZeroInt resolution.
    # action.input uses "action_input_value" context; unknown fields use plain
    # "value" context with last-value-wins.  No global recursive number scan —
    # typed fields are range-checked below, and arbitrary JSON accepts whatever
    # serde_json::Value accepts (finite f64 from huge integer lexemes is fine).
    plan = _normalize_contextual(plan_raw, "root", _KNOWN_ROOT)

    # schema_version: must be exactly integer 1 (bool True == 1 must be rejected)
    schema_version = plan.get("schema_version")
    if schema_version is True or schema_version is False or not isinstance(schema_version, int):
        raise SkillPlanContractError(
            f"Skill schema_version must be an integer (1), got {type(schema_version).__name__}"
        )
    if schema_version != 1:
        raise SkillPlanContractError(f"Unsupported skill schema_version: {schema_version!r} (expected 1)")

    # name: string (Rust String allows empty)
    name = plan.get("name")
    if not isinstance(name, str):
        raise SkillPlanContractError("Skill plan 'name' must be a string")

    # version: u32 (strict integer, not bool)
    version = plan.get("version")
    if not _is_non_neg_u32(version):
        raise SkillPlanContractError("Skill plan 'version' must be a u32 integer (0–4294967295)")

    # kind: template | instance
    kind = plan.get("kind")
    if kind not in _SKILL_KINDS:
        raise SkillPlanContractError(f"Skill plan 'kind' must be one of {sorted(_SKILL_KINDS)}, got {kind!r}")

    # maturity: draft | validated | production
    maturity = plan.get("maturity")
    if maturity not in _SKILL_MATURITIES:
        raise SkillPlanContractError(
            f"Skill plan 'maturity' must be one of {sorted(_SKILL_MATURITIES)}, got {maturity!r}"
        )

    # defaults: required object with model.by_task_type
    defaults = plan.get("defaults")
    if not isinstance(defaults, dict):
        raise SkillPlanContractError("Skill plan 'defaults' must be an object")
    _validate_model_spec("defaults.model", defaults.get("model"))

    # stages: non-empty array
    stages = plan.get("stages")
    if not isinstance(stages, list) or len(stages) == 0:
        raise SkillPlanContractError("Skill plan 'stages' must be a non-empty array")

    # Validate each stage's structural constraints
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise SkillPlanContractError(f"Skill plan stage[{i}] must be an object")

        # stage.id: string (Rust String allows empty)
        stage_id = stage.get("id")
        if not isinstance(stage_id, str):
            raise SkillPlanContractError(f"Skill plan stage[{i}].id must be a string")

        # stage.model: externally-tagged ModelSpec {literal: str} | {by_task_type: task}
        _validate_model_spec(f"stage[{i}].model", stage.get("model"))

        # agent_count: u32 >= 1
        agent_count = stage.get("agent_count")
        if not _is_pos_u32(agent_count):
            raise SkillPlanContractError(
                f"Skill plan stage[{i}].agent_count must be a u32 integer >= 1, got {agent_count!r}"
            )

        # limits: required object with all three required fields
        limits = stage.get("limits")
        if not isinstance(limits, dict):
            raise SkillPlanContractError(f"Skill plan stage[{i}].limits must be an object")

        max_turns = limits.get("max_turns")
        if not _is_non_neg_u32(max_turns):
            raise SkillPlanContractError(
                f"Skill plan stage[{i}].limits.max_turns must be a u32 integer >= 0, got {max_turns!r}"
            )

        max_cost = limits.get("max_cost_usd")
        if not _is_finite_number(max_cost):
            raise SkillPlanContractError(
                f"Skill plan stage[{i}].limits.max_cost_usd must be a finite number, got {max_cost!r}"
            )

        context_budget = limits.get("context_budget_chars")
        if not _is_non_neg_u64(context_budget):
            raise SkillPlanContractError(
                f"Skill plan stage[{i}].limits.context_budget_chars must be a u64 integer >= 0, got {context_budget!r}"
            )

        # tools: Vec<String> — Rust String allows empty
        tools = stage.get("tools")
        if not isinstance(tools, list):
            raise SkillPlanContractError(f"Skill plan stage[{i}].tools must be an array")
        for j, tool in enumerate(tools):
            if not isinstance(tool, str):
                raise SkillPlanContractError(f"Skill plan stage[{i}].tools[{j}] must be a string, got {tool!r}")

        # metrics: list of objects, each with name (non-empty string) and goal (finite number)
        metrics = stage.get("metrics")
        if not isinstance(metrics, list):
            raise SkillPlanContractError(f"Skill plan stage[{i}].metrics must be an array")
        for j, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                raise SkillPlanContractError(f"Skill plan stage[{i}].metrics[{j}] must be an object")
            metric_name = metric.get("name")
            if not isinstance(metric_name, str):
                raise SkillPlanContractError(f"Skill plan stage[{i}].metrics[{j}].name must be a string")
            goal = metric.get("goal")
            if not _is_finite_number(goal):
                raise SkillPlanContractError(
                    f"Skill plan stage[{i}].metrics[{j}].goal must be a finite number, got {goal!r}"
                )

        # action: must be an object
        action = stage.get("action")
        if not isinstance(action, dict):
            raise SkillPlanContractError(f"Skill plan stage[{i}].action must be an object")

        # action.capability: non-empty, non-whitespace string (Rust validate rejects
        # empty/whitespace via trim().is_empty() — the only String field with this constraint)
        capability = action.get("capability")
        if not _is_non_empty_string(capability):
            raise SkillPlanContractError(
                f"Skill plan stage[{i}].action.capability must be a non-empty, non-whitespace string"
            )

        # action.input: required but any JSON value (including null) is valid
        if "input" not in action:
            raise SkillPlanContractError(f"Skill plan stage[{i}].action.input is required")

    # Return only the five proposal fields — no leaking of parsed internals.
    # Keys are consumer-compatible (ARAS skill_hit_from reads name/version/maturity directly).
    return {
        "schema_version": schema_version,
        "name": name,
        "version": version,
        "kind": kind,
        "maturity": maturity,
    }


class _BatchEmbeddingError(Exception):
    def __init__(self, code: BatchIndexErrorCode):
        self.code = code


@dataclass(frozen=True)
class _PreparedDocument:
    position: int
    document: IndexRequest
    chunks: list[dict]
    offset: int
    source_document: dict | None = None
    evidence_document: dict | None = None

    @property
    def end(self) -> int:
        return self.offset + len(self.chunks)


def compute_doc_content_hash(full_content: str) -> str:
    """Whole-document content hash bound at ingest (SRCH-0038 D3 / S1).

    Bound ONCE over the full pre-chunk source content and stored in each chunk's
    `metadata.section.doc_content_hash`. The fetch path only READS this value — it is never
    recomputed over the assembled response, so integrity verification is not theater.
    """
    return "sha256:" + hashlib.sha256(full_content.encode()).hexdigest()


def _stamp_doc_id(
    section: SectionMeta | None,
    namespace: str,
    source_path: str,
    doc_content_hash: str,
) -> dict | None:
    """Finalize a chunk's section dict with its namespace-scoped doc_id and the whole-document
    content hash (SRCH-0038 D3). Both are indexer-only context — the chunker has neither the
    namespace nor the full pre-chunk content.

    ``doc_content_hash`` is ~71 bytes, safely under the ``jsonb_ops`` GIN entry ceiling, and is
    what ``SearchHit.content_hash`` / the fetch path READ. The exact pre-chunk bytes are NOT
    stamped here (that was the SRCH-0038 1a seam, which raised the multi-KB blob into
    ``idx_chunks_metadata`` GIN and hit the ~2704-byte ``jsonb_ops`` entry ceiling on real
    skills). Under 1b they live in the isolated, un-indexed ``source_documents`` table instead
    (see ``_build_source_document`` / ``replace_source_chunks_atomic``)."""
    if section is None:
        return None
    return {
        **section.model_dump(),
        "doc_id": compute_doc_id(namespace, source_path),
        "doc_content_hash": doc_content_hash,
    }


def _build_source_document(namespace: str, source_path: str, full_content: str) -> dict | None:
    """SRCH-0038 1b: build the isolated exact-source row for the skills namespace ONLY, so a skill
    fetch is byte-exact against ``content_hash``. Returns ``None`` for every other namespace (the
    evidence corpus keeps chunk-reassembly, ``content_exact=False``).

    By-construction byte-exactness: ``raw_content`` is the SAME ``full_content`` string that
    ``content_hash`` is computed over — ``compute_doc_content_hash(full_content)`` — with no
    re-encode or normalization between them, so ``sha256(raw_content) == content_hash`` holds.
    The blob lands in ``source_documents`` (NOT ``chunks.metadata``), so it never enters the
    ``idx_chunks_metadata`` GIN index and the ~2704-byte ``jsonb_ops`` entry ceiling never applies.

    Size guard: the single ``POST /v1/index`` path has no per-document byte cap, so bound the
    exact-source blob at the same 256 KB cap the batch endpoint enforces."""
    if namespace != settings.skills_namespace:
        return None
    if len(full_content.encode("utf-8")) > INDEX_BATCH_MAX_DOCUMENT_BYTES:
        raise BatchIndexLimitError(f"skills document exceeds {INDEX_BATCH_MAX_DOCUMENT_BYTES}-byte exact-source cap")
    return {
        "doc_id": compute_doc_id(namespace, source_path),
        "source_path": source_path,
        "content_hash": compute_doc_content_hash(full_content),
        "raw_content": full_content,
    }


def _build_evidence_document(namespace: str, source_path: str, full_content: str) -> dict | None:
    """SRCH-0039 (Mechanism C): build the isolated exact-source row for the EVIDENCE corpus, so an
    evidence fetch can be byte-exact against ``content_hash`` once the flag is flipped.

    Gated by BOTH triggers (the ratified skills-vs-evidence divergence):
    - ``settings.evidence_exact_bytes`` — default-off, so ingest behaviour is byte-identical until
      the operator flips the flag in prod (no row written while OFF).
    - ``namespace != settings.skills_namespace`` — skills keep their OWN ``source_documents`` path
      (always-exact, 256 KB-capped, fail-closed); the evidence builder never claims a skills doc.

    Returns ``None`` for skills or when the flag is OFF. By-construction byte-exactness:
    ``raw_content`` is the SAME ``full_content`` that ``content_hash`` is computed over, with no
    re-encode between them, so ``sha256(raw_content) == content_hash`` holds. The blob lands in the
    un-indexed ``evidence_documents`` table (never ``chunks.metadata``), so no GIN entry-size
    ceiling applies. Unlike ``_build_source_document``, there is deliberately NO 256 KB per-document
    cap — the evidence corpus holds large documents."""
    if not settings.evidence_exact_bytes:
        return None
    if namespace == settings.skills_namespace:
        return None
    return {
        "doc_id": compute_doc_id(namespace, source_path),
        "source_path": source_path,
        "content_hash": compute_doc_content_hash(full_content),
        "raw_content": full_content,
    }


def _chunk_dicts(
    chunk_result,
    namespace: str,
    source_path: str,
    full_content: str,
    skill_metadata: dict[str, object] | None = None,
) -> list[dict]:
    doc_content_hash = compute_doc_content_hash(full_content)
    # SRCH-0038 1b: keep only the ~71-byte `doc_content_hash` in `metadata.section` (safely under
    # the jsonb_ops GIN entry ceiling); the exact bytes go to `source_documents`, never the GIN
    # -indexed `chunks.metadata`. The skills-blob size guard now lives in `_build_source_document`,
    # invoked here so an oversized skills doc is rejected before persistence even if the caller
    # ignores the source-document payload.
    _build_source_document(namespace, source_path, full_content)
    # ARAS-0055: label (never block) each document with an ingest-time injection signal, scanned
    # ONCE over the whole pre-chunk content (fast regex/set-based — no LLM in the hot path). The
    # signal is small (flag + int score + ≤4 short category names), JSONB-safe, and READ back on
    # the fetch/search path. Ingestion proceeds regardless — this is an observability layer.
    injection = scan_injection(full_content)
    # ARAS-0057: JSON skill plans have no markdown headings, so the chunker emits
    # section=None. Stamp a minimal provenance dict so every skills chunk carries
    # nonempty doc_id and doc_content_hash — required by the search projection and
    # fetch-by-doc_id resolution.
    _fallback_section: dict[str, str] | None = None
    if skill_metadata is not None:
        _fallback_section = {
            "doc_id": compute_doc_id(namespace, source_path),
            "doc_content_hash": doc_content_hash,
        }

    return [
        {
            "id": chunk.id,
            "source_path": source_path,
            "source_type": chunk.metadata.source_type,
            "chunk_index": chunk.chunk_index,
            "parent_id": chunk.parent_id,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "token_count": chunk.token_count,
            "metadata": {
                "heading_hierarchy": chunk.metadata.heading_hierarchy,
                "frontmatter": chunk.metadata.frontmatter,
                "wikilinks": chunk.metadata.wikilinks,
                "tags": chunk.metadata.tags,
                "language": chunk.metadata.language,
                "section": _stamp_doc_id(chunk.metadata.section, namespace, source_path, doc_content_hash)
                or _fallback_section,
                "injection": injection,
                **(skill_metadata or {}),
            },
        }
        for chunk in chunk_result.chunks
    ]


async def index_documents(documents: list[IndexRequest]) -> list[BatchIndexSucceeded | BatchIndexFailed]:
    """Chunk and embed a bounded document pack before storing each source."""
    prepared, texts, results = _prepare_documents(documents)
    _enforce_pack_caps(prepared, texts)
    if not prepared:
        return _complete_results(results)

    positions = [(item.position, item.document.source_path) for item in prepared]
    try:
        embeddings, sparse_weights = await _embed_batch(texts)
    except _BatchEmbeddingError as exc:
        _set_failures(results, positions, exc.code)
        return _complete_results(results)

    try:
        namespace_id = await upsert_namespace(documents[0].namespace)
    except Exception:
        logger.error("Batch namespace persistence failed")
        _set_failures(results, positions, "persistence_failed")
        return _complete_results(results)

    for item in prepared:
        results[item.position] = await _persist_prepared(item, embeddings, sparse_weights, namespace_id)
    return _complete_results(results)


def _prepare_documents(
    documents: list[IndexRequest],
) -> tuple[list[_PreparedDocument], list[str], list[BatchIndexSucceeded | BatchIndexFailed | None]]:
    prepared: list[_PreparedDocument] = []
    texts: list[str] = []
    results: list[BatchIndexSucceeded | BatchIndexFailed | None] = [None] * len(documents)
    for position, document in enumerate(documents):
        try:
            # ARAS-0057: validate skill plan BEFORE chunking/embedding,
            # so a malformed plan is rejected without wasted compute.
            # SkillPlanContractError propagates to the endpoint handler → 422 (typed client-input error).
            skill_metadata = _derive_skill_metadata(document.namespace, document.content)
            chunk_result = chunk_document(
                content=document.content,
                source_path=document.source_path,
                source_type=document.source_type,
                max_tokens=document.max_tokens,
                overlap_tokens=document.overlap_tokens,
            )
        except SkillPlanContractError:
            raise
        except Exception:
            logger.error("Batch chunking failed for one source")
            results[position] = BatchIndexFailed(source_path=document.source_path, error_code="chunking_failed")
            continue
        chunk_dicts = _chunk_dicts(
            chunk_result, document.namespace, document.source_path, document.content, skill_metadata
        )
        source_document = _build_source_document(document.namespace, document.source_path, document.content)
        evidence_document = _build_evidence_document(document.namespace, document.source_path, document.content)
        prepared.append(
            _PreparedDocument(position, document, chunk_dicts, len(texts), source_document, evidence_document)
        )
        texts.extend(chunk["content"] for chunk in chunk_dicts)
    return prepared, texts, results


def _enforce_pack_caps(prepared: list[_PreparedDocument], texts: list[str]) -> None:
    if len(texts) > INDEX_BATCH_MAX_CHUNKS:
        raise BatchIndexLimitError("batch chunk limit exceeded")
    if sum(chunk["token_count"] for item in prepared for chunk in item.chunks) > INDEX_BATCH_MAX_TOKENS:
        raise BatchIndexLimitError("batch token limit exceeded")


async def _embed_batch(texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
    try:
        embeddings = await embed_texts(texts)
    except Exception as exc:
        _log_embedding_failure("Dense", exc)
        raise _BatchEmbeddingError("dense_embedding_failed") from None
    if not _valid_dense_embeddings(embeddings, len(texts)):
        raise _BatchEmbeddingError("invalid_dense_embeddings")

    try:
        sparse_weights = await embed_sparse(texts)
    except Exception as exc:
        _log_embedding_failure("Sparse", exc)
        raise _BatchEmbeddingError("sparse_embedding_failed") from None
    if not _valid_sparse_embeddings(sparse_weights, len(texts)):
        raise _BatchEmbeddingError("invalid_sparse_embeddings")
    return embeddings, sparse_weights


def _log_embedding_failure(stage: str, exception: Exception) -> None:
    status_code = getattr(exception, "status_code", None)
    if not isinstance(status_code, int):
        status_code = "none"
    logger.error(
        "%s embedding failed for batch: error_type=%s status_code=%s",
        stage,
        type(exception).__name__,
        status_code,
    )


async def _persist_prepared(
    item: _PreparedDocument,
    embeddings: list[list[float]],
    sparse_weights: list[dict[str, float]],
    namespace_id: int,
) -> BatchIndexSucceeded | BatchIndexFailed:
    try:
        project_id = await upsert_project(namespace_id, item.document.project) if item.document.project else None
        inserted = await replace_source_chunks_atomic(
            item.chunks,
            embeddings[item.offset : item.end],
            sparse_weights[item.offset : item.end],
            namespace_id,
            project_id,
            source_document=item.source_document,
            evidence_document=item.evidence_document,
        )
        return BatchIndexSucceeded(source_path=item.document.source_path, chunks_indexed=inserted)
    except Exception:
        logger.error("Batch persistence failed for one source")
        return BatchIndexFailed(source_path=item.document.source_path, error_code="persistence_failed")


def _set_failures(
    results: list[BatchIndexSucceeded | BatchIndexFailed | None],
    positions: list[tuple[int, str]],
    code: BatchIndexErrorCode,
) -> None:
    for position, path in positions:
        results[position] = BatchIndexFailed(source_path=path, error_code=code)


def _complete_results(
    results: list[BatchIndexSucceeded | BatchIndexFailed | None],
) -> list[BatchIndexSucceeded | BatchIndexFailed]:
    if any(result is None for result in results):
        raise RuntimeError("batch result mapping incomplete")
    return [result for result in results if result is not None]


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _valid_dense_embeddings(embeddings: object, expected_count: int) -> bool:
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        return False
    return all(
        isinstance(vector, list) and len(vector) == _DENSE_DIMENSIONS and all(_finite_number(value) for value in vector)
        for vector in embeddings
    )


def _valid_sparse_embeddings(embeddings: object, expected_count: int) -> bool:
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        return False
    return all(
        isinstance(vector, dict)
        and all(isinstance(token, str) and _finite_number(weight) for token, weight in vector.items())
        for vector in embeddings
    )


async def _embed_single_document(texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
    embeddings = await embed_texts(texts)
    if not _valid_dense_embeddings(embeddings, len(texts)):
        raise ValueError("invalid dense embeddings")
    try:
        sparse_weights = await embed_sparse(texts)
        if not _valid_sparse_embeddings(sparse_weights, len(texts)):
            raise ValueError("invalid sparse embeddings")
    except Exception:
        # The legacy endpoint has always treated sparse indexing as optional.
        # Persist explicit empty weights so dense replacement remains atomic.
        logger.warning("Sparse indexing unavailable for single-source request")
        sparse_weights = [{} for _ in texts]
    return embeddings, sparse_weights


async def index_document(
    content: str,
    source_path: str,
    namespace: str = "arcanada",
    project: str | None = None,
    source_type: str | None = None,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> IndexResponse:
    """Full index pipeline: chunk → embed → store."""
    # ARAS-0057: validate and derive skill proposal metadata BEFORE embedding,
    # so a malformed plan is rejected without wasting compute.
    skill_metadata = _derive_skill_metadata(namespace, content)

    # 1. Chunk the document
    chunk_result = chunk_document(
        content=content,
        source_path=source_path,
        source_type=source_type,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    if not chunk_result.chunks:
        return IndexResponse(chunks_indexed=0, source_path=source_path, namespace=namespace, strategy_used="empty")

    # 2. Embed all chunks
    texts = [c.content for c in chunk_result.chunks]
    embeddings, sparse_weights = await _embed_single_document(texts)

    # 3. Ensure namespace (and project) exist
    namespace_id = await upsert_namespace(namespace)
    project_id = await upsert_project(namespace_id, project) if project else None

    # 4. Replace dense and sparse rows as one source generation.
    chunk_dicts = _chunk_dicts(chunk_result, namespace, source_path, content, skill_metadata)
    source_document = _build_source_document(namespace, source_path, content)
    evidence_document = _build_evidence_document(namespace, source_path, content)
    inserted = await replace_source_chunks_atomic(
        chunk_dicts,
        embeddings,
        sparse_weights,
        namespace_id,
        project_id,
        source_document=source_document,
        evidence_document=evidence_document,
    )

    return IndexResponse(
        chunks_indexed=inserted,
        source_path=source_path,
        namespace=namespace,
        strategy_used=chunk_result.strategy_used,
    )
