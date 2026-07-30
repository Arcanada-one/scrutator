"""Fail-closed tests for the repository-tracked skill promotion registry."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from scrutator.search.skill_promotions import (
    SKILL_PROMOTIONS,
    SkillPromotionRegistryError,
    approved_skill_maturity,
    load_skill_promotions,
)

_VALID_HASH = "sha256:" + ("a" * 64)


def _registry(entries: list[dict[str, object]]) -> str:
    return json.dumps({"schema_version": 1, "promotions": entries}, ensure_ascii=False)


def _entry(
    *,
    source_path: str = "skills/reviewed.json",
    content_hash: str = _VALID_HASH,
    maturity: str = "production",
) -> dict[str, object]:
    return {
        "source_path": source_path,
        "content_hash": content_hash,
        "maturity": maturity,
    }


def test_valid_registry_is_immutable_and_exact_pair_lookup_only() -> None:
    registry = load_skill_promotions(
        _registry(
            [
                _entry(),
                _entry(
                    source_path="skills/validated.json",
                    content_hash="sha256:" + ("b" * 64),
                    maturity="validated",
                ),
            ]
        )
    )

    assert isinstance(registry, MappingProxyType)
    assert approved_skill_maturity(registry, "skills/reviewed.json", _VALID_HASH) == "production"
    assert (
        approved_skill_maturity(
            registry,
            "skills/validated.json",
            "sha256:" + ("b" * 64),
        )
        == "validated"
    )
    assert approved_skill_maturity(registry, "skills/missing.json", _VALID_HASH) == "draft"
    assert approved_skill_maturity(registry, "skills/reviewed.json", "sha256:" + ("c" * 64)) == "draft"
    with pytest.raises(TypeError):
        registry[("skills/new.json", _VALID_HASH)] = "production"  # type: ignore[index]


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "[]",
        json.dumps({"schema_version": 2, "promotions": []}),
        json.dumps({"schema_version": 1, "promotions": [], "extra": True}),
        '{"schema_version":1,"schema_version":1,"promotions":[]}',
        json.dumps({"schema_version": 1, "promotions": "not-a-list"}),
        json.dumps({"schema_version": 1, "promotions": {}}),
        _registry([{"source_path": "skills/a.json", "content_hash": _VALID_HASH}]),
        _registry([_entry() | {"extra": True}]),
        _registry([_entry(content_hash="a" * 64)]),
        _registry([_entry(content_hash="sha256:" + ("A" * 64))]),
        _registry([_entry(maturity="draft")]),
    ],
)
def test_malformed_or_open_schema_registry_fails_closed(raw: str) -> None:
    with pytest.raises(SkillPromotionRegistryError, match=r"^invalid skill promotion registry: [a-z_]+$"):
        load_skill_promotions(raw)


@pytest.mark.parametrize(
    "source_path",
    [
        "",
        ".",
        "..",
        "/skills/a.json",
        "skills/../a.json",
        "skills/./a.json",
        "skills\\a.json",
        "skills/\x00a.json",
        "skills//a.json",
        "skills/a.json/",
        "skills/e\u0301.json",
    ],
)
def test_unsafe_or_noncanonical_paths_fail_closed(source_path: str) -> None:
    with pytest.raises(SkillPromotionRegistryError) as exc_info:
        load_skill_promotions(_registry([_entry(source_path=source_path)]))

    assert str(exc_info.value) == "invalid skill promotion registry: unsafe_path"
    if source_path:
        assert source_path not in str(exc_info.value)


def test_nfc_unicode_path_is_accepted_deterministically() -> None:
    source_path = "skills/é.json"
    registry = load_skill_promotions(_registry([_entry(source_path=source_path)]))

    assert approved_skill_maturity(registry, source_path, _VALID_HASH) == "production"


@pytest.mark.parametrize(
    "entries",
    [
        [_entry(), _entry()],
        [_entry(), _entry(maturity="validated")],
    ],
)
def test_duplicate_or_conflicting_identity_fails_loader_startup(
    entries: list[dict[str, object]],
) -> None:
    with pytest.raises(SkillPromotionRegistryError) as exc_info:
        load_skill_promotions(_registry(entries))

    assert str(exc_info.value) == "invalid skill promotion registry: duplicate_identity"


def test_errors_do_not_echo_attacker_controlled_keys_paths_or_content() -> None:
    secret_key = "attacker-controlled-secret-key"
    secret_path = "attacker-controlled-secret-path"
    raw = json.dumps(
        {
            "schema_version": 1,
            "promotions": [
                {
                    "source_path": secret_path,
                    "content_hash": _VALID_HASH,
                    "maturity": "production",
                    secret_key: "attacker-controlled-secret-content",
                }
            ],
        }
    )

    with pytest.raises(SkillPromotionRegistryError) as exc_info:
        load_skill_promotions(raw)

    rendered = str(exc_info.value)
    assert rendered == "invalid skill promotion registry: entry_schema"
    assert secret_key not in rendered
    assert secret_path not in rendered
    assert "attacker-controlled-secret-content" not in rendered


def test_tracked_registry_authorizes_exact_aras_probe_artifact() -> None:
    registry_path = Path(__file__).parents[1] / "src" / "scrutator" / "search" / "skill_promotions.json"
    tracked = load_skill_promotions(registry_path.read_text(encoding="utf-8"))

    assert tracked == SKILL_PROMOTIONS
    assert (
        approved_skill_maturity(
            tracked,
            "skills/skills-kb-discovery-probe.json",
            "sha256:a568ac9631c86ff90fc4bf5b893f70a7113cbf851e0991c59acb16d087165633",
        )
        == "production"
    )
