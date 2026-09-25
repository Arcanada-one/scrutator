"""The committed HTTP contract must equal the application, and the ECOSYSTEM's verifier must read it.

Two different failures, two different tests.

1. Drift. `contracts/http/scrutator-http.contract.ts` is generated from `app.openapi()`. If a
   pydantic model or a route changes and the contract is not regenerated, the admission gate would
   report "no breaking change" about a document that no longer describes the service. That is worse
   than no contract at all, so the equality is asserted here, byte for byte.

2. Measurability. A contract only helps if the tool that has to diff it can parse it. The parser is
   not mocked and not reimplemented: these tests import the VENDORED `contract_diff` out of
   `.github/graph-admission/` — the same bytes CI runs — and assert on what it actually extracts.
   A fixture written by the same hand as the generator would happily agree with a generator that
   emits zod the real extractor cannot read (A2-287).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "http" / "scrutator-http.contract.ts"
VENDORED = ROOT / ".github" / "graph-admission" / "tools" / "graph"


def _load(name: str):
    """Import a vendored gate tool without putting it permanently on sys.path."""
    if str(VENDORED) not in sys.path:
        sys.path.insert(0, str(VENDORED))
    spec = importlib.util.find_spec(name)
    if spec is None:  # pragma: no cover - the bundle is committed
        pytest.skip(f"vendored {name} is absent")
    return importlib.import_module(name)


def _generator():
    spec = importlib.util.spec_from_file_location("gen_http_contract", ROOT / "tools" / "gen_http_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extracted():
    contract_diff = _load("contract_diff")
    build_graph = _load("build_graph")
    rel = CONTRACT.relative_to(ROOT).as_posix()
    tree = build_graph.Tree(
        {rel: CONTRACT.read_bytes()},
        {
            "source_repo": "scrutator",
            "source_commit": "0" * 40,
            "source_tree": "test",
            "dirty": False,
            "subdir": None,
            "rev": "test",
        },
    )
    return rel, contract_diff.Extractor(tree).contracts()


def test_committed_contract_equals_the_application():
    generated = _generator().build()
    assert CONTRACT.read_text(encoding="utf-8") == generated, (
        "contracts/http/scrutator-http.contract.ts is stale — run `python3 tools/gen_http_contract.py`"
    )


def test_check_mode_exits_zero_on_the_committed_tree():
    result = subprocess.run(
        [sys.executable, "tools/gen_http_contract.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_check_mode_goes_red_when_the_contract_drifts(tmp_path):
    """Prove the check can fail before trusting its green."""
    drifted = tmp_path / "drifted.ts"
    drifted.write_text(CONTRACT.read_text(encoding="utf-8").replace("z.string()", "z.number()", 1), encoding="utf-8")
    assert _generator().main(["--check", "--out", str(drifted)]) == 1


def test_the_real_verifier_extracts_every_component_schema():
    rel, contracts = _extracted()
    kinds = {c["kind"] for c in contracts.values()}
    assert kinds <= {"zod", "type_alias_union", "class_validator_dto", "cross_repo_enum"}
    assert "zod" in kinds and "type_alias_union" in kinds
    # every exported const is a contract the verifier can diff, not an opaque `any`
    opaque = sorted(c["symbol"] for c in contracts.values() if c["schema"].get("kind") in ("any", None))
    assert opaque == [], f"the verifier could not parse: {opaque}"
    assert len(contracts) >= 40, f"only {len(contracts)} contracts extracted from {rel}"


def test_the_route_surface_is_a_diffable_enum():
    rel, contracts = _extracted()
    routes = contracts[f"contract:{rel}#ScrutatorRouteRequest"]
    assert routes["schema"]["kind"] == "enum"
    assert "POST /v1/edges" in routes["schema"]["values"]
    # input direction: adding a route is compatible, removing one is ENUM_VALUE_REMOVED (breaking)
    assert routes["direction"] == "input"


def test_a_removed_response_field_is_a_breaking_change(tmp_path):
    """The verifier's own taxonomy, run over a mutated copy of the committed contract."""
    contract_diff = _load("contract_diff")
    build_graph = _load("build_graph")
    rel = CONTRACT.relative_to(ROOT).as_posix()
    text = CONTRACT.read_text(encoding="utf-8")
    marker = "export const SearchResponseSchema = z.object({\n"
    assert marker in text
    start = text.index(marker) + len(marker)
    mutated = text[:start] + text[text.index("\n", start) + 1 :]  # drop the first field

    def tree(body: str):
        meta = {
            "source_repo": "scrutator",
            "source_commit": "0" * 40,
            "source_tree": "test",
            "dirty": False,
            "subdir": None,
            "rev": "test",
        }
        return build_graph.Tree({rel: body.encode()}, meta)

    result = contract_diff.run_diff(tree(text), tree(mutated), repo_name="scrutator")
    codes = {c["code"] for c in result["breaking"]}
    assert "FIELD_REMOVED" in codes, codes
