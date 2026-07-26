import re
from pathlib import Path

import yaml

from scrutator.health import app

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = re.compile(
    r"\b(?:CONN|AUTH|ARCA|VERD|TRANS|SUP|MUN|LTM|SRCH|DISK|OVER|CONS|VOICE|"
    r"BILL|CONV|ARGA|EMAIL|INFRA|TUNE|DATA|ROB|WEB|DEV|DEVOPS|CONTENT|"
    r"RESEARCH|AGENT|BENCH|MAINT|FIN|QA|SEC)-\d{4}\b"
)
ACTION_REF = re.compile(r"^\s*uses:\s+([^@\s]+)@([^\s#]+)", re.MULTILINE)


def public_text_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "CLAUDE.md", ROOT / "pyproject.toml"]
    files.extend(sorted((ROOT / "documentation").rglob("*.md")))
    files.extend(sorted((ROOT / ".github" / "workflows").glob("*.yml")))
    files.extend(
        [
            ROOT / "benchmark" / "scrutator" / "README.md",
            ROOT / "benchmark" / "scrutator" / "CONSUMERS.md",
            ROOT / "benchmark" / "scrutator" / "harness.py",
        ]
    )
    files.extend(sorted((ROOT / "benchmark" / "scrutator" / "tests").glob("*.py")))
    return files


def test_public_docs_follow_diataxis_layout():
    expected = {
        "tutorials": {"README.md"},
        "how-to": {"README.md", "ltm-reflect.md"},
        "reference": {"README.md", "api.md", "navigation.md"},
        "explanation": {"README.md", "architecture.md"},
    }

    for category, required_files in expected.items():
        category_dir = ROOT / "documentation" / category
        assert category_dir.is_dir(), category
        assert required_files.issubset({path.name for path in category_dir.iterdir()}), category

    assert not (ROOT / "docs").exists()


def test_public_surface_has_no_internal_task_ids_or_retired_domains():
    findings: list[str] = []
    for path in public_text_files():
        text = path.read_text(encoding="utf-8")
        for match in TASK_ID.finditer(text):
            findings.append(f"{path.relative_to(ROOT)}: {match.group(0)}")
        if "arcanada.one" in text:
            findings.append(f"{path.relative_to(ROOT)}: retired arcanada.one domain")

    assert findings == []


def test_public_api_reference_covers_openapi_routes():
    api_reference = (ROOT / "documentation" / "reference" / "api.md").read_text(encoding="utf-8")
    openapi = app.openapi()

    undocumented: list[str] = []
    for route, operations in openapi["paths"].items():
        for method in operations:
            marker = f"`{method.upper()} {route}`"
            if marker not in api_reference:
                undocumented.append(marker)

    assert undocumented == []


def test_workflow_dependencies_are_immutable_and_current():
    findings: list[str] = []
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        for action, ref in ACTION_REF.findall(text):
            if action.startswith("./"):
                continue
            if not re.fullmatch(r"[0-9a-f]{40}", ref):
                findings.append(f"{workflow.name}: {action}@{ref}")
        if re.search(r"^\s*if:\s*false\s*$", text, re.MULTILINE):
            findings.append(f"{workflow.name}: disabled if:false job")

    benchmark = (ROOT / ".github" / "workflows" / "benchmark-scrutator.yml").read_text(encoding="utf-8")
    assert "ci-general" not in benchmark
    assert "arcana-ai" not in benchmark
    assert findings == []


def test_workflow_shell_blocks_do_not_expand_expressions_directly():
    findings: list[str] = []

    def inspect(value: object, workflow: Path) -> None:
        if isinstance(value, dict):
            run = value.get("run")
            if isinstance(run, str) and "${{" in run:
                findings.append(f"{workflow.name}: expression in run block")
            for child in value.values():
                inspect(child, workflow)
        elif isinstance(value, list):
            for child in value:
                inspect(child, workflow)

    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        inspect(yaml.safe_load(workflow.read_text(encoding="utf-8")), workflow)

    assert findings == []


def test_search_benchmark_mints_and_removes_a_reader_token():
    workflow = (ROOT / ".github" / "workflows" / "benchmark-scrutator.yml").read_text(encoding="utf-8")

    assert "KB_OBSERVER_CLIENT_SECRET: ${{ secrets.KB_OBSERVER_CLIENT_SECRET }}" in workflow
    assert "SCRUTATOR_BEARER_TOKEN_FILE: ${{ runner.temp }}/scrutator-search-reader.jwt" in workflow
    assert "--bearer-token-file" not in workflow
    assert "- name: Remove benchmark reader token\n        if: always()" in workflow
    assert 'pip install -e ".[dev]"' in workflow


def test_live_search_benchmark_is_main_only_and_uses_protected_environment():
    workflow_path = ROOT / ".github" / "workflows" / "benchmark-scrutator.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    benchmark = yaml.safe_load(workflow)["jobs"]["benchmark"]

    assert benchmark["if"] == "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'"
    assert benchmark["runs-on"] == {
        "group": "scrutator-prod",
        "labels": ["self-hosted", "linux", "arcana-db", "docker"],
    }
    assert benchmark["environment"] == "kb-production"
    assert "MC_API_KEY" not in workflow


def test_reader_token_files_are_created_exclusively_with_private_mode():
    for filename in ("benchmark-scrutator.yml", "recall-regression.yml"):
        workflow = (ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8")

        assert "os.open(" in workflow, filename
        assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL" in workflow, filename
        assert "0o600" in workflow, filename
        assert "os.fdopen(" in workflow, filename
        assert ".write_text(token)" not in workflow, filename
        assert ".chmod(0o600)" not in workflow, filename


def test_normal_ci_lints_and_tests_the_active_search_benchmark():
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    steps = {
        step["name"]: step for step in yaml.safe_load(workflow)["jobs"]["lint-and-test"]["steps"] if "name" in step
    }

    assert "ruff check src/ tests/ benchmark/scrutator/" in workflow
    assert "ruff format --check src/ tests/ benchmark/scrutator/harness.py benchmark/scrutator/tests/" in workflow
    assert steps["Test"]["run"] == "pytest tests/ -v"
    assert steps["Test"]["env"]["PYTHONPATH"] == "src"
    assert steps["Benchmark tests"]["run"] == "pytest benchmark/scrutator/tests/ -v"
    assert steps["Benchmark tests"]["env"]["PYTHONPATH"] == "src:benchmark/scrutator"


def test_security_policy_matches_ecosystem_response_sla():
    policy = (ROOT / "SECURITY.md").read_text(encoding="utf-8")

    for required in (
        "security@arcanada.ai",
        "72 hours",
        "7 days",
        "90 days",
        "180 days",
        "best effort",
        "14 days",
        "120 days",
        "whichever is sooner",
        "coordinated disclosure",
    ):
        assert required in policy


def test_numpy_determinism_copy_matches_declared_dependency_range():
    copy = (ROOT / "documentation" / "how-to" / "ltm-reflect.md").read_text(encoding="utf-8")

    assert "`numpy>=1.26`" in copy
    assert "not exactly pinned" in copy
    assert "numpy version is pinned" not in copy


def test_dependabot_updates_have_a_seven_day_cooldown():
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    for update in config["updates"]:
        assert update["cooldown"]["default-days"] >= 7


def test_public_repository_discovery_files_exist():
    required = [
        "LICENSE",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        ".github/dependabot.yml",
    ]
    assert [path for path in required if not (ROOT / path).is_file()] == []
