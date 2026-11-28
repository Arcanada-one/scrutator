"""Pin-regression checks for dependency floors and the hash-locked sync runtime.

These assertions read committed requirement files only (no moving git refs).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
REQUIREMENTS_DEV = ROOT / "requirements-dev.txt"
SYNC_LOCK = ROOT / "deploy" / "requirements-muneral-kb-sync.txt"

# Floors this change raises; a regression below these values fails the suite.
RUNTIME_FLOORS = {
    "tenacity": "9.2.1",
}
DEV_FLOORS = {
    "ruff": "0.17.4",
    "pytest": "9.2.0",
    "pytest-asyncio": "1.4.1.post1",
}
SYNC_EXACT = {
    "asyncpg": "0.32.0",
    "anyio": "4.16.0",
    "certifi": "2026.11.20",
    "idna": "3.20",
}


def _requirement_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        if line.startswith("--"):
            continue
        if line.endswith("\\"):
            line = line[:-1].rstrip()
        lines.append(line)
    return lines


def _floor_for(package: str, lines: list[str]) -> Version:
    pattern = re.compile(rf"^{re.escape(package)}(?P<spec>[^\s;]+)?", re.IGNORECASE)
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        spec = SpecifierSet(match.group("spec") or "")
        floors = [Version(sp.version) for sp in spec if sp.operator in {">=", "=="}]
        if floors:
            return max(floors)
    raise AssertionError(f"{package} floor not found in requirement lines")


def _exact_for(package: str, text: str) -> Version:
    match = re.search(
        rf"(?m)^{re.escape(package)}==([0-9A-Za-z_.+]+)\s*(?:\\)?$",
        text,
    )
    assert match, f"{package} exact pin missing from sync lock"
    return Version(match.group(1))


@pytest.mark.parametrize(("package", "minimum"), sorted(RUNTIME_FLOORS.items()))
def test_runtime_dependency_floors(package: str, minimum: str) -> None:
    got = _floor_for(package, _requirement_lines(REQUIREMENTS))
    assert got >= Version(minimum), f"{package} floor {got} < {minimum}"


@pytest.mark.parametrize(("package", "minimum"), sorted(DEV_FLOORS.items()))
def test_dev_dependency_floors(package: str, minimum: str) -> None:
    got = _floor_for(package, _requirement_lines(REQUIREMENTS_DEV))
    assert got >= Version(minimum), f"{package} floor {got} < {minimum}"


@pytest.mark.parametrize(("package", "exact"), sorted(SYNC_EXACT.items()))
def test_sync_runtime_hash_lock_pins(package: str, exact: str) -> None:
    text = SYNC_LOCK.read_text(encoding="utf-8")
    got = _exact_for(package, text)
    assert got == Version(exact), f"{package} sync pin {got} != {exact}"
    assert "--hash=sha256:" in text
