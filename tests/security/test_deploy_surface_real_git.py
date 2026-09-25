"""The deploy surface is judged by REAL git, and it covers every file the deploy job executes.

The fake-host tests in test_deploy_transaction.py hand the transaction a preset `deploy_diff`, so
they prove what the script does WITH a surface verdict, never which paths produce one. A path missing
from `DEPLOY_SURFACE` is invisible there. A2-329 measured exactly that hole: the deploy job runs
`python3 tools/canary_probe.py` on the production host, `tools/` was not in the surface, and a change
to the probe would have deployed by push with no reviewed SHA.

So here the real transaction script runs against a real repository (a bare `origin` and a checkout on
`main`), with only docker/sudo/systemctl/curl faked. Each case commits a change to ONE path and asks
the transaction to deploy it by push, unattested:

* a path inside the surface must be refused with the surface message, before anything is touched;
* a path outside it must pass the surface check (and then stop at the faked docker, which proves it
  got past — nothing is mutated either way).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parents[2]
SCRIPT = REPO / "deploy" / "scrutator-deploy-transaction.sh"
DEPLOY_WORKFLOW = REPO / ".github" / "workflows" / "deploy.yml"
REFUSAL = "deploy-surface change requires an externally reviewed exact target SHA"
PAST_SURFACE = "fake docker reached"


def deploy_surface() -> list[str]:
    block = re.search(r"readonly DEPLOY_SURFACE=\(\n(.*?)\n\)", SCRIPT.read_text(), re.S)
    assert block, "DEPLOY_SURFACE array not found in the transaction"
    return [line.split("#", 1)[0].strip() for line in block.group(1).splitlines() if line.split("#", 1)[0].strip()]


def deploy_job_commands() -> str:
    """The shell the deploy job runs, without its comment lines (which quote paths as prose)."""
    steps = yaml.safe_load(DEPLOY_WORKFLOW.read_text())["jobs"]["deploy"]["steps"]
    lines = "\n".join(step.get("run") or "" for step in steps).splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def covered(path: str, surface: list[str]) -> bool:
    return any(path == entry or path.startswith(entry.rstrip("/") + "/") for entry in surface)


def _git(cwd: Path, *args: str) -> str:
    env = os.environ | {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


@pytest.fixture()
def real_host(tmp_path: Path):
    """A production-shaped checkout on main, one commit behind a bare origin, with faked side effects."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "author"
    root = tmp_path / "srv" / "scrutator"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    for path in ("tools/canary_probe.py", "tools/backfill_sections.py", "src/scrutator/app.py", "deploy/x.sh"):
        (work / path).parent.mkdir(parents=True, exist_ok=True)
        (work / path).write_text("v1\n")
    (work / ".gitignore").write_text(".env\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "base")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-q", "origin", "main")
    subprocess.run(["git", "clone", "-q", str(origin), str(root)], check=True)
    (root / ".env").write_text("SECRET=not-printed\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "docker").write_text(f"#!/bin/sh\necho '{PAST_SURFACE}' >&2\nexit 70\n")
    for name in ("sudo", "systemctl", "curl"):
        (fake_bin / name).write_text("#!/bin/sh\necho 'unexpected side effect' >&2\nexit 70\n")
    for command in fake_bin.iterdir():
        command.chmod(0o755)

    def deploy_change(path: str) -> subprocess.CompletedProcess[str]:
        (work / path).parent.mkdir(parents=True, exist_ok=True)
        (work / path).write_text("v2\n")
        _git(work, "add", "-A")
        _git(work, "commit", "-qm", f"change {path}")
        _git(work, "push", "-q", "origin", "main")
        target = _git(work, "rev-parse", "HEAD")
        before = _git(root, "rev-parse", "HEAD")
        env = os.environ | {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "SCRUTATOR_DEPLOY_ROOT": str(root),
            "SCRUTATOR_DEPLOY_STATE_DIR": str(tmp_path / "state"),
            "SCRUTATOR_REVIEWED_DEPLOY_SURFACE_SHA": "",
            "GITHUB_EVENT_NAME": "push",
        }
        done = subprocess.run(
            [str(SCRIPT), "--target-sha", target], cwd=root, env=env, text=True, capture_output=True, check=False
        )
        assert _git(root, "rev-parse", "HEAD") == before, "the surface check must not mutate the checkout"
        assert "unexpected side effect" not in done.stderr
        return done

    return deploy_change


# Paths the deploy job reads or executes on the production host, out of the TARGET checkout.
MUST_BE_IN_SURFACE = ["tools/canary_probe.py", "deploy/x.sh"]
# Application payload: it ships through the image by an ordinary push deploy, by design.
MUST_STAY_OUTSIDE = ["tools/backfill_sections.py", "src/scrutator/app.py"]


@pytest.mark.parametrize("path", MUST_BE_IN_SURFACE)
def test_a_change_to_a_surface_path_is_refused_on_an_unattested_push(real_host, path):
    done = real_host(path)
    assert done.returncode != 0
    assert REFUSAL in done.stderr, done.stderr
    assert PAST_SURFACE not in done.stderr


@pytest.mark.parametrize("path", MUST_STAY_OUTSIDE)
def test_a_change_outside_the_surface_passes_the_surface_check(real_host, path):
    done = real_host(path)
    assert REFUSAL not in done.stderr, done.stderr
    assert PAST_SURFACE in done.stderr, done.stderr


def test_every_checkout_file_the_deploy_job_executes_or_reads_is_in_the_surface():
    """Derived from the workflow, not from a list someone must remember to extend.

    Any repository path named in a `run:` of the deploy job (a script it executes, a plan it
    reads) is code or data that acts on the production host out of the TARGET checkout. If it is
    not in the surface, it changes by push with no reviewed SHA.
    """
    runs = deploy_job_commands()
    named = set(re.findall(r"(?<![\w/$.])((?:tools|deploy|scripts|src)/[\w./${}-]+)", runs))
    assert "tools/canary_probe.py" in named, "the extraction must see the probe, or this test proves nothing"
    surface = deploy_surface()
    missing = sorted(path for path in named if not covered(path, surface))
    assert not missing, f"executed/read by the deploy job but outside DEPLOY_SURFACE: {missing}"


def test_the_probe_cannot_import_unreviewed_siblings():
    """`python3 tools/x.py` puts tools/ first on sys.path: a new tools/json.py would run inside the
    probe without touching tools/canary_probe.py. Isolated mode (-I) drops the script directory,
    which is what lets ONE file, not the whole tools/ directory, be the surface entry."""
    runs = deploy_job_commands()
    invocations = re.findall(r"python3?\s+([^\n\\]*?)tools/canary_probe\.py", runs)
    assert invocations, "the canary invocation was not found"
    assert all(re.search(r"(^|\s)-I(\s|$)", flags) for flags in invocations), invocations
