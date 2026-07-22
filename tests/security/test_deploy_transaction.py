from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
SCRIPT = REPO / "deploy" / "scrutator-deploy-transaction.sh"
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
DEPLOY_WORKFLOW = REPO / ".github" / "workflows" / "deploy.yml"
RECALL_WORKFLOW = REPO / ".github" / "workflows" / "recall-regression.yml"
OLD_SHA = "1" * 40
TARGET_SHA = "2" * 40
OLD_IMAGE = "sha256:" + "a" * 64
NEW_IMAGE = "sha256:" + "b" * 64
TIMERS = (
    "kb-reconcile.timer",
    "kb-observe.timer",
    "kb-self-improvement-reconcile.timer",
)


FAKE_COMMAND = r"""#!/usr/bin/python3
import json
import os
import pathlib
import sys

name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
state_path = pathlib.Path(os.environ["FAKE_STATE"])
state = json.loads(state_path.read_text())

def save():
    state_path.write_text(json.dumps(state, sort_keys=True))

def record():
    call = name + " " + " ".join(args)
    state.setdefault("calls", []).append(call)
    save()
    if state.get("fail_once_call") and state["fail_once_call"] in call and not state.get("failed_once"):
        state["failed_once"] = True
        save()
        print("injected one-shot failure", file=sys.stderr)
        raise SystemExit(70)
    if state.get("fail_call") and state["fail_call"] in call:
        print("injected failure", file=sys.stderr)
        raise SystemExit(70)

def image_from_override():
    for index, value in enumerate(args):
        if value == "-f" and index + 1 < len(args):
            candidate = pathlib.Path(args[index + 1])
            if candidate.name.endswith(".override.yml"):
                for line in candidate.read_text().splitlines():
                    if line.strip().startswith("image:"):
                        return line.split('"', 2)[1]
    raise SystemExit("override image missing")

record()

if name == "sudo":
    if args and args[0] == "-n":
        args = args[1:]
    os.execvp(args[0], args)

if name == "git":
    if args[:2] == ["status", "--porcelain"]:
        if state.get("dirty"):
            print(" M dirty-file")
    elif args[:3] == ["symbolic-ref", "--quiet", "--short"]:
        print(state.get("branch", "main"))
    elif args[:2] == ["rev-parse", "HEAD"]:
        print(state["source_sha"])
    elif args[:2] == ["rev-parse", "origin/main"]:
        print(state["target_sha"])
    elif args[:2] == ["fetch", "origin"]:
        pass
    elif args[:2] == ["merge-base", "--is-ancestor"]:
        if state.get("diverged"):
            raise SystemExit(1)
    elif args[:2] == ["diff", "--name-only"]:
        print("\n".join(state["deploy_diff"]))
    elif args[:4] == ["ls-tree", "-r", "--name-only", state["source_sha"]]:
        print("\n".join(state["trees"][state["source_sha"]]))
    elif args and args[0] == "show":
        revision, path = args[1].split(":", 1)
        print(state["tree_content"][revision][path], end="")
    elif args[:2] == ["reset", "--hard"]:
        revision = args[2]
        state["source_sha"] = revision
        save()
        for path, content in state["tree_content"][revision].items():
            target = pathlib.Path.cwd() / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    else:
        print("unsupported fake git: " + " ".join(args), file=sys.stderr)
        raise SystemExit(64)
    raise SystemExit(0)

if name == "systemctl":
    action = args[0]
    units = args[1:]
    if action == "is-active":
        unit = units[-1]
        active = state["timers"].get(unit, state["services"].get(unit, False))
        raise SystemExit(0 if active else 3)
    if action in {"start", "stop"}:
        active = action == "start"
        for unit in units:
            if unit in state["timers"]:
                state["timers"][unit] = active
        save()
        raise SystemExit(0)

if name == "docker":
    if args and args[0] == "compose" and "ps" in args:
        print(state["container_id"])
    elif args[:2] == ["inspect", "--format"]:
        print(state["current_image"])
    elif args[:2] == ["image", "inspect"]:
        ref = args[-1]
        if ref in state["tags"]:
            print(state["tags"][ref])
        elif ref.startswith("sha256:"):
            print(ref)
        else:
            raise SystemExit(1)
    elif args[:2] == ["image", "tag"]:
        source, tag = args[2], args[3]
        state["tags"][tag] = state["tags"].get(source, source)
        save()
    elif args[:2] == ["image", "rm"]:
        state["tags"].pop(args[2], None)
        save()
    elif args[0] == "compose" and "build" in args:
        state["built_image"] = state["new_image"]
        try:
            tag = image_from_override()
        except SystemExit:
            tag = ""
        if tag:
            state["tags"][tag] = state["new_image"]
        save()
    elif args[0] == "compose" and "images" in args:
        print(state["compose_images_id"])
    elif args[0] == "compose" and "up" in args:
        tag = image_from_override()
        state["current_image"] = state["tags"][tag]
        state["container_id"] = "container-after-up"
        if tag.startswith("scrutator-deploy:") and state.get("mutate_env_on_candidate_up"):
            (pathlib.Path.cwd() / ".env").write_text("MUTATED=must-be-restored\n")
            (pathlib.Path.cwd() / ".env").chmod(0o644)
        if tag.startswith("scrutator-deploy:") and state.get("delete_env_snapshot_on_candidate_up"):
            for snapshot in pathlib.Path(os.environ["SCRUTATOR_DEPLOY_STATE_DIR"]).glob("checkpoint.*/env.snapshot"):
                snapshot.unlink()
        if tag.startswith("scrutator-deploy:") and state.get("block_outcome_write_on_candidate_up"):
            for checkpoint in pathlib.Path(os.environ["SCRUTATOR_DEPLOY_STATE_DIR"]).glob("checkpoint.*"):
                (checkpoint / "outcome.next").mkdir()
        save()
    else:
        print("unsupported fake docker: " + " ".join(args), file=sys.stderr)
        raise SystemExit(64)
    raise SystemExit(0)

if name == "curl":
    sequence = state.setdefault("health", [True])
    healthy = sequence.pop(0) if len(sequence) > 1 else sequence[0]
    save()
    if healthy:
        print('{"status":"ok"}')
        raise SystemExit(0)
    raise SystemExit(22)

print("unsupported fake command: " + name, file=sys.stderr)
raise SystemExit(64)
"""


@pytest.fixture()
def fake_host(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "srv" / "scrutator"
    root.mkdir(parents=True)
    (root / ".env").write_text("SECRET=not-printed\n")
    (root / ".env").chmod(0o640)
    (root / "docker-compose.yml").write_text("services:\n  scrutator:\n    build: .\n")
    (root / "Dockerfile").write_text("FROM scratch\n")
    (root / "deploy").mkdir()
    (root / "deploy" / "ltm-reflect-state-preflight.sh").write_text("#!/bin/sh\nexit 0\n")
    os.chmod(root / "deploy" / "ltm-reflect-state-preflight.sh", 0o700)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    dispatcher = fake_bin / "dispatcher"
    dispatcher.write_text(FAKE_COMMAND)
    dispatcher.chmod(0o755)
    for command in ("git", "docker", "systemctl", "sudo", "curl"):
        (fake_bin / command).symlink_to(dispatcher)

    old_paths = ["Dockerfile", "docker-compose.yml", "deploy/ltm-reflect-state-preflight.sh"]
    target_paths = [*old_paths, "deploy/scrutator-deploy-transaction.sh"]
    state = {
        "source_sha": OLD_SHA,
        "target_sha": TARGET_SHA,
        "branch": "main",
        "dirty": False,
        "diverged": False,
        "deploy_diff": [
            ".github/workflows/ci.yml",
            ".github/workflows/deploy.yml",
            ".github/workflows/recall-regression.yml",
            "deploy/scrutator-deploy-transaction.sh",
        ],
        "trees": {OLD_SHA: old_paths, TARGET_SHA: target_paths},
        "tree_content": {
            OLD_SHA: {
                "Dockerfile": "FROM scratch\n",
                "docker-compose.yml": "services:\n  scrutator:\n    build: .\n",
                "deploy/ltm-reflect-state-preflight.sh": "#!/bin/sh\nexit 0\n",
            },
            TARGET_SHA: {
                "Dockerfile": "FROM scratch\n",
                "docker-compose.yml": "services:\n  scrutator:\n    build: .\n",
                "deploy/ltm-reflect-state-preflight.sh": "#!/bin/sh\nexit 0\n",
                "deploy/scrutator-deploy-transaction.sh": "#!/bin/sh\nexit 0\n",
            },
        },
        "timers": {TIMERS[0]: True, TIMERS[1]: False, TIMERS[2]: True},
        "services": {"kb-reconcile.service": False, "kb-self-improvement-reconcile.service": False},
        "container_id": "container-before",
        "current_image": OLD_IMAGE,
        "new_image": NEW_IMAGE,
        "built_image": "",
        "compose_images_id": OLD_IMAGE,
        "tags": {},
        "health": [True],
        "calls": [],
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    state_dir = tmp_path / "deploy-state"
    state_dir.mkdir(mode=0o700)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_STATE": str(state_path),
        "SCRUTATOR_DEPLOY_ROOT": str(root),
        "SCRUTATOR_DEPLOY_STATE_DIR": str(state_dir),
        "SCRUTATOR_DEPLOY_HEALTH_ATTEMPTS": "2",
        "SCRUTATOR_DEPLOY_HEALTH_INTERVAL_SECONDS": "0",
        "SCRUTATOR_DEPLOY_RETAIN_IMAGES": "2",
        "SCRUTATOR_REVIEWED_DEPLOY_SURFACE_SHA": TARGET_SHA,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
    }
    return {"root": root, "state_path": state_path, "state_dir": state_dir, "env": env}


def run_deploy(fake_host: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), "--target-sha", TARGET_SHA],
        cwd=fake_host["root"],
        env=fake_host["env"],
        text=True,
        capture_output=True,
        check=False,
    )


def load_state(fake_host: dict[str, object]) -> dict[str, object]:
    return json.loads(Path(fake_host["state_path"]).read_text())


def update_state(fake_host: dict[str, object], **changes: object) -> None:
    state = load_state(fake_host)
    state.update(changes)
    Path(fake_host["state_path"]).write_text(json.dumps(state))


def assert_pre_state_restored(fake_host: dict[str, object]) -> None:
    state = load_state(fake_host)
    assert state["source_sha"] == OLD_SHA
    assert state["current_image"] == OLD_IMAGE
    assert state["timers"] == {TIMERS[0]: True, TIMERS[1]: False, TIMERS[2]: True}
    assert (Path(fake_host["root"]) / ".env").read_text() == "SECRET=not-printed\n"
    assert (Path(fake_host["root"]) / ".env").stat().st_mode & 0o777 == 0o640


def test_success_deploys_target_image_and_restores_exact_timer_set(fake_host: dict[str, object]) -> None:
    result = run_deploy(fake_host)
    state = load_state(fake_host)

    assert result.returncode == 0, result.stderr
    assert state["source_sha"] == TARGET_SHA
    assert state["current_image"] == NEW_IMAGE
    assert state["timers"] == {TIMERS[0]: True, TIMERS[1]: False, TIMERS[2]: True}
    assert state["tags"][f"scrutator-deploy:{TARGET_SHA}"] == NEW_IMAGE
    assert state["tags"][f"scrutator-rollback:{OLD_SHA}"] == OLD_IMAGE
    assert not any(" compose " in call and " images " in call for call in state["calls"])
    assert "SECRET=not-printed" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("change", "value"),
    (("dirty", True), ("diverged", True), ("branch", "feature")),
)
def test_preflight_refuses_unsafe_source_without_mutation(
    fake_host: dict[str, object], change: str, value: object
) -> None:
    update_state(fake_host, **{change: value})

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)
    assert not any(" compose " in call and " up " in call for call in load_state(fake_host)["calls"])


def test_preflight_refuses_active_reconciler_and_restores_timers(fake_host: dict[str, object]) -> None:
    state = load_state(fake_host)
    state["services"]["kb-reconcile.service"] = True
    Path(fake_host["state_path"]).write_text(json.dumps(state))

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)


def test_preflight_requires_preprovisioned_private_state_directory(fake_host: dict[str, object]) -> None:
    Path(fake_host["state_dir"]).rmdir()

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert "preprovisioned state directory" in result.stderr
    assert_pre_state_restored(fake_host)
    assert not any(" install " in call for call in load_state(fake_host)["calls"])
    assert not any(" compose " in call and " up " in call for call in load_state(fake_host)["calls"])


def test_reconcile_probe_error_is_not_misclassified_as_inactive(fake_host: dict[str, object]) -> None:
    update_state(fake_host, fail_call="systemctl is-active --quiet kb-reconcile.service")

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)
    assert not any(" compose " in call and " build " in call for call in load_state(fake_host)["calls"])


def test_preflight_treats_git_diff_failure_as_failure_not_empty_diff(fake_host: dict[str, object]) -> None:
    update_state(fake_host, fail_call="git diff --name-only")

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)
    assert not any(" compose " in call and " build " in call for call in load_state(fake_host)["calls"])


@pytest.mark.parametrize(
    "fail_call",
    (
        "git show",
        "candidate.override.yml build",
        "candidate.override.yml up",
        "curl -fsS",
    ),
)
def test_failure_in_each_mutating_phase_rolls_back_source_image_and_timers(
    fake_host: dict[str, object], fail_call: str
) -> None:
    update_state(fake_host, fail_call=fail_call)
    if fail_call == "curl -fsS":
        update_state(fake_host, health=[False, False, True])

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)


def test_health_failure_rolls_back_then_proves_rollback_health(fake_host: dict[str, object]) -> None:
    update_state(fake_host, health=[False, False, True])

    result = run_deploy(fake_host)
    state = load_state(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)
    assert sum(call.startswith("curl -fsS") for call in state["calls"]) == 3
    rollback_up = next(call for call in state["calls"] if "rollback.override.yml up" in call)
    assert f"--project-directory {fake_host['root']}" in rollback_up
    assert f"-f {fake_host['root']}/docker-compose.yml" in rollback_up
    assert "/raw/docker-compose.yml" not in rollback_up


def test_candidate_preflight_failure_rolls_back_before_container_recreate(fake_host: dict[str, object]) -> None:
    state = load_state(fake_host)
    state["tree_content"][TARGET_SHA]["deploy/ltm-reflect-state-preflight.sh"] = "#!/bin/sh\nexit 74\n"
    Path(fake_host["state_path"]).write_text(json.dumps(state))

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)
    calls = load_state(fake_host)["calls"]
    assert not any(" compose " in call and " build " in call for call in calls)


def test_rollback_is_idempotent_when_previous_tag_already_points_to_image(fake_host: dict[str, object]) -> None:
    state = load_state(fake_host)
    state["tags"][f"scrutator-rollback:{OLD_SHA}"] = OLD_IMAGE
    state["health"] = [False, False, True]
    Path(fake_host["state_path"]).write_text(json.dumps(state))

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)


def test_success_prunes_candidate_and_rollback_tags_but_keeps_in_use_image(fake_host: dict[str, object]) -> None:
    current_tag = f"scrutator-rollback:{OLD_SHA}"
    deploy_tags = [f"scrutator-deploy:{value * 40}" for value in ("3", "4", "5")]
    rollback_tags = [f"scrutator-rollback:{value * 40}" for value in ("6", "7", "8")]
    state = load_state(fake_host)
    state["tags"] = {current_tag: OLD_IMAGE}
    for index, tag in enumerate([*deploy_tags, *rollback_tags], start=1):
        state["tags"][tag] = "sha256:" + f"{index:x}" * 64
    state_dir = Path(fake_host["state_dir"])
    (state_dir / "image-tags").write_text("\n".join([*deploy_tags, *rollback_tags, current_tag]) + "\n")
    Path(fake_host["state_path"]).write_text(json.dumps(state))

    result = run_deploy(fake_host)

    assert result.returncode == 0, result.stderr
    final = load_state(fake_host)
    final_tags = set(final["tags"])
    candidate = f"scrutator-deploy:{TARGET_SHA}"
    assert candidate in final_tags
    assert final["tags"][candidate] == final["current_image"]
    assert len([tag for tag in final_tags if tag.startswith("scrutator-deploy:")]) == 2
    assert len([tag for tag in final_tags if tag.startswith("scrutator-rollback:")]) == 2
    assert current_tag in final_tags


def test_success_bounds_aliases_sharing_running_image_and_keeps_designated_tag(
    fake_host: dict[str, object],
) -> None:
    deploy_tags = [f"scrutator-deploy:{value * 40}" for value in ("3", "4", "5", "6")]
    rollback_tags = [f"scrutator-rollback:{value * 40}" for value in ("7", "8", "9", "a")]
    state = load_state(fake_host)
    for tag in [*deploy_tags, *rollback_tags]:
        state["tags"][tag] = NEW_IMAGE
    state_dir = Path(fake_host["state_dir"])
    (state_dir / "image-tags").write_text("\n".join([*deploy_tags, *rollback_tags]) + "\n")
    Path(fake_host["state_path"]).write_text(json.dumps(state))

    result = run_deploy(fake_host)

    assert result.returncode == 0, result.stderr
    final = load_state(fake_host)
    candidate = f"scrutator-deploy:{TARGET_SHA}"
    assert final["tags"][candidate] == final["current_image"] == NEW_IMAGE
    assert len([tag for tag in final["tags"] if tag.startswith("scrutator-deploy:")]) <= 2
    assert len([tag for tag in final["tags"] if tag.startswith("scrutator-rollback:")]) <= 2


def test_deploy_surface_change_requires_manual_exact_sha_attestation(fake_host: dict[str, object]) -> None:
    state = load_state(fake_host)
    state["deploy_diff"] = ["Dockerfile"]
    Path(fake_host["state_path"]).write_text(json.dumps(state))
    fake_host["env"]["SCRUTATOR_REVIEWED_DEPLOY_SURFACE_SHA"] = ""

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)


def test_refuses_ambiguous_running_container(fake_host: dict[str, object]) -> None:
    state = load_state(fake_host)
    state["container_id"] = "one\ntwo"
    Path(fake_host["state_path"]).write_text(json.dumps(state))

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)


def test_reviewed_manual_deploy_surface_change_can_be_used_after_bootstrap(fake_host: dict[str, object]) -> None:
    state = load_state(fake_host)
    state["tree_content"][OLD_SHA]["deploy/scrutator-deploy-transaction.sh"] = "#!/bin/sh\nexit 0\n"
    state["trees"][OLD_SHA].append("deploy/scrutator-deploy-transaction.sh")
    state["deploy_diff"] = ["Dockerfile"]
    Path(fake_host["state_path"]).write_text(json.dumps(state))

    result = run_deploy(fake_host)

    assert result.returncode == 0, result.stderr
    assert load_state(fake_host)["source_sha"] == TARGET_SHA


def test_deploy_surface_requires_externally_reviewed_exact_target_sha(fake_host: dict[str, object]) -> None:
    fake_host["env"]["SCRUTATOR_REVIEWED_DEPLOY_SURFACE_SHA"] = "3" * 40

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert_pre_state_restored(fake_host)


def test_deploy_surface_requires_manual_dispatch(fake_host: dict[str, object]) -> None:
    fake_host["env"]["GITHUB_EVENT_NAME"] = "push"

    result = run_deploy(fake_host)

    assert result.returncode != 0
    assert "workflow_dispatch" in result.stderr
    assert_pre_state_restored(fake_host)


def test_normal_later_push_deploy_remains_allowed(fake_host: dict[str, object]) -> None:
    state = load_state(fake_host)
    state["deploy_diff"] = []
    state["tree_content"][OLD_SHA]["deploy/scrutator-deploy-transaction.sh"] = "#!/bin/sh\nexit 0\n"
    state["trees"][OLD_SHA].append("deploy/scrutator-deploy-transaction.sh")
    Path(fake_host["state_path"]).write_text(json.dumps(state))
    fake_host["env"]["GITHUB_EVENT_NAME"] = "push"

    result = run_deploy(fake_host)

    assert result.returncode == 0, result.stderr
    assert load_state(fake_host)["source_sha"] == TARGET_SHA


def test_env_mutation_is_fatal_without_replacing_live_file(fake_host: dict[str, object]) -> None:
    env_path = Path(fake_host["root"]) / ".env"
    original = env_path.stat()
    update_state(fake_host, health=[False, False, True], mutate_env_on_candidate_up=True)

    result = run_deploy(fake_host)

    assert result.returncode == 90
    state = load_state(fake_host)
    assert state["source_sha"] == OLD_SHA
    assert state["current_image"] == OLD_IMAGE
    assert state["timers"] == {TIMERS[0]: True, TIMERS[1]: False, TIMERS[2]: True}
    assert env_path.read_text() == "MUTATED=must-be-restored\n"
    assert env_path.stat().st_ino == original.st_ino
    assert env_path.stat().st_uid == original.st_uid
    assert env_path.stat().st_gid == original.st_gid
    snapshots = list(Path(fake_host["state_dir"]).glob("checkpoint.*/env.snapshot"))
    assert len(snapshots) == 1
    assert snapshots[0].stat().st_mode & 0o777 == 0o600
    assert snapshots[0].read_text() == "SECRET=not-printed\n"
    outcome = next(Path(fake_host["state_dir"]).glob("checkpoint.*/outcome")).read_text()
    assert "outcome=rollback_failed" in outcome
    assert "env_restored=0" in outcome


def test_rollback_image_failure_still_restores_source_and_timers_without_faking_env_restore(
    fake_host: dict[str, object],
) -> None:
    update_state(
        fake_host,
        health=[False, False, True],
        mutate_env_on_candidate_up=True,
        fail_call="rollback.override.yml up",
    )

    result = run_deploy(fake_host)

    assert result.returncode == 90
    state = load_state(fake_host)
    assert state["source_sha"] == OLD_SHA
    assert state["current_image"] == NEW_IMAGE
    assert state["timers"] == {TIMERS[0]: True, TIMERS[1]: False, TIMERS[2]: True}
    assert (Path(fake_host["root"]) / ".env").read_text() == "MUTATED=must-be-restored\n"
    assert (Path(fake_host["root"]) / ".env").stat().st_mode & 0o777 == 0o644
    status_files = list(Path(fake_host["state_dir"]).glob("checkpoint.*/outcome"))
    assert len(status_files) == 1
    status = status_files[0].read_text()
    assert "outcome=rollback_failed" in status
    assert "source_restored=1" in status
    assert "env_restored=0" in status
    assert "image_restored=0" in status
    assert "timers_restored=1" in status


def test_checkpoint_metadata_captures_env_uid_gid_without_replacing_env(fake_host: dict[str, object]) -> None:
    env_path = Path(fake_host["root"]) / ".env"
    original = env_path.stat()
    update_state(fake_host, fail_call="candidate.override.yml build")

    result = run_deploy(fake_host)

    assert result.returncode != 0
    metadata = next(Path(fake_host["state_dir"]).glob("checkpoint.*/metadata")).read_text()
    assert f"env_uid={original.st_uid}" in metadata
    assert f"env_gid={original.st_gid}" in metadata
    assert env_path.stat().st_ino == original.st_ino


def test_checkpoint_failure_after_tag_mutation_runs_finalizer(fake_host: dict[str, object]) -> None:
    update_state(fake_host, fail_call=f"git show {OLD_SHA}:Dockerfile")

    result = run_deploy(fake_host)

    assert result.returncode != 0
    status_file = next(Path(fake_host["state_dir"]).glob("checkpoint.*/outcome"))
    assert "outcome=failed_before_mutation" in status_file.read_text()
    assert f"scrutator-rollback:{OLD_SHA}" in load_state(fake_host)["tags"]


def test_outcome_write_failure_is_reported_as_cleanup_failure(fake_host: dict[str, object]) -> None:
    update_state(
        fake_host,
        health=[False, False, True],
        block_outcome_write_on_candidate_up=True,
    )

    result = run_deploy(fake_host)

    assert result.returncode == 90
    assert "cleanup failed" in result.stderr


def require_docker() -> None:
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode != 0:
        if os.environ.get("SCRUTATOR_REQUIRE_DOCKER_SEMANTIC_TEST") == "1":
            pytest.fail("Docker semantic test is mandatory in the deploy verification job")
        pytest.skip("Docker daemon unavailable")


def test_real_compose_override_build_tags_exact_candidate_not_stale_default(tmp_path: Path) -> None:
    require_docker()
    suffix = uuid.uuid4().hex
    default_tag = f"scrutator-deploy-test-default:{suffix}"
    candidate_tag = f"scrutator-deploy-test-candidate:{suffix}"
    compose = tmp_path / "compose.yml"
    override = tmp_path / "override.yml"
    dockerfile = tmp_path / "Dockerfile"
    compose.write_text(f'services:\n  scrutator:\n    build: .\n    image: "{default_tag}"\n')
    override.write_text(f'services:\n  scrutator:\n    image: "{candidate_tag}"\n')
    try:
        dockerfile.write_text('FROM scratch\nLABEL generation="old"\n')
        subprocess.run(["docker", "compose", "-f", compose, "build", "scrutator"], cwd=tmp_path, check=True)
        old_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", default_tag], text=True)
        dockerfile.write_text('FROM scratch\nLABEL generation="target"\n')
        subprocess.run(
            ["docker", "compose", "-f", compose, "-f", override, "build", "--no-cache", "scrutator"],
            cwd=tmp_path,
            check=True,
        )
        candidate_id = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", candidate_tag], text=True
        )
        assert candidate_id.strip() != old_id.strip()
    finally:
        subprocess.run(["docker", "image", "rm", "-f", candidate_tag, default_tag], capture_output=True, check=False)


def test_real_compose_rollback_uses_live_project_identity_and_relative_paths(tmp_path: Path) -> None:
    suffix = uuid.uuid4().hex
    project = tmp_path / f"scrutator-{suffix}"
    checkpoint = tmp_path / "deploy-state" / "checkpoint"
    data = project / "data"
    data.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (data / "marker").write_text("live-root\n")
    (project / ".env").write_text("PROJECT_MARKER=live-root\n")
    compose = project / "docker-compose.yml"
    override = checkpoint / "rollback.override.yml"
    project_name = f"scrutator-live-{suffix}"
    compose.write_text(
        textwrap.dedent(
            f"""
            name: {project_name}
            services:
              scrutator:
                image: alpine:3.20
                env_file: .env
                volumes:
                  - ./data:/data:ro
                command:
                  - sh
                  - -c
                  - 'test "$$PROJECT_MARKER" = live-root && test -f /data/marker && sleep 30'
            """
        ).lstrip()
    )
    override.write_text("services:\n  scrutator:\n    image: alpine:3.20\n")
    command = [
        "docker",
        "compose",
        "--project-directory",
        str(project),
        "-f",
        str(compose),
        "-f",
        str(override),
    ]
    try:
        config = json.loads(subprocess.check_output([*command, "config", "--format", "json"], text=True))
        assert config["name"] == project_name
        assert config["services"]["scrutator"]["volumes"][0]["source"] == str(data)
        assert "$PROJECT_MARKER" in config["services"]["scrutator"]["command"][-1]
        require_docker()
        subprocess.run([*command, "up", "-d", "--no-build", "scrutator"], check=True)
        container_id = subprocess.check_output([*command, "ps", "-q", "scrutator"], text=True).strip()
        labels = json.loads(
            subprocess.check_output(
                ["docker", "inspect", "--format", "{{json .Config.Labels}}", container_id], text=True
            )
        )
        assert labels["com.docker.compose.project"] == project_name
        assert labels["com.docker.compose.project.working_dir"] == str(project)
        subprocess.run([*command, "exec", "-T", "scrutator", "test", "-f", "/data/marker"], check=True)
    finally:
        subprocess.run([*command, "down", "--volumes", "--remove-orphans"], capture_output=True, check=False)


def test_ci_workflow_is_hosted_validation_only() -> None:
    workflow = CI_WORKFLOW.read_text()

    assert "permissions:\n  contents: read" in workflow
    assert "pull_request:" in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert 'SCRUTATOR_REQUIRE_DOCKER_SEMANTIC_TEST: "1"' in workflow
    assert "self-hosted" not in workflow
    assert "scrutator-deploy-transaction.sh" not in workflow
    assert "  deploy:" not in workflow
    assert "workflow_dispatch:" not in workflow


def test_deploy_workflow_uses_hosted_semantic_gate_and_restricted_runner_group() -> None:
    workflow = DEPLOY_WORKFLOW.read_text()

    assert "permissions:\n  contents: read" in workflow
    assert "actions/checkout@" not in workflow or "actions/checkout@v4" not in workflow
    assert "deploy/scrutator-deploy-transaction.sh" in workflow
    assert "docker compose up -d --build" not in workflow
    assert "git pull --ff-only" not in workflow
    assert "systemctl stop" not in workflow
    assert (
        textwrap.dedent(
            """
        --target-sha "${GITHUB_SHA}"
        """
        ).strip()
        in workflow
    )
    uses = re.findall(r"^\s*uses:\s*\S+@([^\s#]+)", workflow, flags=re.MULTILINE)
    assert uses
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in uses)
    assert "runs-on: ubuntu-24.04" in workflow
    assert "uses: Arcanada-one/datarim/.github/workflows/network-exposure-lint.yml@" in workflow
    assert "runner_labels: '[\"ubuntu-24.04\"]'" in workflow
    assert "ruff check src/ tests/" in workflow
    assert "ruff format --check src/ tests/" in workflow
    assert "bandit -ll -ii" in workflow
    assert "pytest tests/ -v" in workflow
    assert 'SCRUTATOR_REQUIRE_DOCKER_SEMANTIC_TEST: "1"' in workflow
    assert "workflow_dispatch:" in workflow
    deploy_job = workflow[workflow.index("  deploy:") :]
    assert "needs: [network-exposure, verify-exact-target]" in deploy_job
    assert "group: scrutator-prod" in deploy_job
    assert "labels: [self-hosted, linux, arcana-db, docker]" in deploy_job
    assert "environment: kb-production" in deploy_job
    assert "github.ref == 'refs/heads/main'" in deploy_job
    assert "github.event_name == 'push' || github.event_name == 'workflow_dispatch'" in deploy_job
    assert "SCRUTATOR_REVIEWED_DEPLOY_SURFACE_SHA" in deploy_job


def test_persistent_live_recall_runner_never_executes_pull_request_code() -> None:
    workflow = RECALL_WORKFLOW.read_text()

    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "workflow_dispatch:" in workflow
    assert "group: scrutator-prod" in workflow
    assert "labels: [self-hosted, linux, arcana-db, docker]" in workflow
    assert "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'" in workflow
    assert "environment: kb-production" in workflow
    assert "C10" in workflow and "dedicated recall client" in workflow
