"""scripts/graph-full-suite.sh reports `not measured` (127), never a failure, when it cannot run the suite."""

import json
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "graph-full-suite.sh"


def test_wrapper_is_executable_and_declared_as_the_full_test():
    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    profile = json.loads((ROOT / ".arcana" / "verify.json").read_text())
    root = profile["deployables"]["."]
    assert root["full_test"] == ["bash", "scripts/graph-full-suite.sh"]
    assert 1 <= root["full_test_timeout_seconds"] <= 7200


def test_unimportable_environment_exits_127_not_failure(tmp_path):
    fake = tmp_path / "python3"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    bash = "/bin/bash"
    # The fake python3 comes first; the directory of `dirname` follows so the wrapper's own
    # `cd "$(dirname ...)"` works as it does in CI. That directory also holds a real python3, which
    # the fake shadows.
    tools = Path(shutil.which("dirname")).parent
    result = subprocess.run(
        [bash, str(SCRIPT)],
        capture_output=True,
        text=True,
        env={"PATH": f"{tmp_path}:{tools}", "HOME": str(tmp_path)},
    )
    assert result.returncode == 127
    assert "FULL_FALLBACK_TEST_NOT_MEASURED" in result.stderr
    assert "dirname: command not found" not in result.stderr
