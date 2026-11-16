#!/usr/bin/env bash
# Complete maintained suite for the graph verifier's global fallback (DEC-AUP-0188: a change to
# requirements*.txt is measured by running this, never by classification alone).
#
# Contract with the verifier: exit 0 only when the whole `tests/` suite ran and passed; a non-zero test
# result is a failure; when the interpreter cannot import pytest or the project, nothing was measured and
# the script exits 127 (verifier verdict `not_measured`), never a false failure.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}/src" PYTHONDONTWRITEBYTECODE=1
if ! python3 -c 'import pytest, pytest_asyncio, scrutator' >/dev/null 2>&1; then
  echo 'FULL_FALLBACK_TEST_NOT_MEASURED: python3 cannot import pytest, pytest_asyncio and scrutator; install requirements-dev.txt' >&2
  exit 127
fi
exec python3 -B -m pytest -q -p no:cacheprovider tests
