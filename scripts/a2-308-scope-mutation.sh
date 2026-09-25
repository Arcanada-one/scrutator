#!/usr/bin/env bash
# A2-308 — prove the per-route scope tests can go red.
#
# A green suite only means something if removing the control it guards turns it red. Each
# arm below deletes or weakens one piece of the A2-308 scope check on a scratch copy of the
# tree and re-runs the guarding tests; an arm that stays green is a hole in the tests, not a
# success. Run from the repository root: scripts/a2-308-scope-mutation.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
# Scratch copies live under the repo, never /tmp: on a shared host a foreign /tmp file can
# shadow a stdlib module for anything run from there (A2-236).
SCRATCH="${ROOT}/.a2-308-mutation"
GUARD_TESTS=(
  tests/security/test_route_scope_enforcement.py
  tests/security/test_route_auth_manifest.py
  tests/test_auth_verifier.py
)

caught=0
total=0
declare -a RESULTS=()

run_arm() {
  local name="$1" description="$2" mutate="$3"
  total=$((total + 1))
  local work
  mkdir -p "${SCRATCH}"
  work="$(mktemp -d "${SCRATCH}/arm-XXXXXX")"
  # Copy only what the suite needs; .venv stays shared via $PYTHON.
  cp -r "${ROOT}/src" "${ROOT}/tests" "${ROOT}/pyproject.toml" "${work}/"

  if ! (cd "${work}" && eval "${mutate}"); then
    RESULTS+=("ERROR  ${name} — mutation could not be applied: ${description}")
    rm -rf "${work}"
    return
  fi

  local output
  output="$(cd "${work}" && PYTHONPATH=src "${PYTHON}" -m pytest "${GUARD_TESTS[@]}" -q 2>&1 | tail -1)"
  rm -rf "${work}"

  if grep -q "failed\|error" <<<"${output}"; then
    caught=$((caught + 1))
    RESULTS+=("CAUGHT ${name} — ${description} :: ${output}")
  else
    RESULTS+=("SURVIVED ${name} — ${description} :: ${output}")
  fi
}

# 1. The headline control: the route-level scope check is a no-op.
run_arm "drop-write-scope-check" \
  "require_ltm_write_scope stops raising — every mutating route is open again" \
  "${PYTHON} - <<'PY'
import pathlib
p = pathlib.Path('src/scrutator/auth/dependency.py')
s = p.read_text()
s = s.replace('    if settings.auth_ltm_write_scope not in ctx.scopes:', '    if False:')
p.write_text(s)
PY"

# 2. The scope is carried but never populated — a token's write grant is invented.
run_arm "grant-every-scope" \
  "the tenant context claims both scopes regardless of the credential" \
  "${PYTHON} - <<'PY'
import pathlib
p = pathlib.Path('src/scrutator/auth/dependency.py')
s = p.read_text()
s = s.replace('        scopes=principal.scopes,', '        scopes=frozenset({\"kb:ltm.read\", \"kb:ltm.write\"}),')
p.write_text(s)
PY"

# 3. The grace window regains its old power to write.
run_arm "grace-window-writes" \
  "an unverified caller is handed the write scope during the auth-enforce grace window" \
  "${PYTHON} - <<'PY'
import pathlib
p = pathlib.Path('src/scrutator/auth/dependency.py')
s = p.read_text()
s = s.replace('            scopes=frozenset(),', '            scopes=frozenset({\"kb:ltm.write\", \"kb:ltm.read\"}),')
p.write_text(s)
PY"

# 4. Fail-closed parsing becomes fail-open: unknown scopes are ignored again.
run_arm "ignore-unknown-scopes" \
  "parse_scopes tolerates an unrecognized scope instead of denying" \
  "${PYTHON} - <<'PY'
import pathlib
p = pathlib.Path('src/scrutator/auth/verifier.py')
s = p.read_text()
s = s.replace('    if unknown:\n        raise Unauthenticated(\"token carries unrecognized scopes\")', '    if False:\n        raise Unauthenticated(\"token carries unrecognized scopes\")')
p.write_text(s)
PY"

# 5. One route quietly reverts to the read-only dependency.
run_arm "revert-one-route" \
  "POST /v1/edges alone goes back to require_tenant_context" \
  "${PYTHON} - <<'PY'
import pathlib
p = pathlib.Path('src/scrutator/health.py')
s = p.read_text()
s = s.replace(
    'async def create_edges(edges: list[EdgeCreate], ctx: TenantContext = Depends(require_ltm_write_scope)) -> dict:',
    'async def create_edges(edges: list[EdgeCreate], ctx: TenantContext = Depends(require_tenant_context)) -> dict:',
)
p.write_text(s)
PY"

# 6. The required read scope stops being required — any scope string passes.
run_arm "drop-required-scope" \
  "parse_scopes no longer requires the read scope to be present" \
  "${PYTHON} - <<'PY'
import pathlib
p = pathlib.Path('src/scrutator/auth/verifier.py')
s = p.read_text()
s = s.replace('    if required not in granted:\n        raise Unauthenticated(\"token scope mismatch\")', '    if False:\n        raise Unauthenticated(\"token scope mismatch\")')
p.write_text(s)
PY"

rmdir "${SCRATCH}" 2>/dev/null || true

printf '\n── A2-308 scope mutation battery ──\n'
for line in "${RESULTS[@]}"; do printf '%s\n' "${line}"; done
printf '\n%d/%d arms caught\n' "${caught}" "${total}"
[ "${caught}" -eq "${total}" ]
