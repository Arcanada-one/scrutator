#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

inject_failure=0
if (($# > 1)); then
    echo "usage: $0 [--inject-failure]" >&2
    exit 2
fi
if (($# == 1)); then
    if [[ "$1" != "--inject-failure" ]]; then
        echo "usage: $0 [--inject-failure]" >&2
        exit 2
    fi
    inject_failure=1
fi

ssh_target="${PG18_DOCKER_SSH_TARGET:-}"
if [[ -n "$ssh_target" && ! "$ssh_target" =~ ^[A-Za-z0-9._@:-]+$ ]]; then
    echo "invalid PG18_DOCKER_SSH_TARGET" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Python runtime not found" >&2
    exit 2
fi

docker_on_target() {
    if [[ -n "$ssh_target" ]]; then
        local remote_command
        printf -v remote_command '%q ' docker "$@"
        ssh -o BatchMode=yes -- "$ssh_target" "$remote_command"
    else
        docker "$@"
    fi
}

suffix="$(date +%s)-$$"
container="ltm0025-task5-$suffix"
database="ltm0025_test_${suffix//-/_}"
temp_dir=$(mktemp -d)
chmod 700 "$temp_dir"
tunnel_pid=""

cleanup() {
    local original_status=$?
    local cleanup_status=0
    trap - EXIT
    set +e

    if [[ -n "$tunnel_pid" ]]; then
        kill "$tunnel_pid" >/dev/null 2>&1
        wait "$tunnel_pid" >/dev/null 2>&1
    fi

    if docker_on_target inspect "$container" >/dev/null 2>&1; then
        docker_on_target exec "$container" psql -v ON_ERROR_STOP=1 -U postgres -d postgres \
            -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$database'" \
            >/dev/null 2>&1 || cleanup_status=1
        docker_on_target exec "$container" psql -v ON_ERROR_STOP=1 -U postgres -d postgres \
            -c "DROP DATABASE IF EXISTS \"$database\"" >/dev/null 2>&1 || cleanup_status=1
        docker_on_target exec "$container" psql -v ON_ERROR_STOP=1 -U postgres -d postgres \
            -c "DROP ROLE IF EXISTS muneral_kb_reader" >/dev/null 2>&1 || cleanup_status=1
        docker_on_target exec "$container" psql -At -U postgres -d postgres \
            -c "SELECT count(*) FROM pg_database WHERE datname = '$database'" | grep -qx '0' || cleanup_status=1
        docker_on_target exec "$container" psql -At -U postgres -d postgres \
            -c "SELECT count(*) FROM pg_roles WHERE rolname = 'muneral_kb_reader'" | grep -qx '0' || cleanup_status=1
        docker_on_target rm -f "$container" >/dev/null 2>&1 || cleanup_status=1
    fi
    if docker_on_target ps -a --format '{{.Names}}' | grep -Fxq "$container"; then
        cleanup_status=1
    fi
    rm -rf -- "$temp_dir"

    if ((cleanup_status == 0)); then
        echo "PG18_SMOKE_ZERO_RESIDUE"
    else
        echo "PG18_SMOKE_CLEANUP_FAILED" >&2
    fi
    if ((original_status != 0)); then
        exit "$original_status"
    fi
    exit "$cleanup_status"
}
trap cleanup EXIT

docker_on_target run -d --rm --name "$container" \
    -e POSTGRES_HOST_AUTH_METHOD=trust \
    -p 127.0.0.1::5432 \
    pgvector/pgvector:pg18 >/dev/null

for _ in {1..100}; do
    if docker_on_target exec "$container" pg_isready -U postgres >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
docker_on_target exec "$container" pg_isready -U postgres >/dev/null
docker_on_target exec "$container" psql -v ON_ERROR_STOP=1 -U postgres -d postgres \
    -c "CREATE DATABASE \"$database\"" >/dev/null

remote_port=$(docker_on_target port "$container" 5432/tcp | sed 's/.*://')
if [[ ! "$remote_port" =~ ^[0-9]+$ ]]; then
    echo "invalid allocated PostgreSQL port" >&2
    exit 1
fi

if [[ -n "$ssh_target" ]]; then
    local_port=$(
        "$python_bin" -c \
            'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
    )
    ssh -o BatchMode=yes -N -L "127.0.0.1:$local_port:127.0.0.1:$remote_port" -- "$ssh_target" &
    tunnel_pid=$!
else
    local_port="$remote_port"
fi

export local_port
"$python_bin" - <<'PY'
import os
import socket
import time

port = int(os.environ["local_port"])
for _ in range(100):
    try:
        with socket.create_connection(("127.0.0.1", port), 0.2):
            break
    except OSError:
        time.sleep(0.1)
else:
    raise SystemExit("PostgreSQL tunnel did not become ready")
PY

umask 077
admin_dsn="postgresql://postgres@127.0.0.1:$local_port/$database"
role_password=$("$python_bin" -c 'import secrets; print(secrets.token_urlsafe(32))')
printf '%s\n' "$admin_dsn" >"$temp_dir/admin-dsn"
printf '%s\n' "$role_password" >"$temp_dir/reader-password"
chmod 600 "$temp_dir/admin-dsn" "$temp_dir/reader-password"

MUNERAL_TEST_DATABASE_URL="$admin_dsn" \
    MUNERAL_TEST_ADMIN_DSN_FILE="$temp_dir/admin-dsn" \
    MUNERAL_TEST_ROLE_PASSWORD_FILE="$temp_dir/reader-password" \
    PYTHONPATH="$repo_root/src" \
    "$repo_root/.venv/bin/pytest" -q "$repo_root/tests/test_muneral_sync_sql.py"

if ((inject_failure == 1)); then
    echo "PG18_SMOKE_INJECTED_FAILURE" >&2
    exit 97
fi

echo "PG18_SMOKE_PASS"
