#!/usr/bin/env bash
set -euo pipefail

umask 077

readonly state_dir="${MUNERAL_KB_SYNC_STATE_DIR:-/var/lib/muneral-kb-sync/runtime}"
readonly credentials_dir="${CREDENTIALS_DIRECTORY:-/etc/muneral-kb-sync}"
readonly dsn_credential="${credentials_dir}/muneral-db-dsn"
readonly writer_credential="${credentials_dir}/ltm-writer-token"
readonly cursor_file="${state_dir}/cursor.json"
readonly endpoint="${MUNERAL_KB_SYNC_ENDPOINT:-http://127.0.0.1:8310/v1/ltm/ingest}"

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly release_root
readonly python_bin="${MUNERAL_KB_SYNC_PYTHON:-${release_root}/venv/bin/python}"

if [[ $# -eq 0 ]]; then
    echo "usage: muneral-kb-sync MODE [OPTIONS]" >&2
    exit 64
fi

for argument in "$@"; do
    case "$argument" in
        --dsn-credential | --dsn-credential=* | --writer-credential | --writer-credential=* | --cursor-file | --cursor-file=* | --endpoint | --endpoint=*)
            echo "credential, cursor, and endpoint paths are controlled by the wrapper" >&2
            exit 64
            ;;
    esac
done

for credential in "$dsn_credential" "$writer_credential"; do
    if [[ -L "$credential" || ! -f "$credential" || ! -r "$credential" ]]; then
        echo "required credential is unavailable" >&2
        exit 78
    fi
done
if [[ ! -x "$python_bin" ]]; then
    echo "configured Python interpreter is not executable" >&2
    exit 69
fi
if [[ ! -d "$state_dir" || -L "$state_dir" ]]; then
    echo "state directory is unavailable" >&2
    exit 73
fi

if [[ ! -f "$release_root/tools/muneral_sync/cli.py" ]]; then
    echo "release payload is incomplete" >&2
    exit 70
fi

# Lock the StateDirectory inode itself. Its parent is root-owned, so the
# service user cannot replace it with a symlink between validation and open.
exec 9<"$state_dir"
if ! flock --nonblock 9; then
    echo "another Muneral KB sync is already running" >&2
    exit 75
fi

export PYTHONPATH="${release_root}/src:${release_root}"
exec "$python_bin" -m tools.muneral_sync.cli "$@" \
    --dsn-credential "$dsn_credential" \
    --writer-credential "$writer_credential" \
    --cursor-file "$cursor_file" \
    --endpoint "$endpoint"
