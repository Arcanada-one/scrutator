#!/usr/bin/env bash
# Start a DISPOSABLE PostgreSQL cluster (owned synthetic data only), run the KB-CONS-01-N native tests,
# stop the cluster and remove its data directory. Needs PostgreSQL server binaries WITH the pgvector
# extension: point KC_PG_BIN at their bin directory (a relocatable tree works).
#
#   KC_PG_BIN=/path/to/postgresql/16/bin tests/kc_generations/run_with_temp_pg.sh [pytest args...]
#
# It never touches another cluster: it picks a free port and a fresh mktemp directory.
set -euo pipefail
: "${KC_PG_BIN:?set KC_PG_BIN to a PostgreSQL bin directory that has pgvector installed}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d)"
PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
cleanup() { "$KC_PG_BIN/pg_ctl" -D "$WORK/data" -m immediate stop >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT
"$KC_PG_BIN/initdb" -D "$WORK/data" -U postgres --auth=trust -E UTF8 >/dev/null
"$KC_PG_BIN/pg_ctl" -D "$WORK/data" -o "-p $PORT -c listen_addresses=127.0.0.1 -c unix_socket_directories=$WORK -c fsync=off" \
  -l "$WORK/pg.log" -w start >/dev/null
export KC_TEST_PG_ADMIN_DSN="postgresql://postgres@127.0.0.1:$PORT/postgres"
cd "$REPO"
PYTHONPATH="$REPO/src:$REPO" python3 -m pytest tests/kc_generations tests/test_kc_router.py -p no:cacheprovider "$@"
