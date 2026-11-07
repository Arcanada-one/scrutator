#!/usr/bin/env bash
# Start a DISPOSABLE PostgreSQL cluster (owned synthetic data only), run the KB-CONS-01-N native tests,
# stop the cluster and remove its data directory. Needs PostgreSQL server binaries WITH the pgvector
# extension: point KC_PG_BIN at their bin directory (a relocatable tree works).
#
#   KC_PG_BIN=/path/to/postgresql/16/bin tests/kc_generations/run_with_temp_pg.sh [pytest args...]
#
# On CI runners that already have Docker (lint-and-test), an equivalent disposable cluster can be
# started from a pgvector image instead of host binaries:
#
#   KC_PG_DOCKER_IMAGE=pgvector/pgvector:pg16 tests/kc_generations/run_with_temp_pg.sh [pytest args...]
#
# It never touches another cluster: host-bin mode picks a free port and a fresh mktemp directory;
# docker mode uses a one-shot container bound to 127.0.0.1 on an ephemeral port.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

start_host_bin_cluster() {
  : "${KC_PG_BIN:?set KC_PG_BIN to a PostgreSQL bin directory that has pgvector installed}"
  WORK="$(mktemp -d)"
  PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  cleanup() { "$KC_PG_BIN/pg_ctl" -D "$WORK/data" -m immediate stop >/dev/null 2>&1 || true; rm -rf "$WORK"; }
  trap cleanup EXIT
  "$KC_PG_BIN/initdb" -D "$WORK/data" -U postgres --auth=trust -E UTF8 >/dev/null
  "$KC_PG_BIN/pg_ctl" -D "$WORK/data" -o "-p $PORT -c listen_addresses=127.0.0.1 -c unix_socket_directories=$WORK -c fsync=off" \
    -l "$WORK/pg.log" -w start >/dev/null
  export KC_TEST_PG_ADMIN_DSN="postgresql://postgres@127.0.0.1:$PORT/postgres"
}

start_docker_cluster() {
  : "${KC_PG_DOCKER_IMAGE:?set KC_PG_DOCKER_IMAGE to a pgvector PostgreSQL image}"
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required when KC_PG_DOCKER_IMAGE is set" >&2
    exit 1
  fi
  NAME="kc-temp-pg-$$-${RANDOM:-0}"
  PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
  cleanup() {
    docker rm -f "$NAME" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  echo "run_with_temp_pg: pulling $KC_PG_DOCKER_IMAGE" >&2
  docker pull "$KC_PG_DOCKER_IMAGE"
  echo "run_with_temp_pg: starting disposable container $NAME on 127.0.0.1:$PORT" >&2
  # Explicit host port (not docker's empty :: ephemeral form): clearer errors on shared runners.
  # POSTGRES_PASSWORD is set even with trust so entrypoint variants that require it still boot.
  if ! docker run -d --name "$NAME" \
      -e POSTGRES_HOST_AUTH_METHOD=trust \
      -e POSTGRES_PASSWORD=postgres \
      -p "127.0.0.1:${PORT}:5432" \
      "$KC_PG_DOCKER_IMAGE"; then
    echo "run_with_temp_pg: docker run failed" >&2
    exit 1
  fi
  ready=0
  for _ in $(seq 1 150); do
    if docker exec "$NAME" pg_isready -U postgres >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 0.1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "run_with_temp_pg: PostgreSQL container did not become ready" >&2
    docker logs "$NAME" >&2 || true
    exit 1
  fi
  export KC_TEST_PG_ADMIN_DSN="postgresql://postgres@127.0.0.1:${PORT}/postgres"
  echo "run_with_temp_pg: disposable DSN ready (port $PORT)" >&2
}

if [[ -n "${KC_PG_BIN:-}" ]]; then
  start_host_bin_cluster
elif [[ -n "${KC_PG_DOCKER_IMAGE:-}" ]]; then
  start_docker_cluster
else
  echo "set KC_PG_BIN (host binaries with pgvector) or KC_PG_DOCKER_IMAGE (disposable container)" >&2
  exit 1
fi

cd "$REPO"
echo "run_with_temp_pg: executing native pytest suite" >&2
PYTHONPATH="$REPO/src:$REPO" python3 -m pytest tests/kc_generations tests/test_kc_router.py -p no:cacheprovider "$@"
