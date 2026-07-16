#!/usr/bin/env bash
set -euo pipefail

umask 077

readonly state_dir="${LTM_REFLECT_STATE_DIR:-/var/lib/scrutator/ltm-reflect}"
readonly state_file="${LTM_REFLECT_STATE_FILE:-${state_dir}/cursor.json}"
readonly namespace="${LTM_REFLECT_NAMESPACE:-wiki}"
readonly max_chunks="${LTM_REFLECT_MAX_CHUNKS:-50}"
readonly container_label="${SCRUTATOR_CONTAINER_LABEL:-com.docker.compose.service=scrutator}"
readonly container_name="${SCRUTATOR_CONTAINER_NAME:-}"

mkdir -p "$state_dir"
if [[ -L "$state_dir" || ! -d "$state_dir" ]]; then
  echo "state directory is unavailable" >&2
  exit 73
fi

exec 9<"$state_dir"
if ! flock --nonblock 9; then
  echo "another LTM reflect run is already active" >&2
  exit 75
fi

container="$container_name"
if [[ -z "$container" ]]; then
  container=$(docker ps --filter "label=${container_label}" --format "{{.Names}}" | head -1)
fi
if [[ -z "$container" ]]; then
  echo "scrutator container not found" >&2
  exit 69
fi

exec docker exec "$container" python -m scrutator.ltm.reflect_runner \
  --namespace "$namespace" \
  --state-file "$state_file" \
  --max-chunks "$max_chunks" "$@"
