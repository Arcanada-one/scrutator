#!/usr/bin/env bash
set -euo pipefail

readonly state_dir="${1:-/var/lib/scrutator/ltm-reflect}"
readonly expected_uid="${LTM_REFLECT_STATE_EXPECTED_UID:-0}"
readonly expected_gid="${LTM_REFLECT_STATE_EXPECTED_GID:-0}"

if [[ -L "$state_dir" || ! -d "$state_dir" ]]; then
  echo "reflect state path must be an existing directory, not a symlink: $state_dir" >&2
  exit 73
fi

resolved=$(realpath -e -- "$state_dir")
if [[ "$resolved" != "$state_dir" ]]; then
  echo "reflect state path contains a symlink: $state_dir -> $resolved" >&2
  exit 73
fi

metadata=$(stat -c '%u:%g:%a' -- "$state_dir")
if [[ "$metadata" != "${expected_uid}:${expected_gid}:700" ]]; then
  echo "reflect state path must be ${expected_uid}:${expected_gid} mode 700; found $metadata" >&2
  exit 77
fi
