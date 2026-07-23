#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 <scrutator-image-tag> <benchmark-source-dir> <output-dir>" >&2
    exit 2
fi

image="$1"
source_dir="$2"
output_dir="$3"
production_container="${SCRUTATOR_PRODUCTION_CONTAINER:-scrutator-scrutator-1}"
benchmark_scope="${SCRUTATOR_BENCHMARK_SCOPE:-deployed}"
off_port="18310"
on_port="18311"
off_id=""
on_id=""

if ! [[ "$image" =~ ^scrutator-deploy:[0-9a-f]{40}$ ]]; then
    echo "image must be an immutable scrutator-deploy:<40-hex-sha> tag" >&2
    exit 2
fi
if [[ "$benchmark_scope" != "deployed" && "$benchmark_scope" != "candidate" ]]; then
    echo "SCRUTATOR_BENCHMARK_SCOPE must be deployed or candidate" >&2
    exit 2
fi
if [[ ! -f "$source_dir/rerank_gate.py" || ! -f "$source_dir/golden-arcanada-v0.jsonl" ]]; then
    echo "benchmark source directory is incomplete" >&2
    exit 2
fi
if [[ ! -f "$source_dir/live/granted_context_app.py" ]]; then
    echo "granted-context wrapper is missing" >&2
    exit 2
fi

for lock_command in flock id stat; do
    if ! command -v "$lock_command" >/dev/null; then
        echo "$lock_command is required for safe ownership of the fixed benchmark ports" >&2
        exit 2
    fi
done
umask 077
runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
lock_file="${SCRUTATOR_BENCHMARK_LOCK_FILE:-$runtime_dir/kb-enh-srch0031.lock}"
if [[ "$lock_file" != /* ]]; then
    echo "SCRUTATOR_BENCHMARK_LOCK_FILE must be absolute" >&2
    exit 2
fi
lock_dir="${lock_file%/*}"
current_uid="$(id -u)"
if [[ ! -d "$lock_dir" || -L "$lock_dir" ]]; then
    echo "benchmark lock directory must be a real directory" >&2
    exit 2
fi
lock_dir_uid="$(stat -c %u -- "$lock_dir")"
lock_dir_mode="$(stat -c %a -- "$lock_dir")"
if [[ "$lock_dir_uid" != "$current_uid" || $((8#$lock_dir_mode & 077)) -ne 0 ]]; then
    echo "benchmark lock directory must be owner-private" >&2
    exit 2
fi
if [[ -L "$lock_file" ]]; then
    echo "lock path must not be a symlink" >&2
    exit 2
fi
if [[ -e "$lock_file" ]]; then
    lock_uid="$(stat -c %u -- "$lock_file")"
    lock_mode="$(stat -c %a -- "$lock_file")"
    if [[ ! -f "$lock_file" || "$lock_uid" != "$current_uid" || $((8#$lock_mode & 077)) -ne 0 ]]; then
        echo "existing lock must be an owner-private regular file" >&2
        exit 2
    fi
fi
if ! exec 9>"$lock_file"; then
    echo "cannot open the SRCH-0031 benchmark lock" >&2
    exit 2
fi
if ! flock -n 9; then
    echo "another SRCH-0031 benchmark owns the fixed ports" >&2
    exit 2
fi

mkdir -p -- "$output_dir"
run_tmp="$(mktemp -d /var/tmp/srch0031.XXXXXX)"
run_suffix="${run_tmp##*.}"
off_container="kb-enh-srch0031-off-${run_suffix}"
on_container="kb-enh-srch0031-on-${run_suffix}"
env_file="$run_tmp/benchmark.env"

cleanup() {
    if [[ -n "$on_id" ]]; then
        docker rm -f "$on_id" >/dev/null 2>&1 || true
    fi
    if [[ -n "$off_id" ]]; then
        docker rm -f "$off_id" >/dev/null 2>&1 || true
    fi
    rm -rf -- "$run_tmp"
}
trap cleanup EXIT

unexpected_failure() {
    trap - ERR
    echo "unexpected shell failure; evidence is invalid" >&2
    exit 2
}
trap unexpected_failure ERR

for command in docker python3 curl sha256sum flock id stat; do
    command -v "$command" >/dev/null
done

if ss -H -ltn "sport = :$off_port or sport = :$on_port" | grep -q .; then
    echo "benchmark loopback port already in use" >&2
    exit 2
fi

production_image="$(docker inspect --format '{{.Config.Image}}' "$production_container")"
if ! [[ "$production_image" =~ ^scrutator-deploy:[0-9a-f]{40}$ ]]; then
    echo "production container does not use an immutable image tag" >&2
    exit 2
fi
if [[ "$benchmark_scope" == "deployed" && "$production_image" != "$image" ]]; then
    echo "deployed-scope image does not match the production image" >&2
    exit 2
fi
production_id_before="$(docker inspect --format '{{.Id}}' "$production_container")"
docker inspect "$production_container" >"$run_tmp/production-inspect.json"

python3 - "$run_tmp/production-inspect.json" "$env_file" <<'PY'
import json
import os
import sys

source, target = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))[0]
environment = {}
for item in payload["Config"].get("Env", []):
    key, _, value = item.partition("=")
    environment[key] = value

overrides = {
    "PYTHONPATH": "/benchmark:/app/src",
    "SCRUTATOR_HOST": "127.0.0.1",
    "SCRUTATOR_DATABASE_POOL_MIN": "1",
    "SCRUTATOR_DATABASE_POOL_MAX": "2",
    "SCRUTATOR_BENCHMARK_PRINCIPAL": "kb-observer",
    "SCRUTATOR_LTM_WRITER_TOKEN": "",
    "SCRUTATOR_FEEDER_TOKEN": "",
    "SCRUTATOR_ROLLBACK_TOKEN": "",
    "SCRUTATOR_OPERATOR_ROLLBACK_TOKEN": "",
}
environment.update(overrides)

descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    for key in sorted(environment):
        stream.write(f"{key}={environment[key]}\n")
PY

env_hash="$(sha256sum "$env_file" | awk '{print $1}')"
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
golden_hash="$(sha256sum "$source_dir/golden-arcanada-v0.jsonl" | awk '{print $1}')"

python3 - "$output_dir/manifest.json" "$image" "$image_id" "$production_image" "$production_id_before" "$env_hash" "$golden_hash" "$benchmark_scope" <<'PY'
import json
import os
import sys

target, image, image_id, production_image, production_id, env_hash, golden_hash, benchmark_scope = sys.argv[1:]
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "image": image,
            "image_id": image_id,
            "production_image": production_image,
            "production_container_id_before": production_id,
            "eligibility_scope": benchmark_scope,
            "redacted_environment_sha256": env_hash,
            "golden_sha256": golden_hash,
            "listeners": ["127.0.0.1:18310", "127.0.0.1:18311"],
            "lifespan": "off",
            "benchmark_principal": "kb-observer",
        },
        stream,
        indent=2,
    )
    stream.write("\n")
PY

off_id="$(docker run -d --rm \
    --name "$off_container" \
    --network host \
    --env-file "$env_file" \
    -e SCRUTATOR_PORT="$off_port" \
    -e SCRUTATOR_RERANK_ENABLED=false \
    -e SCRUTATOR_DATABASE_POOL_MIN=1 \
    -e SCRUTATOR_DATABASE_POOL_MAX=2 \
    -e SCRUTATOR_LTM_WRITER_TOKEN= \
    -e SCRUTATOR_FEEDER_TOKEN= \
    -e SCRUTATOR_ROLLBACK_TOKEN= \
    -e SCRUTATOR_OPERATOR_ROLLBACK_TOKEN= \
    -v "$source_dir:/benchmark:ro" \
    "$image" \
    uvicorn granted_context_app:app --app-dir /benchmark/live \
    --host 127.0.0.1 --port "$off_port" --lifespan off)"

on_id="$(docker run -d --rm \
    --name "$on_container" \
    --network host \
    --env-file "$env_file" \
    -e SCRUTATOR_PORT="$on_port" \
    -e SCRUTATOR_RERANK_ENABLED=true \
    -e SCRUTATOR_DATABASE_POOL_MIN=1 \
    -e SCRUTATOR_DATABASE_POOL_MAX=2 \
    -e SCRUTATOR_LTM_WRITER_TOKEN= \
    -e SCRUTATOR_FEEDER_TOKEN= \
    -e SCRUTATOR_ROLLBACK_TOKEN= \
    -e SCRUTATOR_OPERATOR_ROLLBACK_TOKEN= \
    -v "$source_dir:/benchmark:ro" \
    "$image" \
    uvicorn granted_context_app:app --app-dir /benchmark/live \
    --host 127.0.0.1 --port "$on_port" --lifespan off)"

for port in "$off_port" "$on_port"; do
    ready=0
    for _ in $(seq 1 30); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
            ready=1
            break
        fi
        sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "benchmark listener on port $port did not become healthy" >&2
        exit 2
    fi
done

for mode_port in "off:$off_port" "on:$on_port"; do
    mode="${mode_port%%:*}"
    port="${mode_port##*:}"
    raw_probe="$run_tmp/${mode}-namespaces.json"
    derived_probe="$output_dir/${mode}-auth-probe.json"
    curl -fsS --max-time 10 "http://127.0.0.1:$port/v1/namespaces" >"$raw_probe"
    python3 - "$raw_probe" "$derived_probe" <<'PY'
import json
import os
import sys

source, target = sys.argv[1:]
namespaces = json.load(open(source, encoding="utf-8"))
authorized = any(isinstance(item, dict) and item.get("name") == "arcanada" for item in namespaces)
if not authorized:
    raise SystemExit("benchmark principal is not authorized for arcanada")
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(
        {"principal": "kb-observer", "namespace": "arcanada", "authorized": True},
        stream,
        sort_keys=True,
    )
    stream.write("\n")
PY
done

measurement_rc=0
python3 "$source_dir/rerank_gate.py" \
    --off-endpoint "http://127.0.0.1:$off_port" \
    --on-endpoint "http://127.0.0.1:$on_port" \
    --golden "$source_dir/golden-arcanada-v0.jsonl" \
    --out-dir "$output_dir" \
    --repeats 3 \
    --timeout 45 \
    --latency-p95-budget-ms 5000 \
    --eligibility-scope "$benchmark_scope" >"$output_dir/verdict.stdout" 2>"$output_dir/verdict.stderr" \
    || measurement_rc="$?"

if [[ "$measurement_rc" -eq 0 || "$measurement_rc" -eq 1 ]]; then
    if [[ ! -s "$output_dir/summary.json" ]]; then
        echo "benchmark exited $measurement_rc without a summary; evidence is invalid" >&2
        measurement_rc=2
    elif ! python3 - "$output_dir/summary.json" "$measurement_rc" "$benchmark_scope" <<'PY'
import json
import sys

path, encoded_rc, scope = sys.argv[1:]
status = json.load(open(path, encoding="utf-8")).get("verdict", {}).get("status")
expected_eligible = "ELIGIBLE_TO_FLIP" if scope == "deployed" else "CANDIDATE_ELIGIBLE"
valid = (int(encoded_rc) == 0 and status == expected_eligible) or (int(encoded_rc) == 1 and status == "KEEP_OFF")
raise SystemExit(0 if valid else 1)
PY
    then
        echo "benchmark exit code and summary verdict disagree; evidence is invalid" >&2
        measurement_rc=2
    fi
fi

docker logs "$off_id" >"$output_dir/off-container.log" 2>&1
docker logs "$on_id" >"$output_dir/on-container.log" 2>&1
if grep -Eiq 'ColBERT rerank (failed|unexpected error)' "$output_dir/on-container.log"; then
    echo "ON container logged a ColBERT soft failure" >&2
    exit 2
fi

production_id_after="$(docker inspect --format '{{.Id}}' "$production_container")"
if [[ "$production_id_before" != "$production_id_after" ]]; then
    echo "production container identity changed during benchmark" >&2
    exit 2
fi
curl -fsS --max-time 10 http://127.0.0.1:8310/health >"$output_dir/production-health-after.json"

trap - ERR
exit "$measurement_rc"
