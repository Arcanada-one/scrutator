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
off_container="kb-enh-srch0031-off"
on_container="kb-enh-srch0031-on"
off_port="18310"
on_port="18311"

if ! [[ "$image" =~ ^scrutator-deploy:[0-9a-f]{40}$ ]]; then
    echo "image must be an immutable scrutator-deploy:<40-hex-sha> tag" >&2
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

umask 077
mkdir -p -- "$output_dir"
run_tmp="$(mktemp -d /var/tmp/srch0031.XXXXXX)"
env_file="$run_tmp/benchmark.env"

cleanup() {
    docker rm -f "$on_container" "$off_container" >/dev/null 2>&1 || true
    rm -rf -- "$run_tmp"
}
trap cleanup EXIT

for command in docker python3 curl sha256sum; do
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

python3 - "$output_dir/manifest.json" "$image" "$image_id" "$production_image" "$production_id_before" "$env_hash" "$golden_hash" <<'PY'
import json
import os
import sys

target, image, image_id, production_image, production_id, env_hash, golden_hash = sys.argv[1:]
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "image": image,
            "image_id": image_id,
            "production_image": production_image,
            "production_container_id_before": production_id,
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

docker run -d --rm \
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
    --host 127.0.0.1 --port "$off_port" --lifespan off >"$run_tmp/off.id"

docker run -d --rm \
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
    --host 127.0.0.1 --port "$on_port" --lifespan off >"$run_tmp/on.id"

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

curl -fsS --max-time 10 "http://127.0.0.1:$off_port/v1/namespaces" >"$output_dir/off-auth-probe.json"
curl -fsS --max-time 10 "http://127.0.0.1:$on_port/v1/namespaces" >"$output_dir/on-auth-probe.json"

set +e
python3 "$source_dir/rerank_gate.py" \
    --off-endpoint "http://127.0.0.1:$off_port" \
    --on-endpoint "http://127.0.0.1:$on_port" \
    --golden "$source_dir/golden-arcanada-v0.jsonl" \
    --out-dir "$output_dir" \
    --repeats 3 \
    --timeout 45 \
    --latency-p95-budget-ms 5000 >"$output_dir/verdict.stdout" 2>"$output_dir/verdict.stderr"
measurement_rc="$?"
set -e

docker logs "$off_container" >"$output_dir/off-container.log" 2>&1
docker logs "$on_container" >"$output_dir/on-container.log" 2>&1
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

exit "$measurement_rc"
