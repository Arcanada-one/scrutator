#!/usr/bin/env bash
# Transactional Scrutator deploy with exact rollback of source, image, env hash, and KB timer state.
set -euo pipefail
IFS=$'\n\t'

readonly TIMER_UNITS=(
    kb-reconcile.timer
    kb-observe.timer
    kb-self-improvement-reconcile.timer
)
readonly RECONCILE_UNITS=(
    kb-reconcile.service
    kb-self-improvement-reconcile.service
)
readonly DEPLOY_SURFACE=(
    .github/workflows/ci.yml
    .github/workflows/deploy.yml
    .github/workflows/recall-regression.yml
    Dockerfile
    docker-compose.yml
    docker-compose.yaml
    compose.yml
    compose.yaml
    deploy
    scripts/deploy.sh
)

TARGET_SHA=""
PREVIOUS_SHA=""
PREVIOUS_IMAGE=""
ROLLBACK_TAG=""
CANDIDATE_TAG=""
ENV_SHA=""
ENV_MODE=""
ENV_UID=""
ENV_GID=""
ENV_DEVICE=""
ENV_INODE=""
CHECKPOINT=""
QUIESCED=0
MUTATION_STARTED=0
ACTIVE_TIMERS=()
SOURCE_RESTORED=0
ENV_RESTORED=0
IMAGE_RESTORED=0
TIMERS_RESTORED=0
TAGS_PRUNED=0

readonly PROJECT_ROOT="${SCRUTATOR_DEPLOY_ROOT:-/srv/apps/scrutator}"
readonly STATE_DIR="${SCRUTATOR_DEPLOY_STATE_DIR:-/var/lib/scrutator/deploy-state}"
readonly HEALTH_URL="${SCRUTATOR_DEPLOY_HEALTH_URL:-http://localhost:8310/health}"
readonly HEALTH_ATTEMPTS="${SCRUTATOR_DEPLOY_HEALTH_ATTEMPTS:-12}"
readonly HEALTH_INTERVAL="${SCRUTATOR_DEPLOY_HEALTH_INTERVAL_SECONDS:-5}"
readonly RETAIN_IMAGES="${SCRUTATOR_DEPLOY_RETAIN_IMAGES:-3}"
readonly REVIEWED_DEPLOY_SURFACE_SHA="${SCRUTATOR_REVIEWED_DEPLOY_SURFACE_SHA:-}"
readonly GITHUB_EVENT_NAME="${GITHUB_EVENT_NAME:-}"

log() {
    printf 'scrutator-deploy: %s\n' "$*"
}

die() {
    printf 'scrutator-deploy: ERROR: %s\n' "$*" >&2
    exit 1
}

validate_positive_integer() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$label must be a positive integer"
}

validate_nonnegative_integer() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$label must be a non-negative integer"
}

parse_args() {
    [[ "$#" -eq 2 && "$1" == "--target-sha" ]] || die "usage: $0 --target-sha <40-hex-sha>"
    TARGET_SHA="$2"
    [[ "$TARGET_SHA" =~ ^[0-9a-f]{40}$ ]] || die "target SHA must be 40 lowercase hexadecimal characters"
    validate_positive_integer "health attempts" "$HEALTH_ATTEMPTS"
    validate_nonnegative_integer "health interval" "$HEALTH_INTERVAL"
    validate_positive_integer "retained image count" "$RETAIN_IMAGES"
}

ensure_state_dir() {
    local mode
    [[ "$STATE_DIR" == /* && "$STATE_DIR" != "/" ]] || die "state directory must be a specific absolute path"
    [[ -d "$STATE_DIR" && ! -L "$STATE_DIR" ]] || die "preprovisioned state directory is missing or unsafe"
    [[ -O "$STATE_DIR" ]] || die "preprovisioned state directory is not owned by the deploy user"
    mode="$(stat -c '%a' "$STATE_DIR")"
    [[ "$mode" == "700" ]] || die "preprovisioned state directory must have mode 0700"
    [[ -w "$STATE_DIR" ]] || die "state directory is not writable"
}

validate_repository() {
    local branch remote_sha
    [[ -f .env && ! -L .env ]] || die ".env is missing or unsafe"
    [[ -z "$(git status --porcelain)" ]] || die "deployment checkout is dirty"
    branch="$(git symbolic-ref --quiet --short HEAD)"
    [[ "$branch" == "main" ]] || die "deployment checkout must be on main"
    PREVIOUS_SHA="$(git rev-parse HEAD)"
    [[ "$PREVIOUS_SHA" =~ ^[0-9a-f]{40}$ ]] || die "current source SHA is invalid"
    git fetch origin main
    remote_sha="$(git rev-parse origin/main)"
    [[ "$remote_sha" == "$TARGET_SHA" ]] || die "target SHA does not equal origin/main"
    git merge-base --is-ancestor "$PREVIOUS_SHA" "$TARGET_SHA" || die "deployment checkout diverged from target"
}

validate_deploy_surface() {
    local changed_output
    local -a changed=()
    changed_output="$(git diff --name-only "$PREVIOUS_SHA" "$TARGET_SHA" -- "${DEPLOY_SURFACE[@]}")" || \
        die "could not inspect deploy-surface changes"
    [[ -n "$changed_output" ]] && mapfile -t changed <<<"$changed_output"
    ((${#changed[@]} == 0)) && return 0
    [[ "$REVIEWED_DEPLOY_SURFACE_SHA" =~ ^[0-9a-f]{40}$ ]] || \
        die "deploy-surface change requires an externally reviewed exact target SHA"
    [[ "$REVIEWED_DEPLOY_SURFACE_SHA" == "$TARGET_SHA" ]] || \
        die "reviewed deploy-surface SHA does not match target"
    [[ "$GITHUB_EVENT_NAME" == "workflow_dispatch" ]] || \
        die "reviewed deploy-surface change requires workflow_dispatch"
    log "accepted SHA-attested manual deploy-surface change"
}

compose_project() {
    docker compose --project-directory "$PROJECT_ROOT" -f "$PROJECT_ROOT/docker-compose.yml" "$@"
}

unit_is_active() {
    local unit="$1"
    local status
    if sudo -n systemctl is-active --quiet "$unit"; then
        return 0
    else
        status=$?
    fi
    [[ "$status" -eq 3 ]] && return 1
    printf 'scrutator-deploy: ERROR: could not determine state of %s (status %s)\n' "$unit" "$status" >&2
    return 2
}

current_container_id() {
    local -a containers=()
    mapfile -t containers < <(compose_project ps -q scrutator)
    if ((${#containers[@]} != 1)) || [[ -z "${containers[0]:-}" ]]; then
        printf 'scrutator-deploy: ERROR: expected exactly one Scrutator container\n' >&2
        return 1
    fi
    printf '%s\n' "${containers[0]}"
}

canonical_image_id() {
    local reference="$1"
    local image_id
    image_id="$(docker image inspect --format '{{.Id}}' "$reference")" || return 1
    if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
        printf 'scrutator-deploy: ERROR: image reference did not resolve to one immutable ID\n' >&2
        return 1
    fi
    printf '%s\n' "$image_id"
}

capture_running_image() {
    local container_id
    container_id="$(current_container_id)"
    PREVIOUS_IMAGE="$(docker inspect --format '{{.Image}}' "$container_id")"
    PREVIOUS_IMAGE="$(canonical_image_id "$PREVIOUS_IMAGE")"
    ROLLBACK_TAG="scrutator-rollback:${PREVIOUS_SHA}"
    CANDIDATE_TAG="scrutator-deploy:${TARGET_SHA}"
}

ensure_immutable_tag() {
    local tag="$1"
    local image="$2"
    local existing=""
    if existing="$(docker image inspect --format '{{.Id}}' "$tag" 2>/dev/null)"; then
        [[ "$existing" == "$image" ]] || die "immutable tag $tag already points to another image"
        return 0
    fi
    docker image tag "$image" "$tag"
}

capture_raw_artifacts() {
    local path destination
    local -a paths=()
    mkdir -p "$CHECKPOINT/raw"
    mapfile -t paths < <(git ls-tree -r --name-only "$PREVIOUS_SHA" -- "${DEPLOY_SURFACE[@]:1}")
    ((${#paths[@]} > 0)) || die "no deploy artifacts found in previous source"
    for path in "${paths[@]}"; do
        case "$path" in
            /* | ../* | */../*) die "unsafe deploy artifact path" ;;
        esac
        destination="$CHECKPOINT/raw/$path"
        mkdir -p "$(dirname "$destination")"
        git show "$PREVIOUS_SHA:$path" >"$destination"
    done
    printf '%s\n' "${paths[@]}" >"$CHECKPOINT/raw-paths"
}

capture_timer_state() {
    local timer status
    : >"$CHECKPOINT/timers.active"
    for timer in "${TIMER_UNITS[@]}"; do
        if unit_is_active "$timer"; then
            ACTIVE_TIMERS+=("$timer")
            printf '%s\n' "$timer" >>"$CHECKPOINT/timers.active"
        else
            status=$?
            [[ "$status" -eq 1 ]] || die "timer-state probe failed"
        fi
    done
}

create_checkpoint() {
    ensure_state_dir
    CHECKPOINT="$(mktemp -d "$STATE_DIR/checkpoint.XXXXXXXX")"
    chmod 0700 "$CHECKPOINT"
    ENV_SHA="$(sha256sum .env | awk '{print $1}')"
    [[ "$ENV_SHA" =~ ^[0-9a-f]{64}$ ]] || die "could not hash .env"
    ENV_MODE="$(stat -c '%a' .env)"
    [[ "$ENV_MODE" =~ ^[0-7]{3,4}$ ]] || die "could not capture .env mode"
    ENV_UID="$(stat -c '%u' .env)"
    ENV_GID="$(stat -c '%g' .env)"
    ENV_DEVICE="$(stat -c '%d' .env)"
    ENV_INODE="$(stat -c '%i' .env)"
    [[ "$ENV_UID" =~ ^[0-9]+$ && "$ENV_GID" =~ ^[0-9]+$ ]] || die "could not capture .env ownership"
    [[ "$ENV_DEVICE" =~ ^[0-9]+$ && "$ENV_INODE" =~ ^[0-9]+$ ]] || die "could not capture .env identity"
    cp -- .env "$CHECKPOINT/env.snapshot"
    chmod 0600 "$CHECKPOINT/env.snapshot"
    [[ "$(sha256sum "$CHECKPOINT/env.snapshot" | awk '{print $1}')" == "$ENV_SHA" ]] || \
        die ".env checkpoint verification failed"
    capture_raw_artifacts
    capture_timer_state
    printf 'source_sha=%s\nimage_id=%s\nenv_sha256=%s\nenv_mode=%s\nenv_uid=%s\nenv_gid=%s\nenv_device=%s\nenv_inode=%s\n' \
        "$PREVIOUS_SHA" "$PREVIOUS_IMAGE" "$ENV_SHA" "$ENV_MODE" "$ENV_UID" "$ENV_GID" "$ENV_DEVICE" \
        "$ENV_INODE" >"$CHECKPOINT/metadata"
}

restore_timers() {
    local timer active_timer should_be_active actual status
    sudo -n systemctl stop "${TIMER_UNITS[@]}"
    if ((${#ACTIVE_TIMERS[@]})); then
        sudo -n systemctl start "${ACTIVE_TIMERS[@]}"
    fi
    for timer in "${TIMER_UNITS[@]}"; do
        should_be_active=0
        for active_timer in "${ACTIVE_TIMERS[@]}"; do
            if [[ "$active_timer" == "$timer" ]]; then
                should_be_active=1
                break
            fi
        done
        actual=0
        if unit_is_active "$timer"; then
            actual=1
        else
            status=$?
            [[ "$status" -eq 1 ]] || return 1
        fi
        [[ "$actual" -eq "$should_be_active" ]] || return 1
    done
}

quiesce_kb() {
    local service status
    QUIESCED=1
    sudo -n systemctl stop "${TIMER_UNITS[@]}"
    for service in "${RECONCILE_UNITS[@]}"; do
        if unit_is_active "$service"; then
            die "$service is active; refusing deploy while a reconcile ledger is open"
        else
            status=$?
            [[ "$status" -eq 1 ]] || die "reconcile-state probe failed"
        fi
    done
}

write_override() {
    local destination="$1"
    local tag="$2"
    if [[ ! "$tag" =~ ^scrutator-(deploy|rollback):[0-9a-f]{40}$ ]]; then
        printf 'scrutator-deploy: ERROR: unsafe image tag\n' >&2
        return 1
    fi
    printf 'services:\n  scrutator:\n    image: "%s"\n' "$tag" >"$destination"
}

wait_for_health() {
    local attempt
    for ((attempt = 1; attempt <= HEALTH_ATTEMPTS; attempt++)); do
        if curl -fsS --max-time 10 "$HEALTH_URL" >/dev/null; then
            return 0
        fi
        ((attempt < HEALTH_ATTEMPTS)) && sleep "$HEALTH_INTERVAL"
    done
    return 1
}

verify_running_image() {
    local expected="$1"
    local container_id actual
    container_id="$(current_container_id)" || return 1
    actual="$(docker inspect --format '{{.Image}}' "$container_id")" || return 1
    actual="$(canonical_image_id "$actual")" || return 1
    if [[ "$actual" != "$expected" ]]; then
        printf 'scrutator-deploy: ERROR: running container image does not match expected image\n' >&2
        return 1
    fi
}

verify_env_exact() {
    local actual_sha actual_metadata
    [[ -f .env && ! -L .env ]] || return 1
    actual_sha="$(sha256sum .env | awk '{print $1}')" || return 1
    actual_metadata="$(stat -c '%a:%u:%g:%d:%i' .env)" || return 1
    if [[ "$actual_sha" != "$ENV_SHA" || \
        "$actual_metadata" != "$ENV_MODE:$ENV_UID:$ENV_GID:$ENV_DEVICE:$ENV_INODE" ]]; then
        printf 'scrutator-deploy: ERROR: .env bytes, ownership, mode, or identity changed during deployment\n' >&2
        return 1
    fi
}

verify_raw_artifacts() {
    local path
    while IFS= read -r path; do
        [[ -f "$path" ]] || return 1
        cmp -s "$CHECKPOINT/raw/$path" "$path" || return 1
    done <"$CHECKPOINT/raw-paths"
}

deploy_candidate() {
    local candidate_image
    MUTATION_STARTED=1
    git reset --hard "$TARGET_SHA"
    deploy/ltm-reflect-state-preflight.sh
    write_override "$CHECKPOINT/candidate.override.yml" "$CANDIDATE_TAG"
    if docker image inspect "$CANDIDATE_TAG" >/dev/null 2>&1; then
        die "immutable candidate tag already exists"
    fi
    compose_project -f "$CHECKPOINT/candidate.override.yml" build scrutator
    candidate_image="$(canonical_image_id "$CANDIDATE_TAG")"
    record_image_tag "$CANDIDATE_TAG" || die "could not record candidate image tag"
    compose_project -f "$CHECKPOINT/candidate.override.yml" up -d --no-build scrutator
    wait_for_health || die "candidate failed bounded health checks"
    verify_running_image "$candidate_image"
    verify_env_exact
}

rollback_transaction() {
    local rollback_override="$CHECKPOINT/rollback.override.yml"
    local failed=0
    log "candidate failed; restoring checkpoint"
    if git reset --hard "$PREVIOUS_SHA" && [[ "$(git rev-parse HEAD)" == "$PREVIOUS_SHA" ]] && verify_raw_artifacts; then
        SOURCE_RESTORED=1
    else
        failed=1
    fi
    # The deploy contract forbids touching .env. Never replace a live root-owned
    # file from the runner: unchanged bytes keep their inode and metadata; any
    # mutation is a fatal, honestly reported rollback failure.
    if verify_env_exact; then
        ENV_RESTORED=1
    else
        failed=1
    fi
    if write_override "$rollback_override" "$ROLLBACK_TAG" \
        && [[ -f "$PROJECT_ROOT/docker-compose.yml" ]] \
        && compose_project -f "$rollback_override" up -d --no-build scrutator \
        && verify_running_image "$PREVIOUS_IMAGE" \
        && wait_for_health; then
        IMAGE_RESTORED=1
    else
        failed=1
    fi
    if ((failed == 0)); then
        log "rollback restored source, env, image, and container health"
        return 0
    fi
    return 1
}

write_outcome() {
    local outcome="$1"
    local temporary="$CHECKPOINT/outcome.next"
    [[ -n "$CHECKPOINT" && -d "$CHECKPOINT" ]] || return 1
    printf 'outcome=%s\nsource_restored=%s\nenv_restored=%s\nimage_restored=%s\ntimers_restored=%s\ntags_pruned=%s\n' \
        "$outcome" "$SOURCE_RESTORED" "$ENV_RESTORED" "$IMAGE_RESTORED" "$TIMERS_RESTORED" "$TAGS_PRUNED" \
        >"$temporary" || return 1
    chmod 0600 "$temporary" || return 1
    mv -f -- "$temporary" "$CHECKPOINT/outcome"
}

record_image_tag() {
    local tag="$1"
    local history="$STATE_DIR/image-tags"
    local temporary="$STATE_DIR/image-tags.next"
    local existing
    [[ "$tag" =~ ^scrutator-(deploy|rollback):[0-9a-f]{40}$ ]] || return 1
    : >"$temporary" || return 1
    printf '%s\n' "$tag" >>"$temporary" || return 1
    if [[ -f "$history" ]]; then
        while IFS= read -r existing; do
            [[ -z "$existing" || "$existing" == "$tag" ]] && continue
            [[ "$existing" =~ ^scrutator-(deploy|rollback):[0-9a-f]{40}$ ]] || return 1
            printf '%s\n' "$existing" >>"$temporary" || return 1
        done <"$history"
    fi
    chmod 0600 "$temporary" || return 1
    mv -f -- "$temporary" "$history"
}

prune_image_tags() {
    local history="$STATE_DIR/image-tags"
    local temporary="$STATE_DIR/image-tags.pruned"
    local container_id running_image tag image_id preferred designated=""
    local deploy_count=0 rollback_count=0 failed=0
    [[ -f "$history" ]] || return 0
    container_id="$(current_container_id)" || return 1
    running_image="$(docker inspect --format '{{.Image}}' "$container_id")" || return 1
    running_image="$(canonical_image_id "$running_image")" || return 1
    for preferred in "$CANDIDATE_TAG" "$ROLLBACK_TAG"; do
        [[ "$preferred" =~ ^scrutator-(deploy|rollback):[0-9a-f]{40}$ ]] || continue
        image_id="$(docker image inspect --format '{{.Id}}' "$preferred" 2>/dev/null)" || continue
        if [[ "$image_id" == "$running_image" ]]; then
            designated="$preferred"
            break
        fi
    done
    if [[ -z "$designated" ]]; then
        while IFS= read -r tag; do
            [[ "$tag" =~ ^scrutator-(deploy|rollback):[0-9a-f]{40}$ ]] || continue
            image_id="$(docker image inspect --format '{{.Id}}' "$tag" 2>/dev/null)" || continue
            if [[ "$image_id" == "$running_image" ]]; then
                designated="$tag"
                break
            fi
        done <"$history"
    fi
    [[ -n "$designated" ]] || return 1
    : >"$temporary" || return 1
    printf '%s\n' "$designated" >>"$temporary" || return 1
    if [[ "$designated" == scrutator-deploy:* ]]; then
        deploy_count=1
    else
        rollback_count=1
    fi
    while IFS= read -r tag; do
        [[ "$tag" =~ ^scrutator-(deploy|rollback):[0-9a-f]{40}$ ]] || {
            failed=1
            continue
        }
        [[ "$tag" == "$designated" ]] && continue
        image_id="$(docker image inspect --format '{{.Id}}' "$tag" 2>/dev/null)" || continue
        if [[ "$tag" == scrutator-deploy:* ]] && ((deploy_count < RETAIN_IMAGES)); then
            printf '%s\n' "$tag" >>"$temporary"
            ((deploy_count += 1))
        elif [[ "$tag" == scrutator-rollback:* ]] && ((rollback_count < RETAIN_IMAGES)); then
            printf '%s\n' "$tag" >>"$temporary"
            ((rollback_count += 1))
        elif ! docker image rm "$tag"; then
            printf '%s\n' "$tag" >>"$temporary"
            failed=1
        fi
    done <"$history"
    chmod 0600 "$temporary" || return 1
    mv -f -- "$temporary" "$history" || return 1
    image_id="$(docker image inspect --format '{{.Id}}' "$designated" 2>/dev/null)" || return 1
    [[ "$image_id" == "$running_image" ]] || return 1
    return "$failed"
}

handle_exit() {
    local original_status="$?"
    local rollback_status=0 timer_status=0 cleanup_status=0
    trap - EXIT INT TERM
    set +e
    if ((original_status != 0 && MUTATION_STARTED)); then
        rollback_transaction
        rollback_status=$?
    fi
    if ((QUIESCED)); then
        if restore_timers; then
            TIMERS_RESTORED=1
        else
            timer_status=$?
        fi
    else
        TIMERS_RESTORED=1
    fi
    if [[ -d "$STATE_DIR" ]]; then
        if docker image inspect "$CANDIDATE_TAG" >/dev/null 2>&1; then
            record_image_tag "$CANDIDATE_TAG" || cleanup_status=1
        fi
        record_image_tag "$ROLLBACK_TAG" || cleanup_status=1
        if prune_image_tags; then
            TAGS_PRUNED=1
        else
            cleanup_status=1
        fi
    fi
    if ((original_status != 0)) && [[ -n "$CHECKPOINT" && -d "$CHECKPOINT" ]]; then
        if ((MUTATION_STARTED)); then
            if ((rollback_status == 0 && timer_status == 0)); then
                write_outcome "rolled_back" || cleanup_status=1
            else
                write_outcome "rollback_failed" || cleanup_status=1
            fi
        else
            SOURCE_RESTORED=1
            ENV_RESTORED=1
            IMAGE_RESTORED=1
            write_outcome "failed_before_mutation" || cleanup_status=1
        fi
    fi
    if ((rollback_status != 0 || timer_status != 0 || cleanup_status != 0)); then
        printf 'scrutator-deploy: FATAL: rollback or finalization cleanup failed\n' >&2
        exit 90
    fi
    exit "$original_status"
}

main() {
    parse_args "$@"
    cd "$PROJECT_ROOT"
    validate_repository
    validate_deploy_surface
    capture_running_image
    trap handle_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    ensure_state_dir
    ensure_immutable_tag "$ROLLBACK_TAG" "$PREVIOUS_IMAGE"
    record_image_tag "$ROLLBACK_TAG" || die "could not record rollback image tag"
    create_checkpoint
    quiesce_kb
    deploy_candidate
    prune_image_tags || die "could not enforce bounded deploy image retention"
    TAGS_PRUNED=1
    restore_timers
    QUIESCED=0
    MUTATION_STARTED=0
    rm -rf -- "$CHECKPOINT"
    log "deployed $TARGET_SHA with exact timer-state restoration"
}

main "$@"
