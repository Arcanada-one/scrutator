#!/usr/bin/env bash
set -euo pipefail

umask 077

readonly service_name="muneral-kb-sync"
readonly pilot_task_id="7d2c0e8a-4b7e-4c01-8d33-100000000004"
readonly manifest_name="PAYLOAD_SHA256SUMS"

usage() {
    echo "usage: $0 install --sha SHA --source DIR [--root DIR] [--enable-timer-after-pilot]" >&2
    echo "       $0 rollback [--root DIR]" >&2
    exit 64
}

[[ $# -ge 1 ]] || usage
action="$1"
shift

root="/"
release_sha=""
source_root=""
enable_timer=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            [[ $# -ge 2 ]] || usage
            root="$2"
            shift 2
            ;;
        --sha)
            [[ $# -ge 2 ]] || usage
            release_sha="$2"
            shift 2
            ;;
        --source)
            [[ $# -ge 2 ]] || usage
            source_root="$2"
            shift 2
            ;;
        --enable-timer-after-pilot)
            enable_timer=true
            shift
            ;;
        *) usage ;;
    esac
done

[[ "$root" == /* && "$root" != *$'\n'* ]] || usage
root="${root%/}"
[[ -n "$root" ]] || root="/"
root="$(realpath -m -- "$root")"

root_path() {
    if [[ "$root" == "/" ]]; then
        printf '%s\n' "$1"
    else
        printf '%s%s\n' "$root" "$1"
    fi
}

opt_root="$(root_path /opt/muneral-kb-sync)"
readonly opt_root
readonly releases_dir="$opt_root/releases"
state_base="$(root_path /var/lib/muneral-kb-sync)"
readonly state_base
state_dir="$state_base/runtime"
readonly state_dir
credential_dir="$(root_path /etc/muneral-kb-sync)"
readonly credential_dir
unit_dir="$(root_path /etc/systemd/system)"
readonly unit_dir
lock_dir="$(root_path /run/muneral-kb-sync-install)"
readonly lock_dir
readonly lock_file="$lock_dir/transaction.lock"
staging=""
created_release=""
activation_started=false
transaction_snapshot_started=false
transaction_backup=""
timer_was_enabled=false
timer_was_active=false
original_current=""
original_previous=""
original_current_present=false
original_previous_present=false

cleanup_staging() {
    local status=$?
    if [[ $status -ne 0 ]]; then
        if [[ "$transaction_snapshot_started" == true ]]; then
            restore_transaction_state || true
        fi
        if [[ "$activation_started" == true ]]; then
            if [[ "$original_current_present" == true ]]; then
                atomic_link "$original_current" "$opt_root/current"
            else
                rm -f -- "$opt_root/current"
            fi
            if [[ "$original_previous_present" == true ]]; then
                atomic_link "$original_previous" "$opt_root/previous"
            else
                rm -f -- "$opt_root/previous"
            fi
        fi
        if [[ -n "$staging" && -d "$staging" ]]; then
            chmod -R u+w "$staging" 2>/dev/null || true
            rm -rf -- "$staging"
        fi
        if [[ -n "$created_release" && -d "$created_release" ]]; then
            chmod -R u+w "$created_release"
            rm -rf -- "$created_release"
        fi
    fi
    if [[ -n "$transaction_backup" && -d "$transaction_backup" ]]; then
        rm -rf -- "$transaction_backup"
    fi
    return "$status"
}
trap cleanup_staging EXIT

if [[ "$root" == "/" ]]; then
    [[ $EUID -eq 0 ]] || { echo "production installation requires root" >&2; exit 77; }
    PATH=/usr/sbin:/usr/bin:/sbin:/bin
    export PATH
fi

atomic_link() {
    local target="$1"
    local link="$2"
    local temporary="${link}.tmp.$$"
    ln -s -- "$target" "$temporary"
    mv -Tf -- "$temporary" "$link"
}

expected_root_uid() {
    if [[ "$root" == "/" ]]; then printf '0\n'; else id -u; fi
}

validate_secure_directory() {
    local path="$1" expected_uid="$2" mode owner
    [[ -d "$path" && ! -L "$path" ]] || return 1
    owner="$(stat -c '%u' "$path")"
    mode=$((8#$(stat -c '%a' "$path")))
    [[ "$owner" == "$expected_uid" && $((mode & 8#022)) -eq 0 ]]
}

validate_existing_canonical_paths() {
    local expected_uid path
    expected_uid="$(expected_root_uid)"
    for path in "$opt_root" "$releases_dir" "$state_base" "$credential_dir" "$unit_dir" "$lock_dir"; do
        if [[ -e "$path" || -L "$path" ]]; then
            validate_secure_directory "$path" "$expected_uid" || return 1
        fi
    done
}

validate_service_account() {
    local passwd_record group_record uid gid primary_gid home shell supplementary
    group_record="$(getent group "$service_name")" || return 1
    IFS=: read -r _ _ gid _ <<<"$group_record"
    passwd_record="$(getent passwd "$service_name")" || return 1
    IFS=: read -r _ _ uid primary_gid _ home shell <<<"$passwd_record"
    [[ "$uid" =~ ^[0-9]+$ && "$gid" =~ ^[0-9]+$ && "$primary_gid" =~ ^[0-9]+$ ]] || return 1
    [[ $uid -ne 0 && $gid -ne 0 && "$primary_gid" == "$gid" ]] || return 1
    [[ "$home" == "/var/lib/$service_name" && "$shell" == "/usr/sbin/nologin" ]] || return 1
    supplementary="$(id -G "$service_name")" || return 1
    [[ "$supplementary" == "$primary_gid" ]]
}

acquire_global_lock() {
    local expected_uid mode owner
    expected_uid="$(expected_root_uid)"
    if [[ -e "$lock_dir" || -L "$lock_dir" ]]; then
        validate_secure_directory "$lock_dir" "$expected_uid" || {
            echo "installer lock directory is insecure" >&2
            exit 73
        }
    else
        install -d -m 0700 "$lock_dir"
    fi
    [[ ! -L "$lock_file" ]] || { echo "installer lock file must not be a symlink" >&2; exit 73; }
    exec 8>"$lock_file"
    chmod 0600 "$lock_file"
    owner="$(stat -c '%u' "$lock_file")"
    mode="$(stat -c '%a' "$lock_file")"
    [[ "$owner" == "$expected_uid" && "$mode" == "600" ]] || exit 73
    flock --nonblock 8 || { echo "another install or rollback transaction is running" >&2; exit 75; }
    if [[ "$root" != "/" && -n "${MUNERAL_KB_SYNC_TEST_HOLD_LOCK_FILE:-}" ]]; then
        : >"${MUNERAL_KB_SYNC_TEST_HOLD_LOCK_FILE}.ready"
        while [[ -e "${MUNERAL_KB_SYNC_TEST_HOLD_LOCK_FILE}" ]]; do sleep 0.05; done
    fi
}

snapshot_transaction_state() {
    transaction_backup="$opt_root/.transaction-backup.$$"
    install -d -m 0700 "$transaction_backup"
    for unit in "$service_name.service" "$service_name.timer"; do
        [[ ! -L "$unit_dir/$unit" ]] || { echo "unit file must not be a symlink" >&2; return 73; }
        if [[ -f "$unit_dir/$unit" && ! -L "$unit_dir/$unit" ]]; then
            cp --preserve=mode,timestamps -- "$unit_dir/$unit" "$transaction_backup/$unit"
        fi
    done
    if [[ "$root" == "/" ]]; then
        if systemctl is-enabled --quiet "$service_name.timer"; then timer_was_enabled=true; fi
        if systemctl is-active --quiet "$service_name.timer"; then timer_was_active=true; fi
    else
        [[ -e "$state_dir/timer-enabled" ]] && timer_was_enabled=true
        [[ -e "$state_dir/timer-active" ]] && timer_was_active=true
    fi
    transaction_snapshot_started=true
}

restore_transaction_state() {
    local unit
    for unit in "$service_name.service" "$service_name.timer"; do
        if [[ -f "$transaction_backup/$unit" ]]; then
            install -m "$(stat -c '%a' "$transaction_backup/$unit")" "$transaction_backup/$unit" "$unit_dir/$unit"
        else
            rm -f -- "$unit_dir/$unit"
        fi
    done
    if [[ "$root" == "/" ]]; then
        systemctl daemon-reload
        if [[ "$timer_was_enabled" == true ]]; then systemctl enable "$service_name.timer"; else systemctl disable "$service_name.timer"; fi
        if [[ "$timer_was_active" == true ]]; then systemctl start "$service_name.timer"; else systemctl stop "$service_name.timer"; fi
    else
        if [[ "$timer_was_enabled" == true ]]; then : >"$state_dir/timer-enabled"; else rm -f -- "$state_dir/timer-enabled"; fi
        if [[ "$timer_was_active" == true ]]; then : >"$state_dir/timer-active"; else rm -f -- "$state_dir/timer-active"; fi
    fi
    transaction_snapshot_started=false
}

validate_release_target() {
    local target="$1"
    [[ "$target" =~ ^releases/[0-9a-f]{40}$ ]] || return 1
    validate_release_integrity "$opt_root/$target" "${target#releases/}"
}

validate_release_integrity() {
    local release="$1"
    local expected_sha="$2"
    local recorded_sha
    local -a release_sha_lines
    local expected_uid path mode
    [[ -d "$release" && ! -L "$release" ]] || return 1
    expected_uid="$(expected_root_uid)"
    validate_secure_directory "$opt_root" "$expected_uid" || return 1
    validate_secure_directory "$releases_dir" "$expected_uid" || return 1
    if find "$release" ! -type f ! -type d -print -quit | grep -q .; then
        return 1
    fi
    while IFS= read -r path; do
        [[ "$(stat -c '%u' "$path")" == "$expected_uid" ]] || return 1
        mode="$(stat -c '%a' "$path")"
        if [[ -d "$path" ]]; then
            [[ "$mode" == "555" ]] || return 1
        elif [[ "$path" == "$release/bin/muneral-kb-sync" || "$path" == "$release"/venv/bin/python* ]]; then
            [[ "$mode" == "555" ]] || return 1
        else
            [[ "$mode" == "444" ]] || return 1
        fi
    done < <(find "$release" -type d -o -type f)
    [[ -f "$release/RELEASE_SHA" && ! -L "$release/RELEASE_SHA" ]] || return 1
    [[ -f "$release/$manifest_name" && ! -L "$release/$manifest_name" ]] || return 1
    if find "$release" -type l -print -quit | grep -q .; then
        return 1
    fi
    mapfile -t release_sha_lines <"$release/RELEASE_SHA"
    [[ ${#release_sha_lines[@]} -eq 1 ]] || return 1
    recorded_sha="${release_sha_lines[0]}"
    [[ "$recorded_sha" == "$expected_sha" ]] || return 1
    (
        cd "$release"
        diff -q \
            <(LC_ALL=C find . -type f ! -path "./$manifest_name" -print | LC_ALL=C sort) \
            <(sed -E 's/^[0-9a-f]{64}  //' "$manifest_name" | LC_ALL=C sort) \
            >/dev/null
        sha256sum --check --strict "$manifest_name" >/dev/null
    )
}

validate_source_repo() {
    local top_level actual_sha
    source_root="$(realpath -e -- "$source_root")"
    [[ -d "$source_root" && ! -L "$source_root" ]] || return 1
    top_level="$(git -C "$source_root" rev-parse --show-toplevel 2>/dev/null)" || return 1
    [[ "$(realpath -e -- "$top_level")" == "$source_root" ]] || return 1
    actual_sha="$(git -C "$source_root" rev-parse --verify HEAD 2>/dev/null)" || return 1
    [[ "$actual_sha" == "$release_sha" ]] || return 1
    git -C "$source_root" diff --quiet -- || return 1
    git -C "$source_root" diff --cached --quiet -- || return 1
}

validate_pilot_proof() {
    local proof="$credential_dir/pilot-proven"
    local mode
    local -a proof_lines
    [[ -f "$proof" && ! -L "$proof" ]] || return 1
    mode="$(stat -c '%a' "$proof")"
    [[ "$mode" == "600" ]] || return 1
    if [[ "$root" == "/" ]]; then
        [[ "$(stat -c '%u' "$proof")" == "0" ]] || return 1
    fi
    mapfile -t proof_lines <"$proof"
    [[ ${#proof_lines[@]} -eq 6 ]] || return 1
    [[ "${proof_lines[0]}" == "task_id=$pilot_task_id" ]] || return 1
    [[ "${proof_lines[1]}" == "release_sha=$release_sha" ]] || return 1
    [[ "${proof_lines[2]}" == "principal=$service_name" ]] || return 1
    [[ "${proof_lines[3]}" == "graph_proven=true" ]] || return 1
    [[ "${proof_lines[4]}" == "recall_proven=true" ]] || return 1
    [[ "${proof_lines[5]}" == "idempotent=true" ]] || return 1
}

validate_runtime_credentials() {
    local credential mode
    for credential in "$credential_dir/muneral-db-dsn" "$credential_dir/ltm-writer-token"; do
        [[ -f "$credential" && ! -L "$credential" ]] || return 1
        mode="$(stat -c '%a' "$credential")"
        [[ "$mode" == "600" ]] || return 1
        if [[ "$root" == "/" ]]; then
            [[ "$(stat -c '%u' "$credential")" == "0" ]] || return 1
        fi
    done
}

install_accounts_and_directories() {
    validate_existing_canonical_paths || { echo "canonical install path is insecure" >&2; exit 73; }
    if [[ "$root" == "/" ]]; then
        getent group "$service_name" >/dev/null || groupadd --system "$service_name"
        if ! id -u "$service_name" >/dev/null 2>&1; then
            useradd --system --gid "$service_name" --home-dir "/var/lib/$service_name" \
                --shell /usr/sbin/nologin "$service_name"
        fi
        validate_service_account || { echo "existing service account or group violates the isolation contract" >&2; exit 77; }
        install -d -m 0755 -o root -g root "$opt_root" "$releases_dir" "$unit_dir"
        install -d -m 0755 -o root -g root "$state_base"
        install -d -m 0750 -o "$service_name" -g "$service_name" "$state_dir"
        install -d -m 0700 -o root -g root "$credential_dir"
    else
        install -d -m 0755 "$opt_root" "$releases_dir" "$unit_dir"
        install -d -m 0755 "$state_base"
        install -d -m 0750 "$state_dir"
        install -d -m 0700 "$credential_dir"
    fi
    local expected_uid
    expected_uid="$(expected_root_uid)"
    for path in "$opt_root" "$releases_dir" "$state_base" "$credential_dir" "$unit_dir"; do
        validate_secure_directory "$path" "$expected_uid" || { echo "canonical install path validation failed" >&2; exit 73; }
    done
}

install_release() {
    [[ "$release_sha" =~ ^[0-9a-f]{40}$ ]] || { echo "--sha must be a lowercase 40-character commit SHA" >&2; exit 64; }
    [[ -n "$source_root" && -d "$source_root" && ! -L "$source_root" ]] || { echo "--source must be a real directory" >&2; exit 66; }
    validate_source_repo || { echo "source must be the clean tracked worktree at the requested HEAD SHA" >&2; exit 66; }

    install_accounts_and_directories
    local final_release="$releases_dir/$release_sha"
    [[ ! -L "$final_release" ]] || { echo "release path must not be a symlink" >&2; return 65; }
    staging="$releases_dir/.${release_sha}.tmp.$$"

    if [[ ! -d "$final_release" ]]; then
        install -d -m 0755 "$staging"
        git -C "$source_root" archive --format=tar "$release_sha" -- \
            src/scrutator \
            tools/__init__.py \
            tools/muneral_sync \
            deploy/muneral-kb-sync-run.sh \
            deploy/muneral-kb-sync.service \
            deploy/muneral-kb-sync.timer \
            deploy/requirements-muneral-kb-sync.txt \
            | tar -x -C "$staging"
        for required in \
            "$staging/src/scrutator" \
            "$staging/tools/muneral_sync/cli.py" \
            "$staging/deploy/muneral-kb-sync-run.sh" \
            "$staging/deploy/muneral-kb-sync.service" \
            "$staging/deploy/muneral-kb-sync.timer" \
            "$staging/deploy/requirements-muneral-kb-sync.txt"; do
            [[ -e "$required" && ! -L "$required" ]] || { echo "committed release source is incomplete" >&2; return 66; }
        done
        install -d -m 0755 "$staging/bin"
        install -m 0555 "$staging/deploy/muneral-kb-sync-run.sh" "$staging/bin/muneral-kb-sync"
        printf '%s\n' "$release_sha" >"$staging/RELEASE_SHA"

        if [[ "$root" != "/" && "${MUNERAL_KB_SYNC_TEST_FAIL_AFTER_COPY:-}" == "1" ]]; then
            echo "injected dry-root copy failure" >&2
            return 70
        fi
        if find "$staging" -type l -print -quit | grep -q .; then
            echo "release payload must not contain symlinks" >&2
            return 66
        fi
        python3 -m venv --copies "$staging/venv"
        if [[ -L "$staging/venv/lib64" ]]; then
            rm -- "$staging/venv/lib64"
        fi
        "$staging/venv/bin/python" -m pip install \
            --disable-pip-version-check \
            --no-input \
            --no-deps \
            --require-hashes \
            --only-binary=:all: \
            --requirement "$staging/deploy/requirements-muneral-kb-sync.txt"
        PYTHONPATH="$staging/src:$staging" "$staging/venv/bin/python" -c \
            'import asyncpg, httpx, tools.muneral_sync.cli'
        PYTHONPATH="$staging/src:$staging" "$staging/venv/bin/python" -m tools.muneral_sync.cli --help >/dev/null
        if find "$staging" -type l -print -quit | grep -q .; then
            echo "release payload must not contain symlinks" >&2
            return 66
        fi
        (
            cd "$staging"
            LC_ALL=C find . -type f ! -path "./$manifest_name" -print \
                | LC_ALL=C sort \
                | while IFS= read -r payload_file; do sha256sum "$payload_file"; done \
                >"$manifest_name"
        )
        chmod -R a-w "$staging"
        find "$staging" -type d -exec chmod 0555 {} +
        find "$staging" -type f -exec chmod 0444 {} +
        chmod 0555 "$staging/bin/muneral-kb-sync"
        find "$staging/venv/bin" -maxdepth 1 -type f -name 'python*' -exec chmod 0555 {} +
        mv -- "$staging" "$final_release"
        created_release="$final_release"
    else
        validate_release_integrity "$final_release" "$release_sha" || {
            echo "existing release failed integrity validation" >&2
            return 65
        }
    fi

    validate_release_integrity "$final_release" "$release_sha" || {
        echo "new release failed integrity validation" >&2
        return 65
    }
    snapshot_transaction_state
    install -m 0644 "$final_release/deploy/muneral-kb-sync.service" "$unit_dir/muneral-kb-sync.service"
    install -m 0644 "$final_release/deploy/muneral-kb-sync.timer" "$unit_dir/muneral-kb-sync.timer"

    local current_target=""
    if [[ -L "$opt_root/current" ]]; then
        current_target="$(readlink "$opt_root/current")"
        validate_release_target "$current_target" || { echo "current release link is invalid" >&2; return 65; }
        original_current_present=true
        original_current="$current_target"
    fi
    if [[ -L "$opt_root/previous" ]]; then
        original_previous="$(readlink "$opt_root/previous")"
        validate_release_target "$original_previous" || { echo "previous release link is invalid" >&2; return 65; }
        original_previous_present=true
    fi
    activation_started=true
    if [[ -n "$current_target" && "$current_target" != "releases/$release_sha" ]]; then
        atomic_link "$current_target" "$opt_root/previous"
    fi
    atomic_link "releases/$release_sha" "$opt_root/current"

    if [[ "$root" != "/" && "${MUNERAL_KB_SYNC_TEST_FAIL_AFTER_SWITCH:-}" == "1" ]]; then
        echo "injected dry-root activation failure" >&2
        return 70
    fi

    if [[ "$root" == "/" ]]; then
        systemctl daemon-reload
    fi
    if [[ "$enable_timer" == true ]]; then
        validate_pilot_proof || { echo "timer enable denied: exact root-controlled pilot proof is absent or invalid" >&2; return 78; }
        validate_runtime_credentials || { echo "timer enable denied: root-owned mode-0600 credentials are not ready" >&2; return 78; }
        if [[ "$root" == "/" ]]; then
            systemctl enable --now muneral-kb-sync.timer
        else
            : >"$state_dir/timer-enabled"
            : >"$state_dir/timer-active"
            if [[ "${MUNERAL_KB_SYNC_TEST_FAIL_AFTER_ENABLE:-}" == "1" ]]; then
                echo "injected dry-root post-enable failure" >&2
                return 70
            fi
        fi
    fi
    staging=""
    created_release=""
    activation_started=false
    transaction_snapshot_started=false
}

rollback_release() {
    install_accounts_and_directories
    [[ "$enable_timer" == false && -z "$release_sha" && -z "$source_root" ]] || usage
    [[ -L "$opt_root/current" && -L "$opt_root/previous" ]] || { echo "rollback requires current and previous releases" >&2; exit 65; }
    local current_target previous_target
    current_target="$(readlink "$opt_root/current")"
    previous_target="$(readlink "$opt_root/previous")"
    validate_release_target "$current_target" || { echo "current release link is invalid" >&2; exit 65; }
    validate_release_target "$previous_target" || { echo "previous release link is invalid" >&2; exit 65; }
    [[ "$current_target" != "$previous_target" ]] || { echo "rollback targets are identical" >&2; exit 65; }
    atomic_link "$previous_target" "$opt_root/current"
    atomic_link "$current_target" "$opt_root/previous"
}

acquire_global_lock

case "$action" in
    install) install_release ;;
    rollback) rollback_release ;;
    *) usage ;;
esac
