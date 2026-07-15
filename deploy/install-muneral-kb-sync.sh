#!/usr/bin/env bash
set -euo pipefail

umask 077

readonly service_name="muneral-kb-sync"

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
state_dir="$(root_path /var/lib/muneral-kb-sync)"
readonly state_dir
credential_dir="$(root_path /etc/muneral-kb-sync)"
readonly credential_dir
unit_dir="$(root_path /etc/systemd/system)"
readonly unit_dir
staging=""
created_release=""
activation_started=false
original_current=""
original_previous=""
original_current_present=false
original_previous_present=false

cleanup_staging() {
    local status=$?
    if [[ $status -ne 0 ]]; then
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
            rm -rf -- "$staging"
        fi
        if [[ -n "$created_release" && -d "$created_release" ]]; then
            chmod -R u+w "$created_release"
            rm -rf -- "$created_release"
        fi
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

validate_release_target() {
    local target="$1"
    [[ "$target" =~ ^releases/[0-9a-f]{40}$ && -d "$opt_root/$target" ]]
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
    if [[ "$root" == "/" ]]; then
        getent group "$service_name" >/dev/null || groupadd --system "$service_name"
        if ! id -u "$service_name" >/dev/null 2>&1; then
            useradd --system --gid "$service_name" --home-dir "/var/lib/$service_name" \
                --shell /usr/sbin/nologin "$service_name"
        fi
        install -d -m 0755 -o root -g root "$opt_root" "$releases_dir" "$unit_dir"
        install -d -m 0750 -o "$service_name" -g "$service_name" "$state_dir"
        install -d -m 0700 -o root -g root "$credential_dir"
    else
        install -d -m 0755 "$opt_root" "$releases_dir" "$unit_dir"
        install -d -m 0750 "$state_dir"
        install -d -m 0700 "$credential_dir"
    fi
}

install_release() {
    [[ "$release_sha" =~ ^[0-9a-f]{40}$ ]] || { echo "--sha must be a lowercase 40-character commit SHA" >&2; exit 64; }
    [[ -n "$source_root" && -d "$source_root" && ! -L "$source_root" ]] || { echo "--source must be a real directory" >&2; exit 66; }
    for required in \
        "$source_root/src/scrutator" \
        "$source_root/tools/muneral_sync" \
        "$source_root/deploy/muneral-kb-sync-run.sh" \
        "$source_root/deploy/muneral-kb-sync.service" \
        "$source_root/deploy/muneral-kb-sync.timer"; do
        [[ -e "$required" && ! -L "$required" ]] || { echo "release source is incomplete or contains a top-level symlink" >&2; exit 66; }
    done

    install_accounts_and_directories
    local final_release="$releases_dir/$release_sha"
    [[ ! -L "$final_release" ]] || { echo "release path must not be a symlink" >&2; return 65; }
    staging="$releases_dir/.${release_sha}.tmp.$$"

    if [[ ! -d "$final_release" ]]; then
        install -d -m 0755 "$staging/bin" "$staging/src" "$staging/tools"
        cp -a -- "$source_root/src/scrutator" "$staging/src/"
        cp -a -- "$source_root/tools/muneral_sync" "$staging/tools/"
        if [[ -f "$source_root/tools/__init__.py" ]]; then
            install -m 0444 "$source_root/tools/__init__.py" "$staging/tools/__init__.py"
        fi
        install -m 0555 "$source_root/deploy/muneral-kb-sync-run.sh" "$staging/bin/muneral-kb-sync"
        printf '%s\n' "$release_sha" >"$staging/RELEASE_SHA"

        if [[ "$root" != "/" && "${MUNERAL_KB_SYNC_TEST_FAIL_AFTER_COPY:-}" == "1" ]]; then
            echo "injected dry-root copy failure" >&2
            return 70
        fi
        if find "$staging" -type l -print -quit | grep -q .; then
            echo "release payload must not contain symlinks" >&2
            return 66
        fi
        chmod -R a-w "$staging"
        find "$staging" -type d -exec chmod 0555 {} +
        find "$staging" -type f -exec chmod 0444 {} +
        chmod 0555 "$staging/bin/muneral-kb-sync"
        mv -- "$staging" "$final_release"
        created_release="$final_release"
    fi

    install -m 0644 "$source_root/deploy/muneral-kb-sync.service" "$unit_dir/muneral-kb-sync.service"
    install -m 0644 "$source_root/deploy/muneral-kb-sync.timer" "$unit_dir/muneral-kb-sync.timer"

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
        local proof="$state_dir/pilot-proven"
        [[ -f "$proof" && ! -L "$proof" && -s "$proof" ]] || { echo "timer enable denied: pilot proof marker is absent" >&2; return 78; }
        validate_runtime_credentials || { echo "timer enable denied: root-owned mode-0600 credentials are not ready" >&2; return 78; }
        if [[ "$root" == "/" ]]; then
            systemctl enable --now muneral-kb-sync.timer
        else
            : >"$state_dir/timer-enabled"
        fi
    fi
    staging=""
    created_release=""
    activation_started=false
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

case "$action" in
    install) install_release ;;
    rollback) rollback_release ;;
    *) usage ;;
esac
