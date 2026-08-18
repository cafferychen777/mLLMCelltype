#!/usr/bin/env bash
set -euo pipefail

readonly PLATFORM="x86_64-manylinux_2_28"
readonly PYTHON_VERSION="3.11"
declare -a UPGRADE_ARGS=()

hash_sources() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | cut -d ' ' -f 1
    else
        shasum -a 256 | cut -d ' ' -f 1
    fi
}

source_hash() {
    local source
    for source in "$@"; do
        cat "$source"
    done | hash_sources
}

stored_hash() {
    sed -n 's/^# Sources-SHA256: //p' "$1"
}

compile_lock() {
    local output=$1
    shift
    local sources=("$@")
    local digest temporary

    digest=$(source_hash "${sources[@]}")
    temporary=$(mktemp "${output}.XXXXXX")
    trap 'rm -f -- "${temporary:-}"' RETURN

    # uv preserves compatible pins from an existing output file. Seed the
    # temporary output so routine lock regeneration does not upgrade unrelated
    # dependencies; upgrades must be requested explicitly below.
    if [[ -f "$output" ]]; then
        cp "$output" "$temporary"
    fi

    uv pip compile \
        "${UPGRADE_ARGS[@]}" \
        --python-version "$PYTHON_VERSION" \
        --python-platform "$PLATFORM" \
        --generate-hashes \
        --output-file "$temporary" \
        "${sources[@]}"

    {
        head -n 2 "$temporary"
        printf '# Sources-SHA256: %s\n' "$digest"
        tail -n +3 "$temporary"
    } > "$output"
    rm -f -- "$temporary"
    trap - RETURN
}

check_lock() {
    local output=$1
    shift
    local expected
    expected=$(source_hash "$@")
    [[ -f "$output" ]] || {
        printf 'Missing lock file: %s\n' "$output" >&2
        return 1
    }
    [[ "$(stored_hash "$output")" == "$expected" ]] || {
        printf '%s is stale; run make lock.\n' "$output" >&2
        return 1
    }
}

case "${1:-}" in
    --check)
        check_lock requirements.lock requirements.txt
        check_lock requirements-dev.lock requirements.txt requirements-dev.txt
        exit 0
        ;;
    --upgrade-all)
        UPGRADE_ARGS=(--upgrade)
        ;;
    --upgrade-package)
        [[ -n "${2:-}" ]] || {
            echo "--upgrade-package requires a package name." >&2
            exit 2
        }
        UPGRADE_ARGS=(--upgrade-package "$2")
        ;;
    "")
        ;;
    *)
        echo "Usage: $0 [--check | --upgrade-all | --upgrade-package PACKAGE]" >&2
        exit 2
        ;;
esac

command -v uv >/dev/null 2>&1 || {
    echo "uv is required to compile lock files." >&2
    exit 1
}
compile_lock requirements.lock requirements.txt
compile_lock requirements-dev.lock requirements.txt requirements-dev.txt
