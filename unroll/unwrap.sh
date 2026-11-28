#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
hx=$(cd "$here/.." && pwd)
usage() {
    printf '%s\n' 'Usage: ./run.sh unwrap [--queue] [--from STAGE] [--rotate180] SCROLL[:SCAN] ...'
}
stage=all
options=()
specs=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --queue) stage=queue ;;
        --rotate180) options+=("$1") ;;
        --from)
            if [ "$#" -lt 2 ]; then
                usage >&2
                exit 2
            fi
            options+=("$1" "$2")
            shift
            ;;
        *)
            if [[ ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*(:[A-Za-z0-9][A-Za-z0-9_.-]*)?$ ]]; then
                printf 'Invalid scroll selection: %s\n' "$1" >&2
                usage >&2
                exit 2
            fi
            specs+=("$1")
            ;;
    esac
    shift
done
if [ "${#specs[@]}" -eq 0 ]; then
    usage >&2
    exit 2
fi
cd "$hx"
for spec in "${specs[@]}"; do
    scroll=${spec%%:*}
    if [ "$scroll" != "$spec" ]; then
        scan=${spec#*:}
    else
        read -r held scan < <(./run.sh scans.catalog.scan_choice chosen "$scroll")
        if [ "$held" != "$scroll" ] || [ -z "$scan" ]; then
            printf '%s: no scan chosen\n' "$scroll" >&2
            exit 1
        fi
    fi
    "$here/pipeline.sh" "$scroll" "$scan" "$stage" ${options[@]+"${options[@]}"}
done
