#!/usr/bin/env bash
set -euo pipefail

ash=https://dl.ash2txt.org
paris4_spiral=$ash/datasets/spiral_datasets/PHercParis4
pherc0332=$ash/datasets/spiral_datasets/PHerc0332/20251211183505
instances=$ash/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels
kaggle_api=https://www.kaggle.com/api/v1/competitions/data
kaggle_comp=vesuvius-challenge-surface-detection
all_sets=(kaggle-surface paris4-instances fiber-skeletons paris4-windings pherc0332-lines)

usage() {
    cat >&2 <<'EOF'
Usage: labels.sh [--list] DEST [SET ...]

Pulls the hand-labelled training data, as published, into DEST/SET. Sets (default all):
  kaggle-surface     Kaggle Surface Detection: train.csv, 786 CT sub-volumes and their surface labels, unpacked from
                     the competition's one archive, resumed after a cut (needs the competition rules accepted and a
                     token: KAGGLE_API_TOKEN or ~/.kaggle/access_token, or KAGGLE_USERNAME and KAGGLE_KEY or
                     ~/.kaggle/kaggle.json)
  paris4-instances   PHercParis4 256^3 cubes with hand-made sheet instance labels, raw and harmonized
  fiber-skeletons    Hand-traced papyrus fibres in 256^3 and 512^3 cubes (WEBKNOSSOS), nnU-Net layout
  paris4-windings    PHercParis4 hand-placed same, relative and absolute winding points and 645 eval fibres
  pherc0332-lines    PHerc0332 51 hand-traced slice lines, hand-annotated fibres and the hand-placed umbilicus
--list prints every file and its size without downloading. Reruns skip complete files and resume partial ones.
EOF
    exit 2
}

remote_size() {
    curl -fsIL --retry 5 "$1" | tr -d '\r' | awk 'tolower($1) == "content-length:" { n = $2 } END { print n }'
}

local_size() {
    if [ -f "$1" ]; then wc -c <"$1" | tr -d ' '; else echo -1; fi
}

fetch() {
    local url=$1 out=$2 want
    want=$(remote_size "$url")
    if [ "$listing" = 1 ]; then
        printf '%s\t%s\n' "${want:-?}" "$url"
        total=$((total + ${want:-0}))
        return 0
    fi
    if [ -n "$want" ] && [ "$(local_size "$out")" = "$want" ]; then
        return 0
    fi
    mkdir -p "$(dirname "$out")"
    curl -fL --retry 8 --retry-delay 5 -C - -o "$out.part" "$url"
    if [ -n "$want" ] && [ "$(local_size "$out.part")" != "$want" ]; then
        printf 'size mismatch for %s\n' "$url" >&2
        exit 1
    fi
    mv "$out.part" "$out"
}

entries() {
    curl -fsSL --retry 5 "$1" | grep -o 'href="[^"?#]*"' | sed 's/^href="//; s/"$//' |
        grep -v -e '^\.\./*$' -e '^/' -e '://' || true
}

mirror() {
    local url=$1 dir=$2 entry name
    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        name=$(printf '%b' "${entry//%/\\x}")
        case $entry in
            */) mirror "$url$entry" "$dir/${name%/}" ;;
            *) fetch "$url$entry" "$dir/$name" ;;
        esac
    done < <(entries "$url")
}

unpack() {
    local zip=$1 dir=$2
    [ "$listing" = 0 ] || return 0
    [ ! -f "$dir/.unpacked" ] || return 0
    mkdir -p "$dir"
    unzip -q -o "$zip" -d "$dir"
    touch "$dir/.unpacked"
    rm -f "$zip"
}

kaggle_auth() {
    local token=${KAGGLE_API_TOKEN:-} file=$HOME/.kaggle/kaggle.json user=${KAGGLE_USERNAME:-} key=${KAGGLE_KEY:-}
    kaggle_curl=(-A "kaggle/1.7.4")
    [ -n "$token" ] || [ ! -f "$HOME/.kaggle/access_token" ] || token=$(tr -d '\n\r ' <"$HOME/.kaggle/access_token")
    if [ -n "$token" ]; then
        kaggle_curl+=(-H "Authorization: Bearer $token")
        return 0
    fi
    if [ -z "$user" ] && [ -f "$file" ]; then
        user=$(sed -n 's/.*"username" *: *"\([^"]*\)".*/\1/p' "$file")
        key=$(sed -n 's/.*"key" *: *"\([^"]*\)".*/\1/p' "$file")
    fi
    if [ -z "$user" ] || [ -z "$key" ]; then
        printf 'kaggle-surface needs KAGGLE_API_TOKEN, ~/.kaggle/access_token, or KAGGLE_USERNAME and KAGGLE_KEY\n' >&2
        exit 1
    fi
    kaggle_curl+=(-u "$user:$key")
}

kaggle_files() {
    local page="" json
    while :; do
        json=$(curl -fsS --retry 5 "${kaggle_curl[@]}" "$kaggle_api/list/$kaggle_comp?pageSize=200${page:+&pageToken=$page}") || {
            printf 'Kaggle refused the file list: check the token\n' >&2
            exit 1
        }
        paste <(printf '%s' "$json" | grep -o '"totalBytes":[0-9]*' | cut -d: -f2) \
            <(printf '%s' "$json" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
        page=$(printf '%s' "$json" | sed -n 's/.*"nextPageTokenNullable":"\([^"]*\)".*/\1/p')
        [ -n "$page" ] || break
    done
}

pull_kaggle() {
    local dir=$1/kaggle-surface size name try
    kaggle_auth
    if [ "$listing" = 1 ]; then
        while IFS=$'\t' read -r size name; do
            printf '%s\t%s\n' "$size" "$kaggle_api/download/$kaggle_comp/$name"
            total=$((total + size))
        done < <(kaggle_files | awk -F'\t' '$2 == "train.csv" || $2 ~ /^train_(images|labels)\//')
        return 0
    fi
    [ ! -f "$dir/.unpacked" ] || return 0
    mkdir -p "$dir"
    for try in 1 2 3 4 5 6 7 8; do
        curl -fsSL --connect-timeout 30 --speed-limit 100000 --speed-time 120 -C - "${kaggle_curl[@]}" \
            -o "$dir/archive.zip.part" "$kaggle_api/download-all/$kaggle_comp" && break
        [ "$try" -lt 8 ] || {
            printf 'Kaggle refused the archive: accept the competition rules on kaggle.com and check the token\n' >&2
            exit 1
        }
        sleep 60
    done
    mv "$dir/archive.zip.part" "$dir/archive.zip"
    unzip -q -o "$dir/archive.zip" 'train.csv' 'train_images/*' 'train_labels/*' -d "$dir"
    rm -f "$dir/archive.zip"
    touch "$dir/.unpacked"
}

pull_paris4_instances() {
    local dir=$1/paris4-instances
    fetch "$instances/README.txt" "$dir/README.txt"
    fetch "$instances/instance-labels-harmonized.zip" "$dir/instance-labels-harmonized.zip"
    unpack "$dir/instance-labels-harmonized.zip" "$dir/instance-labels-harmonized"
    mirror "$instances/instance-labels/" "$dir/instance-labels"
}

pull_fiber_skeletons() {
    local dir=$1/fiber-skeletons
    fetch "$ash/datasets/fiber-skeletons/README.txt" "$dir/README.txt"
    fetch "$ash/datasets/fiber-skeletons/fiber-skeletons.zip" "$dir/fiber-skeletons.zip"
    unpack "$dir/fiber-skeletons.zip" "$dir/fiber-skeletons"
}

pull_paris4_windings() {
    local dir=$1/paris4-windings name
    fetch "https://scrollprize.org/data/datasets/spiral-input-PHercParis4-README.md" "$dir/README.md"
    for name in same_windings.json relative_windings.json abs_winding.json umbilicus.json; do
        fetch "$paris4_spiral/$name" "$dir/$name"
    done
    mirror "$paris4_spiral/eval_fibers/" "$dir/eval_fibers"
}

pull_pherc0332_lines() {
    local dir=$1/pherc0332-lines name
    fetch "$pherc0332/README.md" "$dir/README.md"
    fetch "$pherc0332/umbilicus/umbilicus.json" "$dir/umbilicus.json"
    mirror "$pherc0332/slice_lines/" "$dir/slice_lines"
    for name in fibers_2026-10 hub_fibers_266 pre_hub_220; do
        mirror "$pherc0332/fibers/manual/$name/" "$dir/fibers_manual/$name"
    done
}

listing=0
total=0
if [ "${1:-}" = --list ]; then
    listing=1
    shift
fi
[ $# -ge 1 ] || usage
dest=$1
shift
sets=("$@")
[ ${#sets[@]} -gt 0 ] || sets=("${all_sets[@]}")
for set in "${sets[@]}"; do
    case $set in
        kaggle-surface) pull_kaggle "$dest" ;;
        paris4-instances) pull_paris4_instances "$dest" ;;
        fiber-skeletons) pull_fiber_skeletons "$dest" ;;
        paris4-windings) pull_paris4_windings "$dest" ;;
        pherc0332-lines) pull_pherc0332_lines "$dest" ;;
        *) usage ;;
    esac
done
if [ "$listing" = 1 ]; then
    awk -v t="$total" 'BEGIN { printf "total %.2f GB\n", t / 1e9 }'
else
    du -sh "$dest"/* 2>/dev/null || true
fi
