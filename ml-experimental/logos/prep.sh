#!/usr/bin/env bash
set -euo pipefail
code=$(cd "$(dirname "$0")" && pwd)
root=${MODEL_ROOT:-/content/model}
boxes_per_scan=${MODEL_BOXES:-96}
jobs=${MODEL_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}
bin=$root/bin
raw=$root/raw
ctx=$root/ctx
shards=$root/shards
work=$root/work
umb=$code/umbilici
p4url=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
p332url=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr
p139url=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr
p172url=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0172/volumes/20241024131839-7.910um-53keV-masked.zarr
p1667url=https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr
s1aurl=https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr
s1a_axis=https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/umbilici/umbilicus-scroll1a_zyx.txt
bench_slabs=${MODEL_BENCH_SLABS:-320}
bench_controls=${MODEL_BENCH_CONTROLS:-}

usage() {
    cat >&2 <<'EOF'
Usage: prep.sh build | pull | context | labelled | bench | eval | unlabelled | all
build makes the CPU tools (build.sh tools) the other steps use. Writes the model's shards under MODEL_ROOT (default /content/model): shards/mid (hand labels + multi-scroll bench controls when available), shards/eval (held out),
shards/pre (unlabelled CT, MODEL_BOXES boxes of 128x512x512 per scan, default 96).
EOF
    exit 2
}

build() {
    MODEL_ROOT=$root bash "$code/build.sh" tools
}

pull() {
    local set pids=()
    mkdir -p "$raw"
    for set in paris4-instances fiber-skeletons paris4-windings pherc0332-lines; do
        bash "$code/labels.sh" "$raw" "$set" >"$raw/$set.log" 2>&1 &
        pids+=($!)
    done
    for set in "${pids[@]}"; do
        wait "$set"
    done
    mkdir -p "$umb/scroll1a"
    [ -s "$umb/scroll1a/umbilicus-scroll1a_zyx.txt" ] || curl -fsSL --retry 5 -o "$umb/scroll1a/umbilicus-scroll1a_zyx.txt" "$s1a_axis"
}

has_level() {
    curl -fsI --retry 3 "$1/$2/.zarray" >/dev/null 2>&1
}

context_of() {
    local name=$1 url=$2 level=$3
    local want=$((level + 4))
    mkdir -p "$ctx"
    [ -s "$ctx/$name.npy" ] && return 0
    while ! has_level "$url" "$want"; do
        want=$((want - 1))
        [ "$want" -gt "$level" ] || {
            printf 'no context level for %s\n' "$name" >&2
            return 1
        }
    done
    "$bin/fetch" "$url/$want" 0:1000000,0:1000000,0:1000000 "$ctx/$name.npy" 32 >/dev/null
    echo $((1 << (want - level))) >"$ctx/$name.factor"
}

context() {
    local scroll scan url level rest
    while read -r scroll scan url level rest; do
        context_of "${scroll}_$scan" "$url" "$level"
    done <"$code/scans.txt"
    context_of PHerc0332_20251211183505 "$p332url" 2
}

fetch_rows() {
    local plan=$1 url=$2 out=$3 row box
    mkdir -p "$out"
    awk '{ printf "%06d %d:%d,%d:%d,%d:%d\n", NR - 1, $1, $1 + 64, $2, $2 + 128, $3, $3 + 128 }' "$plan" >"$out/boxes.txt"
    while read -r row box; do
        [ -s "$out/$row.npy" ] || printf '%s %s %s 8\n' "$url" "$box" "$out/$row.npy"
    done <"$out/boxes.txt" | xargs -r -P "$jobs" -L 1 "$bin/fetch" >/dev/null
}

labelled() {
    local s1=(--context "$ctx/PHercParis4_20230205180739.npy" --context-factor "$(cat "$ctx/PHercParis4_20230205180739.factor")"
    --umbilicus "$umb/scroll1a/umbilicus-scroll1a_zyx.txt" --umbilicus-format zyx --umbilicus-factor 1 --curl -1
    --voxel 7.91)
    local p4=(--context "$ctx/PHercParis4_20260411134726.npy" --context-factor "$(cat "$ctx/PHercParis4_20260411134726.factor")"
    --umbilicus "$raw/paris4-windings/umbilicus.json" --umbilicus-format vc --umbilicus-factor 1 --curl 1 --voxel 9.6)
    local w=$raw/paris4-windings cubes fibres
    local sets=(--same "$w/same_windings.json" --relative "$w/relative_windings.json" --absolute "$w/abs_winding.json")
    mkdir -p "$shards/mid" "$work"
    cubes=$(dirname "$(dirname "$(find "$raw/paris4-instances/instance-labels-harmonized" -name '*_mask.nrrd' | sort | head -1)")")
    local s5=(--context "$ctx/PHerc0172_20241024131838.npy" --context-factor "$(cat "$ctx/PHerc0172_20241024131838.factor")"
    --umbilicus "$umb/lines/PHerc0172/PHerc0172_20241024131838_finder-v2_umbilicus.json" --umbilicus-format vc
    --umbilicus-factor 1 --curl 1 --voxel 7.91)
    [ -s "$shards/mid/instance.shard" ] ||
        "$bin/prep" instance "$cubes" "$shards/mid/instance.shard" "${s1[@]}" --source 1 --confidence 1
    fibres=$(dirname "$(find "$raw/fiber-skeletons" -type d -name labelsTr -path '*Dataset001*' | sort | head -1)")
    [ -s "$shards/mid/fibre-s1.shard" ] || "$bin/prep" fibre "$fibres" "$shards/mid/fibre-s1.shard" "${s1[@]}" \
        --prefix s1_ --holdout-every 5 --source 2 --confidence 0.7
    [ -s "$shards/mid/fibre-s5.shard" ] || "$bin/prep" fibre "$fibres" "$shards/mid/fibre-s5.shard" "${s5[@]}" \
        --prefix s5_ --holdout-every 5 --source 2 --confidence 0.7
    [ -s "$shards/mid/points.shard" ] && return 0
    "$bin/prep" plan-points "$work/points.plan" "${sets[@]}"
    fetch_rows "$work/points.plan" "$p4url/2" "$work/points"
    "$bin/prep" points "$work/points.plan" "$work/points" "$shards/mid/points.shard" "${sets[@]}" "${p4[@]}" --source 3 \
        --confidence 1
}

find_bench_controls() {
    if [ -n "$bench_controls" ] && [ -d "$bench_controls" ]; then
        printf '%s\n' "$bench_controls"
        return 0
    fi
    local cand
    for cand in "$root/bench-controls" "$code/../../data/outputs/bench/controls"; do
        if [ -d "$cand" ]; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    return 1
}

bench_control() {
    local ctrl=$1 scan_key=$2 url=$3 dense=$4 cdir=$5
    local out=$shards/mid/bench-$ctrl.shard
    [ -s "$out" ] && return 0
    [ -f "$cdir/$ctrl/truth.npz" ] || return 0
    context_of "$scan_key" "$url" 0
    local f plan=$work/bench-$ctrl.plan fetched=$work/bench-$ctrl extra=()
    f=$(cat "$ctx/$scan_key.factor")
    [ "$dense" = 1 ] && extra=(--dense-sheet)
    python3 "$code/bench_to_shard.py" plan "$ctrl" "$plan" --controls-dir "$cdir" --max-slabs "$bench_slabs"
    [ -s "$plan" ] || return 0
    fetch_rows "$plan" "$url/0" "$fetched"
    python3 "$code/bench_to_shard.py" build "$ctrl" "$plan" "$fetched" "$ctx/$scan_key.npy" "$out" \
        --context-factor "$f" --controls-dir "$cdir" "${extra[@]}"
    rm -rf "$fetched" "$plan"
}

bench() {
    local cdir
    if ! cdir=$(find_bench_controls); then
        printf 'prep.sh bench: no bench-controls directory found; sync data/outputs/bench/controls/*-train*/truth.npz to %s/bench-controls (or set MODEL_BENCH_CONTROLS) to add multi-scroll training shards\n' "$root" >&2
        return 0
    fi
    mkdir -p "$shards/mid" "$work"
    bench_control PHerc0139-train PHerc0139_20250728140407 "$p139url" 1 "$cdir"
    bench_control PHerc0172-train-4608 PHerc0172_20241024131839 "$p172url" 0 "$cdir"
    bench_control PHerc0172-train-5760 PHerc0172_20241024131839 "$p172url" 0 "$cdir"
    bench_control PHerc0172-train-9728 PHerc0172_20241024131839 "$p172url" 0 "$cdir"
    bench_control PHerc1667-train-2560 PHerc1667_20231117161658 "$p1667url" 0 "$cdir"
    bench_control PHerc1667-train-7168 PHerc1667_20231117161658 "$p1667url" 0 "$cdir"
    bench_control PHercParis4-train-3072 PHercParis4_20230205180739 "$s1aurl" 0 "$cdir"
    bench_control PHercParis4-train-6144 PHercParis4_20230205180739 "$s1aurl" 0 "$cdir"
    bench_control PHercParis4-train-9216 PHercParis4_20230205180739 "$s1aurl" 0 "$cdir"
}

eval_lines() {
    local name=$1 dir=$2 url=$3 base=$4 boxes=$5 source=$6
    shift 6
    local lines=(--lines "$dir" --line-base "$base" --point-factor 4)
    [ -s "$shards/eval/$name.shard" ] && return 0
    "$bin/prep" plan-points "$work/$name.plan" "${lines[@]}" --max-boxes "$boxes"
    fetch_rows "$work/$name.plan" "$url" "$work/$name"
    "$bin/prep" points "$work/$name.plan" "$work/$name" "$shards/eval/$name.shard" "${lines[@]}" --source "$source" \
        --confidence 1 "$@"
}

evaluation() {
    local p4=(--context "$ctx/PHercParis4_20260411134726.npy" --context-factor "$(cat "$ctx/PHercParis4_20260411134726.factor")"
    --umbilicus "$raw/paris4-windings/umbilicus.json" --umbilicus-format vc --umbilicus-factor 1 --curl 1 --voxel 9.6)
    local p332=(--context "$ctx/PHerc0332_20251211183505.npy" --context-factor "$(cat "$ctx/PHerc0332_20251211183505.factor")"
    --umbilicus "$raw/pherc0332-lines/umbilicus.json" --umbilicus-format vc --umbilicus-factor 1 --curl -1 --voxel 9.6)
    local dir n=0
    mkdir -p "$shards/eval" "$work"
    while read -r dir; do
        eval_lines "p4fibres$n" "$dir" "$p4url/2" $((4000000 + 100000 * n)) 2 4 "${p4[@]}"
        n=$((n + 1))
    done < <(find "$raw/paris4-windings/eval_fibers" -type f -name '*.json' -exec dirname {} \; | sort -u)
    n=0
    while read -r dir; do
        eval_lines "p332lines$n" "$dir" "$p332url/2" $((5000000 + 100000 * n)) 16 5 "${p332[@]}"
        n=$((n + 1))
    done < <(find "$raw/pherc0332-lines/slice_lines" -type f -name '*.json' -exec dirname {} \; | sort -u)
    n=0
    while read -r dir; do
        eval_lines "p332fibres$n" "$dir" "$p332url/2" $((6000000 + 100000 * n)) 1 6 "${p332[@]}"
        n=$((n + 1))
    done < <(find "$raw/pherc0332-lines/fibers_manual" -type f -name '*.json' -exec dirname {} \; | sort -u)
}

unlabelled_scan() {
    local scroll=$1 scan=$2 url=$3 level=$4 voxel=$5 curl=$6 line=$7 format=$8 factor=$9 ez0=${10} ez1=${11}
    local name=${scroll}_$scan f box row=0 z y x
    [ -s "$shards/pre/$name.shard" ] && return 0
    f=$(cat "$ctx/$name.factor")
    mkdir -p "$work/pre/$name" "$shards/pre"
    "$bin/prep" plan-unlabelled "$ctx/$name.npy" --context-factor "$f" --count "$boxes_per_scan" --box-z 128 \
        --box-yx 512 --seed "${scan: -6}" --exclude-z0 "$ez0" --exclude-z1 "$ez1" >"$work/pre/$name/boxes.txt"
    awk '{ printf "%06d %s\n", NR - 1, $1 }' "$work/pre/$name/boxes.txt" >"$work/pre/$name/rows.txt"
    while read -r row box; do
        [ -s "$work/pre/$name/$row.npy" ] || printf '%s %s %s 16\n' "$url/$level" "$box" "$work/pre/$name/$row.npy"
    done <"$work/pre/$name/rows.txt" | xargs -r -P "$jobs" -L 1 "$bin/fetch" >/dev/null || true
    : >"$work/pre/$name/parts.txt"
    while read -r row box; do
        [ -s "$work/pre/$name/$row.npy" ] || continue
        z=${box%%:*}
        y=${box#*,}
        y=${y%%:*}
        x=${box##*,}
        x=${x%%:*}
        "$bin/prep" unlabelled "$work/pre/$name/$row.npy" "$work/pre/$name/$row.shard" --origin "$z,$y,$x" \
            --context "$ctx/$name.npy" --context-factor "$f" --umbilicus "$umb/$line" --umbilicus-format "$format" \
            --umbilicus-factor "$factor" --curl "$curl" --voxel "$voxel" --source 0 --min-fill 0.3 \
            --split $((10#$row % 25 == 24)) 2>/dev/null
        rm -f "$work/pre/$name/$row.npy"
        echo "$work/pre/$name/$row.shard" >>"$work/pre/$name/parts.txt"
    done <"$work/pre/$name/rows.txt"
    xargs "$bin/prep" join "$shards/pre/$name.shard.part" <"$work/pre/$name/parts.txt"
    mv "$shards/pre/$name.shard.part" "$shards/pre/$name.shard"
    rm -rf "$work/pre/$name"
}

unlabelled() {
    local args
    while read -r -a args; do
        unlabelled_scan "${args[@]}"
        printf '%s %s done, %s\n' "${args[0]}" "${args[1]}" "$(du -sh "$shards/pre" | cut -f1)"
    done <"$code/scans.txt"
}

[ $# -ge 1 ] || usage
for step in "$@"; do
    case $step in
        build) build ;;
        pull) pull ;;
        context) context ;;
        labelled) labelled ;;
        bench) bench ;;
        eval) evaluation ;;
        unlabelled) unlabelled ;;
        all) build && pull && context && labelled && bench && evaluation && unlabelled ;;
        *) usage ;;
    esac
done
