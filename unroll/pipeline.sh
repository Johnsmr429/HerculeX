#!/usr/bin/env bash
set -euo pipefail
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
here=$(cd "$(dirname "$0")" && pwd)
hx=$here
until [ -x "$hx/run.sh" ] && [ -d "$hx/level/level" ]; do
    if [ "$hx" = / ]; then
        echo "$0: not inside a HerculeX checkout" >&2
        exit 1
    fi
    hx=$(dirname "$hx")
done

usage() {
    cat >&2 <<'EOF'
Usage: ./run.sh unroll SCROLL SCAN STAGE [ARGS]
  plan [--m7 URL]            keep the scan's umbilicus, write its settings, papyrus mask and unit plan
  trace [UNIT...]            trace each unit on this machine, in z order (every unit by default)
  colab SESSION [UNIT...]    trace the units on a Colab box
  colab-fetch SESSION        pull the box's finished unit graphs
  stitch [--partial] [OPTION...]  stitch planned units; --partial permits missing unit graphs
  layout [OPTION...]         lay the pieces out as one strip; options go to layout.py
  preview [--rotate180]      draw the strip from the CT atlas, without an ink read
  root [--rotate180]         write a HerculeX outputs root with band units and the layout
  read [UNITS]               read the ink on this machine, every band or the named ones
  page                       compose the page and add its card to the start page
  all [--from STAGE] [--rotate180]  run plan, trace, stitch, layout, root, read and page
  queue [--from STAGE] [--rotate180]  queue those stages on the job board with dependencies
Settings: UNROLL_WORK, UNROLL_RUNS, UNROLL_INK, UNROLL_PAGE, UNROLL_SURFACE, UNROLL_SET, UNROLL_ANCHOR,
UNROLL_TRACE_ARGS (README.md).
EOF
    exit "${1:-2}"
}

if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    usage 0
fi
[ $# -ge 3 ] || usage
scroll=$1
scan=$2
stage=$3
shift 3
cd "$hx" || exit 1
library=$(./run.sh paths library)
outputs=$(./run.sh paths outputs)
export UNROLL_WORK=${UNROLL_WORK:-$library/cache/unroll/$scroll-$scan}
work=$UNROLL_WORK
units_dir=$work${UNROLL_SET:+/sets/$UNROLL_SET}
runs=${UNROLL_RUNS:-$outputs/runs/unroll/$scroll-$scan}
model=${UNROLL_INK:-reader-v3}
suffix=-$model
if [ "$model" = reader-v3 ]; then
    suffix=
fi
level=unwrapping/level/$scroll/$scan
cd "$hx" || exit 1

setting() {
    python3 -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$work/scan.json" "$1"
}

unit_names() {
    python3 -I -c 'import json,sys;print(" ".join(u["name"] for u in json.load(open(sys.argv[1]))))' \
        "$work/units_plan.json"
}

build() {
    local bin newer
    for bin in runtime/bin/unroll-trace runtime/bin/unroll-stitch; do
        newer=""
        if [ -x "$bin" ]; then
            newer=$(find "$here" level/level/level_solver.c level/level/sheet_flattening.c \
                level/level/scroll_axis.c level/level/threads_apple level/level/threads_cuda \
                \( -name '*.c' -o -name '*.h' -o -name '*.cu' -o -name build.sh \) -newer "$bin" -type f -print -quit)
        fi
        if [ ! -x "$bin" ] || [ -n "$newer" ]; then
            "$here/build.sh"
            return
        fi
    done
}

plan() {
    mkdir -p "$work"
    ./run.sh level.level.run_levels ready "$scroll" "$scan"
    python3 -I "$here/config.py" "$scroll" "$scan" --out "$work" "$@"
    if [ ! -s "$work/units_plan.json" ]; then
        python3 -I "$here/plan.py" --m7-url "$(setting m7_url)" --out "$work"
    fi
}

trace() {
    local names=("$@")
    if [ ${#names[@]} -eq 0 ]; then
        read -r -a names <<<"$(unit_names)"
    fi
    build
    "$here/units.sh" "${names[@]}"
}

colab() {
    local session=$1
    shift
    local names=("$@")
    if [ ${#names[@]} -eq 0 ]; then
        read -r -a names <<<"$(unit_names)"
    fi
    "$here/colab/drive.sh" up "$session"
    "$here/colab/drive.sh" start "$session" "${names[@]}"
}

stitch() {
    local args=() options=() name z0 z1 graph dims option partial=0 missing=0
    for option in "$@"; do
        if [ "$option" = --partial ]; then
            partial=1
        else
            options+=("$option")
        fi
    done
    while read -r name z0 z1; do
        graph=$units_dir/graphs/$name
        if [ ! -s "$graph/surfels.bin" ] || [ "$(wc -c <"$graph/surfels.bin")" -lt 16 ] ||
            [ ! -s "$units_dir/units/$name/record.json" ]; then
            printf 'Unfinished trace unit: %s\n' "$name" >&2
            missing=$((missing + 1))
            continue
        fi
        if [ "$(wc -c <"$graph/surfels.bin")" -eq 16 ]; then
            continue
        fi
        args+=(--unit "$graph:$z0:$z1")
    done < <(python3 -I -c '
import json, sys
for u in json.load(open(sys.argv[1])):
    print(u["name"], u["z0"], u["z1"])
' "$work/units_plan.json")
    if [ "$missing" -gt 0 ] && [ "$partial" -eq 0 ]; then
        printf '%s unit(s) unfinished; rerun trace, or use stitch --partial for a deliberate subset.\n' "$missing" >&2
        return 1
    fi
    if [ "${#args[@]}" -lt 4 ]; then
        printf '%s\n' 'Stitch needs at least two nonempty unit graphs; existing pieces were kept.' >&2
        return 1
    fi
    dims=$(python3 -I -c 'import numpy, sys; print(",".join(str(int(v)) for v in numpy.load(sys.argv[1])["shape"]))' \
        "$work/papyrus_mask_L4.npz")
    mkdir -p "$runs"
    rm -rf "$runs/pieces"
    build
    runtime/bin/unroll-stitch "${args[@]}" --axis "$work/axis.json" --mask "$work/papyrus_mask_L4.u8" \
        --mask-dims "$dims" --sample "$scroll" --scan "$scan" --curl "$(setting curl)" --stride 8 \
        --voxel "$(setting voxel_um)" --min-nodes 400 --neighbours 4 --neighbours-trust \
        ${options[@]+"${options[@]}"} --out "$runs/pieces"
}

layout() {
    python3 -I "$here/layout.py" --pieces "$runs/pieces" --out "$runs/layout" --axis "$work/axis.json" \
        --consensus-rounds 2 --consensus-min 2.5 --consensus-frac 0 --joint-radius 48 --joint-contacts 10 \
        --pack-mm 60 --pack-tolerance 0.2 "$@"
}

preview() {
    local atlas="" meta
    for meta in "$library/raw/$scroll"/atlas_*/L*/meta.json; do
        if [ -f "$meta" ] && grep -q "$scan" "$meta"; then
            atlas=$(dirname "$meta")
            break
        fi
    done
    if [ -z "$atlas" ]; then
        echo "no CT atlas of $scroll $scan: run ./run.sh scans.transfer.fetch_atlas \"$library/raw/$scroll/atlas_75\"" >&2
        exit 1
    fi
    python3 -I "$here/preview.py" --pieces "$runs/pieces" --layout "$runs/layout/scroll_layout.json" \
        --atlas "$atlas" --out "$runs/layout/preview" "$@"
}

root() {
    local source=()
    if [ -f "$work/axis-source.json" ]; then
        source=(--axis-source "$work/axis-source.json")
    fi
    rm -rf "$runs/root"
    python3 -I "$here/root.py" --pieces "$runs/pieces" --layout "$runs/layout/scroll_layout.json" \
        --root "$runs/root" --scroll "$scroll" --scan "$scan" --axis "$work/axis.json" ${source[@]+"${source[@]}"} \
        --curl "$(setting curl)" --shape "$(setting shape)" "$@"
}

read_ink() {
    mkdir -p "$runs/bricks"
    HERCULEX_OUTPUTS=$runs/root HERCULEX_BRICKS=$runs/bricks HERCULEX_INK_MODEL=$model \
        HERCULEX_KEEP_STACKS=0 HERCULEX_FETCH_AHEAD_GIB=${HERCULEX_FETCH_AHEAD_GIB:-60} \
        HERCULEX_UNITS=${1:-} ./run.sh ink "$scroll" "$scan"
}

page() {
    local faces=$runs/root/$level/faces/level-0$suffix folder name
    ./run.sh runner.keep_faces "$faces" "$runs/faces/level-0$suffix"
    HERCULEX_OUTPUTS=$runs/root HERCULEX_INK_MODEL=$model HERCULEX_SITE=http://127.0.0.1:1 \
        ./run.sh runner.review_page "$scroll" "$scan"
    folder=$(find "$runs/root/website/sheets" -mindepth 1 -maxdepth 1 -name "sheet-*-level$suffix" -print -quit)
    if [ -z "$folder" ] || [ ! -f "$folder/index.html" ]; then
        printf 'No published page found for %s %s (%s).\n' "$scroll" "$scan" "$model" >&2
        return 1
    fi
    name=$(basename "$folder")
    name=${UNROLL_PAGE:-${name/-level/-trace}}
    mkdir -p "$outputs/website/sheets"
    if [ -e "$outputs/website/sheets/$name" ] && [ ! -L "$outputs/website/sheets/$name" ]; then
        printf 'Page path is not a symlink: %s\n' "$outputs/website/sheets/$name" >&2
        return 1
    fi
    ln -sfn "$folder" "$outputs/website/sheets/$name"
    ./run.sh card "$name" "$scroll" --fit "trace 1.5" --surface \
        "${UNROLL_SURFACE:-curl $(printf '%+d' "$(setting curl)")}"
}

all() {
    local mode=$1 from=plan rotate=0 found=0 step previous="" job prefix memory cores
    local options=() settings=() resources=() after=() key
    shift
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --rotate180) rotate=1 ;;
            --from)
                [ "$#" -ge 2 ] || usage
                from=$2
                shift
                ;;
            *) usage ;;
        esac
        shift
    done
    case "$from" in plan | trace | stitch | layout | root | read | page) ;; *) usage ;; esac
    if [ "$rotate" -eq 1 ]; then
        options=(--rotate180)
    fi
    prefix=unroll-$scroll-$scan-$(date +%s)-$$
    while IFS= read -r key; do
        case "$key" in
            HERCULEX_NATIVE_BIN) ;;
            HERCULEX_* | UNROLL_* | OMP_* | STITCH_* | PATH)
                settings+=("$key=${!key}")
                ;;
        esac
    done < <(compgen -e)
    settings+=("HERCULEX_LIBRARY=$library" "HERCULEX_OUTPUTS=$outputs" "UNROLL_RUNS=$runs" "UNROLL_INK=$model"
        "UNROLL_SET=${UNROLL_SET:-}" "UNROLL_PAGE=${UNROLL_PAGE:-}" "UNROLL_SURFACE=${UNROLL_SURFACE:-}"
        "UNROLL_ANCHOR=${UNROLL_ANCHOR:-0}" "UNROLL_TRACE_ARGS=${UNROLL_TRACE_ARGS:-}")
    local online_cpus work_cores read_cores is_linux=0
    online_cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 4)
    work_cores=$((online_cpus > 3 ? online_cpus - 2 : 1))
    read_cores=$((online_cpus > 2 ? online_cpus - 1 : 1))
    if [ "$(uname)" != Darwin ]; then
        is_linux=1
    fi
    for step in plan trace stitch layout root read page; do
        if [ "$step" = "$from" ]; then
            found=1
        fi
        [ "$found" -eq 1 ] || continue
        if [ "$mode" = queue ]; then
            resources=()
            memory=3
            cores=1
            case "$step" in
                plan) resources=(--link) ;;
                trace)
                    memory=6
                    cores=$work_cores
                    resources=(--link)
                    if [ "$is_linux" -eq 1 ]; then
                        resources+=(--gpu)
                    fi
                    ;;
                stitch)
                    memory=10
                    cores=$work_cores
                    if [ "$is_linux" -eq 1 ]; then
                        resources+=(--gpu)
                    fi
                    ;;
                layout)
                    memory=10
                    cores=$work_cores
                    ;;
                read)
                    memory=6
                    cores=$read_cores
                    resources=(--gpu --engine --link)
                    ;;
                page)
                    memory=7
                    cores=$work_cores
                    ;;
            esac
            after=()
            if [ -n "$previous" ]; then
                after=(--after "$previous")
            fi
            job=$prefix-$step
            local command=(./run.sh unroll "$scroll" "$scan" "$step")
            if [ "$step" = root ]; then
                command+=(${options[@]+"${options[@]}"})
            fi
            ./run.sh board add "$job" --memory "$memory" --cores "$cores" \
                ${resources[@]+"${resources[@]}"} ${after[@]+"${after[@]}"} \
                --setting "${settings[@]}" -- "${command[@]}"
            previous=$job
        else
            case "$step" in
                root) root ${options[@]+"${options[@]}"} ;;
                read) read_ink "" ;;
                *) "$step" ;;
            esac
        fi
    done
    if [ "$mode" = queue ]; then
        printf 'Queued through %s. Follow with: ./run.sh board log %s\n' "$previous" "$previous"
    fi
}

case "$stage" in
    plan) plan "$@" ;;
    trace) trace "$@" ;;
    colab) colab "$@" ;;
    colab-fetch) "$here/colab/graphs.sh" "$@" ;;
    stitch) stitch "$@" ;;
    layout) layout "$@" ;;
    preview) preview "$@" ;;
    root) root "$@" ;;
    read) read_ink "${1:-}" ;;
    page) page ;;
    all | queue) all "$stage" "$@" ;;
    *) usage ;;
esac
