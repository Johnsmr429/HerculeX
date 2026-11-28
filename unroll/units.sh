#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
hx=$here
until [ -x "$hx/run.sh" ] && [ -d "$hx/level/level" ]; do
    if [ "$hx" = / ]; then
        echo "$0: not inside a HerculeX checkout" >&2
        exit 1
    fi
    hx=$(dirname "$hx")
done
work=${UNROLL_WORK:?set UNROLL_WORK to the scan work folder, or run pipeline.sh SCROLL SCAN trace}
out=$work${UNROLL_SET:+/sets/$UNROLL_SET}
mkdir -p "$out/graphs" "$out/units" "$out/logs" "$out/tmp"
setting() {
    python3 -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$work/scan.json" "$1"
}
scroll=$(setting scroll)
scan=$(setting scan)
curl=$(setting curl)
voxel=$(setting voxel_um)
read -r -a extra <<<"${UNROLL_TRACE_ARGS:-}"
anchor=()
if [ "${UNROLL_ANCHOR:-0}" = 1 ]; then
    anchor=(--anchor "$out/graphs")
fi
cd "$hx" || exit 1
pending=()
for name in "$@"; do
    if [ -s "$out/graphs/$name/surfels.bin" ] &&
        [ "$(wc -c <"$out/graphs/$name/surfels.bin")" -ge 16 ] && [ -s "$out/units/$name/record.json" ]; then
        echo "SKIP $name"
        continue
    fi
    pending+=("$name")
done
prefetch_pid=""
prefetch_npy=""
cleanup() {
    if [ -n "$prefetch_pid" ]; then
        kill "$prefetch_pid" 2>/dev/null || true
        wait "$prefetch_pid" 2>/dev/null || true
    fi
    if [ -n "$prefetch_npy" ]; then
        rm -f "$prefetch_npy"
    fi
}
trap cleanup EXIT INT TERM
start_fetch() {
    local uname=$1
    prefetch_npy="$out/tmp/$uname.npy"
    python3 -I "$here/fetch.py" unit "$uname" "$prefetch_npy" --work "$work" >"$out/logs/$uname.fetch.log" 2>&1 &
    prefetch_pid=$!
}
failed=0
total=${#pending[@]}
if [ "$total" -gt 0 ]; then
    start_fetch "${pending[0]}"
fi
for ((i = 0; i < total; i++)); do
    name=${pending[$i]}
    box=$(python3 -I -c 'import json,sys;u={x["name"]:x for x in json.load(open(sys.argv[1]))}[sys.argv[2]];print("%d:%d,%d:%d,%d:%d" % (u["z0"], u["z1"], u["y0"], u["y1"], u["x0"], u["x1"]))' "$work/units_plan.json" "$name")
    echo "START $name $box $(date +%s)"
    cur_pid=$prefetch_pid
    cur_npy=$prefetch_npy
    prefetch_pid=""
    prefetch_npy=""
    fetch_ok=0
    if wait "$cur_pid"; then
        fetch_ok=1
    fi
    if [ $((i + 1)) -lt "$total" ]; then
        start_fetch "${pending[$((i + 1))]}"
    fi
    if [ "$fetch_ok" -ne 1 ]; then
        echo "FAIL fetch $name: see $out/logs/$name.fetch.log" >&2
        failed=$((failed + 1))
        rm -f "$cur_npy"
        continue
    fi
    if runtime/bin/unroll-trace --surface "$cur_npy" --axis "$work/axis.json" --box "$box" \
        --curl "$curl" --stride 8 --voxel "$voxel" --min-nodes 200 --sample "$scroll" --scan "$scan" \
        ${anchor[@]+"${anchor[@]}"} ${extra[@]+"${extra[@]}"} \
        --save-graph "$out/graphs/$name" --skip-emit --out "$out/units/$name" >"$out/logs/$name.log" 2>&1; then
        echo "DONE $name $(date +%s)"
    else
        echo "FAIL trace $name: see $out/logs/$name.log" >&2
        failed=$((failed + 1))
    fi
    rm -f "$cur_npy"
done
if [ "$failed" -gt 0 ]; then
    printf '%s unit(s) failed; rerun the trace stage to retry them.\n' "$failed" >&2
    exit 1
fi
