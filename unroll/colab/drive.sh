#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
unroll=$(cd "$here/.." && pwd)
hx=$unroll
until [ -x "$hx/run.sh" ] && [ -d "$hx/level/level" ]; do
    if [ "$hx" = / ]; then
        echo "$0: not inside a HerculeX checkout" >&2
        exit 1
    fi
    hx=$(dirname "$hx")
done
work=${UNROLL_WORK:?set UNROLL_WORK to the scan work folder, or run pipeline.sh SCROLL SCAN colab}

remote() {
    local session=$1 line=$2 wait=${3:-300} try text code
    for try in 1 2 3 4 5; do
        text=$(printf '%%%%bash\n(\n%s\n)\nprintf "\\n__RC__=%%s\\n" "$?"\n' "$line" |
            timeout "$wait" colab exec -s "$session" --timeout 0 2>&1) || true
        code=$(printf '%s\n' "$text" | tr -d '\r' | sed -n 's/^__RC__=//p' | tail -1)
        if [ -n "$code" ]; then
            printf '%s\n' "$text" | tr -d '\r' | sed '/^__RC__=/d' | sed '/^$/d'
            return "$code"
        fi
        printf 'colab exec did not answer (try %s of 5)\n' "$try" >&2
        sleep 5
    done
    return 1
}

pack() {
    local t rel
    t=$(mktemp -d)
    rel=${unroll#"$hx"/}
    mkdir -p "$t/code" "$t/hx"
    (cd "$hx" && COPYFILE_DISABLE=1 tar --no-xattrs -cf - "$rel/build.sh" "$rel/trace.c" "$rel/trace_cuda.cu" \
        "$rel/trace_cuda.h" "$rel/stitch.c" "$rel/stitch_cuda.cu" "$rel/stitch_cuda.h" \
        level/level/level_solver.c level/level/sheet_flattening.c level/level/scroll_axis.c \
        level/level/threads_cuda) | tar -xf - -C "$t/hx"
    cp "$unroll/fetch.py" "$work/units_plan.json" "$work/papyrus_mask_L4.npz" "$work/scan.json" "$work/axis.json" \
        "$t/code/"
    if [ -f "$work/trace_args.txt" ]; then
        cp "$work/trace_args.txt" "$t/code/"
    fi
    if [ "${UNROLL_ANCHOR:-0}" = 1 ]; then
        touch "$t/code/anchor"
    fi
    printf '%s\n' "$rel" >"$t/code/path"
    cp "$here/worker.sh" "$here/setup.sh" "$t/"
    COPYFILE_DISABLE=1 tar --no-xattrs -C "$t" -czf "$work/unroll_code.tgz" .
    rm -rf "$t"
}

case "$1" in
    up)
        s=$2
        if ! colab sessions 2>/dev/null | grep -q "^\[$s\]"; then
            if [ "${3:-}" = L4 ]; then
                colab new -s "$s" --gpu L4
            else
                colab new -s "$s" --high-mem
            fi
        fi
        pack
        remote "$s" "mkdir -p /content/unroll"
        colab upload -s "$s" "$work/unroll_code.tgz" /content/unroll/code.tgz >/dev/null
        remote "$s" "cd /content/unroll && tar -xzf code.tgz && bash setup.sh 2>&1 | tail -3" 900
        ;;
    start)
        s=$2
        shift 2
        args=$(printf "'%s'," "$@")
        printf 'import subprocess\nsubprocess.Popen(["bash","/content/unroll/worker.sh",%s], cwd="/content/unroll", stdout=open("/content/unroll/worker.out","a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)\nprint("started")\n' "$args" |
            timeout 120 colab exec -s "$s" --timeout 60 2>&1 | tail -2
        ;;
    status)
        s=$2
        remote "$s" "cd /content/unroll && tail -4 progress.txt 2>/dev/null; for f in logs/*.log; do [ -f \"\$f\" ] && echo \"== \$f\" && tail -2 \"\$f\"; done 2>/dev/null | tail -12; df -h /content | tail -1; free -g | sed -n 2p"
        ;;
    bg)
        s=$2
        script=$3
        log=$4
        printf 'import subprocess\nsubprocess.Popen(["bash","%s"], cwd="/content", stdout=open("%s","a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)\nprint("started")\n' "$script" "$log" |
            timeout 120 colab exec -s "$s" --timeout 60 2>&1 | tail -1
        ;;
    sh)
        s=$2
        shift 2
        remote "$s" "$*"
        ;;
    *)
        echo "usage: drive.sh up|start|status|bg|sh SESSION ..." >&2
        exit 2
        ;;
esac
