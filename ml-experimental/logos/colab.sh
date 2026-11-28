#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
herculex=$here
until [ -x "$herculex/run.sh" ] && [ -d "$herculex/level/level" ]; do
    if [ "$herculex" = / ]; then
        echo "$0: not inside a HerculeX checkout" >&2
        exit 1
    fi
    herculex=$(dirname "$herculex")
done
session=${MODEL_SESSION:-model}
gpu=${MODEL_GPU:-L4}
key=${MODEL_SSH_KEY:-$HOME/.ssh/id_ed25519}
code=/content/model/code

usage() {
    cat >&2 <<'EOF'
Usage: colab.sh up                    start the runtime (MODEL_GPU, default L4), keep its kernel busy, push the code
       colab.sh keepalive             run a cell that never ends, so Colab does not reclaim a runtime used over ssh
       colab.sh push                  send this folder and the umbilicus lines
       colab.sh sh COMMAND            run a shell line on the runtime
       colab.sh start NAME COMMAND    run a shell line in the background as job NAME (log in /content/model/jobs)
       colab.sh log NAME [LINES]      the end of job NAME's log and its exit code once it ended
       colab.sh get REMOTE LOCAL      bring a file home
       colab.sh status                GPU, disc, jobs and the compute units left
       colab.sh down                  stop the runtime
MODEL_SSH_KEY names the SSH key colab ssh sends (default ~/.ssh/id_ed25519).
EOF
    exit 2
}

remote() {
    timeout "${2:-1800}" colab ssh -s "$session" -i "$key" -- bash -c "set -euo pipefail; mkdir -p $code; cd $code; $1"
}

push() {
    local work
    work=$(mktemp -d)
    mkdir -p "$work/code/umbilici"
    find "$here" -maxdepth 1 -type f ! -name '.DS_Store' -exec cp {} "$work/code/" \;
    cp "$herculex/calibrate/bench/controls.json" "$work/code/controls.json"
    cp -R "$herculex/records/umbilici/vesuvius-center-lines/lines" "$herculex/records/umbilici/vesuvius-center-lines/other-scans" \
        "$work/code/umbilici/"
    COPYFILE_DISABLE=1 tar --no-xattrs -C "$work/code" -czf "$work/code.tgz" .
    remote "tar -xzf - -C $code && ls $code" <"$work/code.tgz"
    rm -r "$work"
}

keepalive() {
    local work
    work=$(mktemp -d)
    printf 'import time\nwhile True:\n    time.sleep(60)\n' >"$work/keepalive.py"
    nohup colab exec -s "$session" --timeout 0 -f "$work/keepalive.py" >"$work/keepalive.log" 2>&1 &
    sleep 20
    printf 'keepalive cell started (local log %s)\n' "$work/keepalive.log"
}

[ $# -ge 1 ] || usage
command=$1
shift
case $command in
    up)
        if ! timeout 120 colab sessions 2>/dev/null | grep -q "^\[$session\]"; then
            timeout 600 colab new -s "$session" --gpu "$gpu"
        fi
        keepalive
        push
        ;;
    keepalive) keepalive ;;
    push) push ;;
    sh)
        [ $# -ge 1 ] || usage
        remote "$*"
        ;;
    start)
        [ $# -ge 2 ] || usage
        name=$1
        shift
        remote "mkdir -p /content/model/jobs; rm -f /content/model/jobs/$name.rc; nohup bash -c $(printf '%q' "cd $code && ( $* ) > /content/model/jobs/$name.log 2>&1; echo \$? > /content/model/jobs/$name.rc") >/dev/null 2>&1 </dev/null & echo started $name"
        ;;
    log)
        [ $# -ge 1 ] || usage
        remote "tail -n ${2:-40} /content/model/jobs/$1.log | tr '\r' '\n' | tail -n ${2:-40}; if [ -f /content/model/jobs/$1.rc ]; then echo EXIT \$(cat /content/model/jobs/$1.rc); else echo RUNNING; fi"
        ;;
    get)
        [ $# -eq 2 ] || usage
        remote "cat '$1'" >"$2.part"
        mv "$2.part" "$2"
        ;;
    status)
        remote "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; df -h /content | tail -1; ls /content/model/jobs 2>/dev/null | tr '\n' ' '; echo" || true
        timeout 120 colab usage 2>&1 | head -3
        ;;
    down) timeout 120 colab stop -s "$session" ;;
    *) usage ;;
esac
