#!/usr/bin/env bash
set -uo pipefail
s=$1
log=${2:-/content/read.log}
stop=${UNROLL_WORK:?set UNROLL_WORK to the scan work folder}/keepalive.$s.stop
rm -f "$stop"
while [ ! -e "$stop" ]; do
    if colab sessions 2>/dev/null | grep -q "^\[$s\]"; then
        line=$(printf '%%%%bash\ntail -c 400 %s 2>/dev/null | tail -1 | cut -c1-160\n' "$log" |
            timeout 100 colab exec -s "$s" --timeout 0 2>&1 | tr -d '\r' | sed '/^$/d' | tail -1)
        echo "$(date -u +%H:%M) $line"
    else
        echo "$(date -u +%H:%M) no session $s"
    fi
    sleep 150
done
