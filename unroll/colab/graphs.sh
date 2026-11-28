#!/usr/bin/env bash
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
base=${UNROLL_WORK:?set UNROLL_WORK to the scan work folder, or run pipeline.sh SCROLL SCAN colab-fetch}
work=$base${UNROLL_SET:+/sets/$UNROLL_SET}
s=$1
done_units=$("$here/drive.sh" sh "$s" "grep '^DONE' /content/unroll/progress.txt | cut -d' ' -f2" | grep '^unit_')
for n in $done_units; do
    if [ -s "$work/graphs/$n/surfels.bin" ] && [ -s "$work/units/$n/record.json" ]; then
        echo "have $n"
        continue
    fi
    mkdir -p "$work/graphs/$n" "$work/units/$n"
    want=$("$here/drive.sh" sh "$s" "stat -c %s /content/unroll/graphs/$n/surfels.bin" | tail -1)
    for _ in 1 2 3; do
        timeout 1800 colab download -s "$s" "/content/unroll/graphs/$n/surfels.bin" "$work/graphs/$n/surfels.bin.part" >/dev/null 2>&1
        have=$(wc -c <"$work/graphs/$n/surfels.bin.part" 2>/dev/null | tr -d ' ' || echo 0)
        if [ "$have" = "$want" ]; then
            mv "$work/graphs/$n/surfels.bin.part" "$work/graphs/$n/surfels.bin"
            break
        fi
        echo "retry $n ($have of $want)"
    done
    timeout 600 colab download -s "$s" "/content/unroll/units/$n/record.json" "$work/units/$n/record.json" >/dev/null 2>&1
    timeout 600 colab download -s "$s" "/content/unroll/logs/$n.log" "$work/logs/$n.log" >/dev/null 2>&1
    echo "got $n $(wc -c <"$work/graphs/$n/surfels.bin" 2>/dev/null | tr -d ' ')"
done
