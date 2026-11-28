#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${MODEL_ROOT:-/content/model}
model=${MODEL_BIN:-$root/bin/model}
stage=${MODEL_STAGE:-mid}

usage() {
    cat >&2 <<'EOF'
Usage: scrolls.sh SCROLL_SCAN ...
Runs the last checkpoint of MODEL_STAGE (default mid) on every unlabelled box of each scan, unmasked, and stitches the
slabs into boxes under MODEL_ROOT/runs/STAGE/scrolls/SCROLL_SCAN. SCROLL_SCAN names a scan of scans.txt whose
shards/pre shard prep.sh wrote, e.g. PHerc1545_20250821151648. A chain.sh still running is waited for first.
EOF
    exit 2
}

[ $# -ge 1 ] || usage
if [ -f "$root/jobs/chain.log" ]; then
    until [ -f "$root/jobs/dump.rc" ]; do sleep 60; done
fi
ckpt=$(find "$root/runs/$stage" -name 'step-*.ckpt' | sort -t- -k2 -n | tail -1)
[ -n "$ckpt" ] || {
    echo "scrolls.sh: no checkpoint in $root/runs/$stage" >&2
    exit 1
}
echo "checkpoint $ckpt"
for scan in "$@"; do
    [ -s "$root/shards/pre/$scan.shard" ] || {
        echo "scrolls.sh: no shard $root/shards/pre/$scan.shard; add the scan to scans.txt and run prep.sh unlabelled" >&2
        exit 1
    }
    set_dir=$root/shards/sets/scroll-$scan
    rm -rf "$set_dir"
    mkdir -p "$set_dir"
    ln -s "$root/shards/pre/$scan.shard" "$set_dir/"
    out=$root/runs/$stage/scrolls/$scan
    rm -rf "$out"
    mkdir -p "$out"
    "$model" dump --ckpt "$ckpt" --shards "$set_dir" --out "$out/slabs" --per-source 1000 --loops 8
    python3 -I "$here/stitch.py" "$out/slabs" "$out" "${scan%%_*}"
    rm -rf "$out/slabs" "$set_dir"
done
du -sh "$root/runs/$stage/scrolls"
