#!/bin/sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
dump=${1:-$root/data/outputs/runs/model/2026-11-26-v0/mid/dump}
context=${2:-$root/data/library/cache/sheet-engine-v3-results/ct_0139_z5576.npy}
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
cc -std=gnu11 -O2 -Wall -Wextra "$here/tiles.c" -lm -o "$build/cut"
mkdir -p "$build/tiles"
"$build/cut" "$dump" "$context" "$build/tiles"
cp "$here/figures.js" "$build/"
"$here/../kit/shoot.sh" "$build" "$here/../../images/logos" hero:1200x724 winding:1200x476 loop:1200x452 \
    training:1200x492 kernel:1200x290 heldout:1200x822
