#!/bin/sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
scan=${1:-PHerc1545-20250821151648}
work=$root/data/library/cache/unroll/$scan
runs=$root/data/outputs/runs/unroll/$scan
atlas=${2:-$root/data/library/raw/PHerc1545/atlas_9.362um/L3}
page=${3:-sheet-PHerc1545-113keV-v1.5-reader-pu-bhand}
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
# shellcheck disable=SC2046
cc -std=gnu11 -O2 -Wall -Wextra "$here/tiles.c" $(pkg-config --cflags --libs libtiff-4 json-c) -lm -o "$build/cut"
mkdir -p "$build/tiles"
"$build/cut" "$work" "$runs" "$atlas" "$build/tiles"
(cd "$root" && ./run.sh viewer.overview "$page" "$build/tiles/strip.jpg" --width 2400 --no-ink --scan 1 >/dev/null)
cp "$runs/passage_3455mm.png" "$build/tiles/passage.png"
cp "$here/figures.js" "$build/"
"$here/../kit/shoot.sh" "$build" "$here/../../images/unroll" hero:1200x704 trace:1200x516 stitch:1200x512 layout:1200x632
