#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
outputs=${HERCULEX_OUTPUTS:-data/outputs}
scan=unwrapping/level/PHerc0211/20250821151803
unit=rows-0000-0012
pieces="15 396 398"
if [ ! -f "$outputs/$scan/sheet/layout.json" ] || [ ! -f "$outputs/$scan/sheet/axis.json" ] ||
    [ ! -d "$outputs/$scan/pieces/$unit" ] || ! command -v jq >/dev/null; then
    printf 'The PHerc0211 pieces, layout and axis or jq are unavailable under %s.\n' "$outputs"
    exit 77
fi
binary=${HERCULEX_NATIVE_BIN:-$PWD/$(./run.sh build)}
source=$(cd "$outputs/$scan" && pwd -P)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
target=$work/outputs/$scan
mkdir -p "$target/sheet" "$work/bricks"
ln -s "$source/pieces" "$target/pieces"
ln -s "$source/sheet/axis.json" "$target/sheet/axis.json"
jq --arg unit "$unit" --argjson keep "[$(echo "$pieces" | tr ' ' ',')]" \
    '.pieces |= map(select(.unit == $unit and (.piece as $p | $keep | index($p))))' \
    "$source/sheet/layout.json" >"$target/sheet/layout.json"
read_ink() {
    HERCULEX_OUTPUTS=$work/outputs HERCULEX_BRICKS=$work/bricks HERCULEX_INK_MODEL=reader-v3 \
        HERCULEX_UNITS=$unit "$binary" ink PHerc0211 20250821151803
}
read_ink
faces=$target/faces/level-0/$unit
for piece in $pieces; do
    name=$faces/$(printf 'piece-%04d' "$piece")
    for suffix in .png -layer0.png -taught.png -taught-side.json; do
        [ -s "$name$suffix" ] || {
            echo "piece $piece has no $suffix"
            exit 1
        }
    done
    size=$(file "$name.png" | sed 's/.*PNG image data, \([0-9]* x [0-9]*\).*/\1/')
    [ "$(file "$name-taught.png" | sed 's/.*PNG image data, \([0-9]* x [0-9]*\).*/\1/')" = "$size" ] ||
        {
            echo "piece $piece ink is not $size"
            exit 1
        }
    written=$(jq -r .written "$name-taught-side.json")
    [ "$written" = near ] || [ "$written" = far ] || {
        echo "piece $piece written face $written"
        exit 1
    }
    printf 'piece %s: %s, written %s\n' "$piece" "$size" "$written"
done
read_ink | grep -q 'no ink maps to update' || {
    echo "a second read redid finished pieces"
    exit 1
}
echo "sheet stage e2e passed"
