#!/bin/sh
set -eu

binary=${HERCULEX_NATIVE_BIN:-$PWD/$(./run.sh build)}
fixture=$(mktemp -d)
fixture=$(cd "$fixture" && pwd -P)
trap 'rm -rf "$fixture"' EXIT
export HERCULEX_OUTPUTS="$fixture/outputs" HERCULEX_LIBRARY="$fixture/library"
export HERCULEX_SCRATCH="$fixture/scratch"
nodes="$HERCULEX_OUTPUTS/unwrapping/nodes/S/V/nodes"
level="$HERCULEX_OUTPUTS/unwrapping/level/S/V"
faces="$level/faces/level-0/rows-0000-0012"
pieces="$level/pieces/rows-0000-0012"
stores="$HERCULEX_LIBRARY/cache/bricks/S"
work="$HERCULEX_OUTPUTS/unwrapping/detection-work/S/V"
mkdir -p "$nodes/rows-0000-0012.building" "$nodes/rows-0011-0023.building" \
    "$nodes/rows-0022-0030" "$faces" "$stores/read" "$stores/unread" "$work/L1-0022-0030" \
    "$work/L1-0000-0012" "$HERCULEX_SCRATCH/old-run" "$HERCULEX_SCRATCH/new-run"
touch -t 201901010000 "$nodes/rows-0000-0012.building" "$nodes/rows-0011-0023.building" \
    "$HERCULEX_SCRATCH/old-run"
touch "$nodes/.rows-0011-0023.claim" "$faces/piece-0001.png" "$faces/piece-0002.png"
printf '{"method":"m"}\n' >"$nodes/rows-0022-0030/record.json"
printf '["%s/piece-0001","%s/piece-0002"]\n' "$pieces" "$pieces" >"$stores/read/planned.json"
printf '["%s/piece-0001","%s/piece-0003"]\n' "$pieces" "$pieces" >"$stores/unread/planned.json"
"$binary" runner.disk reclaimable >"$fixture/disk.txt"
for included in "$nodes/rows-0000-0012.building" "$stores/read" "$work/L1-0022-0030" \
    "$HERCULEX_SCRATCH/old-run"; do
    grep -Fq "$(cd "$included" && pwd -P)" "$fixture/disk.txt"
done
if grep -Eq '(rows-0011-0023.building|unread|L1-0000-0012|new-run)' \
    "$fixture/disk.txt"; then
    exit 1
fi

build() {
    mkdir -p "$1/sheet"
    printf '{"pieces":[],"fixture":"%s"}\n' "$2" >"$1/sheet/layout.json"
    cp "$1/sheet/layout.json" "$1/sheet/placed.json"
    printf '{}\n' >"$1/sheet/review.json"
}

page() {
    mkdir -p "$1"
    touch "$1/card.jpg"
    printf '%s%s%s%s\n' '<script>const M={"scan":"level/S/V","title":"' "$2" \
        '","layers":[]},TOP=32;' 'floor=kept.floor??128,scan=kept.scan??0.25;</script>' \
        >"$1/index.html"
}

sheets="$HERCULEX_OUTPUTS/website/sheets"
staged="$fixture/staged"
build "$level" old
page "$sheets/sheet-S-1keV-level" 'old build'
build "$staged/unwrapping/level/S/V" new
page "$staged/website/sheets/sheet-S-1keV-level" 'new build'
"$binary" runner.place_build "$staged" S
grep -Fq '"new"' "$level/sheet/layout.json"
test "$(cd "$level/nodes" && pwd -P)" = "$(cd "$nodes" && pwd -P)"
grep -Fq 'new build' "$sheets/sheet-S-1keV-level/index.html"
grep -Fq '"old"' "$HERCULEX_OUTPUTS"/archive/*-build/unwrapping/level/S/V/sheet/layout.json
for archived in "$sheets"/sheet-S-1keV-level-*; do
    test -f "$archived/placed.json"
    grep -Fq "part/${archived##*/}" "$archived/index.html"
done
build "$staged/unwrapping/level/S/V" new
page "$staged/website/sheets/sheet-S-1keV-level" 'new build read by m'
"$binary" runner.place_build "$staged" S --model m
test -f "$level/sheet/review-m.json"
test -f "$sheets/sheet-S-1keV-level-m/index.html"
grep -Fq 'new build' "$sheets/sheet-S-1keV-level/index.html"

incomplete="$HERCULEX_OUTPUTS/unwrapping/level/S/NO-PAGE"
staged="$fixture/staged-incomplete"
mkdir -p "$incomplete/pieces"
build "$staged/unwrapping/level/S/NO-PAGE" complete
"$binary" runner.place_build "$staged" S
grep -Fq '"complete"' "$incomplete/sheet/layout.json"
set -- "$HERCULEX_OUTPUTS"/archive/*-build/unwrapping/level/S/NO-PAGE/pieces
test -d "$1"

for scroll in PHercA PHercC; do
    root="$HERCULEX_OUTPUTS/unwrapping/level/$scroll/1"
    mkdir -p "$root"
    printf '{"curl":1,"rule":"set by hand"}\n' >"$root/curl.json"
done
cat >"$fixture/run.sh" <<'EOF2'
#!/bin/sh
test "$1" = level.level.umbilicus && test "$2" = check && test "$3" != PHercC
EOF2
chmod +x "$fixture/run.sh"
cd "$fixture"
result=0
"$binary" level PHercA:1 PHercB:1 PHercC:1 >unwrap.log 2>&1 || result=$?
test "$result" -eq 2
grep -Fq 'run_levels curl PHercB 1' unwrap.log
grep -Fq 'umbilicus PHercC 1' unwrap.log
if grep -q 'PHercA' unwrap.log; then
    exit 1
fi

virtual="$HERCULEX_OUTPUTS/unwrapping/level/PHercV/virtual-L2/sheet"
mkdir -p "$virtual"
printf '[[0,7,8],[50,7,8]]\n' >"$virtual/axis.json"
"$binary" level.level.umbilicus check PHercV virtual-L2
