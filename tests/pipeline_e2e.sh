#!/bin/sh
set -eu

binary=${HERCULEX_NATIVE_BIN:-$PWD/$(./run.sh build)}
fixture=$(mktemp -d)
trap 'rm -rf "$fixture"' EXIT
export HERCULEX_DATA="$fixture/data" HERCULEX_OUTPUTS="$fixture/data/outputs"
export HERCULEX_LIBRARY="$fixture/data/library" HERCULEX_TRACE="$fixture/trace"

cat >"$fixture/run.sh" <<'SH'
#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$HERCULEX_TRACE"
base=$HERCULEX_OUTPUTS/unwrapping/level/PHercT/v1
unit=rows-0000-0002
case "$1" in
    level.detection.detect_nodes)
        nodes=$HERCULEX_OUTPUTS/unwrapping/nodes/PHercT/v1/nodes
        mkdir -p "$nodes/$unit" "$base"
        printf '{"seconds":1}' > "$nodes/$unit/record.json"
        ln -s "$nodes" "$base/nodes"
        ;;
    level.level.run_levels)
        if [ "$2" = ready ]; then
            mkdir -p "$base/sheet"
            printf '[]\n' > "$base/sheet/axis.json"
            exit 0
        fi
        test -f "$base/nodes/$unit/record.json"
        mkdir -p "$base/pieces/$unit"
        printf '{"seconds":2,"pieces":[]}' > "$base/pieces/$unit/record.json"
        ;;
    board) test -f "$base/sheet/axis.json" ;;
    level.layout.mosaic_layout)
        test -f "$base/pieces/$unit/record.json"
        mkdir -p "$base/sheet"
        printf '{"pieces":[{"area":410}]}' > "$base/sheet/layout.json"
        ;;
    ink) test -f "$base/sheet/layout.json" ;;
    publish) test -f "$base/sheet/review.json" ;;
    runner.review_page) printf '{}' > "$base/sheet/review.json" ;;
esac
SH
chmod +x "$fixture/run.sh"

cd "$fixture"
"$binary" level.pipeline run all PHercT --scan v1
grep -q '^fetch run PHercT --scan v1 --since ' "$HERCULEX_TRACE"
grep -Fxq 'ink PHercT v1' "$HERCULEX_TRACE"
grep -q '^runner.review_page PHercT v1 --url ' "$HERCULEX_TRACE"

"$binary" storage.route_database index | grep -Fxq "{'sheet-nodes': 1, 'pieces': 1}"
"$binary" status PHercT --scan v1 | grep -q '^PHercT  *v1 '
"$binary" status PHercT --scan v2 | grep -Fxq 'No completed units for this selection.'

: >"$HERCULEX_TRACE"
"$binary" runner.sheet_stage sheet PHercT --scan v1
grep -Fxq 'publish --current PHercT v1' "$HERCULEX_TRACE"
if grep -q '^ink ' "$HERCULEX_TRACE"; then
    exit 1
fi

if [ "$(uname -s)" = Darwin ]; then
    printf '{"curl":1,"rule":"set by hand"}\n' > \
        "$HERCULEX_OUTPUTS/unwrapping/level/PHercT/v1/curl.json"
    : >"$HERCULEX_TRACE"
    "$binary" level PHercT:v1
    grep -Fxq 'level.level.run_levels ready PHercT v1' "$HERCULEX_TRACE"
    grep -q '^board add PHercT-fetch ' "$HERCULEX_TRACE"
fi
