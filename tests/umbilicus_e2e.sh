#!/bin/sh
set -eu

binary=${HERCULEX_NATIVE_BIN:-$PWD/$(./run.sh build)}
fixture=$(mktemp -d)
fixture=$(cd "$fixture" && pwd -P)
trap 'rm -rf "$fixture"' EXIT
export HERCULEX_OUTPUTS="$fixture/outputs" HERCULEX_LIBRARY="$fixture/library"
lines="$fixture/records/umbilici/vesuvius-center-lines"
mkdir -p "$lines/lines/PHercT" "$HERCULEX_LIBRARY"
entries=
for row in finder-v2:model:1 finder:model:2 drobkov:person:3 bucket:person:4 \
    bucket-mapped:person-mapped:5 old:person:6; do
    source=${row%%:*}
    rest=${row#*:}
    kind=${rest%%:*}
    x=${rest#*:}
    file="lines/PHercT/PHercT_1_${source}_umbilicus.json"
    printf '{"control_points":[{"z":900,"x":%s,"y":5},{"z":100,"x":%s,"y":5}]}\n' "$x" "$x" \
        >"$lines/$file"
    extra=
    if [ "$source" = old ]; then
        extra=',"superseded_by":"bucket"'
    fi
    entries="$entries${entries:+,}{\"scroll\":\"PHercT\",\"scan\":\"1\",\"source\":\"$source\",\"kind\":\"$kind\",\"file\":\"$file\"$extra}"
done
printf '{"files":[],"other_scans":[%s]}\n' "$entries" >"$lines/index.json"
cd "$fixture"

sources() {
    "$binary" level.level.umbilicus PHercT 1 |
        sed -n 's/^ *\[ *"\([^"]*\)",.*/\1/p; s/^ *"\([a-z0-9-]*\)",$/\1/p' |
        grep -v -E '^(person|model|person-mapped)$' | tr '\n' ' '
}

order=$(sources)
test "$order" = "bucket drobkov bucket-mapped finder-v2 finder "

printf '{"PHercT-1":{"source":"finder","maker":"Center Finder","file":"lines/PHercT/PHercT_1_finder_umbilicus.json"}}\n' \
    >records/umbilici/chosen.json
order=$(sources)
case "$order" in
    "finder bucket "*) ;;
    *) exit 1 ;;
esac
rm records/umbilici/chosen.json

sheet="$HERCULEX_OUTPUTS/unwrapping/level/PHercT/1/sheet"
axis() { tr -d ' \n' <"$sheet/axis.json"; }
"$binary" level.level.umbilicus check PHercT 1
"$binary" level.level.umbilicus PHercT 1 drobkov >/dev/null
test "$(axis)" = '[[100.0,3.0,5.0],[900.0,3.0,5.0]]'
grep -q '"source": *"drobkov"' "$sheet/axis-source.json"
"$binary" level.level.umbilicus PHercT 1 bucket >/dev/null
test "$(axis)" = '[[100.0,4.0,5.0],[900.0,4.0,5.0]]'
grep -q '"source": *"bucket"' "$sheet/axis-source.json"
tr -d '\\ \n' <"$sheet/axis-source.json" |
    grep -Fq '"from":"vesuvius-center-lines/lines/PHercT/PHercT_1_bucket_umbilicus.json"'

result=0
"$binary" level.level.umbilicus PHercT 1 nobody >/dev/null 2>nobody.log || result=$?
test "$result" -eq 1
grep -Fq 'PHercT 1: no umbilicus from nobody on hand' nobody.log
test "$(axis)" = '[[100.0,4.0,5.0],[900.0,4.0,5.0]]'

result=0
"$binary" level.level.umbilicus check PHercT 2 || result=$?
test "$result" -eq 1

virtual="$HERCULEX_OUTPUTS/unwrapping/level/PHercV/virtual-L2/sheet"
mkdir -p "$virtual"
printf '[[0.0,7.0,8.0],[50.0,7.0,8.0]]\n' >"$virtual/axis.json"
"$binary" level.level.umbilicus check PHercV virtual-L2
"$binary" level.level.umbilicus PHercV virtual-L2 >shown.json
tr -d ' \n' <shown.json | grep -Fq '"kept":"kept","on_hand":[]'
test "$(tr -d ' \n' <"$virtual/axis.json")" = '[[0.0,7.0,8.0],[50.0,7.0,8.0]]'
echo "umbilicus e2e passed"
