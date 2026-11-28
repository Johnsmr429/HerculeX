#!/bin/sh
set -eu
work=$(mktemp -d "${TMPDIR:-/tmp}/herculex-speed-test.XXXXXXXX")
trap 'find "$work" -depth -delete' EXIT
export HERCULEX_HOME="$work"
export HERCULEX_BOARD="$work/board"
binary=${HERCULEX_NATIVE_BIN:?test requires the native HerculeX binary}
mkdir -p "$work/records"
printf '%s\n' '{"name":"solver","seconds":0.001}' >"$work/records/speed.jsonl"
status=0
"$binary" runner.speed solver --record --compare >"$work/report" 2>"$work/errors" || status=$?
test "$status" = 1
grep -q 'terms.*1000000.*SLOWER' "$work/report"
grep -q '^slower than recorded: solver$' "$work/errors"
test "$(wc -l <"$work/records/speed.jsonl" | tr -d ' ')" = 2
grep -q '"repeats":3.*"terms":1000000.*"commit":' "$work/records/speed.jsonl"
printf '%s\n' 'speed e2e passed'
