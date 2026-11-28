#!/bin/sh
set -eu
kit=$(cd "$(dirname "$0")" && pwd)
build=$1
out=$2
shift 2
chrome=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
cp "$kit/page.html" "$kit/kit.js" "$build/"
for spec in "$@"; do
    name=${spec%%:*}
    w=${spec#*:}
    h=${w#*x}
    w=${w%x*}
    "$chrome" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files \
        --force-device-scale-factor=2 --default-background-color=00000000 --virtual-time-budget=4000 \
        --window-size="$w,$((h + 400))" --screenshot="$build/$name.png" "file://$build/page.html#$name" \
        >/dev/null 2>&1
    sips -c $((h * 2)) $((w * 2)) "$build/$name.png" --out "$out/$name.png" >/dev/null
    echo "$out/$name.png"
done
