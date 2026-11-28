#!/usr/bin/env bash
set -euo pipefail
root=/content/unroll
cd "$root"
export DEBIAN_FRONTEND=noninteractive
if ! pkg-config --exists json-c libtiff-4 2>/dev/null; then
    apt-get -qq update >/dev/null
    apt-get -qq install -y libjson-c-dev libtiff-dev pkg-config >/dev/null
fi
python3 -c "import blosc" 2>/dev/null || pip -q install blosc >/dev/null
if [ -d /usr/local/cuda/bin ]; then
    export PATH=/usr/local/cuda/bin:$PATH
fi
if ! (cd hx && bash "$(cat ../code/path)/build.sh") >build.log 2>&1; then
    tail -20 build.log
    exit 1
fi
ln -sfn "$root/hx/runtime/bin/unroll-trace" "$root/trace"
grep -E 'nvcc|cuda' build.log | head -3 || true
./trace 2>&1 | head -1 || true
echo SETUP_OK
