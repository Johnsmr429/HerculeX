#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${MODEL_ROOT:-/content/model}/bin

usage() {
    cat >&2 <<'EOF'
Usage: build.sh [all | tools | model]
Builds into MODEL_ROOT/bin (default /content/model/bin): fetch and prep, the CPU tools, with tools; model, the CUDA
network, with model; both with all (the default). The model needs an NVIDIA GPU of compute capability 8.0 or newer and
is built for the GPU nvidia-smi reports, or for MODEL_ARCH (e.g. sm_89).
EOF
    exit 2
}

tools() {
    local flags
    mkdir -p "$bin"
    if ! pkg-config --exists json-c libtiff-4 libcurl zlib blosc 2>/dev/null; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get -qq update
        apt-get -qq install -y libjson-c-dev libtiff-dev libcurl4-openssl-dev zlib1g-dev libblosc-dev pkg-config >/dev/null
    fi
    read -r -a flags <<<"$(pkg-config --cflags --libs json-c libtiff-4 libcurl zlib blosc)"
    cc -std=gnu11 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-format-truncation "$here/fetch.c" "${flags[@]}" -lm \
        -o "$bin/fetch"
    cc -std=gnu11 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-format-truncation "$here/prep.c" "${flags[@]}" -lm \
        -o "$bin/prep"
}

model() {
    local nvcc arch=${MODEL_ARCH:-} cap
    nvcc=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
    if [ -z "$arch" ]; then
        cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
        [ -n "$cap" ] || {
            echo "build.sh: no GPU found; set MODEL_ARCH" >&2
            exit 1
        }
        [ "$cap" -ge 80 ] || {
            echo "build.sh: the model needs compute capability 8.0 or newer, this GPU has ${cap:0:1}.${cap:1}" >&2
            exit 1
        }
        arch=sm_$cap
    fi
    mkdir -p "$bin"
    if ! "$nvcc" -O3 -std=c++17 -arch="$arch" -lineinfo "$here/model.cu" -lz -o "$bin/model" 2>/dev/null; then
        local fallback
        for fallback in compute_120 compute_100 compute_90 compute_89 compute_86 compute_80; do
            if "$nvcc" -O3 -std=c++17 -arch="$fallback" -lineinfo "$here/model.cu" -lz -o "$bin/model" 2>/dev/null; then
                printf 'build.sh: %s unsupported by %s, built PTX with -arch=%s\n' "$arch" "$nvcc" "$fallback" >&2
                return 0
            fi
        done
        "$nvcc" -O3 -std=c++17 -arch="$arch" -lineinfo "$here/model.cu" -lz -o "$bin/model"
    fi
}

case ${1:-all} in
    all) tools && model ;;
    tools) tools ;;
    model) model ;;
    *) usage ;;
esac
