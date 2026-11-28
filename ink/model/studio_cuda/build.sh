#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/studio_cuda}
nvcc=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
arch=${STUDIO_ARCH:-}
if [ -z "$arch" ]; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
    [ -n "$cap" ] || {
        echo "build.sh: no GPU found; set STUDIO_ARCH" >&2
        exit 1
    }
    arch=sm_$cap
fi
cc -std=c11 -O3 -Wall -Wextra -c "$here/studio_safetensors.c" -o "$here/studio_safetensors.o"
cc -std=c11 -O3 -Wall -Wextra -c "$here/studio_cuda.c" -o "$here/studio_cuda.o"
"$nvcc" -O3 -std=c++17 -arch="$arch" -lineinfo --use_fast_math -c "$here/studio_kernels.cu" -o "$here/studio_kernels.o"
"$nvcc" -O3 -arch="$arch" "$here/studio_safetensors.o" "$here/studio_cuda.o" "$here/studio_kernels.o" -lm -o "$out"
rm -f "$here/studio_safetensors.o" "$here/studio_cuda.o" "$here/studio_kernels.o"
printf 'built %s (%s)\n' "$out" "$arch"
