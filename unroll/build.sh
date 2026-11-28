#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
hx=$here
until [ -x "$hx/run.sh" ] && [ -d "$hx/level/level" ]; do
    if [ "$hx" = / ]; then
        echo "$0: not inside a HerculeX checkout" >&2
        exit 1
    fi
    hx=$(dirname "$hx")
done
cd "$hx"
lane=threads_apple
extra=(-fblocks)
libs=(json-c libtiff-4)
link=()
trace_cuda=()
stitch_cuda=()
mkdir -p runtime/bin
if [ "$(uname)" = Darwin ]; then
    link=(-framework Accelerate)
else
    lane=threads_cuda
    extra=(-fopenmp -Wno-format-truncation)
    if command -v nvcc >/dev/null 2>&1; then
        codes=$(nvcc --list-gpu-code 2>/dev/null || true)
        arch=()
        for sm in 80 89 120; do
            if grep -qx "sm_$sm" <<<"$codes"; then
                arch+=(-gencode "arch=compute_$sm,code=sm_$sm")
            fi
        done
        if grep -qx sm_75 <<<"$codes"; then
            arch+=(-gencode "arch=compute_75,code=[sm_75,compute_75]")
        fi
        nvcc -O3 ${arch[@]+"${arch[@]}"} -c "level/level/$lane/flow.cu" -o runtime/bin/unroll-flow.o
        nvcc -O3 -fmad=false ${arch[@]+"${arch[@]}"} -c "$here/trace_cuda.cu" -o runtime/bin/unroll-trace.o
        nvcc -O3 -fmad=false ${arch[@]+"${arch[@]}"} -c "$here/stitch_cuda.cu" -o runtime/bin/unroll-stitch.o
        cuda_lib=$(dirname "$(dirname "$(command -v nvcc)")")/lib64
        if [ -f "$cuda_lib/libcudart_static.a" ]; then
            link=(-L"$cuda_lib" -lcudart_static -ldl -lrt -lpthread -lstdc++)
        else
            link=(-L"$cuda_lib" -lcudart -lstdc++)
        fi
        extra+=(-DHERCULEX_CUDA_FLOW runtime/bin/unroll-flow.o)
        trace_cuda=(-DTRACE_CUDA runtime/bin/unroll-trace.o)
        stitch_cuda=(-DSTITCH_CUDA runtime/bin/unroll-stitch.o)
    fi
fi
read -r -a cflags <<<"$(pkg-config --cflags "${libs[@]}")"
read -r -a ldflags <<<"$(pkg-config --libs "${libs[@]}")"
build() {
    local source=$1 out=$2
    shift 2
    "${CC:-cc}" -std=gnu11 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
        -I"level/level/$lane" "${extra[@]}" "$@" "${cflags[@]}" "$source" \
        level/level/level_solver.c level/level/sheet_flattening.c level/level/scroll_axis.c \
        "${ldflags[@]}" ${link[@]+"${link[@]}"} -lm -o "$out"
}
build "$here/trace.c" runtime/bin/unroll-trace ${trace_cuda[@]+"${trace_cuda[@]}"}
build "$here/stitch.c" runtime/bin/unroll-stitch ${stitch_cuda[@]+"${stitch_cuda[@]}"}
printf '%s\n%s\n' runtime/bin/unroll-trace runtime/bin/unroll-stitch
