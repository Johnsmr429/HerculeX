#!/bin/sh
set -eu

if ! command -v clang >/dev/null; then
    echo "parallel blocks golden needs clang"
    exit 77
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/dispatch"
cat >"$work/dispatch/dispatch.h" <<'EOF'
#include <stddef.h>
typedef void *dispatch_queue_t;
typedef enum { QOS_CLASS_USER_INITIATED = 0x19 } qos_class_t;
dispatch_queue_t dispatch_get_global_queue(long identifier, unsigned long flags);
void dispatch_apply_f(size_t iterations, dispatch_queue_t queue, void *context, void (*run)(void *, size_t));
EOF
level=level/level
for source in level_solver.c patch_agglomeration.c piece_kernels.c; do
    for lane in apple cuda; do
        if [ "$lane" = apple ]; then
            set -- -fblocks
            expected="dispatch_apply_f("
            other="pragma omp"
            if [ "$(uname)" != Darwin ]; then
                set -- "$@" "-I$work"
            fi
        else
            set -- -Wno-unknown-pragmas
            expected="pragma omp parallel for"
            other="dispatch_apply"
        fi
        clang -E "$@" "-I$level/threads_$lane" "$level/$source" >"$work/expanded.c"
        grep -Fq "$expected" "$work/expanded.c"
        if grep -Fq "$other" "$work/expanded.c"; then
            echo "$source on $lane expands to $other"
            exit 1
        fi
        clang -fsyntax-only "$@" "-I$level/threads_$lane" "$level/$source"
    done
done
echo "parallel blocks golden passed"
