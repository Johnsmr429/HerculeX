"""Every level kernel builds with each lane's parallel_blocks.h: the Apple lane's dispatch_apply calls and the CUDA
lane's OpenMP loops, so a change made on one lane's machine is checked against the other lane's header too."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LEVEL = Path(__file__).parents[1] / "unwrapping" / "level"
SOURCES = ("level_solver.c", "patch_agglomeration.c", "piece_kernels.c")
DISPATCH = """#include <stddef.h>
typedef void *dispatch_queue_t;
typedef enum { QOS_CLASS_USER_INITIATED = 0x19 } qos_class_t;
dispatch_queue_t dispatch_get_global_queue(long identifier, unsigned long flags);
void dispatch_apply(size_t iterations, dispatch_queue_t queue, void (^block)(size_t));
"""

pytestmark = pytest.mark.skipif(not shutil.which("clang"), reason="needs clang")


def clang(*flags):
    return subprocess.run(["clang", *flags], capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("source", SOURCES)
def test_each_kernel_builds_with_the_apple_lane_loop(tmp_path, source):
    (tmp_path / "dispatch").mkdir()
    (tmp_path / "dispatch" / "dispatch.h").write_text(DISPATCH)
    stand_in = [] if sys.platform == "darwin" else [f"-I{tmp_path}"]
    flags = ["-fblocks", *stand_in, f"-I{LEVEL / 'threads_apple'}", str(LEVEL / source)]
    expanded = clang("-E", *flags)
    assert "dispatch_apply(" in expanded and "pragma omp" not in expanded
    clang("-fsyntax-only", *flags)


@pytest.mark.parametrize("source", SOURCES)
def test_each_kernel_builds_with_the_cuda_lane_loop(source):
    flags = ["-Wno-unknown-pragmas", f"-I{LEVEL / 'threads_cuda'}", str(LEVEL / source)]
    expanded = clang("-E", *flags)
    assert "pragma omp parallel for" in expanded and "dispatch_apply" not in expanded
    clang("-fsyntax-only", *flags)
