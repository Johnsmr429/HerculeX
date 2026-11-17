"""Off macOS, the dispatch_apply side of parallel_blocks.h must still compile and expand to the original Mac calls."""

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

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" or not shutil.which("clang"), reason="a Mac builds these itself; elsewhere needs clang"
)


@pytest.mark.parametrize("source", SOURCES)
def test_the_mac_side_of_each_kernel_compiles_with_blocks(tmp_path, source):
    (tmp_path / "dispatch").mkdir()
    (tmp_path / "dispatch" / "dispatch.h").write_text(DISPATCH)
    flags = ["-fblocks", "-D__APPLE__", f"-I{tmp_path}", f"-I{LEVEL}", str(LEVEL / source)]
    expanded = subprocess.run(["clang", "-E", *flags], capture_output=True, text=True, check=True).stdout
    assert "dispatch_apply(" in expanded and "pragma omp" not in expanded
    subprocess.run(["clang", "-fsyntax-only", *flags], capture_output=True, check=True)
