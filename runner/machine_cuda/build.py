"""How the CUDA lane builds kernels: HERCULEX_CC (cc unless set) with OpenMP and HERCULEX_CFLAGS (-march=native unless
set), into shared objects. A build is kept per processor, since -march=native builds for the one it runs on."""

import os
import platform
import re
from pathlib import Path

CC = os.environ.get("HERCULEX_CC", "cc")
EXTRA = tuple(os.environ.get("HERCULEX_CFLAGS", "-march=native").split())
C_FLAGS = ("-O3",)
SUFFIX = "so"
CPUINFO = Path("/proc/cpuinfo")


def processor():
    """The processor's model name, or its architecture where /proc/cpuinfo names none."""
    found = re.search(r"^model name\s*:\s*(.+)$", CPUINFO.read_text(), re.MULTILINE) if CPUINFO.exists() else None
    return found.group(1).strip() if found else platform.machine()


def key_words(flags, libraries):
    """What decides a build besides its sources."""
    return (*flags, *libraries, CC, *EXTRA, processor())


def command(flags, out, sources, libraries):
    """The compiler call that builds sources into out; Apple-only flags are dropped."""
    kept = [flag for flag in flags if not flag.startswith("-mcpu=apple")]
    return [CC, *kept, *EXTRA, "-fopenmp", "-shared", "-fPIC", "-o", str(out), *map(str, sources), *libraries, "-lm"]
