"""C and Objective-C kernels compiled once per source digest into runtime/kernels, older builds removed, with the
compiler and flags of this machine's lane (runner/machine_apple/build.py, runner/machine_cuda/build.py)."""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

from runner.lane import APPLE
from storage.data_roots import KERNELS

if APPLE:
    from runner.machine_apple import build
else:
    from runner.machine_cuda import build

C_FLAGS = build.C_FLAGS
_LOADED = {}


def load(name, sources, flags=C_FLAGS, headers=(), libraries=()):
    """A process keeps the build it first loaded, whatever lands in the checkout while it runs."""
    if name in _LOADED:
        return _LOADED[name]
    sources = [Path(s) for s in sources]
    key = hashlib.sha256(
        b"".join(s.read_bytes() for s in [*sources, *map(Path, headers)])
        + " ".join(build.key_words(flags, libraries)).encode()
    ).hexdigest()[:20]
    path = KERNELS / f"{name}-{key}.{build.SUFFIX}"
    if not path.exists():
        KERNELS.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        subprocess.run(build.command(flags, temporary, sources, libraries), check=True)
        os.replace(temporary, path)
        for old in KERNELS.glob(f"{name}-*.{build.SUFFIX}"):
            if old != path:
                old.unlink(missing_ok=True)
    _LOADED[name] = ctypes.CDLL(str(path))
    return _LOADED[name]


def pointer(array):
    """A NumPy array's buffer as a ctypes void pointer."""
    return array.ctypes.data_as(ctypes.c_void_p)
