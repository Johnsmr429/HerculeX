"""C and Objective-C kernels compiled once per source digest into runtime/kernels, older builds removed."""

import ctypes
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from storage.data_roots import KERNELS

C_FLAGS = ("-O3", "-mcpu=apple-m1")
APPLE = (
    sys.platform == "darwin"
)  # elsewhere (Linux, as in the submission's container) kernels build as plain shared objects
_LOADED = {}


def load(name, sources, flags=C_FLAGS, headers=()):
    """A process keeps the build it first loaded, whatever lands in the checkout while it runs."""
    if name in _LOADED:
        return _LOADED[name]
    sources = [Path(s) for s in sources]
    key = hashlib.sha256(
        b"".join(s.read_bytes() for s in [*sources, *map(Path, headers)]) + " ".join(flags).encode()
    ).hexdigest()[:20]
    if not APPLE:
        flags = tuple(f for f in flags if f != "-mcpu=apple-m1")
    suffix = "dylib" if APPLE else "so"
    path = KERNELS / f"{name}-{key}.{suffix}"
    if not path.exists():
        KERNELS.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        link = ["clang", *flags, "-dynamiclib"] if APPLE else ["cc", *flags, "-shared", "-fPIC"]
        subprocess.run([*link, "-o", str(temporary), *map(str, sources), *([] if APPLE else ["-lm"])], check=True)
        os.replace(temporary, path)
        for old in KERNELS.glob(f"{name}-*.{suffix}"):
            if old != path:
                old.unlink(missing_ok=True)
    _LOADED[name] = ctypes.CDLL(str(path))
    return _LOADED[name]
