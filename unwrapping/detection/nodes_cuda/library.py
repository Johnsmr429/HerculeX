"""The CUDA lane's sheet-node kernel: kernel.cu, compiled at run time with NVRTC by the C host host.c."""

import ctypes
import ctypes.util
import sys
from pathlib import Path

from runner.compiled_kernels import load

HERE = Path(__file__).parent
KERNEL = HERE / "kernel.cu"
HOST = HERE / "host.c"


def runtime():
    """libcuda, from the NVIDIA driver, and libnvrtc, from PyTorch's CUDA wheels, loaded for the host to call; NVRTC's
    builtins go first, since NVRTC opens them by name, which fails inside a wheel's folder."""
    wheels = [found for entry in sys.path for found in sorted(Path(entry).glob("nvidia/*/lib/libnvrtc.so.*"))]
    nvrtc = wheels[-1] if wheels else Path(ctypes.util.find_library("nvrtc") or "libnvrtc.so")
    for name in ("libcuda.so.1", *sorted(nvrtc.parent.glob("libnvrtc-builtins.so.*")), nvrtc):
        try:
            ctypes.CDLL(str(name), mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            raise RuntimeError(f"sheet detection on the GPU needs {name}: {error}") from error


def host():
    """The compiled host; its open call compiles the kernel's source."""
    runtime()
    return load("sheet-nodes-cuda", [HOST], ("-O2",))
