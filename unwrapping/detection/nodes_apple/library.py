"""The Apple lane's sheet-node kernel: kernel.metal, compiled on the GPU by the Objective-C host host.m."""

from pathlib import Path

from runner.compiled_kernels import load

HERE = Path(__file__).parent
KERNEL = HERE / "kernel.metal"
HOST = HERE / "host.m"


def host():
    """The compiled host; its open call compiles the kernel's source."""
    return load("sheet-nodes-gpu", [HOST], ("-O2", "-fobjc-arc", "-framework", "Metal", "-framework", "Foundation"))
