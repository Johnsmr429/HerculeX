"""The lane this machine runs: apple (Apple silicon: Metal and the Neural Engine) or cuda (Linux, an NVIDIA GPU).

A Mac runs the apple lane and any other machine the cuda lane; HERCULEX_LANE, when set, must name the same one. A
module keeps what only one lane runs in a folder named for it, such as unwrapping/detection/nodes_apple and nodes_cuda.
"""

import os
import sys

LANES = ("apple", "cuda")
MACHINE_LANE = "apple" if sys.platform == "darwin" else "cuda"
LANE = os.environ.get("HERCULEX_LANE") or MACHINE_LANE
if LANE not in LANES:
    raise SystemExit(f"HERCULEX_LANE={LANE}: the lanes are {' and '.join(LANES)}")
if LANE != MACHINE_LANE:
    raise SystemExit(f"HERCULEX_LANE={LANE}: this machine runs the {MACHINE_LANE} lane")
APPLE = LANE == "apple"
CUDA = LANE == "cuda"


def torch_device():
    """This lane's GPU as a PyTorch device, mps or cuda, or cpu when PyTorch sees none."""
    import torch

    if APPLE:
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
