"""The public ink model the pipeline reads with: stock Reader v3, fetched once, checked and converted.

./run.sh ink.model.weights fetch      download Reader v3 (MIT) at its pinned revision and write the tile reader's state
./run.sh ink.model.weights status     where each checkpoint is and whether it is in place

Reader v3 (https://huggingface.co/domenicor046/reader-v3) is a fine-tune of the Vesuvius Challenge's ink_9um network
(a 3D stem over 17 CT layers feeding a 2D U-Net). Its file holds the network under `model`; the tile reader loads the
same tensors with a `model.` prefix. The ink_9um seed-42 checkpoint that configures the network, and the network's code
(ScrollPrize/villa, `vesuvius`), are fetched by ink.model.stock_model the first time a reader is built.
"""

import hashlib
import sys
import urllib.request

from storage.data_roots import PRETRAINED_WEIGHTS

READER_V3_URL = (
    "https://huggingface.co/domenicor046/reader-v3/resolve/aee1d64bced4eb35546348f3d276b9e2a0b99c46/reader-v3.pth"
)
READER_V3_SHA256 = "ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a"
READER_V3 = PRETRAINED_WEIGHTS / "reader_v3" / "reader-v3.pth"
STOCK_READER = PRETRAINED_WEIGHTS / "reader_v3" / "reader-v3-tile-reader-state.pth"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch():
    """Reader v3 downloaded and checked, then saved as the tile reader's state; nothing is done twice."""
    import torch

    READER_V3.parent.mkdir(parents=True, exist_ok=True)
    if not READER_V3.is_file() or sha256(READER_V3) != READER_V3_SHA256:
        part = READER_V3.with_suffix(".pth.part")
        print(f"fetching {READER_V3_URL}", flush=True)
        urllib.request.urlretrieve(READER_V3_URL, part)
        if sha256(part) != READER_V3_SHA256:
            part.unlink()
            raise SystemExit("Reader v3 download does not match its sha256; try again")
        part.replace(READER_V3)
    if not STOCK_READER.is_file():
        network = torch.load(READER_V3, map_location="cpu", weights_only=True)["model"]
        part = STOCK_READER.with_suffix(".pth.part")
        torch.save({f"model.{name}": tensor for name, tensor in network.items()}, part)
        part.replace(STOCK_READER)
    print(f"stock Reader v3 ready: {STOCK_READER}", flush=True)
    return STOCK_READER


def status():
    for name, path in (("Reader v3", READER_V3), ("stock Reader v3, tile reader state", STOCK_READER)):
        print(f"{name:36s} {'ready' if path.is_file() else 'missing'}  {path}")


if __name__ == "__main__":
    {"fetch": fetch, "status": status}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
