"""The mirrors keep every published byte recoverable, and streamed chunks tile a scroll's height exactly and pack without loss."""

import io

import numpy as np
import tifffile
import zstandard

from runner.stage_queue import finish
from scans.transfer.fetch_products import (
    interleaved,
    pack_mesh,
    pack_text,
    stored_path,
)
from scans.transfer.fetch_server import rank
from storage import data_roots
from storage.node_store import pack_chunk, read_cells, read_nodes
from unwrapping.detection import detect_nodes as detector
from unwrapping.detection.detect_nodes import EDGE, chunks
from unwrapping.detection.kernel_bindings import CELL, NODE
from unwrapping.level import run_levels as level_stage


def published_plane(compression=None):
    grid = np.linspace(0, 9000, 300 * 200, dtype=np.float32).reshape(300, 200)
    grid[40:60, 10:90] = -1
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, grid, compression=compression)
    return grid, buffer.getvalue()


def test_mesh_plane_is_smaller_and_bit_exact(tmp_path):
    grid, raw = published_plane()
    target = tmp_path / "x.tif"
    assert pack_mesh(raw, target) == len(raw)
    assert target.stat().st_size < len(raw) / 2
    assert tifffile.imread(target).tobytes() == grid.tobytes()


def test_a_plane_that_cannot_be_read_stays_as_published(tmp_path):
    raw = b"not a tiff at all"
    target = tmp_path / "x.tif"
    pack_mesh(raw, target)
    assert target.read_bytes() == raw


def test_obj_text_survives_packing(tmp_path):
    text = b"v 4894.498047 2601.994628 12.44665\nvt 0.5 0.25\nf 1/1 2/2 3/3\n" * 5000
    target = tmp_path / "mesh.obj.zst"
    assert pack_text(io.BytesIO(text), target) == len(text)
    assert target.stat().st_size < len(text) / 4
    assert zstandard.ZstdDecompressor().stream_reader(target.open("rb")).read() == text


def test_only_obj_changes_name():
    assert stored_path("a/mesh.obj").name == "mesh.obj.zst"
    assert stored_path("a/x.tif").name == "x.tif"


def test_chunks_tile_every_height_exactly():
    for height in [*range(1, 3000, 7), 7010, 9489, 12149, 18977]:
        planned = chunks(height)
        planes = [p for _, _, low, high in planned for p in range(low, high)]
        top = -(-height // CELL) - EDGE
        assert planes == list(range(EDGE, top))
        assert all(low < high for _, _, low, high in planned)


def test_detected_nodes_reach_the_level_route_on_a_new_scan(tmp_path, monkeypatch):
    grown, level = tmp_path / "nodes", tmp_path / "level"
    sample, volume = "PHerc0000", "1"
    scan = {"sample": sample, "volume": volume, "voxel_um": 9.362}
    monkeypatch.setattr(data_roots, "ROUTE_OUTPUTS", level)
    monkeypatch.setattr(detector, "NODE_ROUTE", grown)
    monkeypatch.setattr(detector, "chosen_scans", lambda _: [scan])
    monkeypatch.setattr(detector, "level_shape", lambda *_: (256, 256, 256))
    monkeypatch.setattr(level_stage, "NODE_ROUTE", grown)
    monkeypatch.setattr(level_stage, "LEVEL_ROUTE", level)
    (job,) = detector.jobs({sample}, False)
    nodes = job["out"]
    assert nodes.is_relative_to(grown)
    nodes.mkdir()
    finish(nodes, {"method": detector.METHOD})
    assert list(level_stage.units({sample})) == [(nodes, level / sample / volume / "pieces" / nodes.name)]
    level_stage.share_nodes(nodes)
    assert (level / sample / volume / "nodes" / nodes.name).resolve() == nodes


def test_interleaving_loses_nothing():
    for count in range(12):
        assert sorted(interleaved(list(range(count)))) == list(range(count))


def plain_chunk(folder):
    random = np.random.default_rng(0)
    nodes = np.zeros(5000, dtype=NODE)
    nodes["x"], nodes["prominence"] = random.normal(size=5000), random.integers(0, 255, 5000)
    cells = np.sort(random.integers(0, 10**7, 5000)).astype(np.int32)
    folder.mkdir(parents=True)
    nodes.tofile(folder / "nodes.bin")
    cells.tofile(folder / "cells.bin")
    return nodes, cells


def test_chunks_pack_exactly(tmp_path):
    chunk = tmp_path / "rows-0000-0012"
    nodes, cells = plain_chunk(chunk)
    assert read_nodes(chunk).tobytes() == nodes.tobytes()
    assert read_nodes(chunk, (400, 900)).tobytes() == nodes[400:900].tobytes()
    pack_chunk(chunk)
    assert sorted(p.name for p in chunk.iterdir()) == ["cells.zst", "nodes.zst"]
    assert read_nodes(chunk).tobytes() == nodes.tobytes()
    assert read_nodes(chunk, (400, 900)).tobytes() == nodes[400:900].tobytes()
    assert read_cells(chunk).tobytes() == cells.tobytes()


def test_server_plan_keeps_what_cannot_be_rebuilt():
    segment = "/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20230503225234/20230503225234"
    assert rank(f"{segment}.obj") == 1 and rank(f"{segment}.tif") == 3 and rank(f"{segment}_mask.png") == 0
    assert (
        rank(f"{segment}.ppm") is None
        and rank(f"{segment}_points.obj") is None
        and rank(f"{segment}_cellmap.tif") is None
    )
    assert rank("/community-uploads/someone/ink.zip") is None
