import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def roots(env):
    """The data folders a fresh process sees under these environment variables."""
    code = "import json; from storage import data_roots as d; print(json.dumps([str(d.LIBRARY), str(d.OUTPUTS), str(d.SHEETS)]))"
    clean = {k: v for k, v in os.environ.items() if not k.startswith("HERCULEX_")}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=HERE, env=clean | env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def test_one_data_root_holds_the_library_and_the_outputs(tmp_path):
    library, outputs, sheets = roots({"HERCULEX_DATA": str(tmp_path)})
    assert (library, outputs) == (str(tmp_path.resolve() / "library"), str(tmp_path.resolve() / "outputs"))
    assert sheets == str(tmp_path.resolve() / "outputs" / "website" / "sheets")


def test_either_folder_can_live_elsewhere_and_older_names_still_work(tmp_path):
    library, outputs, _ = roots({"HERCULEX_DATA": str(tmp_path / "data"), "HERCULEX_OUTPUTS": str(tmp_path / "big")})
    assert library == str(tmp_path.resolve() / "data" / "library") and outputs == str(tmp_path.resolve() / "big")
    library, outputs, _ = roots({"HERCULEX_DISC_PCIE": str(tmp_path / "a"), "HERCULEX_DISC_PCLE3": str(tmp_path / "b")})
    assert (library, outputs) == (str(tmp_path.resolve() / "a"), str(tmp_path.resolve() / "b"))


def test_a_symlinked_data_folder_names_its_real_place(tmp_path):
    (tmp_path / "disc").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "outputs").symlink_to(tmp_path / "disc")
    _, outputs, _ = roots({"HERCULEX_DATA": str(tmp_path / "data")})
    assert outputs == str((tmp_path / "disc").resolve())


def test_a_page_names_stock_reader_v3_or_a_fine_tune(tmp_path, monkeypatch):
    from runner import review_page

    assert review_page.model_name(review_page.CHECKPOINT) == "stock Reader v3"
    monkeypatch.delenv("HERCULEX_MODEL_NAME", raising=False)
    assert review_page.model_name(tmp_path / "mine.pth") == "post-trained Reader v3"
    monkeypatch.setenv("HERCULEX_MODEL_NAME", "post-trained Reader v3 (thin strokes)")
    assert review_page.model_name(tmp_path / "mine.pth") == "post-trained Reader v3 (thin strokes)"
    assert not review_page.live("http://127.0.0.1:9")
