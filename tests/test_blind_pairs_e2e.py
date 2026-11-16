"""Blind pairs from selection to score, on made-up pieces and a reader that needs no GPU."""

import json

import numpy as np
from PIL import Image

from calibration.text import blind_pairs as blind


def rows(count=40):
    return [
        {
            "unit": "rows-0000-0012",
            "piece": k,
            "area": 500.0,
            "width_mm": 30.0,
            "height_mm": 20.0,
            "forward": k / 100,
            "reversed": 0.2,
            "scrambled": 0.0,
            "forward_face": "near",
            "fibres": "near",
            "turn": 1,
            "flip_rows": False,
            "flip_cols": False,
            "x_mm": 0.0,
            "y_mm": 0.0,
        }
        for k in range(count)
    ]


def test_the_groups_take_both_tails_and_a_fixed_random_draw():
    pieces = [*rows(), rows(1)[0] | {"piece": 99, "area": 100.0}, rows(1)[0] | {"piece": 98, "scrambled": 0.5}]
    picked = blind.chosen(pieces)
    groups = {g: [r["piece"] for r in picked if r["group"] == g] for g in ("forward tail", "reversed tail", "random")}
    assert groups["forward tail"] == list(range(32, 40)) and groups["reversed tail"] == list(range(8))
    assert len(groups["random"]) == blind.RANDOM and set(groups["random"]) <= set(range(8, 32))
    assert picked == blind.chosen(pieces) and not {98, 99} & {r["piece"] for r in picked}


def test_the_far_view_is_the_same_place_seen_from_behind():
    near, far = (np.random.default_rng(k).random((blind.FIELD, blind.FIELD)).astype(np.float32) for k in (1, 2))
    shown = blind.views(near, far, np.ones_like(near, dtype=bool), (0, 0))
    assert np.array_equal(shown["far"][:, ::-1], np.rint(far * 254).astype(np.uint8) + 1)
    assert np.array_equal(shown["near"], np.rint(near * 254).astype(np.uint8) + 1)


def test_the_square_finds_the_sheet():
    sheet = np.zeros((blind.FIELD + 256, blind.FIELD + 512), dtype=bool)
    sheet[128:, 256:] = True
    assert blind.fullest_square(sheet) == (128, 256)


def test_pairs_say_nothing_and_the_key_scores_them(tmp_path, monkeypatch):
    monkeypatch.setattr(blind, "OUT", tmp_path)
    monkeypatch.setattr(blind, "KEY", tmp_path / "key.json")
    faces = {"near": np.full((4, 4), 200, dtype=np.uint8), "far": np.full((4, 4), 50, dtype=np.uint8)}
    made = [(faces, {"kind": "known text", "forward_face": "near", "forward": 0.3, "reversed": 0.1}) for _ in range(4)]
    made += [
        (faces, {"kind": "random", "forward_face": "near", "fibres": "near", "forward": 0.2, "reversed": 0.2})
        for _ in range(4)
    ]
    blind.sealed(made)
    pairs = json.loads((tmp_path / "pairs.json").read_text())
    assert [p["number"] for p in pairs] == list(range(1, 9)) and set(pairs[0]) == {"number", "a", "b"}
    key = {p["number"]: p for p in json.loads((tmp_path / "key.json").read_text())}
    near_side = {n: "L" if p["sides"]["a"] == "near" else "R" for n, p in key.items()}
    assert all(
        np.asarray(Image.open(tmp_path / f"{n:02d}-{'a' if s == 'L' else 'b'}.png")).max() == 200
        for n, s in near_side.items()
    )
    result = blind.scored(" ".join(f"{n}{s}" for n, s in near_side.items()))
    assert result["known_text"] == {"right": 4, "of": 4} and result["eye_passes"]
    assert result["scroll_picks"]["random"]["forward"] == 4 and result["fibres"]["agree"] == 4
    missed = blind.scored(" ".join(f"{n}N" for n in near_side))
    assert not missed["eye_passes"] and missed["scroll_picks"]["random"]["neither"] == 4


class Reader:
    def read(self, block):
        return block.astype(np.float32).mean(axis=0) / 255


def stack():
    made = np.random.default_rng(3).integers(60, 200, (31, blind.FIELD + 64, blind.FIELD + 128), dtype=np.uint8)
    made[:6] = 0
    return made


def test_a_scroll_pair_keeps_one_place_and_its_numbers(monkeypatch):
    monkeypatch.setattr(blind, "piece_stack", lambda *_: stack())
    faces, entry = blind.scroll_pair(rows(1)[0] | {"group": "random"}, Reader())
    assert faces["near"].shape == faces["far"].shape == (blind.FIELD, blind.FIELD)
    assert (
        entry["kind"] == "random"
        and entry["forward_face"] == "near"
        and set(entry) >= {"open", "forward", "reversed", "band_ratio", "corner"}
    )


def test_a_known_pair_names_the_face_its_labels_score_best(tmp_path):
    made = stack()
    ink = np.arange(4000) % 2 == 0
    r, c = np.arange(4000) % 900 + 40, np.arange(4000) // 900 * 200 + 40
    made[:, r[ink], c[ink]] = 255
    np.savez(tmp_path / "window.npz", stack=made, rows=r, columns=c, ink=ink)
    faces, entry = blind.known_pair("PHerc0841", tmp_path / "window.npz", Reader())
    assert (
        entry["kind"] == "known text"
        and max(entry["auc"].values()) > 0.9
        and faces["far"].shape == (blind.FIELD, blind.FIELD)
    )
