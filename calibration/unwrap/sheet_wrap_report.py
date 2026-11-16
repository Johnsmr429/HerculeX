"""The wrap studies side by side at fixed wrong-wrap rates."""

import json
import sys

from storage.data_roots import SURFACE_RECORDS

RATES = (0.05, 0.10, 0.15, 0.20)


def rows_of(name):
    return {r["control"]: r for r in json.loads((SURFACE_RECORDS / f"wrap-study-{name}.json").read_text())}


def claimed_area(part):
    """Contiguous claimed area on the right wrap, the claim less its measured wrong share."""
    return part["claimed_cm2"] * (1 - part["wrong_rate"])


def verified_area(part):
    """Contiguous area meshes confirm on the right wrap."""
    return part["correct_cm2"]


def at_rate(rows, rate, floor, area):
    """Sum over controls of correct contiguous area, a control counting only where its wrong-wrap rate is within the rate."""
    return sum(area(r[floor]) for r in rows.values() if r[floor]["wrong_rate"] <= rate)


def report(names):
    studies = {n: rows_of(n) for n in names}
    shared = set.intersection(*[set(s) for s in studies.values()])
    measurable = [c for c in sorted(shared) if any(s[c]["usable"]["wrong"] > 0.005 for s in studies.values())]
    lines = [
        "| control | " + " | ".join(f"{n} claimed cm2 / correct / wrong rate" for n in names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    for control in sorted(shared):
        cells = [
            f"{s[control]['usable']['claimed_cm2']:.0f} / {s[control]['usable']['correct']:.3f} / {s[control]['usable']['wrong_rate']:.3f}"
            for s in studies.values()
        ]
        lines.append(f"| {control} | " + " | ".join(cells) + " |")
    for title, area in (
        ("confirmed by meshes", verified_area),
        ("claimed, less the measured wrong share", claimed_area),
    ):
        lines += [
            "",
            f"Contiguous correct-wrap cm2 {title}, summed over the {len(measurable)} controls that can show a wrong wrap, a control counting only within the limit, pieces of at least 1 cm2 and of at least 4 cm2:",
            "",
            "| wrong-wrap rate at most | " + " | ".join(f"{n} 1 cm2 / 4 cm2" for n in names) + " |",
            "| --- |" + " --- |" * len(names),
        ]
        for rate in RATES:
            cells = []
            for s in studies.values():
                chosen = {c: s[c] for c in measurable}
                cells.append(
                    f"{at_rate(chosen, rate, 'usable', area):.0f} / {at_rate(chosen, rate, 'window', area):.0f}"
                )
            lines.append(f"| {rate:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1:]))
