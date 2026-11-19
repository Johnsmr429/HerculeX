"""The Apple lane's unwrap: four job-board jobs per scroll, from its CT to its published page.

`sheets` detects the sheet nodes on the GPU, `pieces` levels them into pieces, `fetch` downloads each unit's CT for
the ink read while later units are still levelled, and `sheet` reads the ink with Metal and the Neural Engine and
publishes the scroll's page. The board starts each job once the jobs it follows are done and the Mac has room for it.
"""

import subprocess
import sys

from ink.model import models
from storage.data_roots import HERE


def jobs(scroll, scan=None):
    """The board jobs of one scroll, as (name, board options, command); a read by another ink model than stock Reader
    v3 gets jobs of its own, named for the model."""
    chosen = ["--scan", scan] if scan else []
    model = models.chosen()
    suffix = model.page_suffix
    reading = [
        "--setting",
        "HERCULEX_KEEP_STACKS=0",
        *(["--setting", f"HERCULEX_INK_MODEL={model.name}"] if suffix else []),
    ]
    after_sheets, after_pieces = ["--after", f"{scroll}-sheets{suffix}"], ["--after", f"{scroll}-pieces{suffix}"]
    options = {
        "sheets": ["--gpu", "--link", "--share", "--memory", "5", "--setting", "HERCULEX_LIST_CT_BRICKS=1"],
        "pieces": ["--memory", "5", "--cores", "3", "--restartable", *after_sheets],
        "fetch": ["--link", "--share", "--memory", "2", "--cores", "2", "--restartable", *after_sheets],
        "sheet": ["--gpu", "--engine", "--link", "--memory", "5", "--cores", "5", *after_pieces, *reading],
    }
    return [
        (f"{scroll}-{stage}{suffix}", board, ["./run.sh", "runner.pipeline", "run", stage, scroll, *chosen])
        for stage, board in options.items()
    ]


def unwrap(scrolls):
    """Each scroll's jobs added to the board, in the order given."""
    for scroll, scan in scrolls:
        for name, options, command in jobs(scroll, scan):
            board = [sys.executable, "-m", "runner.job_board", "add", name, *options, "--", *command]
            subprocess.run(board, cwd=HERE, check=True)
    print("Follow them with ./run.sh board status; each page reaches the start page when its sheet job ends.")
