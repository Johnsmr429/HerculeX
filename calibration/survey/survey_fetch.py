"""A surveyed band's bricks fetched by a job of its own, ahead of its reading."""

import shutil
import sys

from calibration.survey.scroll_survey import PLANS, store_of
from runner.job_board import TRY_LATER
from sampling.piece_images import fetch_bricks

FREE_BYTES = 40 * 2**30


def leave_when_full(store):
    """The store saved and the job ended, not held waiting."""
    store.save()
    if shutil.disk_usage(store.folder).free < FREE_BYTES:
        print("disc full: try later", flush=True)
        sys.exit(TRY_LATER)


def run(kind, item):
    sample, volume, band = item.split("/")
    jobs = PLANS[kind][0](sample, volume, band)
    print(
        sample,
        kind,
        len(jobs),
        "pieces planned:",
        ",".join(job[0].name.removeprefix("piece-") for job in jobs),
        flush=True,
    )
    try:
        if jobs:
            fetch_bricks(sample, volume, jobs, store_of(kind, sample), leave_when_full)
    except OSError as error:
        print("try later:", repr(error)[:300], flush=True)
        sys.exit(TRY_LATER)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
