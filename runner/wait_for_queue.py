"""Ends when none of the named jobs can still run, however they ended."""

import json
import sys
import time

from runner.job_board import ACTIVE, all_jobs, connect

CHECK_SECONDS = 60


def will_run(name, named):
    """Whether a job has run or still can."""
    job = named.get(name)
    if job is None:
        return False
    if job["state"] == "done" or job["state"] in ACTIVE:
        return True
    if job["state"] != "waiting" or job["held"]:
        return False
    return all(will_run(other, named) for other in json.loads(job["after"]))


def left_of(names, jobs):
    named = {job["name"]: job for job in jobs}
    return [name for name in names if name in named and named[name]["state"] != "done" and will_run(name, named)]


def run(names):
    database, shown = connect(), None
    unknown = sorted(set(names) - {job["name"] for job in all_jobs(database)})
    if unknown or not names:
        sys.exit(f"not on the board: {unknown}")
    while left := left_of(names, all_jobs(database)):
        if len(left) != shown:
            shown = len(left)
            print(shown, "jobs can still run, first", left[0], flush=True)
        time.sleep(CHECK_SECONDS)
    print("queue ended", flush=True)


if __name__ == "__main__":
    run(sys.argv[1:])
