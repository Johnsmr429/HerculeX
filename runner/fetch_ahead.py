"""Waiting survey jobs split into a fetch job on the link and a read job on the engine."""

import json
import sys
import time

from runner.job_board import HERE, connect
from storage.data_roots import ct_bricks

MEMORY_GB = 0.7
FREE_GIB = 40
STORES = json.dumps([3, str(ct_bricks("*", "survey*"))])


def splittable(job):
    command = json.loads(job["command"])
    return (
        job["state"] == "waiting"
        and job["link"]
        and "calibration.survey.scroll_survey" in command
        and command[-2] in ("look", "replicate", "reread", "row-sample", "second")
    )


def split(database, job):
    """A job's fetch on the board and the job behind it, or nothing if the job has started meanwhile."""
    name, (mode, item) = job["name"], json.loads(job["command"])[-2:]
    fetch = f"fetch-{name}"
    moved = database.execute(
        "UPDATE jobs SET link=0, after=? WHERE name=? AND state='waiting' AND link=1",
        (json.dumps([*json.loads(job["after"]), fetch]), name),
    ).rowcount
    if moved:
        database.execute(
            """INSERT INTO jobs(name, kind, command, folder, settings, owner, note, memory_gb, cores, gpu, engine, link, priority, nice, background, after, needs, clean, restartable, free_gib, at_most, state, added)
                            VALUES (?, 'calibration.survey.survey_fetch', ?, ?, '{}', 'quality of life', ?, ?, 1, 0, 0, 1, ?, 5, 0, ?, ?, '[]', 1, ?, ?, 'waiting', ?)""",
            (
                fetch,
                json.dumps(["./run.sh", "calibration.survey.survey_fetch", mode, item]),
                str(HERE),
                f"bricks of {name}",
                MEMORY_GB,
                job["priority"],
                job["after"],
                job["needs"],
                FREE_GIB,
                STORES,
                job["added"] - 0.001,
            ),
        )
    return moved


def run(apply):
    database = connect()
    database.execute("BEGIN IMMEDIATE")
    jobs = [dict(row) for row in database.execute("SELECT * FROM jobs ORDER BY added")]
    chosen = [job for job in jobs if splittable(job)]
    for job in chosen:
        print(
            f"{job['name']:30s} priority {job['priority']:3d}  after {json.loads(job['after'])}  needs {len(json.loads(job['needs']))}"
        )
    done = sum(split(database, job) for job in chosen) if apply else 0
    gated = (
        database.execute(
            "UPDATE jobs SET free_gib=?, at_most=? WHERE name LIKE 'fetch-%' AND state='waiting' AND at_most IS NULL",
            (FREE_GIB, STORES),
        ).rowcount
        if apply
        else 0
    )
    database.execute("COMMIT" if apply else "ROLLBACK")
    print(f"{len(chosen)} jobs to split, {done} split, {gated} fetches gated on the board", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    run(sys.argv[1:] == ["apply"])
