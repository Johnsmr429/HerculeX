"""Where a run's time went: the GPU, Neural Engine, link, cores and discs over the run, the board jobs that held them,
how long each slot sat idle while a job that wanted it waited, and under each job the spans it recorded (fetch,
wait-fetch, read, layout, publish, detect) through runner.events.

./run.sh runner.timeline [PATTERN] [--hours H] [--width N] [--json OUT]

PATTERN selects board jobs by name (shell wildcards, such as 'PHerc0826-*'), and the window runs from the first of them
to start to the last to end. Without it, the window is the last H hours (default 6). A job waits "ready" when every
job it runs after has finished, and "behind" while one of them still runs: idle time behind a stage before it is the
cost of the pipeline's stage barriers, and idle time ready is the cost of the board's own decisions.
"""

import argparse
import fnmatch
import json
import sys
import time

from runner import events, job_board

BLOCKS = " ▁▂▃▄▅▆▇█"
SAMPLED = (
    ("GPU %", "gpu", 100.0),
    ("CPU %", "cpu_pct", 100.0),
    ("link MB/s", "net_mb_s", None),
    ("internal MB/s", "internal_mb_s", None),
    ("library MB/s", "library_mb_s", None),
    ("outputs MB/s", "outputs_mb_s", None),
    ("room GB", "room_gb", None),
)
SHOWN_JOBS = 40


def merged(spans):
    """The union of time spans, as sorted disjoint spans."""
    out = []
    for start, end in sorted(span for span in spans if span[1] > span[0]):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [tuple(span) for span in out]


def length(spans):
    return sum(end - start for start, end in spans)


def minus(spans, cut):
    """Spans with the parts inside `cut` removed; both sorted and disjoint."""
    out, cut = [], list(cut)
    for start, end in spans:
        for left, right in cut:
            if right <= start or left >= end:
                continue
            if left > start:
                out.append((start, left))
            start = max(start, right)
        if end > start:
            out.append((start, end))
    return out


def clipped(spans, start, end):
    return [(max(a, start), min(b, end)) for a, b in spans if b > start and a < end]


def ran(job, now):
    """When a job held its slots, or None."""
    if not job.get("started"):
        return None
    end = now if job["state"] in job_board.ACTIVE else job.get("ended")
    return (job["started"], end) if end and end > job["started"] else None


def waits(job, named, now):
    """A job's wait for its turn, split at the time the last job it runs after ended: (behind, ready)."""
    added = job.get("added") or 0.0
    began = job["started"] if job.get("started") else (now if job["state"] == "waiting" else None)
    if began is None or began <= added:
        return None, None
    before = [named[name] for name in json.loads(job.get("after") or "[]") if name in named]
    ends = [other.get("ended") or now for other in before]
    free = min(max([added, *ends]), began)
    return ((added, free) if free > added else None), (free, began)


def slot_report(jobs, slot, start, end, now):
    """A slot's held, idle-ready and idle-behind spans in the window."""
    named = {job["name"]: job for job in jobs}
    held = merged(span for job in jobs if job[slot] and (span := ran(job, now)))
    behind, ready = [], []
    for job in jobs:
        if job[slot]:
            before, turn = waits(job, named, now)
            behind += [before] if before else []
            ready += [turn] if turn else []
    held = merged(clipped(held, start, end))
    idle_ready = minus(merged(clipped(ready, start, end)), held)
    idle_behind = minus(minus(merged(clipped(behind, start, end)), held), idle_ready)
    return {"held": held, "idle_ready": idle_ready, "idle_behind": idle_behind}


def spark(values, top=None):
    """One block a value, scaled to `top` or to the largest value; blank where there is no value."""
    known = [v for v in values if v is not None]
    top = top or max(known, default=0.0) or 1.0
    return "".join(
        " " if v is None else BLOCKS[min(len(BLOCKS) - 1, max(0, round(v / top * (len(BLOCKS) - 1))))] for v in values
    )


def buckets(rows, key, start, end, width):
    """The mean of a sampled value in each of `width` equal parts of the window."""
    step, sums, counts = (end - start) / width, [0.0] * width, [0] * width
    for row in rows:
        value = row.get(key)
        if value is None or not start <= row["time"] < end:
            continue
        i = min(width - 1, int((row["time"] - start) / step))
        sums[i] += value
        counts[i] += 1
    return [s / c if c else None for s, c in zip(sums, counts)]


def lane(spans, start, end, width):
    """A row of the window: full where the spans cover a part, half where they cover some of it."""
    step, out = (end - start) / width, []
    for i in range(width):
        left = start + i * step
        covered = length(clipped(spans, left, left + step)) / step
        out.append("█" if covered > 0.75 else "▌" if covered > 0.05 else "·")
    return "".join(out)


def phases(name, start, end):
    """A job's recorded spans in the window, merged by kind, in the order each kind first began."""
    by_kind = {}
    for row in sorted(events.spans(job_board.events_of(name)), key=lambda row: row["start"]):
        by_kind.setdefault(row["kind"], []).append((row["start"], row["end"]))
    return {kind: merged(clipped(spans, start, end)) for kind, spans in by_kind.items()}


def clock(seconds):
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def at(seconds):
    return time.strftime("%H:%M", time.gmtime(seconds))


def window(jobs, pattern, hours, now):
    """The jobs to show and the window they span."""
    if pattern:
        chosen = [job for job in jobs if fnmatch.fnmatch(job["name"], pattern) and job.get("added")]
        if not chosen:
            return [], now, now
        start = min(job.get("started") or job["added"] for job in chosen)
        end = max((ran(job, now) or (0, job.get("ended") or now))[1] for job in chosen)
        return chosen, start, max(end, start + 60)
    start = now - hours * 3600
    chosen = [job for job in jobs if ((span := ran(job, now)) and span[1] > start) or job["state"] == "waiting"]
    return chosen, start, now


def report(jobs, rows, start, end, now, width):
    """The timeline as text lines and as data."""
    span = end - start
    lines = [f"Window {at(start)}-{at(end)} UTC, {clock(span)}; one column is {span / width / 60:.1f} min"]
    data = {"start": start, "end": end, "width": width, "sampled": {}, "slots": {}, "jobs": []}
    for label, key, top in SAMPLED:
        values = buckets(rows, key, start, end, width)
        if any(v is not None for v in values):
            known = [v for v in values if v is not None]
            lines.append(f"{label:14s}{spark(values, top)}  mean {sum(known) / len(known):.1f}, top {max(known):.1f}")
            data["sampled"][key] = values
    lines.append("")
    for slot in job_board.SLOTS:
        slots = slot_report(jobs, slot, start, end, now)
        data["slots"][slot] = slots
        lines.append(
            f"{slot:14s}{lane(slots['held'], start, end, width)}  held {100 * length(slots['held']) / span:.0f}%, "
            f"idle while a job was ready {clock(length(slots['idle_ready']))}, "
            f"behind a stage {clock(length(slots['idle_behind']))}"
        )
    lines.append("")
    shown = sorted(jobs, key=lambda job: job.get("started") or job.get("added") or 0)[:SHOWN_JOBS]
    recorded = {job["name"]: phases(job["name"], start, end) for job in shown}
    labels = [job["name"] for job in shown] + [f"  {kind}" for kinds in recorded.values() for kind in kinds]
    named = max(map(len, labels), default=0) + 2
    for job in shown:
        spans = [s] if (s := ran(job, now)) else []
        slots = "+".join(slot for slot in job_board.SLOTS if job[slot]) or "cpu"
        took = clock(length(clipped(spans, start, end)))
        lines.append(f"{job['name']:{named}s}{lane(spans, start, end, width)}  {slots}, {took}, {job['state']}")
        kinds = recorded[job["name"]]
        for kind, covered in kinds.items():
            label = f"  {kind}"
            lines.append(f"{label:{named}s}{lane(covered, start, end, width)}  {clock(length(covered))}")
        data["jobs"].append(
            {"name": job["name"], "slots": slots, "spans": spans, "state": job["state"], "phases": kinds}
        )
    if len(jobs) > SHOWN_JOBS:
        lines.append(f"... and {len(jobs) - SHOWN_JOBS} more jobs")
    return lines, data


def main():
    parser = argparse.ArgumentParser(prog="runner.timeline", description=__doc__.split("\n\n")[0])
    parser.add_argument("pattern", nargs="?", help="board jobs by name, shell wildcards")
    parser.add_argument("--hours", type=float, default=6.0, help="the last hours, when no pattern is given")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--json", metavar="OUT", help="also write the timeline's data as JSON")
    args = parser.parse_args()
    database, now = job_board.connect(), time.time()
    jobs, start, end = window(job_board.all_jobs(database), args.pattern, args.hours, now)
    if not jobs:
        sys.exit(f"no board jobs match {args.pattern}")
    rows = [dict(row) for row in database.execute("SELECT * FROM samples WHERE time>=? AND time<?", (start, end))]
    lines, data = report(jobs, rows, start, end, now, args.width)
    print("\n".join(lines))
    if args.json:
        with open(args.json, "w") as out:
            json.dump(data, out)


if __name__ == "__main__":
    main()
