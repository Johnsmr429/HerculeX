"""Spans of a job's work as JSON lines, so the timeline can show what a job did when: fetching, waiting, reading.

The job board sets HERCULEX_EVENTS for every job to its own file in the board's events/ folder; outside the board
nothing is written unless the variable is set.

    with events.span("fetch", store=name):
        ...
"""

import json
import os
import time
from contextlib import contextmanager


def events_path():
    return os.environ.get("HERCULEX_EVENTS")


def record(kind, start, end, **fields):
    """One finished span, appended in a single write so concurrent processes of a job never interleave a line."""
    path = events_path()
    if not path:
        return
    line = json.dumps({"kind": kind, "start": round(start, 3), "end": round(end, 3), "pid": os.getpid(), **fields})
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, (line + "\n").encode())
    finally:
        os.close(descriptor)


@contextmanager
def span(kind, **fields):
    """Record how long the block took, also when it raises."""
    start = time.time()
    try:
        yield
    finally:
        record(kind, start, time.time(), **fields)


def spans(path):
    """The spans in an events file, skipping a line cut short by a killed job."""
    out = []
    try:
        with open(path) as lines:
            for line in lines:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return out
