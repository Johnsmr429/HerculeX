"""Where a program's true footprint peaks, line by line."""

import os
import runpy
import sys
import threading
import time
import traceback

from runner.job_board import FOOTPRINT, HERE, LIFETIME_PEAK, usage_of

EVERY_SECONDS = 0.005
SHOWN = 24


def where(frame):
    """The innermost lines of a stack that are ours, innermost last."""
    lines = [
        f"{os.path.relpath(f.filename, HERE)}:{f.lineno} {f.name}"
        for f in traceback.extract_stack(frame)
        if f.filename.startswith(str(HERE)) and "/runtime/" not in f.filename and "memory_trace" not in f.filename
    ]
    return " < ".join(reversed(lines[-2:]))


def watch(tops, stopped, main):
    """Each line's highest footprint."""
    highest = 0
    while not stopped.is_set():
        usage = usage_of(os.getpid())
        frame = sys._current_frames().get(main)
        if usage and frame is not None:
            key = where(frame)
            top = usage[FOOTPRINT]
            if usage[LIFETIME_PEAK] > highest:
                highest = usage[LIFETIME_PEAK]
                top = highest
            tops[key] = max(tops.get(key, 0), top)
        time.sleep(EVERY_SECONDS)


def main():
    words = sys.argv[1:]
    per = 0
    if words[0] == "--per":
        per, words = float(words[1]), words[2:]
    tops, stopped = {}, threading.Event()
    watcher = threading.Thread(target=watch, args=(tops, stopped, threading.main_thread().ident), daemon=True)
    watcher.start()
    target = words[0]
    sys.argv = words
    sys.path.insert(0, str(HERE))
    start = time.time()
    try:
        runpy.run_path(target, run_name="__main__") if target.endswith(".py") else runpy.run_module(
            target, run_name="__main__", alter_sys=True
        )
    finally:
        stopped.set()
        watcher.join()
        peak = usage_of(os.getpid())[LIFETIME_PEAK]
        print(
            f"peak {peak / 1e9:.2f} GB in {time.time() - start:.0f} s" + (f", {peak / per:.0f} B each" if per else ""),
            file=sys.stderr,
            flush=True,
        )
        for key, value in sorted(tops.items(), key=lambda item: -item[1])[:SHOWN]:
            print(
                f"{value / 1e9:7.2f} GB" + (f"{value / per:7.0f} B" if per else "") + f"  {key}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
