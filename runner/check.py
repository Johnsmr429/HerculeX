"""The test suite with each test file in a fresh process, a few at once: memory a file takes is given back when it
ends, so the suite's peak is that of the files running together, not the sum of all of them.

./run.sh check [cpu|ports] [--workers N]
./run.sh check lint [--fix]         every tracked file through its language's pinned linters (runner/lint.py)

cpu runs everything that needs neither the GPU nor the Neural Engine, three files at a time; ports runs the tests of
the native ports in ports/; with neither, every test runs, one file at a time, since GPU tests share one GPU.
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from runner.job_board import LIFETIME_PEAK, usage_of

HERE = Path(__file__).resolve().parent.parent
NEEDS_GPU = {"test_reader_apple_e2e.py"}
SELECTIONS = {"cpu": ["-k", "not gpu", "-m", "not ports"], "ports": ["-m", "ports"], "all": []}
WORKERS = {"cpu": 3, "ports": 1, "all": 1}
NO_TESTS = 5
COUNTED = re.compile(r"(\d+) (passed|failed|skipped|deselected|errors?|xfailed|xpassed)")


def files(kind):
    found = sorted((HERE / "tests").glob("test_*.py"))
    return [f for f in found if not (kind == "cpu" and f.name in NEEDS_GPU)]


def run_file(path, selection):
    """One file's tests in its own pytest process: its exit code, counts, seconds, peak footprint and output."""
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path), *selection]
    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak, done = [0.0], threading.Event()

    def watch():
        """The kernel's lifetime peak footprint of the file's process, GPU memory included, until it ends."""
        while not done.is_set():
            usage = usage_of(process.pid)
            if usage:
                peak[0] = max(peak[0], usage[LIFETIME_PEAK] / 2**30)
            done.wait(0.2)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    output = process.stdout.read()
    process.stdout.close()
    done.set()
    watcher.join()
    _, status, usage = os.wait4(process.pid, 0)
    process.returncode = code = os.waitstatus_to_exitcode(status)
    peak = max(peak[0], usage.ru_maxrss / 2**30)
    counts = {
        kind.rstrip("s"): int(n) for n, kind in COUNTED.findall(output.strip().splitlines()[-1] if output else "")
    }
    return {"file": path.name, "code": code, "counts": counts, "seconds": time.perf_counter() - start, "peak": peak} | {
        "output": output
    }


def main():
    if sys.argv[1:2] == ["lint"]:
        from runner import lint

        sys.exit(lint.run(fixing="--fix" in sys.argv[2:]))
    parser = argparse.ArgumentParser(prog="check", description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", nargs="?", choices=("cpu", "ports", "all"), default="all")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    workers = args.workers or WORKERS[args.kind]
    start = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda f: run_file(f, SELECTIONS[args.kind]), files(args.kind)))
    failed = [r for r in results if r["code"] not in (0, NO_TESTS)]
    for r in failed:
        print(f"\n===== {r['file']} (exit {r['code']}) =====\n{r['output'].rstrip()}")
    totals = {}
    for r in results:
        for kind, n in r["counts"].items():
            totals[kind] = totals.get(kind, 0) + n
    print("\nslowest files:")
    for r in sorted(results, key=lambda r: -r["seconds"])[:8]:
        print(f"  {r['seconds']:6.1f} s  {r['peak']:5.2f} GB  {r['file']}")
    print(
        f"{', '.join(f'{n} {kind}' for kind, n in sorted(totals.items()))} in {len(results)} files, "
        f"{time.perf_counter() - start:.1f} s with {workers} at once"
        + (f"; FAILED: {', '.join(r['file'] for r in failed)}" if failed else "")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
