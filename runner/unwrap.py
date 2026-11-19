"""Unwrap scrolls from their CT to their published pages on this machine's lane.

./run.sh unwrap SCROLL[:SCAN] ...

On the Apple lane each scroll's jobs join the job board. On the CUDA lane they run here until every page is out, the
cores and GPUs busy at once.
"""

import argparse
import sys

from runner.lane import APPLE, LANE


def main():
    parser = argparse.ArgumentParser(
        prog="./run.sh unwrap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scrolls", nargs="+", metavar="SCROLL[:SCAN]", help="a scroll, such as PHerc0826, and its scan")
    parser.add_argument("--reads-per-gpu", type=int, default=1, help="reads that share one GPU (CUDA lane)")
    args = parser.parse_args()
    scrolls = [(scroll, scan or None) for scroll, _, scan in (word.partition(":") for word in args.scrolls)]
    print(f"{LANE} lane: {', '.join(scroll for scroll, _ in scrolls)}", flush=True)
    if APPLE:
        from runner.machine_apple.unwrap import unwrap

        unwrap(scrolls)
        return 0
    from runner.machine_cuda.unwrap import unwrap

    return unwrap(scrolls, args.reads_per_gpu)


if __name__ == "__main__":
    sys.exit(main())
