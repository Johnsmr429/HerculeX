"""The CUDA lane's unwrap: scrolls from their CT to their published pages on one machine, its cores and GPUs busy at once.

Sheet detection takes one scroll at a time on the GPU. Once a scroll's nodes are done, a pool of level fits takes its
units, largest first, on the cores the reads leave free, and its joins follow. Each GPU then reads one scroll's ink at
a time, and the scroll's page is published on the CPU while the GPU reads the next. Every job logs to
logs/unwrap/, and a rerun skips the work that is done.
"""

import functools
import itertools
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from ink.model import models
from runner.machine_cuda.probes import machine
from storage.data_roots import HERE
from unwrapping.level import run_levels

LOGS = HERE / "logs" / "unwrap"
PIPELINE = ("runner.pipeline", "run")
TRIES = 2
SAMPLERS = int(os.environ.get("HERCULEX_SAMPLERS", "6"))
FIT_GB = 8
CORES_PER_FIT = 3
FIT_THREADS = 12
PUBLISHERS = 2
PAGE = re.compile(r"/sheets/(sheet-[^/]+)/index\.html: ([0-9]+) pieces")
SUFFIX = models.chosen().page_suffix
SETTINGS = {
    "HERCULEX_LIST_CT_BRICKS": "1",
    "HERCULEX_KEEP_STACKS": "0",
    "HERCULEX_COMPOSE_PROCESSES": "8",
    "HERCULEX_FETCH_CONNECTIONS": "32",
}


def gpu_count():
    """The NVIDIA GPUs nvidia-smi lists, at least one."""
    listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False).stdout
    return max(1, len(re.findall(r"^GPU \d+", listed, re.MULTILINE)))


def quota_cores():
    """The cores the container's CPU quota allows, or None without a quota."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except (OSError, ValueError):
        return None
    return None if quota == "max" else math.ceil(float(quota) / float(period))


class Machine:
    """Who works where: reads and publishing on the first cores, level fits on the rest, one read per GPU."""

    def __init__(self, reads_per_gpu=1):
        cores = sorted(os.sched_getaffinity(0))
        cores = cores[: quota_cores() or len(cores)]
        self.gpus = gpu_count()
        self.reads = self.gpus * reads_per_gpu
        reserved = min(len(cores) // 2, (SAMPLERS + 2) * self.reads)
        self.read_cores, self.fit_cores = cores[:reserved], cores[reserved:] or cores
        memory_gb = machine()["room_gb"]
        self.fits = max(1, min(len(self.fit_cores) // CORES_PER_FIT, int(memory_gb // FIT_GB)))
        self.threads = min(FIT_THREADS, len(self.fit_cores))

    def describe(self):
        return (
            f"{self.gpus} GPUs, {self.reads} reads at once on cores {span(self.read_cores)}, "
            f"{self.fits} level fits of {self.threads} threads on cores {span(self.fit_cores)}"
        )


def span(cores):
    return f"{cores[0]}-{cores[-1]}" if len(cores) > 1 else str(cores[0])


class Unwrap:
    """The scrolls' jobs and the threads that run them."""

    def __init__(self, scrolls, place):
        self.scrolls, self.place = scrolls, place
        self.order = {scroll: i for i, (scroll, _) in enumerate(scrolls)}
        self.fits, self.reads, self.publishes = queue.PriorityQueue(), queue.Queue(), queue.Queue()
        self.counter = itertools.count()
        self.outcomes, self.done = {}, threading.Condition()
        self.printing = threading.Lock()

    def say(self, *words):
        with self.printing:
            print(time.strftime("%H:%M:%S"), *words, flush=True)

    def job(self, name, args, cores, env=None):
        """One ./run.sh command, logged and tried again once; its outcome."""
        log = LOGS / f"{name}.log"
        pinned = ["taskset", "-c", ",".join(map(str, cores))] if shutil.which("taskset") else []
        for attempt in range(1, TRIES + 1):
            began = time.time()
            self.say("start", name, f"try {attempt}")
            with open(log, "a") as out:
                code = subprocess.call(
                    [*pinned, str(HERE / "run.sh"), *args],
                    cwd=HERE,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env={**SETTINGS, **os.environ, **(env or {})},
                )
            self.say("end", name, f"{time.time() - began:.0f} s", "ok" if code == 0 else f"code {code}")
            if code == 0:
                return "ok"
        return "failed"

    def settle(self, scroll, outcome):
        with self.done:
            self.outcomes[scroll] = outcome
            self.say("settled", scroll, outcome, f"{len(self.outcomes)} of {len(self.scrolls)}")
            self.done.notify_all()

    def fit_worker(self):
        while True:
            _, _, run, answer = self.fits.get()
            try:
                answer.put(run())
            except Exception as error:
                self.say("error", repr(error)[:300])
                answer.put("failed")

    def level(self, scroll, scan):
        """A scroll's units fitted by the pool, largest first, then its joins; then it waits for a GPU."""
        chosen = ["--scan", scan] if scan else []
        rows = [r for r in run_levels.rows_of(scroll) if scan is None or r.parts[-3] == scan]
        if not rows:
            self.settle(scroll, "no sheet nodes")
            return
        run_levels.share_nodes(rows[0])
        fitting = {"OMP_NUM_THREADS": str(self.place.threads)}
        curl = ["unwrapping.level.run_levels", "curl", scroll]
        if len(rows) > 1 and self.job(f"{scroll}-curl", curl, self.place.fit_cores, fitting) != "ok":
            self.settle(scroll, "curl failed")
            return
        answers = []
        for unit in sorted(rows, key=size_of, reverse=True):
            answer = queue.Queue(1)
            args = ["unwrapping.level.run_levels", "unit", scroll, unit.name]
            run = functools.partial(self.job, f"{scroll}-{unit.name}", args, self.place.fit_cores, fitting)
            self.fits.put(((self.order[scroll], -size_of(unit)), next(self.counter), run, answer))
            answers.append((unit.name, answer))
        outcomes = {name: answer.get() for name, answer in answers}
        failed = sorted(name for name, outcome in outcomes.items() if outcome != "ok")
        if failed:
            self.settle(scroll, f"level fits failed: {', '.join(failed)}")
        elif (
            self.job(f"{scroll}-pieces", [*PIPELINE, "pieces", scroll, *chosen], self.place.fit_cores, fitting) != "ok"
        ):
            self.settle(scroll, "joins failed")
        else:
            self.reads.put((scroll, scan))

    def detect(self):
        """Each scroll's sheet nodes in turn, its levelling started as soon as they are done."""
        for scroll, scan in self.scrolls:
            chosen = ["--scan", scan] if scan else []
            if self.job(f"{scroll}-sheets", [*PIPELINE, "sheets", scroll, *chosen], self.place.read_cores) == "ok":
                threading.Thread(target=self.level, args=(scroll, scan), daemon=True).start()
            else:
                self.settle(scroll, "detection failed")

    def read(self, lane):
        gpu = {"CUDA_VISIBLE_DEVICES": str(lane % self.place.gpus), "HERCULEX_SAMPLERS": str(SAMPLERS)}
        while True:
            scroll, scan = self.reads.get()
            chosen = ["--scan", scan] if scan else []
            outcome = self.job(f"{scroll}-ink{SUFFIX}", [*PIPELINE, "ink", scroll, *chosen], self.place.read_cores, gpu)
            if outcome == "ok":
                self.publishes.put((scroll, scan))
            else:
                self.settle(scroll, "ink read failed")

    def publish(self):
        while True:
            scroll, scan = self.publishes.get()
            chosen = ["--scan", scan] if scan else []
            name = f"{scroll}-sheet{SUFFIX}"
            if self.job(name, [*PIPELINE, "sheet", scroll, *chosen], self.place.read_cores) != "ok":
                self.settle(scroll, "publishing failed")
                continue
            pages = PAGE.findall((LOGS / f"{name}.log").read_text())
            self.settle(scroll, "published " + ", ".join(f"{page} ({count} pieces)" for page, count in pages[-1:]))

    def run(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.say("unwrap", len(self.scrolls), "scrolls on", self.place.describe())
        for _ in range(self.place.fits):
            threading.Thread(target=self.fit_worker, daemon=True).start()
        for lane in range(self.place.reads):
            threading.Thread(target=self.read, args=(lane,), daemon=True).start()
        for _ in range(PUBLISHERS):
            threading.Thread(target=self.publish, daemon=True).start()
        threading.Thread(target=self.detect, daemon=True).start()
        with self.done:
            self.done.wait_for(lambda: len(self.outcomes) == len(self.scrolls))
        for scroll, _ in self.scrolls:
            self.say(scroll, self.outcomes[scroll])
        return 0 if all(o.startswith("published") for o in self.outcomes.values()) else 1


def size_of(unit):
    store = unit / "nodes.zst"
    return store.stat().st_size if store.exists() else 0


def unwrap(scrolls, reads_per_gpu=1):
    """The scrolls, each with its chosen scan or None, unwrapped, read and published; 0 when every page is out."""
    return Unwrap(scrolls, Machine(reads_per_gpu)).run()
