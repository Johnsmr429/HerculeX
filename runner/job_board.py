"""One queue for every long job on this machine."""

import argparse
import collections
import contextlib
import ctypes
import fcntl
import glob
import json
import os
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from storage.data_roots import HERE, LIBRARY, OUTPUTS, SCRATCH

BOARD = Path(os.environ.get("HERCULEX_BOARD", HERE / "runtime" / "board"))
MARGIN_GB = float(os.environ.get("HERCULEX_BOARD_MARGIN_GB", "1.0"))
CORES = int(os.environ.get("HERCULEX_BOARD_CORES", "10"))
DEFAULT_GB = 2.0
SMALL_GB = 1.0
NEGLIGIBLE_GB = 0.2
SWAP_START_GB, SWAP_STOP_GB = 4.2, 4.9
ALONE_BESIDE_GB = 0.5
GPU_BUSY = 50
DISPLAY = "WindowServer"
TICK_SECONDS = float(os.environ.get("HERCULEX_BOARD_TICK", "10"))
SAMPLE_SECONDS = 60
GRACE_SECONDS = 30
RETRY_SECONDS = float(os.environ.get("HERCULEX_BOARD_RETRY", "240"))
MOST_TRIES = 3
TRY_LATER = 75
MOST_LATER = 60
SHOWN_STRANGER_GB = 0.05
SHOWN_WAITING = 12
UNKNOWN_MINUTES = 10
GATE = "runner.wait_for_queue"
PEAK_MARGIN = 1.1
SLOTS = ("gpu", "engine", "link")
SHARED_SLOTS = {"gpu": 1, "link": 2}  # sharing jobs a slot takes beside the one job holding it whole
MAKE_WAY = 2
MAKE_WAY_SECONDS = 60
ACTIVE = ("starting", "running", "stopping")
FINISHED = ("done", "failed", "stopped", "lost", "cancelled")
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(name TEXT PRIMARY KEY, kind TEXT, command TEXT, folder TEXT, settings TEXT, owner TEXT, note TEXT, memory_gb REAL, cores REAL,
    gpu INTEGER, engine INTEGER, link INTEGER, priority INTEGER, nice INTEGER, background INTEGER, after TEXT, needs TEXT, clean TEXT, restartable INTEGER,
    alone INTEGER DEFAULT 0, last INTEGER DEFAULT 0, free_gib REAL DEFAULT 0, at_most TEXT, share INTEGER DEFAULT 0, adopted INTEGER DEFAULT 0, state TEXT, held INTEGER DEFAULT 0, again INTEGER DEFAULT 0, tries INTEGER DEFAULT 0, not_before REAL DEFAULT 0, pid INTEGER, born INTEGER,
    wrapper INTEGER, added REAL, started REAL, stopping REAL, ended REAL, code INTEGER, peak_gb REAL DEFAULT 0, cpu_hours REAL DEFAULT 0, read_gb REAL DEFAULT 0,
    written_gb REAL DEFAULT 0, why TEXT);
CREATE TABLE IF NOT EXISTS samples(time REAL PRIMARY KEY, gpu REAL, room_gb REAL, swap_gb REAL, swapped_out_gb REAL, load REAL, running INTEGER, waiting INTEGER, gpu_waiting INTEGER);
"""
APPLE = sys.platform == "darwin"  # elsewhere (Linux) the board reads /proc and nvidia-smi
LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib") if APPLE else None
RUSAGE_V4, READ, WRITTEN, START, FOOTPRINT, LIFETIME_PEAK = 4, 18, 19, 10, 9, 30


def connect():
    (BOARD / "logs").mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(BOARD / "board.sqlite", timeout=30, isolation_level=None)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=NORMAL")
    database.executescript(SCHEMA)
    columns = [row[1] for row in database.execute("PRAGMA table_info(jobs)")]
    for name, kind in (
        ("last", "INTEGER DEFAULT 0"),
        ("free_gib", "REAL DEFAULT 0"),
        ("at_most", "TEXT"),
        ("share", "INTEGER DEFAULT 0"),
    ):
        if name not in columns:
            database.execute(f"ALTER TABLE jobs ADD COLUMN {name} {kind}")
    return database


def all_jobs(database):
    return [dict(row) for row in database.execute("SELECT * FROM jobs ORDER BY added")]


def job_named(database, name):
    row = database.execute("SELECT * FROM jobs WHERE name=?", (name,)).fetchone()
    if row is None:
        sys.exit(f"no job named {name}")
    return dict(row)


def log_of(name):
    return BOARD / "logs" / f"{name.replace('/', '-')}.log"


def text_of(*command):
    return subprocess.run(command, capture_output=True, text=True, check=False).stdout


_gpu_seen = {}


def gpu_clients():
    """GPU time used so far by each process, in ns."""
    used = collections.defaultdict(int)
    for node in text_of("ioreg", "-r", "-c", "AGXDeviceUserClient", "-l", "-w0").split("+-o ")[1:]:
        creator = re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"', node)
        if creator:
            used[(int(creator.group(1)), creator.group(2))] += sum(
                int(n) for n in re.findall(r'"accumulatedGPUTime"=(\d+)', node)
            )
    return used


def gpu_of_others(device):
    """Percent of the GPU used since the last look by every process but the display."""
    if not _gpu_seen:
        _gpu_seen.update(at=time.time(), clients=gpu_clients())
        time.sleep(0.5)
    now, clients = time.time(), gpu_clients()
    span, before = now - _gpu_seen["at"], _gpu_seen["clients"]
    _gpu_seen.update(at=now, clients=clients)
    used = sum(n - before[key] for key, n in clients.items() if key in before and key[1] != DISPLAY)
    return min(device, max(0, round(used / max(span, 1e-3) / 1e7)))


def machine():
    """What the board goes by, in GB: memory no process holds, swap, swap written so far."""
    if not APPLE:
        memory = {
            key: int(value.split()[0]) * 1024
            for key, value in (line.split(":") for line in Path("/proc/meminfo").read_text().splitlines())
        }
        swapped = int(re.search(r"^pswpout (\d+)", Path("/proc/vmstat").read_text(), re.MULTILINE).group(1))
        query = ("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")
        found = re.search(r"\d+", text_of(*query) if shutil.which(query[0]) else "")
        device = int(found.group()) if found else 0
        return {
            "room_gb": memory["MemAvailable"] / 1e9 - MARGIN_GB,
            "swap_gb": (memory["SwapTotal"] - memory["SwapFree"]) / 1e9,
            "swapped_out_gb": swapped * os.sysconf("SC_PAGE_SIZE") / 1e9,
            "gpu": device,  # the display's share is not known here, so none is taken off
            "gpu_device": device,
            "load": os.getloadavg()[0],
        }
    stats = text_of("vm_stat")
    page = int(re.search(r"page size of (\d+)", stats).group(1))
    pages = {name: int(value) for name, value in re.findall(r'^"?(.+?)"?:\s+(\d+)\.$', stats, re.MULTILINE)}
    held = pages["Pages wired down"] + pages["Anonymous pages"] + pages["Pages occupied by compressor"]
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    swap = float(re.search(r"used = ([0-9.]+)M", text_of("sysctl", "-n", "vm.swapusage")).group(1)) * 2**20 / 1e9
    found = re.search(r'"Device Utilization %"=(\d+)', text_of("ioreg", "-r", "-c", "IOAccelerator", "-d", "1"))
    device = int(found.group(1)) if found else 0
    return {
        "room_gb": (total - held * page) / 1e9 - MARGIN_GB,
        "swap_gb": swap,
        "swapped_out_gb": pages["Swapouts"] * page / 1e9,
        "gpu": gpu_of_others(device),
        "gpu_device": device,
        "load": os.getloadavg()[0],
    }


def usage_of(pid):
    """A process's kernel counters, read without touching it."""
    if APPLE:
        words = (ctypes.c_uint64 * 42)()
        return list(words) if LIBPROC.proc_pid_rusage(pid, RUSAGE_V4, ctypes.byref(words)) == 0 else None
    try:
        stat = stat_of(pid)
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    if stat[0] == "Z":  # ended and not yet collected by its parent
        return None
    words = [0] * 42
    words[START] = int(stat[19])
    for index, key in ((FOOTPRINT, "VmRSS"), (LIFETIME_PEAK, "VmHWM")):
        found = re.search(rf"^{key}:\s+(\d+) kB", status, re.MULTILINE)
        words[index] = int(found.group(1)) * 1024 if found else 0
    with contextlib.suppress(OSError):  # another user's process keeps its I/O counters to itself
        counters = dict(line.split(": ") for line in Path(f"/proc/{pid}/io").read_text().splitlines())
        words[READ], words[WRITTEN] = int(counters["read_bytes"]), int(counters["write_bytes"])
    return words


def stat_of(pid):
    """Linux: the fields of /proc/PID/stat that follow the command name (state, parent, group, ...)."""
    return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()


def group_of(pid):
    if APPLE:
        pids = (ctypes.c_int * 1024)()
        return list(pids[: max(LIBPROC.proc_listpgrppids(pid, pids, ctypes.sizeof(pids)), 0)])
    found = []
    for entry in os.listdir("/proc"):
        with contextlib.suppress(OSError):
            if entry.isdecimal() and int(stat_of(entry)[2]) == pid:
                found.append(int(entry))
    return found


def alive(job):
    """Whether a job's process is still the one that was started."""
    usage = usage_of(job["pid"]) if job["pid"] else None
    return usage is not None and usage[START] == job["born"]


def footprint_of(job):
    """Footprint now of a job's whole process group, or of an adopted process alone, in GB."""
    if not job["pid"]:
        return 0.0
    usages = [usage_of(pid) for pid in ([job["pid"]] if job["adopted"] else group_of(job["pid"]))]
    return sum(u[FOOTPRINT] for u in usages if u) / 1e9


def room_left(jobs, state):
    """Memory free now, less what running jobs were promised and have not yet reached, now or at their peak."""
    return state["room_gb"] - sum(
        max(job["memory_gb"] - max(footprint_of(job), job["peak_gb"]), 0.0) for job in jobs if job["state"] in ACTIVE
    )


def can_still_run(job, named):
    """Whether a job not yet ended may run."""
    if job["state"] in ACTIVE:
        return True
    if job["state"] != "waiting" or job["held"]:
        return False
    return all(
        name in named and (named[name]["state"] == "done" or can_still_run(named[name], named))
        for name in json.loads(job["after"])
    )


def beside(job, holders, slot):
    """Sharing jobs run beside at most one job that holds the slot whole, up to the slot's number of sharers (one
    on the GPU, two on the link); a whole job runs beside sharing jobs only."""
    if slot not in SHARED_SLOTS:
        return False
    whole = sum(not other.get("share") for other in holders)
    sharers = len(holders) - whole
    if job.get("share"):
        return whole <= 1 and sharers < SHARED_SLOTS[slot]
    return whole == 0 and sharers <= SHARED_SLOTS[slot]


def blocked(job, jobs, state, room):
    """Why a waiting job cannot start now, or None."""
    if job["held"]:
        return "held"
    if time.time() < job["not_before"]:
        return f"rests until {time.strftime('%H:%M', time.gmtime(job['not_before']))} UTC"
    named = {other["name"]: other for other in jobs}
    for name in json.loads(job["after"]):
        if name not in named:
            return f"after {name}, not on the board"
        if named[name]["state"] != "done":
            return f"after {name}, {named[name]['state']}"
    for path in json.loads(job["needs"]):
        if not (Path(job["folder"]) / path).exists():
            return f"needs {path}"
    if job["last"]:
        ahead = [
            other["name"]
            for other in jobs
            if other["added"] < job["added"] and not other["last"] and can_still_run(other, named)
        ]
        if ahead:
            return f"last: {len(ahead)} jobs ahead, first {ahead[0]}"
    free = shutil.disk_usage("/").free / 2**30
    if free < job["free_gib"]:
        return f"needs {job['free_gib']:g} GiB free on the Mac, has {free:.0f}"
    if job["at_most"]:
        count, pattern = json.loads(job["at_most"])
        if len(glob.glob(pattern)) >= count:
            return f"at most {count - 1} of {Path(pattern).name} before it, {len(glob.glob(pattern))} there"
    active = [other for other in jobs if other["state"] in ACTIVE]
    if any(other["alone"] for other in active):
        return "the machine is with " + next(other["name"] for other in active if other["alone"])
    if job["alone"] and (active or state.get("strangers")):
        return "wants the machine to itself: " + (
            active[0]["name"] if active else f"{len(state['strangers'])} processes off the board"
        )
    for slot in SLOTS:
        holders = [other for other in active if other[slot]]
        if job[slot] and holders and not beside(job, holders, slot):
            return f"{slot} is with {holders[0]['name']}"
    if job["gpu"] and state["gpu"] >= GPU_BUSY and not any(other["gpu"] for other in active):
        return f"GPU {state['gpu']}% busy off the board"
    used = sum(other["cores"] for other in active)
    if used + job["cores"] > CORES:
        return f"needs {job['cores']:g} cores, {used:g} of {CORES} taken"
    if job["memory_gb"] > room and any(other["memory_gb"] > NEGLIGIBLE_GB for other in active):
        return f"needs {job['memory_gb']:.1f} GB, room {room:.1f}"
    if job["memory_gb"] > SMALL_GB and state["swap_gb"] > SWAP_START_GB:
        return f"swap {state['swap_gb']:.1f} GB, over {SWAP_START_GB}"
    return None


def decisions(jobs, state, room):
    """Each waiting job in turn, with why it cannot start or None."""
    holding = [(job["priority"], job["memory_gb"]) for job in jobs if job["state"] in ACTIVE]
    ceiling, first = room + sum(gb for _, gb in holding), None
    for job in sorted((j for j in jobs if j["state"] == "waiting"), key=lambda j: (-j["priority"], j["added"])):
        free = room
        if first and job["priority"] < first["priority"]:
            free = min(
                room, ceiling - first["memory_gb"] - sum(gb for priority, gb in holding if priority < first["priority"])
            )
        reason = blocked(job, jobs, state, free)
        if reason is None:
            room -= job["memory_gb"]
            holding.append((job["priority"], job["memory_gb"]))
        elif first is None and "GB, room" in reason:
            first = job
        yield job, reason


def wrapped(job):
    command = json.loads(job["command"])
    if job["background"]:
        command = ["taskpolicy", "-b", *command] if APPLE else ["nice", "-n", "19", *command]
    return ["nice", "-n", str(job["nice"]), *command] if job["nice"] else command


def wrap(name):
    """Run one job in its own process group, follow its footprint and leave how it ended on the board."""
    database = connect()
    job = job_named(database, name)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    environment = (
        os.environ
        | {"PYTHONPATH": str(HERE), "PYTHONDONTWRITEBYTECODE": "1", "MallocLargeCache": "0", "TMPDIR": str(SCRATCH)}
        | json.loads(job["settings"])
    )
    with open(log_of(name), "ab", buffering=0) as log:
        log.write(
            f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC, try {job['tries'] + 1}: {' '.join(wrapped(job))}\n".encode()
        )
        process = subprocess.Popen(
            wrapped(job),
            cwd=job["folder"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    first = usage_of(process.pid)
    database.execute(
        "UPDATE jobs SET state='running', pid=?, born=?, wrapper=?, started=?, tries=tries+1, why=NULL WHERE name=?",
        (process.pid, first[START] if first else 0, os.getpid(), time.time(), name),
    )
    peak, seen = job["peak_gb"], {}
    while True:
        finished, status, spent = os.wait4(process.pid, os.WNOHANG)
        if finished:
            break
        usages = {pid: usage_of(pid) for pid in group_of(process.pid)}
        seen |= {pid: (u[READ], u[WRITTEN]) for pid, u in usages.items() if u}
        now = sum(u[FOOTPRINT] for u in usages.values() if u) / 1e9
        peak = max(peak, now, *[u[LIFETIME_PEAK] / 1e9 for u in usages.values() if u], 0.0)
        database.execute("UPDATE jobs SET peak_gb=? WHERE name=?", (peak, name))
        time.sleep(min(TICK_SECONDS, 2))
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGTERM)
    job = job_named(database, name)
    code = os.waitstatus_to_exitcode(status)
    state, rest = ("done" if code == 0 else "failed"), 0.0
    if code == TRY_LATER and job["state"] == "running" and job["tries"] < MOST_LATER:
        state, rest = "waiting", time.time() + RETRY_SECONDS
    if job["state"] == "stopping":
        again = job["again"] == MAKE_WAY or (job["again"] and job["tries"] < MOST_TRIES)
        state = "waiting" if again else "stopped"
        rest = time.time() + RETRY_SECONDS if state == "waiting" else 0.0
        for path in json.loads(job["clean"]):
            target = Path(job["folder"]) / path
            shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink(missing_ok=True)
    database.execute(
        "UPDATE jobs SET state=?, not_before=?, ended=?, code=?, peak_gb=?, cpu_hours=?, read_gb=?, written_gb=?, pid=NULL WHERE name=?",
        (
            state,
            rest,
            time.time(),
            code,
            peak,
            job["cpu_hours"] + (spent.ru_utime + spent.ru_stime) / 3600,
            job["read_gb"] + sum(r for r, _ in seen.values()) / 1e9,
            job["written_gb"] + sum(w for _, w in seen.values()) / 1e9,
            name,
        ),
    )
    serving()


def launch(database, job):
    database.execute("UPDATE jobs SET state='starting', started=? WHERE name=?", (time.time(), job["name"]))
    subprocess.Popen(
        [sys.executable, "-m", "runner.job_board", "wrap", job["name"]],
        cwd=HERE,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=os.environ | {"PYTHONPATH": str(HERE), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    job["state"] = "starting"


def stop(database, job, again, why):
    """TERM to a job's whole group."""
    database.execute(
        "UPDATE jobs SET state='stopping', stopping=?, again=?, why=? WHERE name=?",
        (time.time(), int(again), why, job["name"]),
    )
    job["state"] = "stopping"
    with contextlib.suppress(ProcessLookupError, PermissionError, TypeError):
        os.killpg(job["pid"], signal.SIGTERM)


def follow(database, job):
    """Peak of an adopted process and its end."""
    now = time.time()
    if job["adopted"]:
        usage = usage_of(job["pid"]) if alive(job) else None
        if usage:
            database.execute(
                "UPDATE jobs SET peak_gb=max(peak_gb, ?) WHERE name=?", (usage[LIFETIME_PEAK] / 1e9, job["name"])
            )
        else:
            database.execute(
                "UPDATE jobs SET state='done', ended=?, pid=NULL, why='adopted process ended' WHERE name=?",
                (now, job["name"]),
            )
            job["state"] = "done"
    elif job["state"] == "stopping" and now - job["stopping"] > GRACE_SECONDS:
        with contextlib.suppress(ProcessLookupError, PermissionError, TypeError):
            os.killpg(job["pid"], signal.SIGKILL)
    elif job["state"] == "running" and job["wrapper"] and usage_of(job["wrapper"]) is None:
        again = job["restartable"] and job["tries"] < MOST_TRIES
        vanished = database.execute(
            "UPDATE jobs SET state=?, ended=?, pid=NULL, why='its wrapper vanished' WHERE name=? AND state='running'",
            ("waiting" if again else "lost", now, job["name"]),
        ).rowcount
        job["state"] = ("waiting" if again else "lost") if vanished else job_named(database, job["name"])["state"]
    elif job["state"] == "starting" and now - job["started"] > GRACE_SECONDS:
        database.execute("UPDATE jobs SET state='lost', ended=?, why='never started' WHERE name=?", (now, job["name"]))
        job["state"] = "lost"


def relieve(database, jobs, state):
    """Past the swap limit the largest restartable job of the lowest priority stops and comes back later, never the job the machine is there for."""
    if state["swap_gb"] < SWAP_STOP_GB or any(job["state"] == "stopping" for job in jobs):
        return
    running = [job for job in jobs if job["state"] == "running" and not job["adopted"]]
    top = max((job["priority"] for job in running), default=0)
    candidates = [job for job in running if job["restartable"] and job["priority"] < top]
    if candidates:
        lowest = min(job["priority"] for job in candidates)
        stop(
            database,
            max((job for job in candidates if job["priority"] == lowest), key=footprint_of),
            True,
            f"swap {state['swap_gb']:.1f} GB",
        )


_made_way = {"at": 0.0}


def feeds_gpu(job, jobs, steps=2):
    """Whether a job holds the GPU whole, or a waiting GPU job waits for it at most `steps` jobs on (a read's layout
    and fetch, not the long stages before them, which would stop one another)."""
    if job["gpu"] and not job.get("share"):
        return True
    return steps > 0 and any(
        job["name"] in json.loads(other["after"]) and feeds_gpu(other, jobs, steps - 1)
        for other in jobs
        if other["state"] == "waiting"
    )


def make_way(database, jobs, told, room):
    """The first job on the GPU's path (a GPU job, or one a waiting GPU job waits for) held back only by cores or
    memory that lower restartable jobs hold stops one of them: the lowest (the latest started first) that alone makes
    room, else the lowest; a stopped job waits and comes back on its own. One at a time, a minute apart. Only a GPU
    job stops whole jobs; the jobs it waits for stop sharing ones alone (a detection resumes from its bricks, a long
    unit of pieces would be lost)."""
    if any(job["state"] == "stopping" for job in jobs) or time.time() - _made_way["at"] < MAKE_WAY_SECONDS:
        return None
    first = next(
        (
            job
            for job, reason in told
            if reason and ("cores," in reason or "GB, room" in reason) and feeds_gpu(job, jobs)
        ),
        None,
    )
    if first is None:
        return None
    active = [job for job in jobs if job["state"] in ACTIVE]
    short_cores = sum(job["cores"] for job in active) + first["cores"] - CORES
    short_gb = first["memory_gb"] - room
    candidates = sorted(
        (
            job
            for job in active
            if job["state"] == "running"
            and job["restartable"]
            and not job["adopted"]
            and job["priority"] < first["priority"]
            and (job.get("share") or (first["gpu"] and not first.get("share")))
        ),
        key=lambda job: (job["priority"], -(job["started"] or 0)),
    )
    held = {job["name"]: max(job["memory_gb"], footprint_of(job)) for job in candidates}
    if not candidates or sum(job["cores"] for job in candidates) < short_cores or sum(held.values()) < short_gb:
        return None
    victim = next(
        (job for job in candidates if job["cores"] >= short_cores and held[job["name"]] >= short_gb), candidates[0]
    )
    _made_way["at"] = time.time()
    stop(database, victim, MAKE_WAY, f"made way for {first['name']}")
    return victim["name"]


def tick(database):
    jobs, state = all_jobs(database), machine()
    for job in jobs:
        if job["state"] in ACTIVE:
            follow(database, job)
    relieve(database, jobs, state)
    room = room_left(jobs, state)
    if any(job["alone"] and job["state"] == "waiting" for job in jobs):
        state["strangers"] = [found for found in strangers(jobs) if found[0] >= ALONE_BESIDE_GB]
    reasons = []
    told = []
    # decisions is lazy on purpose: each job launched is running before the next is weighed, so slots and cores count it
    for job, reason in decisions(jobs, state, room):
        told.append((job, reason))
        if reason != job["why"]:
            reasons.append((reason, job["name"]))
        if reason is None:
            launch(database, job)
    make_way(database, jobs, told, room)
    database.execute("BEGIN")
    database.executemany("UPDATE jobs SET why=? WHERE name=?", reasons)
    database.execute("COMMIT")
    return jobs, state, room


def serve():
    """The dispatcher."""
    BOARD.mkdir(parents=True, exist_ok=True)
    lock = open(BOARD / "dispatcher.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    lock.truncate(0)
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    database, sampled = connect(), 0.0
    while True:
        try:
            jobs, state, room = tick(database)
            if time.time() - sampled >= SAMPLE_SECONDS:
                sampled = time.time()
                waiting = [job for job in jobs if job["state"] == "waiting"]
                database.execute(
                    "INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sampled,
                        state["gpu"],
                        room,
                        state["swap_gb"],
                        state["swapped_out_gb"],
                        state["load"],
                        sum(job["state"] in ACTIVE for job in jobs),
                        len(waiting),
                        sum(job["gpu"] for job in waiting),
                    ),
                )
        except Exception:
            print(time.strftime("%H:%M:%S"), traceback.format_exc(), flush=True)
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
        if not (BOARD / "dispatcher.lock").exists():
            return
        time.sleep(TICK_SECONDS)


def serving():
    """Start the dispatcher unless one holds the lock."""
    BOARD.mkdir(parents=True, exist_ok=True)
    with open(BOARD / "dispatcher.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    with open(BOARD / "dispatcher.log", "ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "runner.job_board", "serve"],
            cwd=HERE,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ | {"PYTHONPATH": str(HERE), "PYTHONDONTWRITEBYTECODE": "1"},
        )


def kind_of(command):
    """What jobs share a memory history."""
    modules = [word for word in command if re.fullmatch(r"[a-z_]+(\.[a-z_]+)+", word)]
    return modules[0] if modules else Path(command[0]).name


def learned_memory(database, kind):
    peaks = [
        row[0]
        for row in database.execute(
            "SELECT peak_gb FROM jobs WHERE kind=? AND state='done' AND peak_gb>0 ORDER BY ended DESC LIMIT 3", (kind,)
        )
    ]
    return round(max(peaks) * PEAK_MARGIN, 2) if peaks else DEFAULT_GB


def add(database, arguments, command):
    kind = arguments.kind or kind_of(command)
    memory = arguments.memory if arguments.memory is not None else learned_memory(database, kind)
    settings = dict(pair.split("=", 1) for pair in arguments.setting)
    try:
        database.execute(
            """INSERT INTO jobs(name, kind, command, folder, settings, owner, note, memory_gb, cores, gpu, engine, link, priority, nice, background, after, needs, clean, restartable, alone, last, free_gib, at_most, share, state, held, added)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'waiting', ?, ?)""",
            (
                arguments.name,
                kind,
                json.dumps(command),
                str(Path(arguments.folder).resolve()),
                json.dumps(settings),
                arguments.owner,
                arguments.note,
                memory,
                arguments.cores,
                int(arguments.gpu),
                int(arguments.engine),
                int(arguments.link),
                arguments.priority,
                (0 if arguments.last else 5) if arguments.nice is None else arguments.nice,
                int(arguments.background),
                json.dumps(arguments.after),
                json.dumps(arguments.needs),
                json.dumps(arguments.clean),
                int(arguments.restartable),
                int(arguments.alone),
                int(arguments.last),
                arguments.free,
                json.dumps([int(arguments.at_most[0]), arguments.at_most[1]]) if arguments.at_most else None,
                int(arguments.share),
                int(arguments.held),
                time.time(),
            ),
        )
    except sqlite3.IntegrityError:
        sys.exit(f"a job named {arguments.name} is already on the board")
    serving()
    print(f"{arguments.name}: {kind}, {memory:.1f} GB, waiting")


def adopt(database, arguments):
    """A process started elsewhere."""
    usage = usage_of(arguments.pid)
    if usage is None:
        sys.exit(f"no process {arguments.pid}")
    command = text_of("ps", "-o", "command=", "-p", str(arguments.pid)).split()
    database.execute(
        """INSERT INTO jobs(name, kind, command, folder, settings, owner, note, memory_gb, cores, gpu, engine, link, priority, nice, background, after, needs, clean, restartable,
                        adopted, state, pid, born, added, started, peak_gb) VALUES (?, ?, ?, ?, '{}', ?, ?, 0, ?, ?, ?, ?, 0, 0, 0, '[]', '[]', '[]', 0, 1, 'running', ?, ?, ?, ?, ?)""",
        (
            arguments.name,
            arguments.kind or kind_of(command),
            json.dumps(command),
            str(HERE),
            arguments.owner,
            arguments.note,
            arguments.cores,
            int(arguments.gpu),
            int(arguments.engine),
            int(arguments.link),
            arguments.pid,
            usage[START],
            time.time(),
            time.time(),
            usage[LIFETIME_PEAK] / 1e9,
        ),
    )
    serving()
    print(f"{arguments.name}: process {arguments.pid} adopted, {usage[FOOTPRINT] / 1e9:.1f} GB now")


def strangers(jobs):
    """HerculeX processes running outside the board, with their footprint."""
    known = (
        {job["pid"] for job in jobs if job["pid"]} | {job["wrapper"] for job in jobs if job["wrapper"]} | {os.getpid()}
    )
    for job in jobs:
        if job["pid"] and not job["adopted"]:
            known |= set(group_of(job["pid"]))
    found = []
    for line in text_of("ps", "-Ao", "pid=,command=").splitlines():
        pid, command = line.split(None, 1)
        if "runtime/python/bin/python" in command:
            usage = usage_of(int(pid))
            if int(pid) not in known and "runner.job_board" not in command and usage:
                found.append((usage[FOOTPRINT] / 1e9, int(pid), command))
    return sorted(found, reverse=True)


def clock(seconds):
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def slots_of(job):
    return (
        "+".join(slot + ("~" if job.get("share") and slot in SHARED_SLOTS else "") for slot in SLOTS if job[slot])
        or "-"
    )


def status(database, everything=False):
    jobs, state = all_jobs(database), machine()
    room, now = room_left(jobs, state), time.time()
    discs = ", ".join(
        f"{name} {shutil.disk_usage(path).free / 2**30:.0f} GiB"
        for name, path in (("internal", "/"), ("library", LIBRARY), ("outputs", OUTPUTS))
        if Path(path).exists()
    )
    others = strangers(jobs)
    print(
        f"{time.strftime('%H:%M', time.gmtime())} UTC  room {room:.1f} GB  off board {sum(other[0] for other in others):.1f} GB  swap {state['swap_gb']:.1f} GB  GPU {state['gpu_device']}%, display {state['gpu_device'] - state['gpu']}%  load {state['load']:.1f}  free {discs}"
    )
    for job in jobs:
        if job["state"] in ACTIVE:
            print(
                f"  {job['state']:10s}{job['name']:34s}{slots_of(job):12s}{footprint_of(job):5.1f} GB, peak {job['peak_gb']:.1f}  {clock(now - job['started'])}{'  adopted' if job['adopted'] else ''}"
            )
    waiting = list(decisions(jobs, state, room))
    for job, reason in waiting if everything else waiting[:SHOWN_WAITING]:
        print(
            f"  {'waiting':10s}{job['name']:34s}{slots_of(job):12s}{job['memory_gb']:5.1f} GB  {reason or 'starting'}"
        )
    if not everything and len(waiting) > SHOWN_WAITING:
        print(f"  ... and {len(waiting) - SHOWN_WAITING} more waiting: status --all")
    ended = sorted((j for j in jobs if j["state"] in FINISHED), key=lambda j: j["ended"] or 0)
    for job in ended if everything else ended[-6:]:
        took = max((job["ended"] or now) - (job["started"] or job["ended"] or now), 0)
        print(
            f"  {job['state']:10s}{job['name']:34s}{slots_of(job):12s}peak {job['peak_gb']:.1f} GB  {clock(took)}  cpu {job['cpu_hours']:.2f} h  read {job['read_gb']:.1f} wrote {job['written_gb']:.1f} GB{'  ' + job['why'] if job['why'] else ''}"
        )
    for footprint, pid, command in others:
        if footprint >= SHOWN_STRANGER_GB:
            print(f"  off board {pid:<8d}{footprint:5.1f} GB  {command[:90]}")


def share_held(database, slot, since, now):
    """Share of a period when some job held a slot."""
    held, reached = 0.0, since
    for started, ended in sorted(
        (max(job["started"], since), job["ended"] or now)
        for job in database.execute(
            f"SELECT started, ended FROM jobs WHERE {slot}=1 AND started>0 AND (ended IS NULL OR ended>?)", (since,)
        )
    ):
        held += max(ended - max(started, reached), 0.0)
        reached = max(reached, ended)
    return held / (now - since)


def usage_report(database, hours):
    """How the machine was used."""
    now = time.time()
    since = now - hours * 3600
    rows = database.execute("SELECT * FROM samples WHERE time>=? ORDER BY time", (since,)).fetchall()
    if not rows:
        print("no samples yet")
        return
    busy = sum(r["gpu"] >= GPU_BUSY for r in rows)
    idle_waiting = sum(r["gpu"] < GPU_BUSY and r["gpu_waiting"] > 0 for r in rows)
    ended = database.execute(
        "SELECT COUNT(*), SUM(cpu_hours) FROM jobs WHERE ended>=? AND state='done'", (since,)
    ).fetchone()
    print(
        f"{len(rows)} minutes sampled: GPU busy {100 * busy / len(rows):.0f}%, idle with GPU work waiting {100 * idle_waiting / len(rows):.0f}%, swap peak {max(r['swap_gb'] for r in rows):.1f} GB, "
        f"swapped out {rows[-1]['swapped_out_gb'] - rows[0]['swapped_out_gb']:.1f} GB, least room {min(r['room_gb'] for r in rows):.1f} GB, {ended[0]} jobs done, {ended[1] or 0:.1f} CPU-hours"
    )
    print(", ".join(f"{slot} held {100 * share_held(database, slot, since, now):.0f}%" for slot in SLOTS))


def typical_runs(database):
    """Median minutes and memory promised of finished jobs, by name without its last word and by kind."""
    runs = {}
    for job in database.execute(
        "SELECT name, kind, memory_gb, started, ended FROM jobs WHERE state='done' AND ended>started"
    ):
        for key in (job["name"].rpartition("-")[0], job["kind"]):
            runs.setdefault(key, []).append(((job["ended"] - job["started"]) / 60, job["memory_gb"]))
    return {
        key: (statistics.median(run[0] for run in found), statistics.median(run[1] for run in found))
        for key, found in runs.items()
    }


def minutes_of(job, typical):
    """A job's expected minutes."""
    for key in (job["name"].rpartition("-")[0], job["kind"]):
        if key in typical:
            minutes, memory = typical[key]
            return minutes * job["memory_gb"] / memory if memory else minutes
    return UNKNOWN_MINUTES


def forecast(jobs, typical, room, now):
    """Minutes from now when each job starts and ends, by the board's own decisions on a simulated clock."""
    jobs = [dict(job, not_before=0, needs="[]", free_gib=0, at_most=None) for job in jobs]
    named = {job["name"]: job for job in jobs}
    left = {
        job["name"]: max(minutes_of(job, typical) - (now - job["started"]) / 60, 1.0)
        for job in jobs
        if job["state"] in ACTIVE
    }
    ceiling = room + sum(named[name]["memory_gb"] for name in left)
    started, ended, clock = {}, {}, 0.0
    while True:
        room = ceiling - sum(named[name]["memory_gb"] for name in left)
        for job, reason in decisions(jobs, {"gpu": 0, "swap_gb": 0.0}, room):
            if reason is None:
                job["state"], started[job["name"]], left[job["name"]] = "running", clock, minutes_of(job, typical)
        timed = [name for name in left if named[name]["kind"] != GATE]
        if not left:
            return started, ended, clock
        step = min((left[name] for name in timed), default=0.0)
        clock += step
        for name in timed or list(left):
            left[name] -= step
            if not timed or left[name] <= 1e-9:
                named[name]["state"], ended[name] = "done", clock
                del left[name]


def forecast_report(database, name):
    jobs, state, now = all_jobs(database), machine(), time.time()
    started, ended, clock = forecast(jobs, typical_runs(database), room_left(jobs, state), now)

    def at(minutes):
        return time.strftime("%H:%M UTC", time.gmtime(now + minutes * 60))

    if name:
        print(
            f"{name} starts about {at(started[name])}, ends about {at(ended[name])}"
            if name in started
            else f"{name} ends about {at(ended[name])}"
            if name in ended
            else f"{name} will not run as things stand: it is held, ended, or waits behind one that is"
        )
        return
    named = {job["name"]: job for job in jobs}
    stuck = sum(job["state"] == "waiting" and job["name"] not in started for job in jobs)
    print(
        f"the queue should end about {at(clock)}, {clock / 60:.1f} h from now, by past times of each sort of job; disc limits are not counted"
        + (f"; {stuck} held or behind one held are left out" if stuck else "")
    )
    if clock:
        print(
            ", ".join(
                f"{slot} busy {100 * sum(ended[n] - started.get(n, 0.0) for n in ended if named[n][slot]) / clock:.0f}%"
                for slot in SLOTS
            )
        )
    for owner in sorted({named[n]["owner"] or "no owner" for n in ended}):
        print(f"  {owner:24s}{at(max(ended[n] for n in ended if (named[n]['owner'] or 'no owner') == owner))}")


def wait_for(database, name):
    """Block until a job ends, for a background shell."""
    while True:
        job = job_named(database, name)
        if job["state"] in FINISHED:
            print(name, job["state"], job["code"], f"peak {job['peak_gb']:.1f} GB")
            sys.exit(0 if job["state"] == "done" else job["code"] or 1)
        time.sleep(5)


def main():
    words = sys.argv[1:]
    command = words[words.index("--") + 1 :] if "--" in words else []
    words = words[: words.index("--")] if "--" in words else words
    parser = argparse.ArgumentParser(prog="runner.job_board")
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("add", "adopt"):
        action = actions.add_parser(name)
        if name == "adopt":
            action.add_argument("pid", type=int)
        action.add_argument("name")
        action.add_argument("--kind")
        action.add_argument("--owner")
        action.add_argument("--note")
        action.add_argument("--cores", type=float, default=1.0)
        for slot in SLOTS:
            action.add_argument(f"--{slot}", action="store_true")
    action = actions.choices["add"]
    action.add_argument("--memory", type=float, help="peak GB of the whole job; left out, the last peaks of its kind")
    action.add_argument("--priority", type=int, default=0)
    action.add_argument("--nice", type=int, help="5, or 0 for a --last job")
    action.add_argument("--background", action="store_true", help="efficiency cores and slow disc")
    action.add_argument("--after", nargs="*", default=[], help="jobs that must be done")
    action.add_argument("--needs", nargs="*", default=[], help="paths that must exist")
    action.add_argument("--clean", nargs="*", default=[], help="paths removed when a job is stopped")
    action.add_argument("--restartable", action="store_true", help="may be stopped under swap and run again")
    action.add_argument(
        "--alone",
        action="store_true",
        help="the machine to itself: waits for every other job, and none starts beside it",
    )
    action.add_argument(
        "--last",
        action="store_true",
        help="at full speed once nothing added before it can still run, however those end",
    )
    action.add_argument(
        "--free", type=float, default=0.0, metavar="GIB", help="free on the Mac's disc before it starts"
    )
    action.add_argument(
        "--at-most", nargs=2, metavar=("N", "PATTERN"), help="starts only while fewer than N paths match"
    )
    action.add_argument(
        "--share",
        action="store_true",
        help="runs beside one other job on the GPU and the link (short GPU bursts, mostly downloading)",
    )
    action.add_argument("--held", action="store_true", help="waits for release")
    action.add_argument("--setting", nargs="*", action="extend", default=[], help="KEY=VALUE for the environment")
    action.add_argument("--folder", default=str(HERE))
    actions.add_parser("status").add_argument("--all", action="store_true")
    actions.add_parser("room").add_argument("gb", type=float)
    actions.add_parser("usage").add_argument("hours", type=float, nargs="?", default=24.0)
    actions.add_parser("serve")
    for name in ("wrap", "wait", "hold", "release", "cancel", "retry", "remove"):
        actions.add_parser(name).add_argument("name")
    action = actions.add_parser(
        "resize", help="a job's memory, once better known; lower a running job's only if you know its peak is behind it"
    )
    action.add_argument("name")
    action.add_argument("gb", type=float)
    action = actions.add_parser("change", help="a job's priority, cores, sharing or restarting, as the queue changes")
    action.add_argument("name")
    action.add_argument("--priority", type=int)
    action.add_argument("--cores", type=float)
    action.add_argument("--share", type=int, choices=(0, 1))
    action.add_argument("--restartable", type=int, choices=(0, 1))
    action = actions.add_parser("forecast", help="when the queue, or one job, should end")
    action.add_argument("name", nargs="?")
    action = actions.add_parser("log")
    action.add_argument("name")
    action.add_argument("lines", type=int, nargs="?", default=30)
    arguments = parser.parse_args(words)
    if arguments.action == "serve":
        serve()
        return
    if arguments.action == "wrap":
        wrap(arguments.name)
        return
    database = connect()
    serving()
    if arguments.action == "add":
        if not command:
            sys.exit("add needs a command after --")
        add(database, arguments, command)
    elif arguments.action == "adopt":
        adopt(database, arguments)
    elif arguments.action == "status":
        status(database, arguments.all)
    elif arguments.action == "forecast":
        forecast_report(database, arguments.name)
    elif arguments.action == "usage":
        usage_report(database, arguments.hours)
    elif arguments.action == "room":
        jobs, state = all_jobs(database), machine()
        room = room_left(jobs, state)
        fits = arguments.gb <= room and (arguments.gb <= SMALL_GB or state["swap_gb"] <= SWAP_START_GB)
        print(f"{'fits' if fits else 'wait'}: {arguments.gb:g} GB against room {room:.1f}, swap {state['swap_gb']:.1f}")
        sys.exit(0 if fits else 1)
    elif arguments.action == "log":
        print(
            "".join(open(log_of(arguments.name), errors="replace").readlines()[-arguments.lines :])
            if log_of(arguments.name).exists()
            else f"{arguments.name} has no log yet\n",
            end="",
        )
    elif arguments.action == "wait":
        wait_for(database, arguments.name)
    else:
        job = job_named(database, arguments.name)
        if arguments.action == "hold":
            database.execute("UPDATE jobs SET held=1 WHERE name=?", (arguments.name,))
        elif arguments.action == "change":
            fields = ("priority", "cores", "share", "restartable")
            changed = {field: getattr(arguments, field) for field in fields if getattr(arguments, field) is not None}
            if not changed:
                sys.exit("change needs --priority, --cores, --share or --restartable")
            database.execute(
                f"UPDATE jobs SET {', '.join(f'{field}=?' for field in changed)} WHERE name=?",
                (*changed.values(), arguments.name),
            )
        elif arguments.action == "resize" and (job["state"] == "waiting" or job["state"] in ACTIVE):
            database.execute("UPDATE jobs SET memory_gb=? WHERE name=?", (arguments.gb, arguments.name))
        elif arguments.action == "release":
            database.execute("UPDATE jobs SET held=0, not_before=0 WHERE name=?", (arguments.name,))
            serving()
        elif arguments.action == "cancel" and job["state"] == "running" and not job["adopted"]:
            stop(database, job, False, "cancelled")
        elif arguments.action == "cancel" and job["state"] == "waiting":
            database.execute("UPDATE jobs SET state='cancelled', ended=? WHERE name=?", (time.time(), arguments.name))
        elif arguments.action == "retry" and job["state"] in FINISHED:
            database.execute(
                "UPDATE jobs SET state='waiting', tries=0, not_before=0, code=NULL, why=NULL WHERE name=?",
                (arguments.name,),
            )
            serving()
        elif arguments.action == "remove" and job["state"] in FINISHED:
            database.execute("DELETE FROM jobs WHERE name=?", (arguments.name,))
        else:
            sys.exit(
                f"cannot {arguments.action} {arguments.name}: it is {job['state']}{', adopted' if job['adopted'] else ''}"
            )


if __name__ == "__main__":
    main()
