"""This Linux machine as the job board sees it, from /proc and nvidia-smi, in the Apple lane's terms: a process's
counters sit at the places of the Mac's rusage words, so the board reads both lanes alike."""

import contextlib
import os
import re
import shutil
import subprocess
from pathlib import Path

USER, SYSTEM, READ, WRITTEN, START, FOOTPRINT, LIFETIME_PEAK = 2, 3, 18, 19, 10, 9, 30
TICK_NS = 1e9 / os.sysconf("SC_CLK_TCK")
GPU_QUERY = ("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")


def text_of(*command):
    return subprocess.run(command, capture_output=True, text=True, check=False).stdout


def background(command):
    """A command run at the lowest CPU priority."""
    return ["nice", "-n", "19", *command]


def gpu_use():
    """The busiest GPU's use in percent, 0 without nvidia-smi."""
    found = re.findall(r"\d+", text_of(*GPU_QUERY) if shutil.which(GPU_QUERY[0]) else "")
    return max(map(int, found), default=0)


def machine():
    """Memory no process holds, swap in use and written so far, in GB, the GPU's use in percent, and the load."""
    memory = {
        key: int(value.split()[0]) * 1024
        for key, value in (line.split(":") for line in Path("/proc/meminfo").read_text().splitlines())
    }
    swapped = int(re.search(r"^pswpout (\d+)", Path("/proc/vmstat").read_text(), re.MULTILINE).group(1))
    device = gpu_use()
    return {
        "room_gb": memory["MemAvailable"] / 1e9,
        "swap_gb": (memory["SwapTotal"] - memory["SwapFree"]) / 1e9,
        "swapped_out_gb": swapped * os.sysconf("SC_PAGE_SIZE") / 1e9,
        "gpu": device,
        "gpu_device": device,
        "load": os.getloadavg()[0],
    }


def cpu_ticks():
    """CPU ticks spent busy and idle on every core since boot."""
    user, nice, system, idle = (int(v) for v in Path("/proc/stat").read_text().split("\n")[0].split()[1:5])
    return user + system + nice, idle


def received_bytes():
    """Bytes received since boot on every network interface but the loopback."""
    total = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, _, counters = line.partition(":")
        if not name.strip().startswith("lo"):
            total += int(counters.split()[0])
    return total


def disc_megabytes():
    """Disc counters by role; none are read on this lane."""
    return {}


def stat_of(pid):
    """The fields of /proc/PID/stat that follow the command name (state, parent, group, ...)."""
    return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()


def usage_of(pid):
    """A process's counters from /proc, at the places of the Mac's rusage words, or None once it has ended."""
    try:
        stat = stat_of(pid)
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    if stat[0] == "Z":
        return None
    words = [0] * 42
    words[START] = int(stat[19])
    words[USER], words[SYSTEM] = int(stat[11]), int(stat[12])
    for index, key in ((FOOTPRINT, "VmRSS"), (LIFETIME_PEAK, "VmHWM")):
        found = re.search(rf"^{key}:\s+(\d+) kB", status, re.MULTILINE)
        words[index] = int(found.group(1)) * 1024 if found else 0
    with contextlib.suppress(OSError):
        counters = dict(line.split(": ") for line in Path(f"/proc/{pid}/io").read_text().splitlines())
        words[READ], words[WRITTEN] = int(counters["read_bytes"]), int(counters["write_bytes"])
    return words


def group_of(pid):
    """The processes of a process group."""
    found = []
    for entry in os.listdir("/proc"):
        with contextlib.suppress(OSError):
            if entry.isdecimal() and int(stat_of(entry)[2]) == pid:
                found.append(int(entry))
    return found
