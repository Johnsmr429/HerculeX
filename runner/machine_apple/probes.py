"""This Mac as the job board sees it: memory from vm_stat, CPU ticks from the host, GPU time per process from the
IOKit registry, network and disc counters, and each process's kernel counters from libproc."""

import collections
import ctypes
import os
import re
import subprocess
import time
from pathlib import Path

from storage.data_roots import LIBRARY, OUTPUTS

LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
LIBSYSTEM = ctypes.CDLL("/usr/lib/libSystem.dylib")
LIBSYSTEM.mach_host_self.restype = ctypes.c_uint
HOST_CPU_LOAD_INFO, CPU_STATES = 3, 4
RUSAGE_V4, USER, SYSTEM, READ, WRITTEN, START, FOOTPRINT, LIFETIME_PEAK = 4, 2, 3, 18, 19, 10, 9, 30
DISPLAY = "WindowServer"


class Timebase(ctypes.Structure):
    _fields_ = (("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32))


_timebase = Timebase()
LIBSYSTEM.mach_timebase_info(ctypes.byref(_timebase))
TICK_NS = _timebase.numer / _timebase.denom
_gpu_seen = {}
_discs = {}


def text_of(*command):
    return subprocess.run(command, capture_output=True, text=True, check=False).stdout


def background(command):
    """A command run at background priority: throttled CPU, disc and network."""
    return ["taskpolicy", "-b", *command]


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
    """Memory no process holds, swap in use and written so far, in GB, the GPU's use in percent, and the load."""
    stats = text_of("vm_stat")
    page = int(re.search(r"page size of (\d+)", stats).group(1))
    pages = {name: int(value) for name, value in re.findall(r'^"?(.+?)"?:\s+(\d+)\.$', stats, re.MULTILINE)}
    held = pages["Pages wired down"] + pages["Anonymous pages"] + pages["Pages occupied by compressor"]
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    swap = float(re.search(r"used = ([0-9.]+)M", text_of("sysctl", "-n", "vm.swapusage")).group(1)) * 2**20 / 1e9
    found = re.search(r'"Device Utilization %"=(\d+)', text_of("ioreg", "-r", "-c", "IOAccelerator", "-d", "1"))
    device = int(found.group(1)) if found else 0
    return {
        "room_gb": (total - held * page) / 1e9,
        "swap_gb": swap,
        "swapped_out_gb": pages["Swapouts"] * page / 1e9,
        "gpu": gpu_of_others(device),
        "gpu_device": device,
        "load": os.getloadavg()[0],
    }


def cpu_ticks():
    """CPU ticks spent busy and idle on every core since boot."""
    ticks, count = (ctypes.c_uint * CPU_STATES)(), ctypes.c_uint(CPU_STATES)
    LIBSYSTEM.host_statistics(LIBSYSTEM.mach_host_self(), HOST_CPU_LOAD_INFO, ticks, ctypes.byref(count))
    user, system, idle, nice = ticks
    return user + system + nice, idle


def received_bytes():
    """Bytes received since boot on every network interface but the loopback."""
    total = 0
    for line in text_of("netstat", "-ibn").splitlines()[1:]:
        words = line.split()
        if len(words) >= 10 and words[2].startswith("<Link#") and not words[0].startswith("lo"):
            total += int(words[-5])
    return total


def disc_of(path):
    """The physical disc under a folder's volume, such as disk4, or None."""
    if path not in _discs:
        lines = text_of("df", "-P", str(path)).splitlines()
        info = text_of("diskutil", "info", lines[1].split()[0]) if len(lines) > 1 else ""
        found = re.search(r"APFS Physical Store:\s+(disk\d+)", info) or re.search(r"Part of Whole:\s+(disk\d+)", info)
        _discs[path] = found.group(1) if found else None
    return _discs[path]


def disc_megabytes():
    """Megabytes read and written since boot on the discs of the Mac, the library and the outputs."""
    lines = text_of("iostat", "-Id").splitlines()
    if len(lines) < 3:
        return {}
    names, values = lines[0].split(), lines[2].split()
    moved = {name: float(values[3 * i + 2]) for i, name in enumerate(names) if 3 * i + 2 < len(values)}
    folders = {"internal": Path("/"), "library": LIBRARY, "outputs": OUTPUTS}
    return {role: moved.get(disc_of(folder)) for role, folder in folders.items() if folder.exists()}


def usage_of(pid):
    """A process's kernel counters, read without touching it."""
    words = (ctypes.c_uint64 * 42)()
    return list(words) if LIBPROC.proc_pid_rusage(pid, RUSAGE_V4, ctypes.byref(words)) == 0 else None


def group_of(pid):
    """The processes of a process group."""
    pids = (ctypes.c_int * 1024)()
    return list(pids[: max(LIBPROC.proc_listpgrppids(pid, pids, ctypes.sizeof(pids)), 0)])
