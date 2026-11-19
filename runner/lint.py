"""Every tracked file through the pinned linters of its language: ruff for Python; clang-format for C, C++,
Objective-C, Metal and CUDA, and clang-tidy for C; Biome for JavaScript, CSS and HTML; shellcheck and shfmt for shell.

./run.sh check lint          every linter on every tracked file; exits 1 when one finds anything
./run.sh check lint --fix    the formatters and safe fixes rewrite the files first

The tools are pinned in pyproject.toml's dev group, and Biome here (installed into runtime/node with npm on first use).
Their settings are pyproject.toml, .clang-format, .clang-tidy and biome.json. ports/scrollfiesta keeps its upstream form,
and logs/runs keeps the scripts other machines ran as they ran.
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from runner.lane import APPLE, LANE

HERE = Path(__file__).resolve().parent.parent
TOOLS = Path(sys.executable).parent
BIOME = "2.6.4"
NODE = HERE / "runtime" / "node"
C_FAMILY = ("*.c", "*.h", "*.m", "*.mm", "*.metal", "*.cu", "*.cpp", "*.hpp", "*.hxx", "*.inc")
KEPT = ("ports/scrollfiesta/", "logs/runs/")
SHFMT = ("-i", "4", "-ci")


def tracked(*patterns):
    """Tracked files that match, but those kept as they came."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *patterns], cwd=HERE, capture_output=True, text=True, check=True
    )
    return sorted(f for f in listed.stdout.split("\0") if f and not f.startswith(KEPT) and (HERE / f).is_file())


def biome():
    """The pinned Biome."""
    binary = NODE / "node_modules" / ".bin" / "biome"
    if not binary.exists():
        subprocess.run(["npm", "install", "--prefix", str(NODE), "--save-exact", f"@biomejs/biome@{BIOME}"], check=True)
    return binary


def tidy_flags(path):
    """How clang-tidy compiles a C file: its folder, the lane's threads header, the node kernel's cell, the Mac SDK."""
    threads = HERE / "unwrapping" / "level" / f"threads_{LANE}"
    flags = ["-std=gnu11", f"-I{(HERE / path).parent}", f"-I{threads}", "-DCELL=8", "-DMARGIN=8"]
    if APPLE:
        sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=True).stdout.strip()
        flags += ["-fblocks", "-isysroot", sdk]
    return flags


def checks(keep=lambda path: True):
    """(name, command) of every check on the tracked files `keep` accepts; a check left with no file is left out."""
    python = [f for f in tracked("*.py") if keep(f)]
    c_family = [f for f in tracked(*C_FAMILY) if keep(f)]
    shell = [f for f in tracked("*.sh") if keep(f)]
    found = [
        ("ruff check", [sys.executable, "-m", "ruff", "check", "--no-cache", "--force-exclude"], python),
        ("ruff format", [sys.executable, "-m", "ruff", "format", "--check", "--no-cache", "--force-exclude"], python),
        ("clang-format", [TOOLS / "clang-format", "--dry-run", "--Werror"], c_family),
        ("shellcheck", [TOOLS / "shellcheck"], shell),
        ("shfmt", [TOOLS / "shfmt", *SHFMT, "-d"], shell),
    ]
    tidy = [
        (f"clang-tidy {f}", [TOOLS / "clang-tidy", "--quiet", f, "--", *tidy_flags(f)])
        for f in c_family
        if f.endswith(".c")
    ]
    return [(name, [*command, *files]) for name, command, files in found if files] + tidy


def fix():
    """The formatters and the safe fixes, on every tracked file."""
    c_family, shell = tracked(*C_FAMILY), tracked("*.sh")
    for command in (
        [sys.executable, "-m", "ruff", "check", "--fix", "--no-cache", "--force-exclude", *tracked("*.py")],
        [sys.executable, "-m", "ruff", "format", "--no-cache", "--force-exclude", *tracked("*.py")],
        [TOOLS / "clang-format", "-i", *c_family],
        [biome(), "check", "--write", "."],
        [TOOLS / "shfmt", *SHFMT, "-w", *shell],
    ):
        subprocess.run(command, cwd=HERE, capture_output=True, check=False)


def run(fixing=False):
    """Every check, the failures printed whole; 1 when any fails."""
    if fixing:
        fix()
    every = [*checks(), ("biome", [biome(), "ci", "--colors=off", "."])]

    def one(check):
        name, command = check
        done = subprocess.run(command, cwd=HERE, capture_output=True, text=True, check=False)
        return name, done.returncode, (done.stdout + done.stderr).strip()

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(one, every))
    failed = [(name, output) for name, code, output in results if code]
    for name, output in failed:
        print(f"===== {name} =====\n{output}\n")
    print(f"{len(every)} checks, {len(failed)} failed{': ' + ', '.join(n for n, _ in failed) if failed else ''}")
    return 1 if failed else 0
