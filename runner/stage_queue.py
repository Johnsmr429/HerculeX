"""What every stage shares: method digests, atomic outputs, unit claims and stage locks."""

import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

RECORD = "record.json"


def method_of(*sources):
    """Digest of the code that decides a stage's output."""
    digest = hashlib.sha256()
    for source in sources:
        digest.update(Path(source).read_bytes())
    return digest.hexdigest()[:12]


def done(folder, method=None, same_output=()):
    """A unit is done when its record, written last, exists and names this method, or one proven to make the same output."""
    folder = Path(folder)
    path = folder / RECORD
    if folder.name.endswith(".building") or not path.exists():
        return False
    record = json.loads(path.read_text())
    return method is None or record.get("method") in (method, *same_output)


def record_of(folder):
    return json.loads((Path(folder) / RECORD).read_text())


@contextmanager
def building(final):
    """Write a unit into a sibling folder and swap it in whole, so a reader or a killed run never sees half of it."""
    final = Path(final)
    part = final.with_name(final.name + ".building")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    yield part
    shutil.rmtree(final, ignore_errors=True)
    part.rename(final)


def finish(folder, record):
    """The record goes last."""
    temporary = Path(folder) / (RECORD + ".part")
    temporary.write_text(json.dumps(record, indent=1))
    temporary.rename(Path(folder) / RECORD)


@contextmanager
def claimed(folder):
    """One worker per unit."""
    folder = Path(folder)
    path = folder.with_name(f".{folder.name}.claim")
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        claim = open(path, "a")
        try:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            claim.close()
            yield False
            return
        if path.exists() and os.fstat(claim.fileno()).st_ino == path.stat().st_ino:
            break
        claim.close()
    try:
        yield True
    finally:
        path.unlink(missing_ok=True)
        claim.close()


def stage_lock(path):
    """Hold for the process's life."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lock = open(path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(f"{Path(path).stem} is already running in {Path(path).parent}") from error
    return lock
