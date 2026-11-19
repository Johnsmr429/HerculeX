"""Many HTTP GETs over kept-alive connections."""

import http.client
import os
import queue
import threading
import time
from urllib.parse import urlsplit

import numpy as np

# a background fetch (ahead of a read, a detection) asks for fewer, so a running read keeps most of the link
CONNECTIONS = int(os.environ.get("HERCULEX_FETCH_CONNECTIONS", "32"))
RETRIES = 6
# curl: brick lists go through the libcurl-multi client (scans/chunks/fetch_urls.c), 25 vs 19 MB/s at half the CPU on
# PHerc0826 level 1 at 32 connections; python: always the threads below
FETCHER = os.environ.get("HERCULEX_FETCHER", "curl")
REPORT_EVERY = 500


class Fetcher(threading.Thread):
    def __init__(self, host, jobs, deliver, failures, connect=http.client.HTTPSConnection):
        super().__init__(daemon=True)
        self.host, self.jobs, self.deliver, self.failures, self.connect = host, jobs, deliver, failures, connect
        self.connection = None

    def get(self, path, span=None):
        headers = {"Range": f"bytes={span[0]}-{span[1]}"} if span else {}
        for attempt in range(RETRIES):
            try:
                if self.connection is None:
                    self.connection = self.connect(self.host, timeout=60)
                self.connection.request("GET", path, headers=headers)
                response = self.connection.getresponse()
                body = response.read()
                if response.status in (200, 206):
                    return body
                if response.status in (403, 404):
                    return None
                raise OSError(f"HTTP {response.status}")
            except (OSError, http.client.HTTPException):
                if self.connection is not None:
                    self.connection.close()
                self.connection = None
                time.sleep(min(30, 2**attempt))
        raise OSError(f"giving up on {path}")

    def run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            key, path, *span = job
            try:
                self.deliver(key, self.get(path, *span))
            except OSError as error:
                self.failures.append((key, str(error)))


def bricks_of(items):
    """(M, 3) keys and (M, 2) ranges or None when every item is a brick at its z/y/x path, all with a byte range or all
    without; else None, for the Python client."""
    keys, spans = [], []
    for key, path, *span in items:
        if not (isinstance(key, tuple) and len(key) == 3 and path == "/".join(map(str, key))):
            return None
        keys.append(key)
        spans.append(span[0] if span else None)
    if any(s is None for s in spans) and not all(s is None for s in spans):
        return None
    return np.array(keys, dtype=np.int64).reshape(-1, 3), None if spans and spans[0] is None else np.array(spans)


def fetch_bricks_native(base_url, keys, ranges, deliver, connections, report):
    """The same contract as fetch_all, through the libcurl-multi client."""
    from scans.chunks.fetch_urls import UrlFetch

    failures, done, answered = [], 0, np.zeros(len(keys), dtype=bool)
    with UrlFetch(base_url.rstrip("/"), keys, separator="/", connections=connections, ranges=ranges) as fetch:
        for bodies, buffer in fetch.batches():
            for body in bodies:
                answered[body["index"]] = True
                key = tuple(int(v) for v in keys[body["index"]])
                try:
                    if body["status"] == 200:
                        deliver(key, buffer[body["offset"] : body["offset"] + body["length"]].tobytes())
                    elif body["status"] in (403, 404):
                        deliver(key, None)
                    else:
                        failures.append((key, f"status {int(body['status'])} after {int(body['attempts'])} tries"))
                except OSError as error:
                    failures.append((key, str(error)))
                done += 1
                if report and done % REPORT_EVERY == 0:
                    report(done)
        cause = "the fetch gave up after failing in a row" if fetch.stats()["aborted"] else "no answer"
    # a fetch that gives up drops the keys it never reached: they failed, they are not absent
    failures += [(tuple(int(v) for v in keys[i]), cause) for i in np.flatnonzero(~answered)]
    return failures


def fetch_all(base_url, items, deliver, connections=CONNECTIONS, report=None):
    """deliver(key, bytes or None) for every (key, path) or (key, path, (first byte, last byte))."""
    items = list(items)
    bricks = bricks_of(items) if FETCHER == "curl" and items else None
    if bricks is not None:
        return fetch_bricks_native(base_url, *bricks, deliver, connections, report)
    parts = urlsplit(base_url)
    jobs, failures = queue.Queue(maxsize=connections * 4), []
    connect = http.client.HTTPConnection if parts.scheme == "http" else http.client.HTTPSConnection
    workers = [Fetcher(parts.netloc, jobs, deliver, failures, connect) for _ in range(connections)]
    for worker in workers:
        worker.start()
    for count, (key, path, *span) in enumerate(items):
        jobs.put((key, parts.path.rstrip("/") + "/" + path.lstrip("/"), *span))
        if report and count % REPORT_EVERY == REPORT_EVERY - 1:
            report(count + 1)
    for _ in workers:
        jobs.put(None)
    for worker in workers:
        worker.join()
    return failures
