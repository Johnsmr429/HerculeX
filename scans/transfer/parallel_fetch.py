"""Many HTTP GETs over kept-alive connections."""

import http.client
import os
import queue
import threading
import time
from urllib.parse import urlsplit

# a background fetch (ahead of a read, a detection) asks for fewer, so a running read keeps most of the link
CONNECTIONS = int(os.environ.get("HERCULEX_FETCH_CONNECTIONS", "32"))
RETRIES = 6


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


def fetch_all(base_url, items, deliver, connections=CONNECTIONS, report=None):
    """deliver(key, bytes or None) for every (key, path) or (key, path, (first byte, last byte))."""
    parts = urlsplit(base_url)
    jobs, failures = queue.Queue(maxsize=connections * 4), []
    connect = http.client.HTTPConnection if parts.scheme == "http" else http.client.HTTPSConnection
    workers = [Fetcher(parts.netloc, jobs, deliver, failures, connect) for _ in range(connections)]
    for worker in workers:
        worker.start()
    for count, (key, path, *span) in enumerate(items):
        jobs.put((key, parts.path.rstrip("/") + "/" + path.lstrip("/"), *span))
        if report and count % 500 == 499:
            report(count + 1)
    for _ in workers:
        jobs.put(None)
    for worker in workers:
        worker.join()
    return failures
