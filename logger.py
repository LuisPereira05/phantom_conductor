"""
Phantom Conductor — Logger
==========================
Thread-safe ring-buffer logger with severity levels.
All pipeline components import this; the UI drains it each frame.
"""

import threading
import time
import collections


class Logger:
    def __init__(self, maxlen: int = 200):
        self._lock  = threading.Lock()
        self._lines = collections.deque(maxlen=maxlen)
        self._start = time.time()

    def log(self, msg: str, level: str = "info"):
        t  = time.time() - self._start
        ts = f"{int(t//60):02d}:{t%60:05.2f}"
        with self._lock:
            self._lines.appendleft((ts, level, msg))

    def info(self, msg): self.log(msg, "info")
    def ok(self,   msg): self.log(msg, "ok")
    def warn(self, msg): self.log(msg, "warn")
    def err(self,  msg): self.log(msg, "err")

    def lines(self, n: int = 80) -> list:
        with self._lock: return list(self._lines)[:n]

    def clear(self):
        with self._lock: self._lines.clear()
