import os
import threading


class TrackQueue:
    """Fila FIFO para tracks de audio con metadatos de BPM"""

    def __init__(self):
        self._lock = threading.Lock()
        self._tracks = []  # list de dicts: {path, name, bpm, duration}
        self._index = 0

    def add(self, path: str, bpm: float | None = None, duration: float = 0.0):
        entry = {
            "path": path,
            "name": os.path.basename(path),
            "bpm": bpm,
            "duration": duration,
        }
        with self._lock:
            self._tracks.append(entry)

    def remove(self, idx: int):
        with self._lock:
            if 0 <= idx < len(self._tracks):
                self._tracks.pop(idx)
                if idx < self._index:
                    self._index = max(0, self._index - 1)
                elif self._index >= len(self._tracks):
                    self._index = max(0, len(self._tracks) - 1)

    def move_up(self, idx: int):
        with self._lock:
            if 0 < idx < len(self._tracks):
                self._tracks[idx - 1], self._tracks[idx] = (
                    self._tracks[idx],
                    self._tracks[idx - 1],
                )
                if self._index == idx:
                    self._index = idx - 1
                elif self._index == idx - 1:
                    self._index = idx

    def move_down(self, idx: int):
        with self._lock:
            if 0 <= idx < len(self._tracks) - 1:
                self._tracks[idx], self._tracks[idx + 1] = (
                    self._tracks[idx + 1],
                    self._tracks[idx],
                )
                if self._index == idx:
                    self._index = idx + 1
                elif self._index == idx + 1:
                    self._index = idx

    def set_bpm(self, idx: int, bpm: float):
        with self._lock:
            if 0 <= idx < len(self._tracks):
                self._tracks[idx]["bpm"] = bpm

    def get_current(self) -> dict | None:
        with self._lock:
            if self._tracks and 0 <= self._index < len(self._tracks):
                return dict(self._tracks[self._index])
            return None

    def next_track(self) -> dict | None:
        with self._lock:
            if not self._tracks:
                return None
            self._index = (self._index + 1) % len(self._tracks)
            return dict(self._tracks[self._index])

    def prev_track(self) -> dict | None:
        with self._lock:
            if not self._tracks:
                return None
            self._index = (self._index - 1) % len(self._tracks)
            return dict(self._tracks[self._index])

    def select(self, idx: int) -> dict | None:
        with self._lock:
            if 0 <= idx < len(self._tracks):
                self._index = idx
                return dict(self._tracks[idx])
            return None

    def snapshot(self) -> tuple[list, int]:
        with self._lock:
            return [dict(t) for t in self._tracks], self._index

    def __len__(self):
        with self._lock:
            return len(self._tracks)
