"""
Phantom Conductor — Persistent Track List
==========================================
Wraps TrackQueue with automatic JSON persistence.

File layout (tracklist.json, next to the script):
[
  {"name": "song.mp3", "path": "music/song.mp3", "bpm": 128.0, "duration": 214.5},
  ...
]

Paths are stored relative to the directory that contains tracklist.json
so the project stays portable when moved.

Public API
----------
    from tracklist import PersistentQueue
    q = PersistentQueue()          # loads saved list automatically
    q.add("/abs/path/to/song.mp3", bpm=128.0, duration=214.5)
    q.remove(idx)
    q.snapshot()   # → (list_of_dicts, current_index)
    q.load_state() # reload from disk (e.g. after external edit)

All TrackQueue methods still work; PersistentQueue sub-classes it.
"""

import json
import os
import threading

from track_queue import TrackQueue

LIST_PATH = os.path.join(os.path.dirname(__file__), "tracklist.json")


def _to_rel(path: str) -> str:
    """Convert an absolute path to one relative to the script directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        return os.path.relpath(path, base)
    except ValueError:
        # On Windows, relpath fails across drives — keep absolute
        return path


def _to_abs(rel: str) -> str:
    """Resolve a (possibly relative) stored path to an absolute path."""
    if os.path.isabs(rel):
        return rel
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base, rel))


class PersistentQueue(TrackQueue):
    """
    TrackQueue with automatic JSON persistence.

    Every mutating operation (add / remove / move / set_bpm) calls
    _save() immediately so the file stays in sync.
    """

    def __init__(self):
        super().__init__()
        self._save_lock = threading.Lock()
        self._load_state()

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _save(self):
        """Persist the current track list to tracklist.json."""
        with self._lock:
            tracks = list(self._tracks)
        serializable = []
        for t in tracks:
            serializable.append({
                "name":     t.get("name", os.path.basename(t["path"])),
                "path":     _to_rel(t["path"]),
                "bpm":      t.get("bpm"),
                "duration": t.get("duration", 0.0),
            })
        try:
            with self._save_lock:
                with open(LIST_PATH, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"[tracklist] save failed: {e}")

    def _load_state(self):
        """Load tracklist.json into the queue on startup."""
        if not os.path.exists(LIST_PATH):
            return
        try:
            with open(LIST_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except Exception as e:
            print(f"[tracklist] load failed: {e}")
            return

        loaded = 0
        for entry in saved:
            rel  = entry.get("path", "")
            abs_ = _to_abs(rel)
            if not os.path.isfile(abs_):
                print(f"[tracklist] skipping missing file: {abs_}")
                continue
            # Use parent-class add to avoid recursive _save during init
            record = {
                "path":     abs_,
                "name":     entry.get("name") or os.path.basename(abs_),
                "bpm":      entry.get("bpm"),
                "duration": entry.get("duration", 0.0),
            }
            with self._lock:
                self._tracks.append(record)
            loaded += 1
        print(f"[tracklist] loaded {loaded} track(s) from {LIST_PATH}")

    # ── Overrides that trigger _save ──────────────────────────────────────────

    def add(self, path: str, bpm: float | None = None, duration: float = 0.0):
        super().add(path, bpm=bpm, duration=duration)
        self._save()

    def remove(self, idx: int):
        super().remove(idx)
        self._save()

    def move_up(self, idx: int):
        super().move_up(idx)
        self._save()

    def move_down(self, idx: int):
        super().move_down(idx)
        self._save()

    def set_bpm(self, idx: int, bpm: float):
        super().set_bpm(idx, bpm)
        self._save()

    def clear(self):
        """Remove all tracks and persist."""
        with self._lock:
            self._tracks.clear()
            self._index = 0
        self._save()