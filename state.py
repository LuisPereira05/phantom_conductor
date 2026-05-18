"""
Phantom Conductor — Shared State & Track Queue
===============================================
Thread-safe data containers shared by all pipeline components.
No I/O, no threads, no GUI — pure data.
"""

import threading
import time
import os


# ═══════════════════════════════════════════════════════════════════════════════
#  TRACK QUEUE
# ═══════════════════════════════════════════════════════════════════════════════

class TrackQueue:
    """Thread-safe FIFO queue for audio tracks with BPM metadata."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._tracks = []   # list of dicts: {path, name, bpm, duration}
        self._index  = 0

    def add(self, path: str, bpm: float | None = None, duration: float = 0.0):
        entry = {"path": path, "name": os.path.basename(path),
                 "bpm": bpm, "duration": duration}
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
                self._tracks[idx-1], self._tracks[idx] = \
                    self._tracks[idx], self._tracks[idx-1]
                if   self._index == idx:     self._index = idx - 1
                elif self._index == idx - 1: self._index = idx

    def move_down(self, idx: int):
        with self._lock:
            if 0 <= idx < len(self._tracks) - 1:
                self._tracks[idx], self._tracks[idx+1] = \
                    self._tracks[idx+1], self._tracks[idx]
                if   self._index == idx:     self._index = idx + 1
                elif self._index == idx + 1: self._index = idx

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
            if not self._tracks: return None
            self._index = (self._index + 1) % len(self._tracks)
            return dict(self._tracks[self._index])

    def prev_track(self) -> dict | None:
        with self._lock:
            if not self._tracks: return None
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


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ═══════════════════════════════════════════════════════════════════════════════

class PhantomState:
    """Thread-safe container — worker threads write, UI reads every frame."""

    def __init__(self):
        self._lock = threading.Lock()

        # BPM
        self.bpm_live: float | None      = None
        self.bpm_original: float         = 120.0
        self.bpm_raw: float | None       = None
        self.bpm_corrected: float | None = None
        self.stretch_ratio: float        = 1.0
        self.bpm_source: str             = "audio"
        self.onset_max: float            = 0.0
        self.last_bpm_dbg: dict          = {}

        # Playback
        self.is_playing: bool            = False
        self.is_looping: bool            = False
        self.track_path: str             = ""
        self.track_duration: float       = 0.0
        self.track_position: float       = 0.0
        self.markers: list[float]        = []

        # Signals to backing_track_thread
        self.load_new_track: dict | None = None
        self.skip_to_next: bool          = False
        self.skip_to_prev: bool          = False

        # I/O — set by UI, consumed by io_manager
        self.dev_in:  int | None         = None
        self.dev_out: int | None         = None
        self.gain: float                 = 0.85
        self.io_restart_requested: bool  = False

        # Audio levels — set by audio_input callback
        self.rms: float                  = 0.0
        self.peak: float                 = 0.0
        self.waveform: list[float]       = [0.0] * 64
        self.buffer_fill: float          = 0.0

        # Gesture
        self.gesture_name: str           = "NO HAND"
        self.gesture_confidence: float   = 0.0
        self.gesture_hold_frames: int    = 0
        self.gesture_hold_target: int    = 8
        self.pending_command: str | None = None
        self.last_command: str | None    = None
        self.hands_detected: int         = 0
        self.camera_active: bool         = False

        # System
        self.running: bool               = True
        self.start_time: float           = time.time()

        # Track queue (thread-safe itself)
        self.queue: TrackQueue           = TrackQueue()

    # ── Playback ──────────────────────────────────────────────────────────────
    def play(self):
        with self._lock: self.is_playing = True

    def pause(self):
        with self._lock: self.is_playing = False

    def toggle(self) -> bool:
        with self._lock:
            self.is_playing = not self.is_playing
            return self.is_playing

    def playing(self) -> bool:
        with self._lock: return self.is_playing

    def set_playback(self, v: bool):
        with self._lock: self.is_playing = v

    # ── BPM ───────────────────────────────────────────────────────────────────
    def get_bpm(self) -> float | None:
        with self._lock: return self.bpm_live

    def set_bpm(self, bpm: float, raw: float | None = None,
                corrected: float | None = None, onset_max: float = 0.0):
        with self._lock:
            self.bpm_live      = bpm
            self.bpm_raw       = raw       if raw       is not None else bpm
            self.bpm_corrected = corrected if corrected is not None else bpm
            self.stretch_ratio = bpm / self.bpm_original if self.bpm_original else 1.0
            self.onset_max     = onset_max

    def set_bpm_original(self, bpm: float):
        """Update the reference BPM (from UI). Recalculates stretch ratio."""
        with self._lock:
            self.bpm_original  = bpm
            self.stretch_ratio = ((self.bpm_live / bpm)
                                  if self.bpm_live and bpm else 1.0)

    # ── Gesture ───────────────────────────────────────────────────────────────
    def set_gesture(self, name: str, confidence: float = 0.0,
                    hold_frames: int = 0, hands: int = 1):
        with self._lock:
            self.gesture_name        = name
            self.gesture_confidence  = confidence
            self.gesture_hold_frames = hold_frames
            self.hands_detected      = hands
            self.camera_active       = (hands > 0)

    def get_gesture(self) -> str:
        with self._lock: return self.gesture_name

    def set_command(self, cmd: str | None):
        with self._lock:
            self.last_command    = cmd
            self.pending_command = cmd

    # ── Audio ─────────────────────────────────────────────────────────────────
    def push_waveform(self, rms_val: float):
        with self._lock:
            v = min(1.0, float(rms_val))
            self.waveform.append(v)
            if len(self.waveform) > 64:
                self.waveform.pop(0)
            self.rms  = v
            self.peak = max(self.peak * 0.98, v)

    def set_position(self, pos: float):
        with self._lock: self.track_position = pos

    # ── I/O ───────────────────────────────────────────────────────────────────
    def request_io_restart(self, dev_in, dev_out):
        with self._lock:
            self.dev_in               = dev_in
            self.dev_out              = dev_out
            self.io_restart_requested = True

    def consume_io_restart(self):
        """Called by io_manager; returns (dev_in, dev_out) and clears flag."""
        with self._lock:
            self.io_restart_requested = False
            return self.dev_in, self.dev_out

    # ── System ────────────────────────────────────────────────────────────────
    def stop(self):
        with self._lock: self.running = False

    def alive(self) -> bool:
        with self._lock: return self.running

    def snapshot(self) -> dict:
        with self._lock:
            d = self.__dict__.copy()
            d.pop("_lock", None)
            d.pop("queue", None)
            return d
