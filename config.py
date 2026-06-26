"""
Phantom Conductor — Configuration
===================================
Single source of truth for all user-adjustable settings.
Persisted to config.json next to the script.

All other modules should import from here rather than hard-coding values.
The UI's Settings panel reads and writes this object directly.
"""

import json
import os
import threading

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "phantom_config.json")

_DEFAULTS: dict = {
    # ── Audio I/O ─────────────────────────────────────────────────────────────
    "dev_in": None,  # sounddevice device index (None = system default)
    "dev_out": None,
    "input_gain": 1.0,  # mic pre-gain applied before ring buffer
    "output_gain": 0.85,  # backing-track output gain
    # ── Video ─────────────────────────────────────────────────────────────────
    "cam_index": 0,  # OpenCV camera index
    # ── Gesture / hand command mapper ─────────────────────────────────────────
    # Maps gesture name → transport command
    "gesture_map": {
        "PLAY": "play",
        "PAUSE": "pause",
    },
    "gesture_hold_frames": 8,  # frames a gesture must be held before firing
    # ── Inference ─────────────────────────────────────────────────────────────
    "inference_skip_enabled": False,
    "inference_skip_frames": 2,
    # ── Pedal ─────────────────────────────────────────────────────────────────
    "use_pedal": False,
    "pedal_key": "space",  # keyboard key that simulates pedal press
    # ── Tempo tapper ──────────────────────────────────────────────────────────
    "use_tempo_tapper": False,
    "tap_key": "t",  # keyboard key for tap-tempo
    # ── BPM analysis ──────────────────────────────────────────────────────────
    "smooth_alpha": 0.3,
    "min_bpm": 60,
    "max_bpm": 200,
    "analyze_every": 0.25,
    # seconds between analysis passes
    "bpm_median_window": 6,  # increase for stability, decrease for faster response
    "rms_threshold": 0.025,  # raise if detecting in silence, lower if missing quiet playing
    "timing_model_path": "./models/timing_aligner_v1.pt",
    "timing_seq_len": 32,
    # Speed controller
    "speed_k_p": 2.0,
    "speed_k_i": 0.1,
    "speed_k_d": 0.5,
    "max_playback_speed": 1.15,
    "min_playback_speed": 0.85,
    "speed_smoothing_alpha": 0.15,
    "delta_median_window": 8,
    "use_neural_timing": False,  # Toggle between BPM and neural timing
}


class Config:
    """
    Thread-safe configuration container.

    Usage
    -----
    from config import CFG
    CFG.output_gain          # read
    CFG.set("output_gain", 0.9)   # write + auto-save
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict = dict(_DEFAULTS)
        self.load()

    # ── Persistence ───────────────────────────────────────────────────────────
    def load(self):
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            with self._lock:
                # Merge: saved values win, but new defaults are preserved
                merged = dict(_DEFAULTS)
                merged.update(saved)
                self._data = merged
        except Exception as e:
            print(f"[config] load failed: {e}")

    def save(self):
        """Persist current settings to JSON."""
        try:
            with self._lock:
                snapshot = dict(self._data)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            print(f"[config] save failed: {e}")

    # ── Attribute-style access ─────────────────────────────────────────────────
    def __getattr__(self, name: str):
        # Only called when normal attribute lookup fails
        with object.__getattribute__(self, "_lock"):
            data = object.__getattribute__(self, "_data")
            if name in data:
                return data[name]
        raise AttributeError(f"Config has no field '{name}'")

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value, autosave: bool = True):
        with self._lock:
            self._data[key] = value
        if autosave:
            self.save()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def reset_to_defaults(self):
        with self._lock:
            self._data = dict(_DEFAULTS)
        self.save()


# Module-level singleton — import this everywhere
CFG = Config()
