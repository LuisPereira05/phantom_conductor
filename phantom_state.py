# phantom_state.py — Shared state and utilities (no circular dependencies)

import threading
import time
import os
from collections import deque, Counter
from queue import Queue, Empty
from scipy.signal import butter, lfilter

try:
    from mutagen.mp3 import MP3
    from mutagen import File as MutagenFile
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

try:
    import pyrubberband as pyrb
    HAS_PYRB = True
except ImportError:
    HAS_PYRB = False

import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG GLOBAL (moved here to break circular dependency)
# ═══════════════════════════════════════════════════════════════════════════════
SR             = 44100          # sample rate — 44.1 kHz has wider device support
BUFFER_SEC     = 10
HOP_STREAM_SEC = 0.05
ANALYZE_EVERY  = 0.25
HOP_LENGTH     = 256
MIN_BPM        = 70
MAX_BPM        = 190
SMOOTH_ALPHA   = 0.3
BANDPASS       = (40, 3000)

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Landmark indices
WRIST                         = 0
THUMB_IP,    THUMB_TIP        = 3,  4
INDEX_PIP,   INDEX_TIP        = 6,  8
MIDDLE_MCP,  MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP,    RING_TIP         = 14, 16
PINKY_PIP,   PINKY_TIP        = 18, 20

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  TRACK QUEUE (moved here - from Citation 1)
# ═══════════════════════════════════════════════════════════════════════════════

class TrackQueue:
    """Thread-safe FIFO queue for audio tracks with BPM metadata."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._tracks = []           # list of dicts: {path, name, bpm, duration}
        self._index  = 0            # currently loaded track index (-1 = none)

    def add(self, path: str, bpm: float | None = None, duration: float = 0.0):
        name = os.path.basename(path)
        entry = {"path": path, "name": name, "bpm": bpm, "duration": duration}
        with self._lock:
            self._tracks.append(entry)

    def remove(self, idx: int):
        with self._lock:
            if 0 <= idx < len(self._tracks):
                removed_before_current = idx < self._index
                self._tracks.pop(idx)
                if removed_before_current:
                    self._index = max(0, self._index - 1)
                elif self._index >= len(self._tracks):
                    self._index = max(0, len(self._tracks) - 1)

    def get_item(self, idx: int):
        """Get track by index (thread-safe)."""
        with self._lock:
            if 0 <= idx < len(self._tracks):
                return self._tracks[idx]
            return None

    def __len__(self):
        with self._lock:
            return len(self._tracks)

    def items(self):
        """Return iterator over tracks (thread-safe)."""
        with self._lock:
            return iter(self._tracks.copy())

    def update_item(self, path: str, bpm: float = None, duration: float = None):
        """Update existing track's BPM/duration."""
        with self._lock:
            for entry in self._tracks:
                if entry["path"] == path:
                    if bpm is not None:
                        entry["bpm"] = bpm
                    if duration is not None:
                        entry["duration"] = duration
                    return True
            return False

    def clear(self):
        with self._lock:
            self._tracks.clear()
            self._index = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE CLASS (moved here)
# ═══════════════════════════════════════════════════════════════════════════════

class PhantomState:
    """Central application state."""

    def __init__(self):
        self._lock = threading.Lock()
        
        # Core data structures
        self.queue = TrackQueue()
        self.audio_buffer = deque(maxlen=int(SR * BUFFER_SEC))  # ring buffer for BPM analysis
        self.audio_queue = Queue(maxsize=60)                    # blocks ready for playback
        
        # Playback state
        self._playing = False
        self._paused = False
        self._position = 0.0  # seconds
        self._bpm_live = None
        self._bpm_original = 120.0  # default from Citation 2
        
        # Gesture state
        self._gesture_active = False
        self._last_gesture = "NONE"
        
        # Load state (from Citation 2)
        self._load_new_track = None

    def alive(self):
        """Check if application is running."""
        return not (self._playing and not self._paused) or len(self.queue) > 0

    def playing(self):
        """Is currently playing?"""
        with self._lock:
            return self._playing and not self._paused

    def set_playing(self, value: bool):
        with self._lock:
            self._playing = value
            if value:
                self._paused = False

    def pause(self):
        with self._lock:
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False

    def get_bpm(self) -> float | None:
        with self._lock:
            return self._bpm_live

    def set_bpm_live(self, bpm: float):
        with self._lock:
            self._bpm_live = bpm

    def set_position(self, pos: float):
        with self._lock:
            self._position = max(0.0, pos)

    def get_position(self) -> float:
        with self._lock:
            return self._position

    def set_load_new_track(self, track_info: dict | None):
        with self._lock:
            self._load_new_track = track_info

    def get_load_new_track(self) -> dict | None:
        with self._lock:
            return self._load_new_track.copy() if self._load_new_track else None


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGER CLASS (moved here)
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    """Thread-safe logger with colored output."""

    def __init__(self):
        self._lock = threading.Lock()
        self._history = []

    def ok(self, msg: str):
        with self._lock:
            print(f"\033[92m✓ {msg}\033[0m")
            self._history.append(("ok", msg))

    def err(self, msg: str):
        with self._lock:
            print(f"\033[91m✗ {msg}\033[0m")
            self._history.append(("err", msg))

    def warn(self, msg: str):
        with self._lock:
            print(f"\033[93m⚠ {msg}\033[0m")
            self._history.append(("warn", msg))

    def info(self, msg: str):
        with self._lock:
            print(f"\033[94mℹ {msg}\033[0m")
            self._history.append(("info", msg))

    def debug(self, msg: str):
        with self._lock:
            print(f"\033[96m🔍 {msg}\033[0m")
            self._history.append(("debug", msg))


# ═══════════════════════════════════════════════════════════════════════════════
#  INITIALIZATION (moved here)
# ═══════════════════════════════════════════════════════════════════════════════

def init_model():
    """Initialize hand landmarking model."""
    global MODEL_PATH
    
    if not os.path.exists(MODEL_PATH):
        LOG.info(f"Downloading hand landmarking model...")
        try:
            urllib.request.urlretrieve(
                MODEL_URL,
                MODEL_PATH
            )
            LOG.ok(f"Model downloaded to {MODEL_PATH}")
        except Exception as e:
            LOG.warn(f"Could not download model: {e}. Using simulated data.")

# Initialize shared state at module level (safe for imports)
STATE = PhantomState()
LOG = Logger()


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTED NAMES
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'PhantomState',
    'Logger', 
    'TrackQueue',
    'STATE',
    'LOG',
    'SR',
    'BUFFER_SEC',
    'MIN_BPM',
    'MAX_BPM',
    'HAS_MUTAGEN',
    'HAS_PYRB',
]
