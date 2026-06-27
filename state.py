"""
Phantom Conductor — Shared State & Track Queue
===============================================
Thread-safe data containers shared by all pipeline components.
No I/O, no threads, no GUI — pure data.

Changes from v0.5.1
--------------------
* Added loop-section fields: loop_section_index, loop_sections (derived
  from sorted markers), and helpers get_loop_section / step_loop_section.
* Added dispatch_command() — single place that maps command strings
  ("play", "pause", "next", "prev", "loop_toggle", "loop_next",
  "loop_prev") to state mutations.  gesture_recognition calls this
  instead of calling play()/pause() directly.
* skip_to_next / skip_to_prev flags honoured by backing_track_thread.

Changes from v0.5.2 (tempo tapper)
-----------------------------------
* Added a tap-override window: apply_tap_bpm() / apply_audio_bpm() let
  manual foot-taps "win" over the smoothed audio BPM for a short
  period, instead of having both writers fight over set_bpm().
  set_bpm() itself is untouched so existing callers keep working.
* Added tap_connected for the Settings/HUD panel to show serial status.

Changes from v0.5.3 (tempo tapper gating fix)
-----------------------------------------------
* Added last_bpm_analysis_dbg — audio_analysis_thread now keeps this
  updated every pass regardless of whether CFG.use_tempo_tapper gates
  the actual write, so HUD/debug views don't go stale while tapper
  mode is active.
* apply_tap_bpm / apply_audio_bpm behaviour unchanged; the *live*
  CFG.use_tempo_tapper check now lives in audio_analysis.py and
  tempo_tapper.py themselves (checked every loop pass in both), so
  the override window here is a secondary safety net rather than the
  only thing keeping the two sources from fighting.
"""

import collections
import threading
import time

from config import CFG
from tracklist import PersistentQueue

# How long a manual tap takes priority over audio-derived BPM
TAP_OVERRIDE_WINDOW_S = 4.0


class PhantomState:
    """Thread-safe container — worker threads write, UI reads every frame."""

    def __init__(self):
        self._lock = threading.Lock()

        # BPM
        self.bpm_live: float | None = None
        self.bpm_original: float = 120.0
        self.bpm_raw: float | None = None
        self.bpm_corrected: float | None = None
        self.stretch_ratio: float = 1.0
        self.bpm_source: str = "audio"
        self.onset_max: float = 0.0
        self.last_bpm_dbg: dict = {}
        self.last_bpm_analysis_dbg: dict = {}  # always fresh, even when
        # tapper mode gates the write
        self._tap_override_until: float = 0.0

        # Playback
        self.is_playing: bool = False
        self.is_looping: bool = False
        self.track_path: str = ""
        self.track_duration: float = 0.0
        self.track_position: float = 0.0

        # Markers & loop sections
        # markers: raw list of timestamps (seconds) added by user
        # loop_section_index: which gap between markers is active (-1 = whole track)
        self.markers: list[float] = []
        self.loop_section_index: int = -1

        # Signals to backing_track_thread
        self.load_new_track: dict | None = None
        self.skip_to_next: bool = False
        self.skip_to_prev: bool = False

        # I/O
        self.dev_in: int | None = CFG.dev_in
        self.dev_out: int | None = CFG.dev_out
        self.gain: float = CFG.output_gain
        self.input_gain: float = CFG.input_gain
        self.io_restart_requested: bool = False
        self.latest_frame = None

        # Audio levels
        self.rms: float = 0.0
        self.peak: float = 0.0
        self.waveform: list[float] = [0.0] * 64
        self.buffer_fill: float = 0.0

        # Gesture
        self.gesture_name: str = "NO HAND"
        self.gesture_confidence: float = 0.0
        self.gesture_hold_frames: int = 0
        self.gesture_hold_target: int = CFG.gesture_hold_frames
        self.pending_command: str | None = None
        self.last_command: str | None = None
        self.hands_detected: int = 0
        self.camera_active: bool = False

        # Tempo tapper (serial)
        self.tap_connected: bool = False

        # Beat timestamps fed to the PLL by OnsetNet (or any other source)
        self.recent_beat_times: collections.deque = collections.deque(maxlen=32)
        self.onset_net_available: bool = False

        # System
        self.running: bool = True
        self.start_time: float = time.time()

        # Persistent track queue
        self.queue: PersistentQueue = PersistentQueue()

    # ── Playback ──────────────────────────────────────────────────────────────
    def play(self):
        with self._lock:
            self.is_playing = True

    def pause(self):
        with self._lock:
            self.is_playing = False

    def toggle(self) -> bool:
        with self._lock:
            self.is_playing = not self.is_playing
            return self.is_playing

    def playing(self) -> bool:
        with self._lock:
            return self.is_playing

    def set_playback(self, v: bool):
        with self._lock:
            self.is_playing = v

    # ── BPM ───────────────────────────────────────────────────────────────────
    def get_bpm(self) -> float | None:
        with self._lock:
            return self.bpm_live

    def set_bpm(
        self,
        bpm: float,
        raw: float | None = None,
        corrected: float | None = None,
        onset_max: float = 0.0,
    ):
        """
        Unconditional setter — unchanged behaviour for existing callers
        (gesture HUD, manual UI overrides, etc.). Does NOT touch
        bpm_source or the tap-override window; the two competing
        writer threads should use apply_tap_bpm() / apply_audio_bpm()
        instead so they don't stomp on each other.
        """
        with self._lock:
            self.bpm_live = bpm
            self.bpm_raw = raw if raw is not None else bpm
            self.bpm_corrected = corrected if corrected is not None else bpm
            self.stretch_ratio = bpm / self.bpm_original if self.bpm_original else 1.0
            self.onset_max = onset_max

    def apply_tap_bpm(self, bpm: float):
        """
        Called by tempo_tapper_thread when a valid BPM: line arrives.
        Always wins immediately, and opens a window during which
        apply_audio_bpm() will refuse to overwrite it.
        """
        with self._lock:
            self.bpm_live = bpm
            self.bpm_raw = bpm
            self.bpm_corrected = bpm
            self.bpm_source = "tap"
            self.stretch_ratio = bpm / self.bpm_original if self.bpm_original else 1.0
            self._tap_override_until = time.time() + TAP_OVERRIDE_WINDOW_S

    def apply_audio_bpm(
        self,
        bpm: float,
        raw: float | None = None,
        corrected: float | None = None,
        onset_max: float = 0.0,
    ) -> bool:
        """
        Called by bpm_analysis_thread instead of set_bpm() directly.
        Returns False (no-op) while a recent manual tap is still inside
        its override window, so the mic's smoothing doesn't immediately
        erase what the foot just set. Returns True if it wrote the value.
        """
        with self._lock:
            if time.time() < self._tap_override_until:
                return False
            self.bpm_live = bpm
            self.bpm_raw = raw if raw is not None else bpm
            self.bpm_corrected = corrected if corrected is not None else bpm
            self.stretch_ratio = bpm / self.bpm_original if self.bpm_original else 1.0
            self.onset_max = onset_max
            self.bpm_source = "audio"
            return True

    def set_bpm_original(self, bpm: float):
        with self._lock:
            self.bpm_original = bpm
            self.stretch_ratio = (self.bpm_live / bpm) if self.bpm_live and bpm else 1.0

    # ── Markers & loop sections ───────────────────────────────────────────────
    def get_loop_sections(self) -> list[tuple[float, float]]:
        """
        Return sorted adjacent pairs from markers, bookended by 0 and duration.
        E.g. markers [10, 30] on a 60s track → [(0,10), (10,30), (30,60)]
        Returns [] when there are no markers.
        """
        with self._lock:
            pts = sorted(set(self.markers))
            dur = self.track_duration
        if not pts:
            return []
        bounds = [0.0] + pts + [dur]
        return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    def get_active_loop_section(self) -> tuple[float, float] | None:
        """
        Returns (start, end) of the currently selected loop section,
        or None if loop_section_index is -1 (whole track).
        """
        sections = self.get_loop_sections()
        with self._lock:
            idx = self.loop_section_index
        if idx < 0 or not sections:
            return None
        return sections[idx % len(sections)]

    def step_loop_section(self, delta: int):
        """Advance (+1) or rewind (-1) the active loop section."""
        sections = self.get_loop_sections()
        if not sections:
            return
        with self._lock:
            if self.loop_section_index < 0:
                # First activation: go to section 0 on next, last on prev
                self.loop_section_index = 0 if delta > 0 else len(sections) - 1
            else:
                self.loop_section_index = (self.loop_section_index + delta) % len(
                    sections
                )
            # Jump playhead to the start of the new section
            sec = sections[self.loop_section_index]
            self.track_position = sec[0]

    # ── Command dispatcher ────────────────────────────────────────────────────
    def dispatch_command(self, cmd: str, logger=None) -> bool:
        """
        Execute a transport command by name.  Returns True if handled.

        Supported commands
        ------------------
        play          — resume / start playback
        pause         — pause playback
        toggle        — flip play/pause
        next          — load next track in queue
        prev          — load previous track in queue
        loop_toggle   — toggle loop on/off; if markers exist, activates
                        loop-section mode (section 0) on first enable
        loop_next     — advance to next loop section (enables loop if off)
        loop_prev     — rewind to previous loop section (enables loop if off)
        """
        cmd = cmd.lower().strip()

        if cmd == "play":
            self.play()
        elif cmd == "pause":
            self.pause()
        elif cmd == "toggle":
            self.toggle()
        elif cmd == "next":
            with self._lock:
                self.skip_to_next = True
        elif cmd == "prev":
            with self._lock:
                self.skip_to_prev = True
        elif cmd == "loop_toggle":
            with self._lock:
                self.is_looping = not self.is_looping
                if self.is_looping and self.markers and self.loop_section_index < 0:
                    self.loop_section_index = 0
                elif not self.is_looping:
                    self.loop_section_index = -1
        elif cmd in ("loop_next", "loop_prev"):
            delta = 1 if cmd == "loop_next" else -1
            with self._lock:
                self.is_looping = True  # implicitly enable loop
            self.step_loop_section(delta)
        else:
            return False

        with self._lock:
            self.last_command = cmd
            self.pending_command = cmd
        if logger:
            logger.info(f"cmd: {cmd}")
        return True

    # ── Gesture ───────────────────────────────────────────────────────────────
    def set_gesture(
        self, name: str, confidence: float = 0.0, hold_frames: int = 0, hands: int = 1
    ):
        with self._lock:
            self.gesture_name = name
            self.gesture_confidence = confidence
            self.gesture_hold_frames = hold_frames
            self.hands_detected = hands
            self.camera_active = hands > 0

    def get_gesture(self) -> str:
        with self._lock:
            return self.gesture_name

    def set_command(self, cmd: str | None):
        with self._lock:
            self.last_command = cmd
            self.pending_command = cmd

    # ── Audio ─────────────────────────────────────────────────────────────────
    def push_waveform(self, rms_val: float):
        with self._lock:
            v = min(1.0, float(rms_val))
            self.waveform.append(v)
            if len(self.waveform) > 64:
                self.waveform.pop(0)
            self.rms = v
            self.peak = max(self.peak * 0.98, v)

    def set_position(self, pos: float):
        with self._lock:
            self.track_position = pos

    # ── I/O ───────────────────────────────────────────────────────────────────
    def request_io_restart(self, dev_in, dev_out, input_gain=None, output_gain=None):
        with self._lock:
            self.dev_in = dev_in
            self.dev_out = dev_out
            if input_gain is not None:
                self.input_gain = input_gain
            if output_gain is not None:
                self.gain = output_gain
            self.io_restart_requested = True
        CFG.set("dev_in", dev_in, autosave=False)
        CFG.set("dev_out", dev_out, autosave=False)
        if input_gain is not None:
            CFG.set("input_gain", input_gain, autosave=False)
        if output_gain is not None:
            CFG.set("output_gain", output_gain, autosave=False)
        CFG.save()

    def consume_io_restart(self):
        with self._lock:
            self.io_restart_requested = False
            return self.dev_in, self.dev_out

    # ── System ────────────────────────────────────────────────────────────────
    def stop(self):
        with self._lock:
            self.running = False

    def alive(self) -> bool:
        with self._lock:
            return self.running

    def snapshot(self) -> dict:
        with self._lock:
            d = self.__dict__.copy()
            d.pop("_lock", None)
            d.pop("queue", None)
            return d
