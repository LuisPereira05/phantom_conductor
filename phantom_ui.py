"""
Phantom Conductor — UI + Shared State
======================================
Dark rack-unit style interface using Dear PyGui.

Install:
    pip install dearpygui mutagen sounddevice

Run standalone (demo mode):
    python phantom_ui.py
"""

import threading
import time
import math
import random
import collections
import os
import dearpygui.dearpygui as dpg

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

try:
    from mutagen import File as MutagenFile
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False


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

        # I/O — set by UI, consumed by merged_systems
        self.dev_in:  int | None         = None
        self.dev_out: int | None         = None
        self.gain: float                 = 0.85
        self.io_restart_requested: bool  = False

        # Audio levels — set by audio_callback
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
        """Called by merged_systems; returns (dev_in, dev_out) and clears flag."""
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


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":          (14,  14,  14,  255),
    "panel":       (22,  22,  22,  255),
    "panel2":      (28,  28,  28,  255),
    "border":      (42,  42,  42,  255),
    "border2":     (55,  55,  55,  255),
    "text":        (212, 207, 200, 255),
    "text_dim":    (110, 106, 98,  255),
    "amber":       (239, 159, 39,  255),
    "amber_dim":   (186, 117, 23,  255),
    "amber_faint": (65,  36,  2,   255),
    "green":       (99,  197, 71,  255),
    "green_dim":   (59,  109, 17,  255),
    "red":         (226, 75,  74,  255),
    "red_faint":   (45,  16,  16,  255),
    "blue":        (55,  138, 221, 255),
    "blue_dim":    (24,  95,  165, 255),
    "select":      (35,  55,  90,  255),
    "white":       (255, 255, 255, 255),
}

GESTURE_ICONS = {"NO HAND": " — ", "PLAY": "[O]", "PAUSE": "[F]"}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sd_devices() -> tuple[list, list]:
    """Return (input_list, output_list) each as [(index, label), ...]."""
    if not HAS_SD:
        stub = [(-1, "sounddevice not installed")]
        return stub, stub
    inputs, outputs = [], []
    try:
        for i, d in enumerate(sd.query_devices()):
            lbl = f"{i}: {d['name']}"
            if d["max_input_channels"]  > 0: inputs.append((i, lbl))
            if d["max_output_channels"] > 0: outputs.append((i, lbl))
    except Exception:
        pass
    if not inputs:  inputs  = [(-1, "No input device found")]
    if not outputs: outputs = [(-1, "No output device found")]
    return inputs, outputs


# ═══════════════════════════════════════════════════════════════════════════════
#  PHANTOM UI
# ═══════════════════════════════════════════════════════════════════════════════

class PhantomUI:
    WIN_W, WIN_H = 1420, 840

    def __init__(self, state: PhantomState, logger: Logger):
        self.state  = state
        self.logger = logger
        self._tick  = 0

        # Queue selection
        self._queue_sel: int      = -1
        self._last_queue_sig: tuple = (-1, -1, -1, -1)

        # I/O device lists
        self._in_devices:  list = []
        self._out_devices: list = []
        self._in_sel:  int = 0
        self._out_sel: int = 0

    # ── theme helpers ─────────────────────────────────────────────────────────
    def _btn(self, fg, bg, bd):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Text,          fg)
                dpg.add_theme_color(dpg.mvThemeCol_Button,        bg)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, bg)
                dpg.add_theme_color(dpg.mvThemeCol_Border,        bd)
        return t

    # ── setup ─────────────────────────────────────────────────────────────────
    def setup(self):
        dpg.create_context()
        dpg.create_viewport(
            title="Phantom Conductor",
            width=self.WIN_W, height=self.WIN_H,
            min_width=1100, min_height=700,
            resizable=True,
        )
        self._apply_theme()

        self._th_idle = self._btn(C["text"],    C["panel2"],      C["border2"])
        self._th_play = self._btn(C["green"],   (21,48,16,255),   C["green_dim"])
        self._th_loop = self._btn(C["amber"],   C["amber_faint"], C["amber_dim"])
        self._th_amb  = self._btn(C["amber"],   C["amber_faint"], C["amber_dim"])
        self._th_grn  = self._btn(C["green"],   (21,48,16,255),   C["green_dim"])
        self._th_red  = self._btn(C["red"],     C["red_faint"],   (100,40,40,255))
        self._th_blue = self._btn(C["blue"],    (15,30,55,255),   C["blue_dim"])
        self._th_dim  = self._btn(C["text_dim"],C["panel"],       C["border"])

        self._in_devices, self._out_devices = _sd_devices()
        self._build_ui()
        self._setup_file_dialog()
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self):
        self.setup()
        while dpg.is_dearpygui_running() and self.state.alive():
            self._tick += 1
            snap = self.state.snapshot()
            self._update_bpm(snap)
            self._update_waveform(snap)
            self._update_transport(snap)
            self._update_gesture(snap)
            self._update_queue_panel()
            self._update_io_status(snap)
            self._sync_gain()
            self._update_log()
            self._update_clock(snap)
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    # ── global theme ──────────────────────────────────────────────────────────
    def _apply_theme(self):
        with dpg.theme() as g:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,        C["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,         C["panel"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,         C["panel2"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,  C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,   C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_Border,          C["border"])
                dpg.add_theme_color(dpg.mvThemeCol_Text,            C["text"])
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg,         C["panel"])
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,   C["panel2"])
                dpg.add_theme_color(dpg.mvThemeCol_Button,          C["panel2"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,   C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,    C["border"])
                dpg.add_theme_color(dpg.mvThemeCol_Header,          C["select"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,   C["border"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,    C["select"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,     C["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,   C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,         C["panel2"])
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,      C["amber"])
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,C["amber_dim"])
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   2)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,   2)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,   10, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,    6, 4)
        dpg.bind_theme(g)

    # ── file dialog ───────────────────────────────────────────────────────────
    # ── file dialog ───────────────────────────────────────────────────────────
    def _setup_file_dialog(self):
        with dpg.file_dialog(
            label="Add Track(s)", tag="file_dlg",
            width=740, height=540, show=False,
            callback=self._cb_file_dialog,
            cancel_callback=lambda s, a, u: None,
            file_count=100,
        ):
            # .* first so ALL files show by default (no filter needed)
            dpg.add_file_extension(".*",    color=C["text_dim"])
            dpg.add_file_extension(".wav",  color=C["green"])
            dpg.add_file_extension(".mp3",  color=C["amber"])
            dpg.add_file_extension(".flac", color=C["blue"])
            dpg.add_file_extension(".ogg",  color=C["text"])

    def _cb_file_dialog(self, sender, app_data, user_data):
        # Log raw payload — visible in the System Log panel
        self.logger.info(f"file_dlg app_data keys: {list(app_data.keys())}")

        # DearPyGui gives us:
        #   "current_path"   — directory the dialog is browsing
        #   "file_path_name" — full path of highlighted/typed file (may be dir)
        #   "selections"     — dict {display_name: full_path} of checked items
        #                      NOTE: on some DPG builds the value is just the
        #                      filename, not the full path — we handle both.
        current_path = app_data.get("current_path", "")
        selections   = app_data.get("selections", {})
        file_path    = app_data.get("file_path_name", "")

        self.logger.info(
            f"file_dlg current_path={current_path!r}  "
            f"file_path={file_path!r}  "
            f"selections={selections}"
        )

        candidates = []

        # 1. Items the user checked in the multi-select list
        for display_name, sel_path in selections.items():
            if sel_path and os.path.sep in sel_path:
                # Full path provided
                candidates.append(sel_path)
            else:
                # Only filename provided — join with current directory
                name = sel_path or display_name
                candidates.append(os.path.join(current_path, name))

        # 2. The typed / highlighted path
        if file_path:
            if os.path.isabs(file_path):
                candidates.append(file_path)
            else:
                candidates.append(os.path.join(current_path, file_path))

        # De-duplicate, normalise
        seen, paths = set(), []
        for p in candidates:
            p = os.path.normpath(p)
            if p not in seen:
                seen.add(p)
                paths.append(p)

        added = 0
        for path in sorted(paths):
            if not os.path.isfile(path):
                self.logger.warn(f"skipping (not a file): {path}")
                continue
            bpm, dur = _read_audio_meta(path)
            self.state.queue.add(path, bpm=bpm, duration=dur)
            flag = f"  BPM={bpm:.1f}" if bpm else "  BPM=? (set manually)"
            self.logger.ok(f"added: {os.path.basename(path)}{flag}")
            added += 1

        if added == 0:
            self.logger.warn("no valid files found in selection")
        self._force_queue_redraw()

    # ═════════════════════════════════════════════════════════════════════════
    #  LAYOUT
    # ═════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        with dpg.window(label="Phantom Conductor", tag="main_win",
                        no_title_bar=True, no_resize=False,
                        no_move=True, no_scrollbar=True):
            dpg.set_primary_window("main_win", True)
            self._build_header()
            dpg.add_spacer(height=4)

            with dpg.group(horizontal=True):
                # LEFT
                with dpg.child_window(width=710, height=-138,
                                      border=False, tag="left_col"):
                    self._build_bpm_module()
                    dpg.add_spacer(height=4)
                    self._build_waveform_module()
                    dpg.add_spacer(height=4)
                    self._build_transport_module()
                    dpg.add_spacer(height=4)
                    self._build_io_module()
                    dpg.add_spacer(height=4)
                    self._build_controls_module()

                dpg.add_spacer(width=4)

                # RIGHT
                with dpg.child_window(width=-1, height=-138,
                                      border=False, tag="right_col"):
                    self._build_gesture_module()
                    dpg.add_spacer(height=4)
                    self._build_queue_module()

            dpg.add_spacer(height=4)
            self._build_log_panel()

    # ── header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        with dpg.child_window(height=34, border=True, tag="hdr"):
            with dpg.group(horizontal=True):
                dpg.add_text("PHANTOM CONDUCTOR", color=C["amber"])
                dpg.add_text("  v0.5.0", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_text("●", tag="sys_led", color=C["green"])
                dpg.add_text("RUNNING", tag="sys_state", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_text("00:00:00", tag="sys_clock", color=C["text_dim"])

    # ── BPM ───────────────────────────────────────────────────────────────────
    def _build_bpm_module(self):
        with dpg.child_window(height=130, border=True, tag="bpm_panel"):
            dpg.add_text("BPM DETECTION", color=C["text_dim"])
            dpg.add_spacer(height=2)
            with dpg.group(horizontal=True):
                with dpg.group():
                    with dpg.drawlist(width=120, height=56, tag="bpm_draw"):
                        dpg.draw_text((0,0), "---", tag="bpm_draw_text",
                                      color=C["amber"], size=50)
                dpg.add_spacer(width=16)
                with dpg.group():
                    dpg.add_text("SOURCE:", color=C["text_dim"])
                    dpg.add_text("audio input", tag="bpm_source", color=C["text"])
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text("[AUDIO]", tag="pill_audio", color=C["green"])
                        dpg.add_spacer(width=6)
                        dpg.add_text("[TAP]",   tag="pill_tap",   color=C["text_dim"])
                        dpg.add_spacer(width=6)
                        dpg.add_text("[SYNC]",  tag="pill_sync",  color=C["text_dim"])
                    dpg.add_spacer(height=4)
                    dpg.add_text("raw: ---  corr: ---", tag="bpm_debug",
                                 color=C["text_dim"])
                dpg.add_spacer(width=20)
                with dpg.group():
                    dpg.add_text("RATIO", color=C["text_dim"])
                    dpg.add_text("1.000", tag="ratio_val", color=C["amber"])
                    dpg.add_spacer(height=4)
                    dpg.add_text("ONSET", color=C["text_dim"])
                    dpg.add_text("0.000", tag="onset_val", color=C["text_dim"])

            dpg.add_spacer(height=4)
            with dpg.drawlist(width=680, height=12, tag="ratio_bar_draw"):
                dpg.draw_rectangle((0,4),(680,10), color=C["border"],
                                   fill=C["panel2"], tag="ratio_track")
                dpg.draw_line((340,2),(340,14), color=C["border2"],
                              tag="ratio_center")
                dpg.draw_rectangle((340,4),(340,10), color=C["amber"],
                                   fill=C["amber"], tag="ratio_fill")
            with dpg.group(horizontal=True):
                dpg.add_text("0.5×", color=C["text_dim"])
                dpg.add_spacer(width=289)
                dpg.add_text("1.0×", color=C["amber"])
                dpg.add_spacer(width=289)
                dpg.add_text("2.0×", color=C["text_dim"])

    # ── Input level ───────────────────────────────────────────────────────────
    def _build_waveform_module(self):
        with dpg.child_window(height=80, border=True, tag="wave_panel"):
            dpg.add_text("INPUT LEVEL", color=C["text_dim"])
            with dpg.group(horizontal=True):
                with dpg.drawlist(width=590, height=50, tag="waveform_draw"):
                    pass
                dpg.add_spacer(width=6)
                with dpg.group():
                    dpg.add_text("RMS:", color=C["text_dim"])
                    dpg.add_text("0.000", tag="rms_val", color=C["green"])
                dpg.add_spacer(width=6)
                with dpg.group():
                    dpg.add_text("PK:", color=C["text_dim"])
                    dpg.add_text("0.000", tag="peak_val", color=C["amber"])

    # ── Transport ─────────────────────────────────────────────────────────────
    def _build_transport_module(self):
        with dpg.child_window(height=118, border=True, tag="transport_panel"):
            dpg.add_text("BACKING TRACK", color=C["text_dim"])
            with dpg.group(horizontal=True):
                dpg.add_text("no file loaded", tag="file_name", color=C["amber"])
                dpg.add_spacer(width=10)
                dpg.add_text("—", tag="file_meta", color=C["text_dim"])
            dpg.add_spacer(height=4)
            with dpg.drawlist(width=688, height=20, tag="timeline_draw"):
                dpg.draw_rectangle((0,0),(688,20), color=C["border"],
                                   fill=C["panel2"], tag="tl_bg")
                dpg.draw_rectangle((0,0),(0,20), color=C["amber_dim"],
                                   fill=C["amber_faint"], tag="tl_fill")
                dpg.draw_line((0,0),(0,20), color=C["amber"], tag="tl_head")
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(label=" |<  ", tag="btn_prev",
                               callback=self._cb_prev, width=44)
                dpg.bind_item_theme("btn_prev", self._th_dim)

                dpg.add_button(label=" PLAY  ", tag="btn_play",
                               callback=self._cb_play, width=74)

                dpg.add_button(label=" >|  ", tag="btn_next",
                               callback=self._cb_next, width=44)
                dpg.bind_item_theme("btn_next", self._th_dim)

                dpg.add_spacer(width=6)
                dpg.add_button(label=" LOOP ", tag="btn_loop",
                               callback=self._cb_loop, width=62)

                dpg.add_spacer(width=6)
                dpg.add_button(label=" + MARK ", tag="btn_marker",
                               callback=self._cb_add_marker, width=74)
                dpg.bind_item_theme("btn_marker", self._th_dim)

                dpg.add_spacer(width=80)
                dpg.add_text("0:00 / 0:00", tag="time_display",
                             color=C["text_dim"])

    # ── AUDIO I/O panel ───────────────────────────────────────────────────────
    def _build_io_module(self):
        in_names  = [n for _, n in self._in_devices]
        out_names = [n for _, n in self._out_devices]

        with dpg.child_window(height=114, border=True, tag="io_panel"):
            dpg.add_text("AUDIO I/O  — select devices then click APPLY",
                         color=C["text_dim"])
            dpg.add_spacer(height=4)

            with dpg.group(horizontal=True):
                # Input device
                with dpg.group(width=290):
                    dpg.add_text("🎤 INPUT / MICROPHONE", color=C["text_dim"])
                    dpg.add_combo(
                        items=in_names,
                        tag="io_in_combo",
                        default_value=in_names[0] if in_names else "",
                        width=282,
                        callback=self._cb_io_in,
                    )

                dpg.add_spacer(width=8)

                # Output device
                with dpg.group(width=290):
                    dpg.add_text("🔊 OUTPUT / SPEAKER", color=C["text_dim"])
                    dpg.add_combo(
                        items=out_names,
                        tag="io_out_combo",
                        default_value=out_names[0] if out_names else "",
                        width=282,
                        callback=self._cb_io_out,
                    )

                dpg.add_spacer(width=8)

                # Gain
                with dpg.group(width=110):
                    dpg.add_text("GAIN", color=C["text_dim"])
                    dpg.add_slider_float(
                        tag="io_gain",
                        default_value=0.85,
                        min_value=0.0, max_value=2.0,
                        width=100,
                    )

                dpg.add_spacer(width=8)

                # Apply button
                with dpg.group():
                    dpg.add_spacer(height=18)
                    dpg.add_button(
                        label=" ▶ APPLY ",
                        tag="io_apply_btn",
                        callback=self._cb_io_apply,
                        width=90,
                    )
                    dpg.bind_item_theme("io_apply_btn", self._th_grn)

            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text("●", tag="io_led", color=C["text_dim"])
                dpg.add_text("not started — click APPLY to activate streams",
                             tag="io_status", color=C["text_dim"])

    def _cb_io_in(self, sender, app_data, user_data):
        self._in_sel = next(
            (i for i, (_, n) in enumerate(self._in_devices) if n == app_data), 0)

    def _cb_io_out(self, sender, app_data, user_data):
        self._out_sel = next(
            (i for i, (_, n) in enumerate(self._out_devices) if n == app_data), 0)

    def _cb_io_apply(self):
        dev_in  = self._in_devices[self._in_sel][0]  if self._in_devices  else None
        dev_out = self._out_devices[self._out_sel][0] if self._out_devices else None
        if isinstance(dev_in,  int) and dev_in  < 0: dev_in  = None
        if isinstance(dev_out, int) and dev_out < 0: dev_out = None
        self.state.request_io_restart(dev_in, dev_out)
        in_lbl  = self._in_devices[self._in_sel][1]  if self._in_devices  else "default"
        out_lbl = self._out_devices[self._out_sel][1] if self._out_devices else "default"
        self.logger.ok(f"I/O apply: in=[{in_lbl}]  out=[{out_lbl}]")
        dpg.configure_item("io_led", color=C["amber"])
        dpg.set_value("io_status", "applying…")

    # ── Controls ──────────────────────────────────────────────────────────────
    def _build_controls_module(self):
        with dpg.child_window(height=96, border=True, tag="ctrl_panel"):
            dpg.add_text("TIME-STRETCH / REFERENCE BPM", color=C["text_dim"])
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                # Editable reference BPM for the current track
                with dpg.group(width=220):
                    dpg.add_text("Track Ref BPM (editable)", color=C["text_dim"])
                    with dpg.group(horizontal=True):
                        dpg.add_input_float(
                            tag="ctrl_bpm_input",
                            default_value=120.0,
                            min_value=20.0, max_value=300.0,
                            step=0.5, step_fast=5.0,
                            width=128,
                            format="%.1f",
                        )
                        dpg.add_button(label=" SET ", tag="ctrl_bpm_set",
                                       callback=self._cb_set_ref_bpm, width=52)
                        dpg.bind_item_theme("ctrl_bpm_set", self._th_amb)

                dpg.add_spacer(width=10)
                with dpg.group(width=110):
                    dpg.add_text("Stretch", color=C["text_dim"])
                    dpg.add_text("pyrubberband", color=C["text"])

                with dpg.group(width=110):
                    dpg.add_text("Buffer Fill", color=C["text_dim"])
                    dpg.add_text("0%", tag="ctrl_buf", color=C["text"])

                with dpg.group(width=90):
                    dpg.add_text("SR", color=C["text_dim"])
                    dpg.add_text("44100 Hz", color=C["text"])

                with dpg.group(width=100):
                    dpg.add_text("Smooth α", color=C["text_dim"])
                    dpg.add_slider_float(tag="slider_alpha",
                                         default_value=0.3,
                                         min_value=0.0, max_value=1.0,
                                         width=90)

    def _cb_set_ref_bpm(self):
        bpm = dpg.get_value("ctrl_bpm_input")
        if bpm and bpm > 0:
            self.state.set_bpm_original(bpm)
            if self._queue_sel >= 0:
                self.state.queue.set_bpm(self._queue_sel, bpm)
            self.logger.ok(f"ref BPM → {bpm:.1f}")
            self._force_queue_redraw()

    # ── Gesture ───────────────────────────────────────────────────────────────
    def _build_gesture_module(self):
        with dpg.child_window(width=-1, height=270, border=True,
                              tag="gesture_panel"):
            dpg.add_text("GESTURE CONTROL", color=C["text_dim"])
            dpg.add_separator()
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                with dpg.drawlist(width=180, height=110, tag="gesture_draw"):
                    dpg.draw_rectangle((0,0),(180,110), color=C["border"],
                                       fill=(10,10,10,255), tag="gest_bg")
                    dpg.draw_text((60,14), "[ ]", tag="gest_icon_draw",
                                  color=C["text_dim"], size=36)
                    dpg.draw_text((40,72), "NO HAND", tag="gest_name_draw",
                                  color=C["text_dim"], size=14)
                    dpg.draw_text((130,96), "—", tag="gest_conf_draw",
                                  color=C["text_dim"], size=11)
                dpg.add_spacer(width=10)
                with dpg.group():
                    dpg.add_text("HOLD", color=C["text_dim"])
                    dpg.add_progress_bar(tag="hold_bar", default_value=0.0,
                                         width=130, height=8, overlay="0 / 8")
                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text("CAM:", color=C["text_dim"])
                        dpg.add_text("●", tag="cam_led", color=C["text_dim"])
                        dpg.add_text("OFFLINE", tag="cam_state",
                                     color=C["text_dim"])
                    dpg.add_text("HANDS: 0", tag="cam_hands",   color=C["text"])
                    dpg.add_text("CMD:   —", tag="cam_lastcmd", color=C["text"])
                    dpg.add_spacer(height=4)
                    dpg.add_text("🖐 Open = PLAY",  color=C["green"])
                    dpg.add_text("✊ Fist  = PAUSE", color=C["blue"])
                    dpg.add_spacer(height=4)
                    dpg.add_button(label=" CLEAR CMD ", tag="btn_clear_cmd",
                                   callback=lambda: self.state.set_command(None),
                                   width=120)
                    dpg.bind_item_theme("btn_clear_cmd", self._th_red)

    # ── Queue ─────────────────────────────────────────────────────────────────
    def _build_queue_module(self):
        with dpg.child_window(width=-1, height=-1, border=True,
                              tag="queue_panel"):
            dpg.add_text("TRACK QUEUE", color=C["text_dim"])
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # toolbar
            with dpg.group(horizontal=True):
                dpg.add_button(label=" + ADD ", tag="btn_add",
                               callback=lambda: dpg.show_item("file_dlg"),
                               width=64)
                dpg.bind_item_theme("btn_add", self._th_grn)

                dpg.add_button(label=" ▲ ", tag="btn_q_up",
                               callback=self._cb_q_up, width=36)
                dpg.bind_item_theme("btn_q_up", self._th_dim)

                dpg.add_button(label=" ▼ ", tag="btn_q_down",
                               callback=self._cb_q_down, width=36)
                dpg.bind_item_theme("btn_q_down", self._th_dim)

                dpg.add_button(label=" ▶ LOAD ", tag="btn_q_load",
                               callback=self._cb_q_load, width=76)
                dpg.bind_item_theme("btn_q_load", self._th_amb)

                dpg.add_button(label=" ✕ REM ", tag="btn_q_rem",
                               callback=self._cb_q_remove, width=64)
                dpg.bind_item_theme("btn_q_rem", self._th_red)

                dpg.add_button(label=" CLEAR ALL ", tag="btn_q_clear",
                               callback=self._cb_q_clear, width=84)
                dpg.bind_item_theme("btn_q_clear", self._th_red)

            dpg.add_spacer(height=6)

            # ── Inline BPM editor ──────────────────────────────────────────
            with dpg.child_window(height=52, border=True, tag="bpm_edit_box"):
                dpg.add_text("TRACK BPM EDITOR", color=C["text_dim"])
                with dpg.group(horizontal=True):
                    dpg.add_text("BPM:", color=C["text_dim"])
                    dpg.add_input_float(
                        tag="q_bpm_input",
                        default_value=120.0,
                        min_value=20.0, max_value=300.0,
                        step=0.5, step_fast=5.0,
                        width=112,
                        format="%.1f",
                    )
                    dpg.add_button(label=" SET BPM ", tag="q_bpm_set",
                                   callback=self._cb_q_set_bpm, width=76)
                    dpg.bind_item_theme("q_bpm_set", self._th_amb)
                    dpg.add_spacer(width=6)
                    dpg.add_text("← select a row, then SET BPM",
                                 tag="q_bpm_hint", color=C["text_dim"])

            dpg.add_spacer(height=4)

            # column headers
            with dpg.group(horizontal=True):
                dpg.add_text("   #  FILE",            color=C["text_dim"])
                dpg.add_spacer(width=140)
                dpg.add_text("BPM",                   color=C["text_dim"])
                dpg.add_spacer(width=36)
                dpg.add_text("DURATION",              color=C["text_dim"])
            dpg.add_separator()

            # scrollable list — plain group so delete_item(children_only=True) works reliably
            with dpg.child_window(tag="queue_list_outer", height=-1, border=False,
                                  horizontal_scrollbar=False):
                dpg.add_group(tag="queue_list")
                dpg.add_text("— empty —", tag="queue_empty_label",
                             color=C["text_dim"], parent="queue_list_outer")

    # ── Log ───────────────────────────────────────────────────────────────────
    def _build_log_panel(self):
        with dpg.child_window(height=130, border=True, tag="log_panel"):
            with dpg.group(horizontal=True):
                dpg.add_text("SYSTEM LOG", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_button(label=" CLR ", tag="btn_clr_log",
                               callback=lambda: self.logger.clear(), width=50)
                dpg.bind_item_theme("btn_clr_log", self._th_amb)
            dpg.add_separator()
            with dpg.child_window(tag="log_scroll", height=-1, border=False):
                dpg.add_text("", tag="log_text",
                             color=C["text_dim"], wrap=1350)

    # ═════════════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ═════════════════════════════════════════════════════════════════════════

    # transport
    def _cb_play(self):
        with self.state._lock:
            has_track = bool(self.state.track_path)
        if not has_track:
            track = self.state.queue.get_current()
            if track:
                with self.state._lock:
                    self.state.load_new_track = track
                self.logger.info(f"auto-load: {track['name']}")
            return
        new_state = self.state.toggle()
        self.logger.log(f"transport: {'PLAY' if new_state else 'PAUSE'}",
                        "ok" if new_state else "warn")

    def _cb_prev(self):
        track = self.state.queue.prev_track()
        if track:
            with self.state._lock:
                self.state.load_new_track = track
            self.logger.info(f"transport: prev → {track['name']}")

    def _cb_next(self):
        track = self.state.queue.next_track()
        if track:
            with self.state._lock:
                self.state.load_new_track = track
            self.logger.info(f"transport: next → {track['name']}")

    def _cb_loop(self):
        with self.state._lock:
            self.state.is_looping = not self.state.is_looping
            looping = self.state.is_looping
        self.logger.log(f"loop: {'ON' if looping else 'OFF'}",
                        "ok" if looping else "info")

    def _cb_add_marker(self):
        with self.state._lock:
            pos = self.state.track_position
            self.state.markers.append(pos)
        self.logger.info(f"marker at {pos:.2f}s")

    # queue
    def _cb_q_up(self):
        if self._queue_sel > 0:
            self.state.queue.move_up(self._queue_sel)
            self._queue_sel -= 1
            self._force_queue_redraw()

    def _cb_q_down(self):
        tracks, _ = self.state.queue.snapshot()
        if 0 <= self._queue_sel < len(tracks) - 1:
            self.state.queue.move_down(self._queue_sel)
            self._queue_sel += 1
            self._force_queue_redraw()

    def _cb_q_remove(self):
        if self._queue_sel < 0: return
        tracks, _ = self.state.queue.snapshot()
        if self._queue_sel < len(tracks):
            name = tracks[self._queue_sel]["name"]
            self.state.queue.remove(self._queue_sel)
            tracks2, _ = self.state.queue.snapshot()
            self._queue_sel = min(self._queue_sel, len(tracks2) - 1)
            self.logger.info(f"queue: removed {name}")
            self._force_queue_redraw()

    def _cb_q_clear(self):
        q = self.state.queue
        with q._lock:
            q._tracks.clear()
            q._index = 0
        self._queue_sel = -1
        self.logger.info("queue: cleared")
        self._force_queue_redraw()

    def _cb_q_load(self):
        if self._queue_sel < 0: return
        track = self.state.queue.select(self._queue_sel)
        if track:
            with self.state._lock:
                self.state.load_new_track = track
            if track.get("bpm"):
                dpg.set_value("q_bpm_input", track["bpm"])
                dpg.set_value("ctrl_bpm_input", track["bpm"])
            self.logger.ok(f"queue: loading → {track['name']}")

    def _cb_q_row(self, sender, app_data, user_data):
        self._queue_sel = user_data
        tracks, _ = self.state.queue.snapshot()
        if 0 <= user_data < len(tracks):
            bpm = tracks[user_data].get("bpm") or 120.0
            dpg.set_value("q_bpm_input", bpm)
            dpg.set_value("q_bpm_hint", "")
        self._force_queue_redraw()

    def _cb_q_set_bpm(self):
        if self._queue_sel < 0:
            self.logger.warn("select a track row first")
            return
        bpm = dpg.get_value("q_bpm_input")
        if not bpm or bpm <= 0: return
        self.state.queue.set_bpm(self._queue_sel, bpm)
        _, current_idx = self.state.queue.snapshot()
        if self._queue_sel == current_idx:
            self.state.set_bpm_original(bpm)
            dpg.set_value("ctrl_bpm_input", bpm)
        tracks, _ = self.state.queue.snapshot()
        name = tracks[self._queue_sel]["name"] if self._queue_sel < len(tracks) else "?"
        self.logger.ok(f"BPM set: {name} → {bpm:.1f}")
        self._force_queue_redraw()

    # ═════════════════════════════════════════════════════════════════════════
    #  PER-FRAME UPDATES
    # ═════════════════════════════════════════════════════════════════════════

    def _sync_gain(self):
        gain = dpg.get_value("io_gain")
        with self.state._lock:
            self.state.gain = gain

    def _update_io_status(self, snap: dict):
        if snap.get("io_restart_requested"):
            dpg.configure_item("io_led", color=C["amber"])
            dpg.set_value("io_status", "applying…")
        else:
            dev_in  = snap.get("dev_in")
            dev_out = snap.get("dev_out")
            gain    = snap.get("gain", 0.85)
            if dev_in is not None or dev_out is not None:
                dpg.configure_item("io_led", color=C["green"])
                dpg.set_value(
                    "io_status",
                    f"active  in={dev_in}  out={dev_out}  gain={gain:.2f}")
            else:
                dpg.configure_item("io_led", color=C["text_dim"])
                dpg.set_value("io_status",
                              "not started — select devices and click APPLY")

    def _update_clock(self, snap):
        e = int(time.time() - snap["start_time"])
        dpg.set_value("sys_clock",
                      f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}")

    def _update_bpm(self, snap):
        bpm   = snap["bpm_live"]
        orig  = snap["bpm_original"]
        ratio = snap["stretch_ratio"]
        onset = snap["onset_max"]
        raw   = snap["bpm_raw"]
        corr  = snap["bpm_corrected"]

        bpm_str = f"{bpm:.1f}" if bpm else "---"
        synced  = bool(bpm and abs(ratio - 1.0) < 0.03)
        col     = C["green"] if synced else C["amber"]

        try:
            dpg.delete_item("bpm_draw_text")
            dpg.draw_text((0,0), bpm_str, parent="bpm_draw",
                          tag="bpm_draw_text", color=col, size=50)
        except Exception: pass

        dpg.set_value("bpm_source", snap["bpm_source"])
        dpg.set_value("ratio_val",  f"{ratio:.3f}")
        dpg.set_value("onset_val",  f"{onset:.3f}")
        dpg.set_value("ctrl_buf",   f"{int(snap['buffer_fill']*100)}%")
        if raw and corr:
            dpg.set_value("bpm_debug", f"raw: {raw:.1f}  corr: {corr:.1f}")
        dpg.configure_item("pill_sync",
                           color=C["green"] if synced else C["text_dim"])

        # Keep ctrl_bpm_input in sync (unless focused)
        try:
            if not dpg.is_item_focused("ctrl_bpm_input"):
                dpg.set_value("ctrl_bpm_input", orig)
        except Exception: pass

        def rx(r):
            return 340 + (math.log2(max(0.5, min(2.0, r))) * 340)

        x = rx(ratio)
        fc = C["green"] if synced else (C["amber"] if ratio >= 1.0 else C["blue_dim"])
        try:
            dpg.delete_item("ratio_fill")
            lo, hi = (340, x) if ratio >= 1.0 else (x, 340)
            dpg.draw_rectangle((lo, 4), (hi, 10), color=fc, fill=fc,
                               parent="ratio_bar_draw", tag="ratio_fill")
        except Exception: pass

    def _update_waveform(self, snap):
        wave = snap["waveform"][-64:]
        rms  = snap["rms"]
        peak = snap["peak"]
        dpg.set_value("rms_val",  f"{rms:.3f}")
        dpg.set_value("peak_val", f"{peak:.3f}")
        dpg.configure_item("rms_val",
                           color=C["green"] if rms > 0.01 else C["text_dim"])
        dpg.configure_item("peak_val",
                           color=C["red"]   if peak > 0.9 else C["amber"])
        try:
            dpg.delete_item("wave_bars", children_only=True)
            dpg.delete_item("wave_bars")
        except Exception: pass
        try:
            bw = 590 // max(1, len(wave))
            with dpg.draw_node(parent="waveform_draw", tag="wave_bars"):
                for i, v in enumerate(wave):
                    h = max(1, int(v * 48))
                    x = i * bw
                    shade = (C["amber"] if v > 0.7
                             else C["amber_dim"] if v > 0.3
                             else C["amber_faint"])
                    dpg.draw_rectangle((x, 25 - h//2),
                                       (x + max(1, bw-1), 25 + h//2),
                                       color=shade, fill=shade)
        except Exception: pass

    def _update_transport(self, snap):
        playing = snap["is_playing"]
        looping = snap["is_looping"]
        pos     = snap["track_position"]
        dur     = snap["track_duration"]
        fname   = snap["track_path"]
        orig    = snap["bpm_original"]

        dpg.configure_item("btn_play",
                           label=" PAUSE " if playing else " PLAY  ")
        dpg.bind_item_theme("btn_play",
                            self._th_play if playing else self._th_idle)
        dpg.bind_item_theme("btn_loop",
                            self._th_loop if looping else self._th_idle)

        if fname:
            dpg.set_value("file_name", os.path.basename(fname))
            dpg.set_value("file_meta",
                          f"BPM ref: {orig:.1f}  ·  {_fmt(dur)}")

        dpg.set_value("time_display", f"{_fmt(pos)} / {_fmt(dur)}")

        x = int((pos / dur) * 688) if dur > 0 else 0
        try:
            dpg.delete_item("tl_fill"); dpg.delete_item("tl_head")
            dpg.draw_rectangle((0,0),(x,20), color=C["amber_dim"],
                               fill=C["amber_faint"],
                               parent="timeline_draw", tag="tl_fill")
            dpg.draw_line((x,0),(x,20), color=C["amber"],
                          parent="timeline_draw", tag="tl_head")
        except Exception: pass

        try:
            dpg.delete_item("tl_markers", children_only=True)
            dpg.delete_item("tl_markers")
        except Exception: pass
        if dur > 0:
            try:
                with dpg.draw_node(parent="timeline_draw", tag="tl_markers"):
                    for m in snap.get("markers", []):
                        mx = int((m / dur) * 688)
                        dpg.draw_line((mx,0),(mx,20),
                                      color=C["blue"], thickness=2)
            except Exception: pass

    def _update_gesture(self, snap):
        name   = snap["gesture_name"]
        conf   = snap["gesture_confidence"]
        hold   = snap["gesture_hold_frames"]
        target = snap["gesture_hold_target"]
        hands  = snap["hands_detected"]
        active = snap["camera_active"]
        cmd    = snap["last_command"]

        icon  = GESTURE_ICONS.get(name, "[ ]")
        gcol  = (C["green"] if name == "PLAY"
                 else C["blue"] if name == "PAUSE"
                 else C["text_dim"])
        try:
            dpg.delete_item("gest_icon_draw")
            dpg.delete_item("gest_name_draw")
            dpg.delete_item("gest_conf_draw")
            dpg.draw_text((48,12), icon, parent="gesture_draw",
                          tag="gest_icon_draw", color=gcol, size=36)
            dpg.draw_text((max(4, 90 - len(name)*4), 72), name,
                          parent="gesture_draw",
                          tag="gest_name_draw", color=gcol, size=13)
            dpg.draw_text((130,96),
                          f"{conf:.0%}" if active else "—",
                          parent="gesture_draw",
                          tag="gest_conf_draw", color=C["text_dim"], size=11)
        except Exception: pass

        dpg.set_value("hold_bar", min(1.0, hold / max(1, target)))
        dpg.configure_item("hold_bar", overlay=f"{hold} / {target}")
        dpg.set_value("cam_led", "●")
        dpg.configure_item("cam_led",
                           color=C["green"] if active else C["text_dim"])
        dpg.set_value("cam_state", "ACTIVE" if active else "WAITING")
        dpg.configure_item("cam_state",
                           color=C["green"] if active else C["text_dim"])
        dpg.set_value("cam_hands",   f"HANDS: {hands}")
        dpg.set_value("cam_lastcmd", f"CMD:   {cmd}" if cmd else "CMD:   —")
        dpg.configure_item("cam_lastcmd",
                           color=C["green"] if cmd else C["text"])

    def _update_queue_panel(self):
        tracks, cur = self.state.queue.snapshot()
        # Include a content hash so any add/remove/BPM-edit triggers a redraw
        content_hash = hash(tuple(
            (t["name"], t.get("bpm"), t.get("duration", 0.0))
            for t in tracks
        ))
        sig = (len(tracks), cur, self._queue_sel, content_hash)
        if sig != self._last_queue_sig:
            self._last_queue_sig = sig
            self._refresh_queue_table(tracks, cur)

        # update BPM hint
        try:
            dpg.set_value("q_bpm_hint",
                          "" if self._queue_sel >= 0
                          else "← select a row, then SET BPM")
        except Exception: pass

    def _force_queue_redraw(self):
        """Mark signature dirty so the next frame unconditionally redraws."""
        self._last_queue_sig = (-1, -1, -1, -1)

    def _refresh_queue_table(self, tracks, current_idx):
        # children_only=True is the only DPG API that reliably clears a group atomically
        try:
            dpg.delete_item("queue_list", children_only=True)
        except Exception:
            pass

        # Also hide/show the "empty" label that lives in the outer scroll window
        try:
            if tracks:
                dpg.hide_item("queue_empty_label")
            else:
                dpg.show_item("queue_empty_label")
        except Exception:
            pass

        for i, t in enumerate(tracks):
            is_cur  = (i == current_idx)
            is_sel  = (i == self._queue_sel)
            bpm     = t.get("bpm")
            bpm_str = f"{bpm:.0f}" if bpm is not None else "??"
            dur     = t.get("duration", 0.0)
            dur_str = f"{int(dur)//60}:{int(dur)%60:02d}" if dur else "—"

            row_col = (C["amber"] if is_cur
                       else C["white"] if is_sel
                       else C["text_dim"])

            prefix   = "▶ " if is_cur else f"{i+1:>2}."
            name_col = t["name"][:38].ljust(38)
            bpm_col  = bpm_str.ljust(7)
            label    = f"{prefix}  {name_col}  {bpm_col}  {dur_str}"

            sel = dpg.add_selectable(
                label=label,
                default_value=is_sel,
                callback=self._cb_q_row,
                user_data=i,
                width=-1,
                parent="queue_list",
            )
            with dpg.theme() as rt:
                with dpg.theme_component(dpg.mvSelectable):
                    dpg.add_theme_color(dpg.mvThemeCol_Text, row_col)
                    if is_sel:
                        dpg.add_theme_color(dpg.mvThemeCol_Header, C["select"])
            dpg.bind_item_theme(sel, rt)

    def _update_log(self):
        if self._tick % 6 != 0: return
        lines = self.logger.lines(80)
        dpg.set_value("log_text",
                      "\n".join(f"[{ts}] [{lvl.upper():4s}] {msg}"
                                for ts, lvl, msg in lines))

    @staticmethod
    def _fmt_time(s: float) -> str:
        return _fmt(s)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(s: float) -> str:
    s = int(s)
    return f"{s//60}:{s%60:02d}"


def _read_audio_meta(path: str) -> tuple[float | None, float]:
    """Read BPM tag and duration from any audio file mutagen supports."""
    bpm, dur = None, 0.0
    if not HAS_MUTAGEN:
        return bpm, dur
    try:
        af = MutagenFile(path)
        if af and hasattr(af.info, "length"):
            dur = float(af.info.length)
        tags = af.tags if af else None
        if tags:
            for key in ("TBPM", "bpm", "BPM",
                        "TXXX:BPM", "----:com.apple.iTunes:BPM"):
                if key in tags:
                    raw = tags[key]
                    val = str(raw[0] if (hasattr(raw, "__iter__")
                                         and not isinstance(raw, str))
                              else raw)
                    try:
                        bpm = float(val.strip())
                        break
                    except ValueError:
                        pass
    except Exception:
        pass
    return bpm, dur


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def _sim_thread(state: PhantomState, logger: Logger):
    logger.ok("phantom conductor — demo mode (no real audio)")
    logger.info("1) Use +ADD in the Queue panel to add tracks")
    logger.info("2) Set BPM for WAV files using the BPM editor")
    logger.info("3) Select I/O devices in the AUDIO I/O panel")
    logger.info("4) Click APPLY, then PLAY")

    state.bpm_original   = 120.0
    state.track_duration = 0.0

    gestures = ["NO HAND","NO HAND","PLAY","PLAY","PLAY",
                "NO HAND","PAUSE","PAUSE","NO HAND"]
    g_idx = hold = tick = 0
    bpm_c = bpm_t = 120.0

    while state.alive():
        tick += 1

        with state._lock:
            nt = state.load_new_track
            if nt:
                state.load_new_track  = None
                state.track_path      = nt["path"]
                state.track_duration  = nt.get("duration", 0.0)
                state.track_position  = 0.0
                bo = nt.get("bpm") or 120.0
                state.bpm_original    = bo
                state.bpm_live        = bo
                state.stretch_ratio   = 1.0
                bpm_c = bpm_t = bo
                logger.ok(f"demo loaded: {nt['name']}  BPM={bo:.1f}")

        bpm_t += random.gauss(0, 0.2)
        bpm_t  = max(80.0, min(160.0, bpm_t))
        bpm_c  = bpm_c * 0.95 + bpm_t * 0.05
        state.set_bpm(bpm_c,
                      raw=bpm_c + random.gauss(0, 1.2),
                      corrected=bpm_c,
                      onset_max=0.1 + random.random() * 0.7)

        if state.playing():
            state.push_waveform(0.08 + random.random() * 0.4)
            with state._lock:
                state.buffer_fill = min(1.0, tick / 120)
                dur = state.track_duration
                if dur > 0:
                    state.track_position = min(dur, state.track_position + 0.1)
                    if state.track_position >= dur:
                        if state.is_looping:
                            state.track_position = 0.0
                        else:
                            state.is_playing = False
                            logger.ok("demo: track finished")
        else:
            state.push_waveform(0.0)

        if tick % 18 == 0:
            g = gestures[g_idx % len(gestures)]
            g_idx += 1
            active = g != "NO HAND"
            hold   = min(hold + 1, 12) if active else 0
            state.set_gesture(g, confidence=0.9 if active else 0.0,
                              hold_frames=hold, hands=1 if active else 0)
            if active and hold == 8:
                if g == "PLAY":
                    state.play(); state.set_command("PLAY")
                    logger.ok("gesture: PLAY")
                elif g == "PAUSE":
                    state.pause(); state.set_command("PAUSE")
                    logger.ok("gesture: PAUSE")

        time.sleep(0.1)


def main():
    state  = PhantomState()
    logger = Logger()
    threading.Thread(target=_sim_thread, args=(state, logger),
                     daemon=True).start()
    ui = PhantomUI(state, logger)
    try:
        ui.run()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop()
        logger.ok("shutdown")


if __name__ == "__main__":
    main()