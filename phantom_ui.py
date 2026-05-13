"""
Phantom Conductor — UI + Shared State
======================================
Dark rack-unit style interface using Dear PyGui.

Install:
    pip install dearpygui mutagen

Run standalone (simulated data):
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
    from mutagen.mp3 import MP3
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

    def move_up(self, idx: int):
        with self._lock:
            if idx > 0 and idx < len(self._tracks):
                self._tracks[idx-1], self._tracks[idx] = \
                    self._tracks[idx], self._tracks[idx-1]
                if self._index == idx:
                    self._index = idx - 1
                elif self._index == idx - 1:
                    self._index = idx

    def move_down(self, idx: int):
        with self._lock:
            if idx >= 0 and idx < len(self._tracks) - 1:
                self._tracks[idx], self._tracks[idx+1] = \
                    self._tracks[idx+1], self._tracks[idx]
                if self._index == idx:
                    self._index = idx + 1
                elif self._index == idx + 1:
                    self._index = idx

    def get_current(self) -> dict | None:
        with self._lock:
            if self._tracks and 0 <= self._index < len(self._tracks):
                return self._tracks[self._index].copy()
            return None

    def next_track(self) -> dict | None:
        with self._lock:
            if not self._tracks:
                return None
            self._index = (self._index + 1) % len(self._tracks)
            return self._tracks[self._index].copy()

    def prev_track(self) -> dict | None:
        with self._lock:
            if not self._tracks:
                return None
            self._index = (self._index - 1) % len(self._tracks)
            return self._tracks[self._index].copy()

    def select(self, idx: int) -> dict | None:
        with self._lock:
            if 0 <= idx < len(self._tracks):
                self._index = idx
                return self._tracks[idx].copy()
            return None

    def snapshot(self) -> tuple[list, int]:
        with self._lock:
            return list(self._tracks), self._index

    def __len__(self):
        with self._lock:
            return len(self._tracks)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ═══════════════════════════════════════════════════════════════════════════════

class PhantomState:
    """
    Thread-safe container for all runtime state.
    All worker threads write here; the UI reads every frame.
    """

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

        # Signals to worker threads (set by UI callbacks)
        self.load_new_track: dict | None = None   # set to track dict → picked up by worker
        self.skip_to_next: bool          = False
        self.skip_to_prev: bool          = False

        # Audio level
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

        # Track queue (shared object, itself thread-safe)
        self.queue: TrackQueue           = TrackQueue()

    # ── Playback control ─────────────────────────────────────────────────────
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

    # ── BPM ──────────────────────────────────────────────────────────────────
    def get_bpm(self) -> float | None:
        with self._lock:
            return self.bpm_live

    def set_bpm(self, bpm: float, raw: float | None = None,
                corrected: float | None = None, onset_max: float = 0.0):
        with self._lock:
            self.bpm_live      = bpm
            self.bpm_raw       = raw if raw is not None else bpm
            self.bpm_corrected = corrected if corrected is not None else bpm
            self.stretch_ratio = bpm / self.bpm_original if self.bpm_original else 1.0
            self.onset_max     = onset_max

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
        with self._lock:
            return self.gesture_name

    def set_command(self, cmd: str | None):
        with self._lock:
            self.last_command    = cmd
            self.pending_command = cmd

    # ── Audio ─────────────────────────────────────────────────────────────────
    def push_waveform(self, rms_val: float):
        with self._lock:
            self.waveform.append(min(1.0, float(rms_val)))
            if len(self.waveform) > 64:
                self.waveform.pop(0)
            self.rms  = float(rms_val)
            self.peak = max(self.peak * 0.98, float(rms_val))

    def set_position(self, pos: float):
        with self._lock:
            self.track_position = pos

    def set_playback(self, playing: bool):
        with self._lock:
            self.is_playing = playing

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
        m  = int(t // 60)
        s  = t % 60
        ts = f"{m:02d}:{s:05.2f}"
        with self._lock:
            self._lines.appendleft((ts, level, msg))

    def info(self, msg):  self.log(msg, "info")
    def ok(self, msg):    self.log(msg, "ok")
    def warn(self, msg):  self.log(msg, "warn")
    def err(self, msg):   self.log(msg, "err")

    def lines(self, n: int = 60) -> list:
        with self._lock:
            return list(self._lines)[:n]

    def clear(self):
        with self._lock:
            self._lines.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":           (14,  14,  14,  255),
    "panel":        (22,  22,  22,  255),
    "panel2":       (28,  28,  28,  255),
    "border":       (42,  42,  42,  255),
    "border2":      (55,  55,  55,  255),
    "text":         (212, 207, 200, 255),
    "text_dim":     (110, 106, 98,  255),
    "amber":        (239, 159, 39,  255),
    "amber_dim":    (186, 117, 23,  255),
    "amber_faint":  (65,  36,  2,   255),
    "green":        (99,  197, 71,  255),
    "green_dim":    (59,  109, 17,  255),
    "red":          (226, 75,  74,  255),
    "red_faint":    (45,  16,  16,  255),
    "blue":         (55,  138, 221, 255),
    "blue_dim":     (24,  95,  165, 255),
    "black":        (0,   0,   0,   255),
    "white":        (255, 255, 255, 255),
    "select":       (35,  55,  90,  255),
}

GESTURE_ICONS = {
    "NO HAND":  " — ",
    "PLAY":     "[O]",
    "PAUSE":    "[F]",
}
GESTURE_COMMANDS = {
    "PLAY":  "PLAY",
    "PAUSE": "PAUSE",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════

class PhantomUI:
    WIN_W, WIN_H = 1280, 780

    def __init__(self, state: PhantomState, logger: Logger):
        self.state   = state
        self.logger  = logger
        self._tick   = 0
        self._queue_sel: int = -1          # selected row in queue table
        self._last_queue_len: int = -1     # track when queue changes for redraw

    # ── themes ────────────────────────────────────────────────────────────────
    def _make_btn_theme(self, text_color, bg_color, border_color):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Text,          text_color)
                dpg.add_theme_color(dpg.mvThemeCol_Button,        bg_color)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, bg_color)
                dpg.add_theme_color(dpg.mvThemeCol_Border,        border_color)
        return t

    # ── bootstrap ─────────────────────────────────────────────────────────────
    def setup(self):
        dpg.create_context()
        dpg.create_viewport(
            title="Phantom Conductor",
            width=self.WIN_W, height=self.WIN_H,
            min_width=1000, min_height=650,
            resizable=True,
        )
        self._apply_theme()

        self._theme_play_idle   = self._make_btn_theme(C["text"],  C["panel2"],    C["border2"])
        self._theme_play_active = self._make_btn_theme(C["green"], (21,48,16,255), C["green_dim"])
        self._theme_loop_idle   = self._make_btn_theme(C["text"],  C["panel2"],    C["border2"])
        self._theme_loop_active = self._make_btn_theme(C["amber"], C["amber_faint"], C["amber_dim"])
        self._theme_amb  = self._make_btn_theme(C["amber"], C["amber_faint"], C["amber_dim"])
        self._theme_grn  = self._make_btn_theme(C["green"], (21,48,16,255),  C["green_dim"])
        self._theme_red  = self._make_btn_theme(C["red"],   C["red_faint"],  (100,40,40,255))
        self._theme_blue = self._make_btn_theme(C["blue"],  (15,30,55,255),  C["blue_dim"])
        self._theme_dim  = self._make_btn_theme(C["text_dim"], C["panel"], C["border"])

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
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg,       C["panel"])
                dpg.add_theme_color(dpg.mvThemeCol_Button,          C["panel2"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,   C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,    C["border"])
                dpg.add_theme_color(dpg.mvThemeCol_Header,          C["select"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,   C["border"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,    C["select"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,     C["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,   C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,         C["panel2"])
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   2)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,   2)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,   10, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,    6, 4)
        dpg.bind_theme(g)

    # ── file dialog ───────────────────────────────────────────────────────────
    def _setup_file_dialog(self):
        with dpg.file_dialog(
            label="Add Track(s)",
            tag="file_dlg",
            width=700, height=500,
            show=False,
            callback=self._cb_file_dialog,
            cancel_callback=lambda s, a, u: None,
            file_count=100,
        ):
            dpg.add_file_extension(".mp3",  color=C["amber"])
            dpg.add_file_extension(".wav",  color=C["green"])
            dpg.add_file_extension(".flac", color=C["blue"])
            dpg.add_file_extension(".ogg",  color=C["text"])
            dpg.add_file_extension(".*",    color=C["text_dim"])

    def _cb_file_dialog(self, sender, app_data, user_data):
        selections = app_data.get("selections", {})
        paths = list(selections.values())
        if not paths:
            # single-file path
            fp = app_data.get("file_path_name", "")
            if fp:
                paths = [fp]
        for path in sorted(paths):
            if not os.path.isfile(path):
                continue
            bpm, dur = self._read_audio_meta(path)
            self.state.queue.add(path, bpm=bpm, duration=dur)
            self.logger.ok(f"queue: added {os.path.basename(path)}"
                           + (f"  BPM={bpm:.1f}" if bpm else ""))
        self._refresh_queue_table()

    @staticmethod
    def _read_audio_meta(path: str) -> tuple[float | None, float]:
        bpm, dur = None, 0.0
        if not HAS_MUTAGEN:
            return bpm, dur
        try:
            af = MutagenFile(path)
            if af:
                dur = float(af.info.length) if hasattr(af.info, "length") else 0.0
            tags = af.tags if af else None
            if tags:
                for key in ("TBPM", "bpm", "BPM"):
                    if key in tags:
                        val = tags[key]
                        bpm = float(str(val[0] if hasattr(val, "__iter__")
                                        and not isinstance(val, str) else val))
                        break
        except Exception:
            pass
        return bpm, dur

    # ── UI layout ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        with dpg.window(
            label="Phantom Conductor",
            tag="main_win",
            no_title_bar=True,
            no_resize=False,
            no_move=True,
            no_scrollbar=True,
        ):
            dpg.set_primary_window("main_win", True)
            self._build_header()
            dpg.add_spacer(height=4)

            with dpg.group(horizontal=True):
                # ── LEFT column ───────────────────────────────────────────────
                with dpg.child_window(width=680, height=-145, border=False, tag="left_col"):
                    self._build_bpm_module()
                    dpg.add_spacer(height=4)
                    self._build_waveform_module()
                    dpg.add_spacer(height=4)
                    self._build_transport_module()
                    dpg.add_spacer(height=4)
                    self._build_controls_module()

                dpg.add_spacer(width=4)

                # ── RIGHT column ──────────────────────────────────────────────
                with dpg.child_window(width=-1, height=-145, border=False, tag="right_col"):
                    self._build_gesture_module()
                    dpg.add_spacer(height=4)
                    self._build_queue_module()

            dpg.add_spacer(height=4)
            self._build_log_panel()

    # ── header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        with dpg.child_window(height=34, border=True, tag="header_panel"):
            with dpg.group(horizontal=True):
                dpg.add_text("PHANTOM CONDUCTOR", color=C["amber"])
                dpg.add_text("  v0.4.0", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_text("●", tag="sys_led", color=C["green"])
                dpg.add_text("RUNNING", tag="sys_state", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_text("00:00:00", tag="sys_clock", color=C["text_dim"])

    # ── BPM module ────────────────────────────────────────────────────────────
    def _build_bpm_module(self):
        with dpg.child_window(height=130, border=True, tag="bpm_panel"):
            dpg.add_text("BPM DETECTION", color=C["text_dim"])
            dpg.add_spacer(height=2)
            with dpg.group(horizontal=True):
                with dpg.group():
                    with dpg.drawlist(width=120, height=56, tag="bpm_draw"):
                        dpg.draw_text((0, 0), "---",
                                      tag="bpm_draw_text",
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
                    dpg.add_text("raw: ---  corr: ---", tag="bpm_debug", color=C["text_dim"])

                dpg.add_spacer(width=20)

                with dpg.group():
                    dpg.add_text("RATIO", color=C["text_dim"])
                    dpg.add_text("1.000", tag="ratio_val", color=C["amber"])
                    dpg.add_spacer(height=4)
                    dpg.add_text("ONSET", color=C["text_dim"])
                    dpg.add_text("0.000", tag="onset_val", color=C["text_dim"])

            dpg.add_spacer(height=4)
            with dpg.drawlist(width=640, height=12, tag="ratio_bar_draw"):
                dpg.draw_rectangle((0, 4), (640, 10), color=C["border"], fill=C["panel2"],
                                   tag="ratio_track")
                dpg.draw_line((320, 2), (320, 14), color=C["border2"], tag="ratio_center")
                dpg.draw_rectangle((320, 4), (320, 10), color=C["amber"], fill=C["amber"],
                                   tag="ratio_fill")
            with dpg.group(horizontal=True):
                dpg.add_text("0.5×", color=C["text_dim"])
                dpg.add_spacer(width=265)
                dpg.add_text("1.0×", color=C["amber"])
                dpg.add_spacer(width=265)
                dpg.add_text("2.0×", color=C["text_dim"])

    # ── Waveform ──────────────────────────────────────────────────────────────
    def _build_waveform_module(self):
        with dpg.child_window(height=80, border=True, tag="wave_panel"):
            dpg.add_text("INPUT LEVEL", color=C["text_dim"])
            with dpg.group(horizontal=True):
                with dpg.drawlist(width=580, height=50, tag="waveform_draw"):
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
            with dpg.drawlist(width=660, height=20, tag="timeline_draw"):
                dpg.draw_rectangle((0, 0), (660, 20), color=C["border"], fill=C["panel2"],
                                   tag="tl_bg")
                dpg.draw_rectangle((0, 0), (0, 20), color=C["amber_dim"], fill=C["amber_faint"],
                                   tag="tl_fill")
                dpg.draw_line((0, 0), (0, 20), color=C["amber"], tag="tl_head")
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(label=" |<  ", tag="btn_prev",
                               callback=self._cb_prev, width=42)
                dpg.bind_item_theme("btn_prev", self._theme_dim)

                dpg.add_button(label=" PLAY ", tag="btn_play",
                               callback=self._cb_play, width=70)

                dpg.add_button(label=" >|  ", tag="btn_next",
                               callback=self._cb_next, width=42)
                dpg.bind_item_theme("btn_next", self._theme_dim)

                dpg.add_spacer(width=6)
                dpg.add_button(label=" LOOP ", tag="btn_loop",
                               callback=self._cb_loop, width=60)

                dpg.add_spacer(width=6)
                dpg.add_button(label=" + MARKER ", tag="btn_marker",
                               callback=self._cb_add_marker, width=90)
                dpg.bind_item_theme("btn_marker", self._theme_dim)

                dpg.add_spacer(width=80)
                dpg.add_text("0:00 / 0:00", tag="time_display", color=C["text_dim"])

    # ── Controls ──────────────────────────────────────────────────────────────
    def _build_controls_module(self):
        with dpg.child_window(height=90, border=True, tag="controls_panel"):
            dpg.add_text("AUDIO / TIME-STRETCH", color=C["text_dim"])
            with dpg.group(horizontal=True):
                with dpg.group(width=130):
                    dpg.add_text("Original BPM", color=C["text_dim"])
                    dpg.add_text("120.0", tag="ctrl_orig_bpm", color=C["amber"])
                with dpg.group(width=130):
                    dpg.add_text("Stretch Mode", color=C["text_dim"])
                    dpg.add_text("pyrubberband", color=C["text"])
                with dpg.group(width=130):
                    dpg.add_text("Block Size", color=C["text_dim"])
                    dpg.add_text("1 beat", color=C["text"])
                with dpg.group(width=130):
                    dpg.add_text("Buffer Fill", color=C["text_dim"])
                    dpg.add_text("0%", tag="ctrl_buf", color=C["text"])
                with dpg.group(width=130):
                    dpg.add_text("SR", color=C["text_dim"])
                    dpg.add_text("48000 Hz", color=C["text"])
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text("Gain:", color=C["text_dim"])
                dpg.add_slider_float(tag="slider_gain", default_value=0.7,
                                     min_value=0.0, max_value=2.0, width=120)
                dpg.add_spacer(width=12)
                dpg.add_text("Mix:", color=C["text_dim"])
                dpg.add_slider_float(tag="slider_mix", default_value=0.5,
                                     min_value=0.0, max_value=1.0, width=120)
                dpg.add_spacer(width=12)
                dpg.add_text("Smooth α:", color=C["text_dim"])
                dpg.add_slider_float(tag="slider_alpha", default_value=0.3,
                                     min_value=0.0, max_value=1.0, width=120)

    # ── Gesture module ────────────────────────────────────────────────────────
    def _build_gesture_module(self):
        with dpg.child_window(width=-1, height=260, border=True, tag="gesture_panel"):
            dpg.add_text("GESTURE CONTROL", color=C["text_dim"])
            dpg.add_separator()
            dpg.add_spacer(height=4)

            with dpg.group(horizontal=True):
                # Big gesture box
                with dpg.drawlist(width=180, height=110, tag="gesture_draw"):
                    dpg.draw_rectangle((0, 0), (180, 110), color=C["border"],
                                       fill=(10,10,10,255), tag="gest_bg")
                    dpg.draw_text((60, 14), "[ ]", tag="gest_icon_draw",
                                  color=C["text_dim"], size=36)
                    dpg.draw_text((40, 72), "NO HAND", tag="gest_name_draw",
                                  color=C["text_dim"], size=14)
                    dpg.draw_text((130, 96), "—", tag="gest_conf_draw",
                                  color=C["text_dim"], size=11)

                dpg.add_spacer(width=10)

                with dpg.group():
                    dpg.add_text("HOLD", color=C["text_dim"])
                    dpg.add_progress_bar(tag="hold_bar", default_value=0.0,
                                         width=120, height=8, overlay="0 / 8")
                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text("CAM:", color=C["text_dim"])
                        dpg.add_text("●", tag="cam_led", color=C["text_dim"])
                        dpg.add_text("OFFLINE", tag="cam_state", color=C["text_dim"])
                    dpg.add_text("HANDS: 0",   tag="cam_hands",   color=C["text"])
                    dpg.add_text("CMD:   —",   tag="cam_lastcmd", color=C["text"])
                    dpg.add_spacer(height=4)
                    dpg.add_text("🖐 Open = PLAY", color=C["green"])
                    dpg.add_text("✊ Fist  = PAUSE", color=C["blue"])
                    dpg.add_spacer(height=4)
                    dpg.add_button(label=" CLEAR CMD ", tag="btn_clear_cmd",
                                   callback=lambda: self.state.set_command(None), width=120)
                    dpg.bind_item_theme("btn_clear_cmd", self._theme_red)

    # ── Queue module ──────────────────────────────────────────────────────────
    def _build_queue_module(self):
        with dpg.child_window(width=-1, height=-1, border=True, tag="queue_panel"):
            dpg.add_text("TRACK QUEUE", color=C["text_dim"])
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # toolbar
            with dpg.group(horizontal=True):
                dpg.add_button(label=" + ADD ", tag="btn_add",
                               callback=lambda: dpg.show_item("file_dlg"), width=60)
                dpg.bind_item_theme("btn_add", self._theme_grn)

                dpg.add_button(label=" ▲ ", tag="btn_q_up",
                               callback=self._cb_q_up, width=36)
                dpg.bind_item_theme("btn_q_up", self._theme_dim)

                dpg.add_button(label=" ▼ ", tag="btn_q_down",
                               callback=self._cb_q_down, width=36)
                dpg.bind_item_theme("btn_q_down", self._theme_dim)

                dpg.add_button(label=" ▶ LOAD ", tag="btn_q_load",
                               callback=self._cb_q_load, width=72)
                dpg.bind_item_theme("btn_q_load", self._theme_amb)

                dpg.add_button(label=" ✕ REM ", tag="btn_q_rem",
                               callback=self._cb_q_remove, width=60)
                dpg.bind_item_theme("btn_q_rem", self._theme_red)

                dpg.add_button(label=" CLEAR ALL ", tag="btn_q_clear",
                               callback=self._cb_q_clear, width=80)
                dpg.bind_item_theme("btn_q_clear", self._theme_red)

            dpg.add_spacer(height=4)

            # Table header (static)
            with dpg.group(horizontal=True, tag="queue_header"):
                dpg.add_text("#",       color=C["text_dim"], indent=4)
                dpg.add_spacer(width=14)
                dpg.add_text("FILE",    color=C["text_dim"])
                dpg.add_spacer(width=130)
                dpg.add_text("BPM",     color=C["text_dim"])
                dpg.add_spacer(width=30)
                dpg.add_text("DURATION",color=C["text_dim"])
            dpg.add_separator()

            # Scrollable list area
            with dpg.child_window(tag="queue_list", height=-1, border=False,
                                  horizontal_scrollbar=False):
                dpg.add_text("— empty —", tag="queue_empty_label",
                             color=C["text_dim"])

    # ── Log panel ─────────────────────────────────────────────────────────────
    def _build_log_panel(self):
        with dpg.child_window(height=135, border=True, tag="log_panel"):
            with dpg.group(horizontal=True):
                dpg.add_text("SYSTEM LOG", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_button(label=" CLR ", callback=lambda: self.logger.clear(),
                               tag="btn_clr_log", width=50)
                dpg.bind_item_theme("btn_clr_log", self._theme_amb)
            dpg.add_separator()
            with dpg.child_window(tag="log_scroll", height=-1, border=False):
                dpg.add_text("", tag="log_text", color=C["text_dim"], wrap=1200)

    # ── Queue callbacks ───────────────────────────────────────────────────────
    def _cb_q_up(self):
        if self._queue_sel >= 0:
            self.state.queue.move_up(self._queue_sel)
            self._queue_sel = max(0, self._queue_sel - 1)
            self._refresh_queue_table()

    def _cb_q_down(self):
        tracks, _ = self.state.queue.snapshot()
        if self._queue_sel >= 0 and self._queue_sel < len(tracks) - 1:
            self.state.queue.move_down(self._queue_sel)
            self._queue_sel += 1
            self._refresh_queue_table()

    def _cb_q_remove(self):
        if self._queue_sel >= 0:
            tracks, _ = self.state.queue.snapshot()
            if self._queue_sel < len(tracks):
                name = tracks[self._queue_sel]["name"]
                self.state.queue.remove(self._queue_sel)
                tracks2, _ = self.state.queue.snapshot()
                self._queue_sel = min(self._queue_sel, len(tracks2) - 1)
                self.logger.info(f"queue: removed {name}")
                self._refresh_queue_table()

    def _cb_q_clear(self):
        q = self.state.queue
        with q._lock:
            q._tracks.clear()
            q._index = 0
        self._queue_sel = -1
        self.logger.info("queue: cleared")
        self._refresh_queue_table()

    def _cb_q_load(self):
        """Load the selected track immediately."""
        if self._queue_sel < 0:
            return
        track = self.state.queue.select(self._queue_sel)
        if track:
            with self.state._lock:
                self.state.load_new_track = track
            self.logger.ok(f"queue: loading → {track['name']}")

    def _cb_q_row(self, sender, app_data, user_data):
        """Called when a selectable row is clicked."""
        self._queue_sel = user_data
        self._refresh_queue_table()

    def _cb_q_row_dbl(self, sender, app_data, user_data):
        """Double-click a row to load it."""
        self._queue_sel = user_data
        self._cb_q_load()

    def _refresh_queue_table(self):
        """Rebuild the queue list widget from current queue state."""
        tracks, current_idx = self.state.queue.snapshot()

        # Delete all existing rows
        try:
            children = dpg.get_item_children("queue_list", slot=1)
            if children:
                for ch in children:
                    dpg.delete_item(ch)
        except Exception:
            pass

        if not tracks:
            dpg.add_text("— empty —", tag="queue_empty_label",
                         color=C["text_dim"], parent="queue_list")
            return

        for i, t in enumerate(tracks):
            is_current  = (i == current_idx)
            is_selected = (i == self._queue_sel)
            name        = t["name"]
            bpm_str     = f"{t['bpm']:.0f}" if t.get("bpm") else "—"
            dur         = t.get("duration", 0.0)
            dur_str     = f"{int(dur)//60}:{int(dur)%60:02d}" if dur else "—"

            row_color   = (C["amber"]  if is_current
                           else C["text"] if is_selected
                           else C["text_dim"])
            prefix      = "▶ " if is_current else f"{i+1:>2}."

            with dpg.group(horizontal=True, parent="queue_list",
                           tag=f"qrow_{i}"):
                sel = dpg.add_selectable(
                    label=f"{prefix}  {name:<40}  {bpm_str:<7}  {dur_str}",
                    tag=f"qsel_{i}",
                    default_value=is_selected,
                    callback=self._cb_q_row,
                    user_data=i,
                    width=-1,
                )
                dpg.configure_item(f"qsel_{i}", span_columns=True)
                # Color the text
                with dpg.theme() as row_theme:
                    with dpg.theme_component(dpg.mvSelectable):
                        dpg.add_theme_color(dpg.mvThemeCol_Text, row_color)
                        if is_selected:
                            dpg.add_theme_color(dpg.mvThemeCol_Header, C["select"])
                dpg.bind_item_theme(f"qsel_{i}", row_theme)

    # ── Transport callbacks ───────────────────────────────────────────────────
    def _cb_play(self):
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
        self.logger.info(f"marker added at {pos:.2f}s")

    # ── Frame updates ─────────────────────────────────────────────────────────
    def _update_clock(self, snap: dict):
        elapsed = int(time.time() - snap["start_time"])
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        dpg.set_value("sys_clock", f"{h:02d}:{m:02d}:{s:02d}")

    def _update_bpm(self, snap: dict):
        bpm   = snap["bpm_live"]
        orig  = snap["bpm_original"]
        ratio = snap["stretch_ratio"]
        onset = snap["onset_max"]
        raw   = snap["bpm_raw"]
        corr  = snap["bpm_corrected"]

        bpm_str   = f"{bpm:.1f}" if bpm else "---"
        synced    = bpm and abs(ratio - 1.0) < 0.03
        bpm_color = C["green"] if synced else C["amber"]
        try:
            dpg.delete_item("bpm_draw_text")
            dpg.draw_text((0, 0), bpm_str, parent="bpm_draw",
                          tag="bpm_draw_text", color=bpm_color, size=50)
        except Exception:
            pass

        dpg.set_value("bpm_source",    snap["bpm_source"])
        dpg.set_value("ratio_val",     f"{ratio:.3f}")
        dpg.set_value("onset_val",     f"{onset:.3f}")
        dpg.set_value("ctrl_orig_bpm", f"{orig:.1f}")
        dpg.set_value("ctrl_buf",      f"{int(snap['buffer_fill']*100)}%")
        if raw and corr:
            dpg.set_value("bpm_debug", f"raw: {raw:.1f}  corr: {corr:.1f}")

        dpg.configure_item("pill_sync",
                           color=C["green"] if synced else C["text_dim"])

        # ratio bar
        def ratio_to_x(r):
            v = max(0.5, min(2.0, r))
            return 320 + (math.log2(v) * 320)

        x = ratio_to_x(ratio)
        fill_color = (C["green"] if synced
                      else C["amber"] if ratio >= 1.0
                      else C["blue_dim"])
        try:
            dpg.delete_item("ratio_fill")
            if ratio >= 1.0:
                dpg.draw_rectangle((320, 4), (x, 10),
                                   color=fill_color, fill=fill_color,
                                   parent="ratio_bar_draw", tag="ratio_fill")
            else:
                dpg.draw_rectangle((x, 4), (320, 10),
                                   color=fill_color, fill=fill_color,
                                   parent="ratio_bar_draw", tag="ratio_fill")
        except Exception:
            pass

    def _update_waveform(self, snap: dict):
        wave = snap["waveform"][-64:]
        rms  = snap["rms"]
        peak = snap["peak"]
        dpg.set_value("rms_val",  f"{rms:.3f}")
        dpg.set_value("peak_val", f"{peak:.3f}")
        dpg.configure_item("rms_val",  color=C["green"] if rms > 0.01 else C["text_dim"])
        dpg.configure_item("peak_val", color=C["red"]   if peak > 0.9  else C["amber"])

        try:
            dpg.delete_item("wave_bars", children_only=True)
            dpg.delete_item("wave_bars")
        except Exception:
            pass
        try:
            bar_w = 580 // max(1, len(wave))
            mid   = 25
            with dpg.draw_node(parent="waveform_draw", tag="wave_bars"):
                for i, v in enumerate(wave):
                    h = max(1, int(v * 48))
                    x = i * bar_w
                    shade = (C["amber"] if v > 0.7
                             else C["amber_dim"] if v > 0.3
                             else C["amber_faint"])
                    dpg.draw_rectangle(
                        (x, mid - h//2),
                        (x + max(1, bar_w-1), mid + h//2),
                        color=shade, fill=shade)
        except Exception:
            pass

    def _update_transport(self, snap: dict):
        playing = snap["is_playing"]
        looping = snap["is_looping"]
        pos     = snap["track_position"]
        dur     = snap["track_duration"]
        fname   = snap["track_path"]

        dpg.configure_item("btn_play", label=" PAUSE " if playing else " PLAY  ")
        dpg.bind_item_theme("btn_play",
                            self._theme_play_active if playing else self._theme_play_idle)
        dpg.bind_item_theme("btn_loop",
                            self._theme_loop_active if looping else self._theme_loop_idle)

        if fname:
            name = os.path.basename(fname)
            dpg.set_value("file_name", name)
            bpm_orig = snap["bpm_original"]
            dpg.set_value("file_meta",
                          f"BPM: {bpm_orig:.0f}  ·  {self._fmt_time(dur)}")

        dpg.set_value("time_display",
                      f"{self._fmt_time(pos)} / {self._fmt_time(dur)}")

        progress = (pos / dur) if dur > 0 else 0.0
        x = int(progress * 660)
        try:
            dpg.delete_item("tl_fill")
            dpg.delete_item("tl_head")
            dpg.draw_rectangle((0, 0), (x, 20),
                               color=C["amber_dim"], fill=C["amber_faint"],
                               parent="timeline_draw", tag="tl_fill")
            dpg.draw_line((x, 0), (x, 20), color=C["amber"],
                          parent="timeline_draw", tag="tl_head")
        except Exception:
            pass

        try:
            dpg.delete_item("tl_markers", children_only=True)
            dpg.delete_item("tl_markers")
        except Exception:
            pass
        if dur > 0:
            try:
                with dpg.draw_node(parent="timeline_draw", tag="tl_markers"):
                    for m in snap.get("markers", []):
                        mx = int((m / dur) * 660)
                        dpg.draw_line((mx, 0), (mx, 20),
                                      color=C["blue"], thickness=2)
            except Exception:
                pass

    def _update_gesture(self, snap: dict):
        name    = snap["gesture_name"]
        conf    = snap["gesture_confidence"]
        hold    = snap["gesture_hold_frames"]
        target  = snap["gesture_hold_target"]
        hands   = snap["hands_detected"]
        active  = snap["camera_active"]
        cmd     = snap["last_command"]

        icon_str = GESTURE_ICONS.get(name, "[ ]")
        g_color  = (C["green"] if name == "PLAY"
                    else C["blue"] if name == "PAUSE"
                    else C["text_dim"])

        try:
            dpg.delete_item("gest_icon_draw")
            dpg.delete_item("gest_name_draw")
            dpg.delete_item("gest_conf_draw")
            dpg.draw_text((48, 12), icon_str, parent="gesture_draw",
                          tag="gest_icon_draw", color=g_color, size=36)
            name_x = max(4, 90 - len(name) * 4)
            dpg.draw_text((name_x, 72), name, parent="gesture_draw",
                          tag="gest_name_draw", color=g_color, size=13)
            conf_str = f"{conf:.0%}" if active else "—"
            dpg.draw_text((130, 96), conf_str, parent="gesture_draw",
                          tag="gest_conf_draw", color=C["text_dim"], size=11)
        except Exception:
            pass

        dpg.set_value("hold_bar", min(1.0, hold / max(1, target)))
        dpg.configure_item("hold_bar", overlay=f"{hold} / {target}")

        dpg.set_value("cam_led", "●")
        dpg.configure_item("cam_led",
                           color=C["green"] if active else C["text_dim"])
        dpg.set_value("cam_state",
                      "ACTIVE" if active else "WAITING")
        dpg.configure_item("cam_state",
                           color=C["green"] if active else C["text_dim"])
        dpg.set_value("cam_hands",
                      f"HANDS: {hands}")
        dpg.set_value("cam_lastcmd",
                      f"CMD:   {cmd}" if cmd else "CMD:   —")
        dpg.configure_item("cam_lastcmd",
                           color=C["green"] if cmd else C["text"])

    def _update_queue_panel(self):
        """Redraw queue table only when length changes (cheap check)."""
        tracks, _ = self.state.queue.snapshot()
        if len(tracks) != self._last_queue_len:
            self._last_queue_len = len(tracks)
            self._refresh_queue_table()

    def _update_log(self):
        if self._tick % 6 != 0:
            return
        lines = self.logger.lines(80)
        parts = []
        for ts, level, msg in lines:
            parts.append(f"[{ts}] [{level.upper():4s}] {msg}")
        dpg.set_value("log_text", "\n".join(parts))

    @staticmethod
    def _fmt_time(s: float) -> str:
        s = int(s)
        return f"{s//60}:{s%60:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def _sim_thread(state: PhantomState, logger: Logger):
    logger.ok("phantom conductor initialized")
    logger.info("standalone demo mode — no real audio")
    logger.ok("add tracks via the queue panel (+ADD button)")

    state.bpm_original = 120.0
    state.track_duration = 0.0

    gestures = ["NO HAND","NO HAND","PLAY","PLAY","PLAY","NO HAND",
                "PAUSE","PAUSE","NO HAND"]
    g_idx  = 0
    hold   = 0
    tick   = 0
    bpm_cur = 120.0
    bpm_target = 120.0

    while state.alive():
        tick += 1

        # Check for newly loaded track
        with state._lock:
            new_track = state.load_new_track
            if new_track:
                state.load_new_track = None
                state.track_path     = new_track["path"]
                state.track_duration = new_track.get("duration", 0.0)
                state.track_position = 0.0
                bpm_orig = new_track.get("bpm") or 120.0
                state.bpm_original   = bpm_orig
                state.bpm_live       = bpm_orig
                state.stretch_ratio  = 1.0
                logger.ok(f"loaded: {new_track['name']}")

        # BPM simulation
        bpm_target += random.gauss(0, 0.2)
        bpm_target  = max(100.0, min(160.0, bpm_target))
        bpm_cur     = bpm_cur * 0.95 + bpm_target * 0.05
        onset       = 0.1 + random.random() * 0.7
        state.set_bpm(bpm_cur,
                      raw=bpm_cur + random.gauss(0,1.2),
                      corrected=bpm_cur,
                      onset_max=onset)

        # Fake audio level when playing
        if state.playing():
            rms = 0.08 + random.random() * 0.4
            state.push_waveform(rms)
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
                            logger.ok("track finished")
        else:
            state.push_waveform(0.0)

        # Gesture simulation
        if tick % 18 == 0:
            g_name = gestures[g_idx % len(gestures)]
            g_idx += 1
            active = g_name != "NO HAND"
            hold   = min(hold + 1, 12) if active else 0
            conf   = 0.72 + random.random() * 0.26 if active else 0.0
            state.set_gesture(g_name, confidence=conf,
                              hold_frames=hold, hands=1 if active else 0)
            if active and hold == 8:
                if g_name == "PLAY":
                    state.play()
                    state.set_command("PLAY")
                    logger.ok("gesture: PLAY confirmed")
                elif g_name == "PAUSE":
                    state.pause()
                    state.set_command("PAUSE")
                    logger.ok("gesture: PAUSE confirmed")

        time.sleep(0.1)


def main():
    state  = PhantomState()
    logger = Logger()

    sim = threading.Thread(target=_sim_thread, args=(state, logger), daemon=True)
    sim.start()

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