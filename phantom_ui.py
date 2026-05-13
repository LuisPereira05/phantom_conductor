"""
Phantom Conductor — Python UI
==============================
Dark rack-unit style interface using Dear PyGui.

Install:
    pip install dearpygui

Run standalone (simulated data):
    python phantom_ui.py

Integration:
    Import PhantomState and update it from your audio/gesture threads.
    The UI reads state every frame — no extra wiring needed.

Architecture:
    PhantomState   — shared state object (thread-safe via threading.Lock)
    Logger         — ring-buffer log, written from any thread
    PhantomUI      — Dear PyGui window, reads state + renders every frame
"""

import threading
import time
import math
import random
import collections
import dearpygui.dearpygui as dpg


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE  — update from your audio/gesture/BPM threads
# ═══════════════════════════════════════════════════════════════════════════════

class PhantomState:
    """
    Thread-safe container for all runtime state.
    Your worker threads write here; the UI reads here every frame.

    Usage from bpm_analysis_thread():
        STATE.set_bpm(bpm_new)

    Usage from gesture_vision_thread():
        STATE.set_gesture("OPEN HAND", confidence=0.94)
        STATE.set_command("PLAY")
    """
    def __init__(self):
        self._lock = threading.Lock()

        # BPM
        self.bpm_live: float | None = None
        self.bpm_original: float    = 120.0
        self.bpm_raw: float | None  = None
        self.bpm_corrected: float | None = None
        self.stretch_ratio: float   = 1.0
        self.bpm_source: str        = "audio"   # "audio" | "tap"

        # Playback
        self.is_playing: bool       = False
        self.is_looping: bool       = False
        self.track_path: str        = ""
        self.track_duration: float  = 0.0
        self.track_position: float  = 0.0      # seconds elapsed
        self.markers: list[float]   = []        # list of marker positions in seconds

        # Audio level  (0.0–1.0)
        self.rms: float             = 0.0
        self.peak: float            = 0.0
        self.waveform: list[float]  = [0.0] * 64   # last 64 short-window RMS values

        # Gesture
        self.gesture_name: str      = "NO HAND"
        self.gesture_confidence: float = 0.0
        self.gesture_hold_frames: int  = 0
        self.gesture_hold_target: int  = 8
        self.pending_command: str | None = None
        self.last_command: str | None    = None
        self.hands_detected: int    = 0
        self.camera_active: bool    = False

        # System
        self.running: bool          = True
        self.start_time: float      = time.time()
        self.onset_max: float       = 0.0
        self.buffer_fill: float     = 0.0      # 0–1, how full the audio ring buffer is

    # ── Setters (called from worker threads) ─────────────────────────────────
    def set_bpm(self, bpm: float, raw: float | None = None,
                corrected: float | None = None, onset_max: float = 0.0):
        with self._lock:
            self.bpm_live      = bpm
            self.bpm_raw       = raw if raw is not None else bpm
            self.bpm_corrected = corrected if corrected is not None else bpm
            self.stretch_ratio = bpm / self.bpm_original if self.bpm_original else 1.0
            self.onset_max     = onset_max

    def set_gesture(self, name: str, confidence: float = 0.0,
                    hold_frames: int = 0, hands: int = 1):
        with self._lock:
            self.gesture_name        = name
            self.gesture_confidence  = confidence
            self.gesture_hold_frames = hold_frames
            self.hands_detected      = hands
            self.camera_active       = (hands > 0)

    def set_command(self, cmd: str | None):
        with self._lock:
            self.last_command    = cmd
            self.pending_command = cmd

    def set_playback(self, playing: bool):
        with self._lock:
            self.is_playing = playing

    def set_position(self, pos: float):
        with self._lock:
            self.track_position = pos

    def push_waveform(self, rms_val: float):
        with self._lock:
            self.waveform.append(min(1.0, rms_val))
            if len(self.waveform) > 64:
                self.waveform.pop(0)
            self.rms  = rms_val
            self.peak = max(self.peak * 0.98, rms_val)

    def stop(self):
        with self._lock:
            self.running = False

    def alive(self) -> bool:
        with self._lock:
            return self.running

    # ── Snapshot (read from UI thread) ───────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            return self.__dict__.copy()


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGER  — ring buffer, written from any thread
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    LEVELS = {"info": 0, "ok": 1, "warn": 2, "err": 3}
    COLORS = {
        "info": (130, 127, 120, 255),
        "ok":   (99,  197, 71,  255),
        "warn": (239, 159, 39,  255),
        "err":  (226, 75,  74,  255),
    }

    def __init__(self, maxlen: int = 200):
        self._lock  = threading.Lock()
        self._lines = collections.deque(maxlen=maxlen)
        self._start = time.time()

    def log(self, msg: str, level: str = "info"):
        t   = time.time() - self._start
        m   = int(t // 60)
        s   = t % 60
        ts  = f"{m:02d}:{s:05.2f}"
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
    "bg":         (14,  14,  14,  255),
    "panel":      (22,  22,  22,  255),
    "panel2":     (28,  28,  28,  255),
    "border":     (42,  42,  42,  255),
    "border2":    (55,  55,  55,  255),
    "text":       (212, 207, 200, 255),
    "text_dim":   (110, 106, 98,  255),
    "amber":      (239, 159, 39,  255),
    "amber_dim":  (186, 117, 23,  255),
    "amber_faint":(65,  36,  2,   255),
    "green":      (99,  197, 71,  255),
    "green_dim":  (59,  109, 17,  255),
    "red":        (226, 75,  74,  255),
    "blue":       (55,  138, 221, 255),
    "blue_dim":   (24,  95,  165, 255),
    "black":      (0,   0,   0,   255),
    "white":      (255, 255, 255, 255),
}

GESTURE_ICONS = {
    "NO HAND":   " — ",
    "OPEN HAND": "[O]",
    "FIST":      "[F]",
    "POINTING":  "[^]",
    "ROCK":      "[R]",
    "PEACE":     "[V]",
    "THUMBS UP": "[T]",
}
GESTURE_COMMANDS = {
    "OPEN HAND": "PLAY",
    "FIST":      "PAUSE",
    "POINTING":  "NEXT",
    "ROCK":      "LOOP",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════

class PhantomUI:
    WIN_W, WIN_H = 1020, 700

    def __init__(self, state: PhantomState, logger: Logger):
        self.state  = state
        self.logger = logger
        self._tick  = 0
        self._waveform_buf: list[float] = [0.0] * 64

    # ── per-item button themes (built once, swapped at runtime) ───────────────
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
            min_width=900, min_height=600,
            resizable=True,
        )
        self._apply_theme()
        # Button state themes — built before UI so we can bind them
        self._theme_play_idle    = self._make_btn_theme(C["text"],  C["panel2"],      C["border2"])
        self._theme_play_active  = self._make_btn_theme(C["green"], (21,48,16,255),   C["green_dim"])
        self._theme_loop_idle    = self._make_btn_theme(C["text"],  C["panel2"],      C["border2"])
        self._theme_loop_active  = self._make_btn_theme(C["amber"], C["amber_faint"], C["amber_dim"])
        self._build_ui()
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
            self._update_log()
            self._update_clock(snap)
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    # ── theme ──────────────────────────────────────────────────────────────────
    def _apply_theme(self):
        with dpg.theme() as global_theme:
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
                dpg.add_theme_color(dpg.mvThemeCol_Header,          C["panel2"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,   C["border"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,     C["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,   C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,         C["panel2"])
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   2)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,   2)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,   10, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,    6, 4)
        dpg.bind_theme(global_theme)

    def _amber_btn_theme(self):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        C["amber_faint"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 45, 5, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text,          C["amber"])
                dpg.add_theme_color(dpg.mvThemeCol_Border,        C["amber_dim"])
        return t

    def _green_btn_theme(self):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (21, 48, 16, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (30, 65, 20, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text,          C["green"])
                dpg.add_theme_color(dpg.mvThemeCol_Border,        C["green_dim"])
        return t

    def _red_btn_theme(self):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (45, 16, 16, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (65, 22, 22, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text,          C["red"])
                dpg.add_theme_color(dpg.mvThemeCol_Border,        (100, 40, 40, 255))
        return t

    # ── UI layout ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        amb  = self._amber_btn_theme()
        grn  = self._green_btn_theme()
        red  = self._red_btn_theme()

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

            # ── two-column layout ─────────────────────────────────────────────
            with dpg.group(horizontal=True):
                # LEFT  (700px wide)
                with dpg.child_window(width=680, height=-140, border=False, tag="left_col"):
                    self._build_bpm_module()
                    dpg.add_spacer(height=4)
                    self._build_waveform_module()
                    dpg.add_spacer(height=4)
                    self._build_transport_module()
                    dpg.add_spacer(height=4)
                    self._build_controls_module()

                dpg.add_spacer(width=4)

                # RIGHT (gesture + camera status)
                with dpg.child_window(width=-1, height=-140, border=False, tag="right_col"):
                    self._build_gesture_module(amb, grn, red)

            # BOTTOM — log panel
            dpg.add_spacer(height=4)
            self._build_log_panel(amb)

    # ── header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        with dpg.child_window(height=34, border=True, tag="header_panel"):
            with dpg.group(horizontal=True):
                dpg.add_text("PHANTOM CONDUCTOR", color=C["amber"])
                dpg.add_text("  v0.3.0", color=C["text_dim"])
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
                # Big BPM number
                with dpg.group():
                    dpg.add_text("---", tag="bpm_big", color=C["amber"])
                    # Make it visually big via draw layer
                    with dpg.drawlist(width=120, height=56, tag="bpm_draw"):
                        dpg.draw_text((0, 0), "---",
                                      tag="bpm_draw_text",
                                      color=C["amber"], size=50)

                dpg.add_spacer(width=16)

                # BPM meta column
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

                # Ratio
                with dpg.group():
                    dpg.add_text("RATIO", color=C["text_dim"])
                    dpg.add_text("1.000", tag="ratio_val", color=C["amber"])
                    dpg.add_spacer(height=4)
                    dpg.add_text("ONSET", color=C["text_dim"])
                    dpg.add_text("0.000", tag="onset_val", color=C["text_dim"])

            dpg.add_spacer(height=4)
            # Ratio bar (drawn each frame)
            with dpg.drawlist(width=640, height=12, tag="ratio_bar_draw"):
                # track
                dpg.draw_rectangle((0, 4), (640, 10), color=C["border"], fill=C["panel2"],
                                   tag="ratio_track")
                # center line
                dpg.draw_line((320, 2), (320, 14), color=C["border2"], tag="ratio_center")
                # fill (updated each frame)
                dpg.draw_rectangle((320, 4), (320, 10), color=C["amber"], fill=C["amber"],
                                   tag="ratio_fill")
            with dpg.group(horizontal=True):
                dpg.add_text("0.5×", color=C["text_dim"])
                dpg.add_spacer(width=265)
                dpg.add_text("1.0×", color=C["amber"])
                dpg.add_spacer(width=265)
                dpg.add_text("2.0×", color=C["text_dim"])

    # ── Waveform / VU module ─────────────────────────────────────────────────
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

    # ── Transport module ─────────────────────────────────────────────────────
    def _build_transport_module(self):
        with dpg.child_window(height=118, border=True, tag="transport_panel"):
            dpg.add_text("BACKING TRACK", color=C["text_dim"])
            with dpg.group(horizontal=True):
                dpg.add_text("no file loaded", tag="file_name", color=C["amber"])
                dpg.add_spacer(width=10)
                dpg.add_text("—", tag="file_meta", color=C["text_dim"])
            dpg.add_spacer(height=4)
            # Timeline
            with dpg.drawlist(width=660, height=20, tag="timeline_draw"):
                dpg.draw_rectangle((0, 0), (660, 20), color=C["border"], fill=C["panel2"],
                                   tag="tl_bg")
                dpg.draw_rectangle((0, 0), (0, 20), color=C["amber_dim"], fill=C["amber_faint"],
                                   tag="tl_fill")
                dpg.draw_line((0, 0), (0, 20), color=C["amber"], tag="tl_head")
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                # Transport buttons
                dpg.add_button(label=" |<  ", tag="btn_prev",
                               callback=self._cb_prev, width=42)
                dpg.add_button(label=" PLAY ", tag="btn_play",
                               callback=self._cb_play, width=70)
                dpg.add_button(label=" >|  ", tag="btn_next",
                               callback=self._cb_next, width=42)
                dpg.add_spacer(width=6)
                dpg.add_button(label=" LOOP ", tag="btn_loop",
                               callback=self._cb_loop, width=60)
                dpg.add_spacer(width=6)
                dpg.add_button(label=" + MARKER ", tag="btn_marker",
                               callback=self._cb_add_marker, width=90)
                dpg.add_spacer(width=140)
                dpg.add_text("0:00 / 0:00", tag="time_display", color=C["text_dim"])

    # ── Controls module ───────────────────────────────────────────────────────
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
    def _build_gesture_module(self, amb, grn, red):
        with dpg.child_window(width=-1, height=-1, border=True, tag="gesture_panel"):
            dpg.add_text("GESTURE CONTROL", color=C["text_dim"])
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Big gesture display box
            with dpg.drawlist(width=290, height=120, tag="gesture_draw"):
                dpg.draw_rectangle((0, 0), (290, 120), color=C["border"],
                                   fill=(10, 10, 10, 255), tag="gest_bg")
                dpg.draw_text((100, 18), "[  ]", tag="gest_icon_draw",
                              color=C["text_dim"], size=40)
                dpg.draw_text((82, 80), "NO HAND", tag="gest_name_draw",
                              color=C["text_dim"], size=16)
                dpg.draw_text((170, 104), "—", tag="gest_conf_draw",
                              color=C["text_dim"], size=12)

            dpg.add_spacer(height=8)

            # Debounce hold bar
            dpg.add_text("HOLD FRAMES", color=C["text_dim"])
            dpg.add_progress_bar(tag="hold_bar", default_value=0.0,
                                 width=-1, height=8,
                                 overlay="0 / 8")
            dpg.add_spacer(height=6)

            # Camera status
            dpg.add_separator()
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text("CAMERA", color=C["text_dim"])
                dpg.add_spacer(width=10)
                dpg.add_text("●", tag="cam_led", color=C["text_dim"])
                dpg.add_text("OFFLINE", tag="cam_state", color=C["text_dim"])
            with dpg.group(horizontal=True):
                dpg.add_text("HANDS:", color=C["text_dim"])
                dpg.add_text("0", tag="cam_hands", color=C["text"])
            with dpg.group(horizontal=True):
                dpg.add_text("PENDING:", color=C["text_dim"])
                dpg.add_text("—", tag="cam_pending", color=C["text"])
            with dpg.group(horizontal=True):
                dpg.add_text("LAST CMD:", color=C["text_dim"])
                dpg.add_text("—", tag="cam_lastcmd", color=C["text"])

            dpg.add_spacer(height=8)
            dpg.add_separator()
            dpg.add_spacer(height=4)
            dpg.add_text("GESTURE MAP", color=C["text_dim"])

            rows = [
                ("[O] Open Hand", "PLAY",  C["green"]),
                ("[F] Fist",      "PAUSE", C["blue"]),
                ("[^] Pointing",  "NEXT",  C["text_dim"]),
                ("[R] Rock",      "LOOP",  C["amber"]),
                ("[V] Peace",     "PREV",  C["text_dim"]),
            ]
            for icon_label, cmd, color in rows:
                with dpg.group(horizontal=True):
                    dpg.add_text(icon_label, color=C["text"])
                    dpg.add_spacer(width=6)
                    dpg.add_text(f"→ {cmd}", color=color)

            dpg.add_spacer(height=6)
            dpg.add_button(label="   CLEAR COMMAND   ", tag="btn_clear_cmd",
                           callback=lambda: self.state.set_command(None), width=-1)
            dpg.bind_item_theme("btn_clear_cmd", red)

    # ── Log panel ─────────────────────────────────────────────────────────────
    def _build_log_panel(self, amb):
        with dpg.child_window(height=130, border=True, tag="log_panel"):
            with dpg.group(horizontal=True):
                dpg.add_text("SYSTEM LOG", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_button(label=" CLR ", callback=lambda: self.logger.clear(),
                               tag="btn_clr_log", width=50)
                dpg.bind_item_theme("btn_clr_log", amb)
            dpg.add_separator()
            with dpg.child_window(tag="log_scroll", height=-1, border=False,
                                  horizontal_scrollbar=False):
                dpg.add_text("", tag="log_text",
                             color=C["text_dim"], wrap=990)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _cb_play(self):
        snap = self.state.snapshot()
        new_state = not snap["is_playing"]
        self.state.set_playback(new_state)
        self.logger.log(f"transport: {'PLAY' if new_state else 'PAUSE'}", "ok" if new_state else "warn")

    def _cb_prev(self):
        self.state.set_command("PREV")
        self.logger.log("transport: previous marker", "info")

    def _cb_next(self):
        self.state.set_command("NEXT")
        self.logger.log("transport: next marker", "info")

    def _cb_loop(self):
        with self.state._lock:
            self.state.is_looping = not self.state.is_looping
            looping = self.state.is_looping
        self.logger.log(f"loop: {'ON' if looping else 'OFF'}", "ok" if looping else "info")

    def _cb_add_marker(self):
        with self.state._lock:
            pos = self.state.track_position
            self.state.markers.append(pos)
        self.logger.log(f"marker added at {pos:.2f}s", "info")

    # ── Frame updates ─────────────────────────────────────────────────────────
    def _update_clock(self, snap: dict):
        elapsed = int(time.time() - snap["start_time"])
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        dpg.set_value("sys_clock", f"{h:02d}:{m:02d}:{s:02d}")

    def _update_bpm(self, snap: dict):
        bpm = snap["bpm_live"]
        orig = snap["bpm_original"]
        ratio = snap["stretch_ratio"]
        onset = snap["onset_max"]
        raw   = snap["bpm_raw"]
        corr  = snap["bpm_corrected"]

        # BPM draw text
        bpm_str = f"{bpm:.1f}" if bpm else "---"
        synced  = bpm and abs(ratio - 1.0) < 0.03
        bpm_color = C["green"] if synced else C["amber"]
        try:
            dpg.delete_item("bpm_draw_text")
            dpg.draw_text((0, 0), bpm_str, parent="bpm_draw",
                          tag="bpm_draw_text", color=bpm_color, size=50)
        except Exception:
            pass

        # metadata
        dpg.set_value("bpm_source",  snap["bpm_source"])
        dpg.set_value("ratio_val",   f"{ratio:.3f}")
        dpg.set_value("onset_val",   f"{onset:.3f}")
        dpg.set_value("ctrl_orig_bpm", f"{orig:.1f}")
        dpg.set_value("ctrl_buf",    f"{int(snap['buffer_fill']*100)}%")
        if raw and corr:
            dpg.set_value("bpm_debug", f"raw: {raw:.1f}  corr: {corr:.1f}")

        dpg.configure_item("pill_sync", color=C["green"] if synced else C["text_dim"])

        # Ratio bar (ratio maps 0.5–2.0 → 0–640)
        def ratio_to_x(r):
            # log2 scale centered at 1.0: range [−1, 1] maps to [0, 640]
            v = max(0.5, min(2.0, r))
            return 320 + (math.log2(v) * 320)   # log2(0.5)=−1, log2(2)=1

        x = ratio_to_x(ratio)
        fill_color = C["green"] if synced else (C["amber"] if ratio >= 1.0 else C["blue_dim"])
        try:
            dpg.delete_item("ratio_fill")
            if ratio >= 1.0:
                dpg.draw_rectangle((320, 4), (x, 10), color=fill_color, fill=fill_color,
                                   parent="ratio_bar_draw", tag="ratio_fill")
            else:
                dpg.draw_rectangle((x, 4), (320, 10), color=fill_color, fill=fill_color,
                                   parent="ratio_bar_draw", tag="ratio_fill")
        except Exception:
            pass

    def _update_waveform(self, snap: dict):
        wave  = snap["waveform"][-64:]
        rms   = snap["rms"]
        peak  = snap["peak"]
        dpg.set_value("rms_val",  f"{rms:.3f}")
        dpg.set_value("peak_val", f"{peak:.3f}")

        dpg.configure_item("rms_val",  color=C["green"] if rms > 0.01 else C["text_dim"])
        dpg.configure_item("peak_val", color=C["red"]   if peak > 0.9  else C["amber"])
        # redraw waveform
        try:
            dpg.delete_item("wave_bars", children_only=True)
            dpg.delete_item("wave_bars")
        except Exception:
            pass
        try:
            w_total = 580
            bar_w   = w_total // len(wave)
            mid     = 25
            with dpg.draw_node(parent="waveform_draw", tag="wave_bars"):
                for i, v in enumerate(wave):
                    h = max(1, int(v * 48))
                    x = i * bar_w
                    shade = C["amber"] if v > 0.7 else (C["amber_dim"] if v > 0.3 else C["amber_faint"])
                    dpg.draw_rectangle((x, mid - h//2), (x + max(1, bar_w-1), mid + h//2),
                                       color=shade, fill=shade)
        except Exception:
            pass

    def _update_transport(self, snap: dict):
        playing  = snap["is_playing"]
        looping  = snap["is_looping"]
        pos      = snap["track_position"]
        dur      = snap["track_duration"]
        fname    = snap["track_path"]

        # play button label + theme swap
        dpg.configure_item("btn_play", label=" PAUSE " if playing else " PLAY  ")
        dpg.bind_item_theme("btn_play",
                            self._theme_play_active if playing else self._theme_play_idle)

        # loop button theme swap
        dpg.bind_item_theme("btn_loop",
                            self._theme_loop_active if looping else self._theme_loop_idle)

        # file info
        if fname:
            name = fname.split("/")[-1].split("\\")[-1]
            dpg.set_value("file_name", name)
            dpg.set_value("file_meta",
                          f"BPM: {snap['bpm_original']:.0f}  ·  {self._fmt_time(dur)}")

        # time display
        dpg.set_value("time_display",
                      f"{self._fmt_time(pos)} / {self._fmt_time(dur)}")

        # timeline
        progress = (pos / dur) if dur > 0 else 0.0
        x = int(progress * 660)
        try:
            dpg.delete_item("tl_fill")
            dpg.delete_item("tl_head")
            dpg.draw_rectangle((0, 0), (x, 20), color=C["amber_dim"], fill=C["amber_faint"],
                               parent="timeline_draw", tag="tl_fill")
            dpg.draw_line((x, 0), (x, 20), color=C["amber"],
                          parent="timeline_draw", tag="tl_head")
        except Exception:
            pass

        # draw markers
        try:
            dpg.delete_item("tl_markers", children_only=True)
            dpg.delete_item("tl_markers")
        except Exception:
            pass
        if dur > 0:
            try:
                with dpg.draw_node(parent="timeline_draw", tag="tl_markers"):
                    for m in snap["markers"]:
                        mx = int((m / dur) * 660)
                        dpg.draw_line((mx, 0), (mx, 20), color=C["blue"], thickness=2)
            except Exception:
                pass

    def _update_gesture(self, snap: dict):
        name   = snap["gesture_name"]
        conf   = snap["gesture_confidence"]
        hold   = snap["gesture_hold_frames"]
        target = snap["gesture_hold_target"]
        hands  = snap["hands_detected"]
        active = snap["camera_active"]
        cmd    = snap["last_command"]
        pending = snap["pending_command"]

        icon_str = GESTURE_ICONS.get(name, "[ ]")
        cmd_for_gesture = GESTURE_COMMANDS.get(name)
        g_color = (C["green"] if cmd_for_gesture == "PLAY"
                   else C["blue"] if cmd_for_gesture == "PAUSE"
                   else C["amber"] if cmd_for_gesture in ("LOOP",)
                   else C["text_dim"])

        try:
            dpg.delete_item("gest_icon_draw")
            dpg.delete_item("gest_name_draw")
            dpg.delete_item("gest_conf_draw")
            dpg.draw_text((90, 14), icon_str, parent="gesture_draw",
                          tag="gest_icon_draw", color=g_color, size=40)
            dpg.draw_text((max(4, 145 - len(name)*5), 78), name,
                          parent="gesture_draw",
                          tag="gest_name_draw", color=g_color, size=16)
            conf_str = f"{conf:.0%}" if active else "—"
            dpg.draw_text((200, 104), conf_str, parent="gesture_draw",
                          tag="gest_conf_draw", color=C["text_dim"], size=12)
        except Exception:
            pass

        # hold bar
        hold_frac = min(1.0, hold / max(1, target))
        dpg.set_value("hold_bar", hold_frac)
        dpg.configure_item("hold_bar", overlay=f"{hold} / {target}")

        # camera status
        dpg.set_value("cam_led",     "●")
        dpg.configure_item("cam_led",
                           color=C["green"] if active else C["text_dim"])
        dpg.set_value("cam_state",
                      "ACTIVE" if active else "WAITING")
        dpg.configure_item("cam_state",
                           color=C["green"] if active else C["text_dim"])
        dpg.set_value("cam_hands",   str(hands))
        dpg.set_value("cam_pending", str(pending) if pending else "—")
        dpg.configure_item("cam_pending",
                           color=C["amber"] if pending else C["text_dim"])
        dpg.set_value("cam_lastcmd", str(cmd) if cmd else "—")
        dpg.configure_item("cam_lastcmd",
                           color=C["green"] if cmd else C["text_dim"])

    def _update_log(self):
        if self._tick % 6 != 0:
            return
        lines = self.logger.lines(60)
        parts = []
        for ts, level, msg in lines:
            color_name = {
                "info": "DIM",
                "ok":   "GRN",
                "warn": "AMB",
                "err":  "RED",
            }.get(level, "DIM")
            parts.append(f"[{ts}] [{level.upper():4s}] {msg}")
        dpg.set_value("log_text", "\n".join(parts))

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _fmt_time(s: float) -> str:
        s = int(s)
        return f"{s//60}:{s%60:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO / SIMULATION  (remove or replace with real threads)
# ═══════════════════════════════════════════════════════════════════════════════

def _sim_thread(state: PhantomState, logger: Logger):
    """Simulates BPM detection, audio levels, and gestures for standalone demo."""
    logger.ok("phantom conductor initialized")
    logger.info("audio device: default input  SR=48000")
    logger.ok("backing track: demo_120bpm.mp3")
    logger.info("original BPM: 120.0")
    logger.ok("mediapipe model: hand_landmarker.task loaded")
    logger.ok("bpm_analysis_thread started")
    logger.ok("gesture_vision_thread started")
    logger.info("waiting for audio signal...")

    state.track_path     = "/tracks/demo_120bpm.mp3"
    state.track_duration = 243.0
    state.bpm_original   = 120.0
    state.markers        = [44.0, 110.0, 175.0]

    gestures = [
        "NO HAND", "NO HAND", "NO HAND",
        "OPEN HAND", "OPEN HAND", "OPEN HAND",
        "NO HAND",
        "FIST", "FIST",
        "NO HAND",
        "POINTING",
        "NO HAND",
        "ROCK",
        "NO HAND",
    ]
    g_idx = 0

    bpm_target = 128.0
    bpm_cur    = 120.0
    hold       = 0
    tick       = 0

    while state.alive():
        tick += 1

        # ── BPM simulation ────────────────────────────────────────────────────
        bpm_target += random.gauss(0, 0.3)
        bpm_target  = max(100.0, min(160.0, bpm_target))
        bpm_cur     = bpm_cur * 0.92 + bpm_target * 0.08
        raw         = bpm_cur + random.gauss(0, 1.5)
        onset       = 0.15 + random.random() * 0.65
        state.set_bpm(bpm_cur, raw=raw, corrected=bpm_cur, onset_max=onset)

        if tick % 20 == 0:
            logger.info(
                f"bpm: live={bpm_cur:.1f}  raw={raw:.1f}  "
                f"ratio={bpm_cur/state.bpm_original:.3f}  onset={onset:.3f}"
            )

        # ── Audio level simulation ────────────────────────────────────────────
        rms = 0.05 + random.random() * (0.55 if state.is_playing else 0.12)
        state.push_waveform(rms)
        state.buffer_fill = min(1.0, tick / 240)

        # ── Playback position ─────────────────────────────────────────────────
        if state.is_playing:
            state.track_position = min(
                state.track_duration,
                state.track_position + 0.1
            )
            if state.track_position >= state.track_duration:
                if state.is_looping:
                    state.track_position = 0.0
                    logger.info("loop: restarted")
                else:
                    state.is_playing = False
                    logger.ok("track finished")

        # ── Gesture simulation ────────────────────────────────────────────────
        if tick % 15 == 0:
            g_name = gestures[g_idx % len(gestures)]
            g_idx += 1
            active = g_name != "NO HAND"
            hold   = min(hold + 1, 12) if active else 0
            conf   = 0.70 + random.random() * 0.28 if active else 0.0
            state.set_gesture(g_name, confidence=conf,
                              hold_frames=hold, hands=1 if active else 0)

            if active and hold == 8:
                cmd = GESTURE_COMMANDS.get(g_name)
                if cmd:
                    state.set_command(cmd)
                    logger.ok(f"gesture: {g_name} → {cmd}")
                    if cmd == "PLAY":
                        state.set_playback(True)
                    elif cmd == "PAUSE":
                        state.set_playback(False)
                    elif cmd == "LOOP":
                        state.is_looping = not state.is_looping
                        logger.ok(f"loop: {'ON' if state.is_looping else 'OFF'}")
            elif active:
                logger.info(f"gesture: {g_name}  hold={hold}/8  conf={conf:.2f}")

        time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    state  = PhantomState()
    logger = Logger()

    # ── Start the simulation thread (replace with your real threads) ──────────
    sim = threading.Thread(target=_sim_thread, args=(state, logger), daemon=True)
    sim.start()

    # ── Run UI (blocking, must be on main thread) ─────────────────────────────
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