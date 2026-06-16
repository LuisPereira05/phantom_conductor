"""
Phantom Conductor — UI  (Dear PyGui)
=====================================
Dark rack-unit style interface.  Must run on the main thread.

Panels
------
* BPM Detection    — live BPM readout, ratio bar, debug row
* Input Level      — waveform bars, RMS / peak meters
* Backing Track    — transport controls, timeline scrubber, markers
* Settings         — audio I/O, gain sliders, video device, hand
                     command mapper, pedal toggle, tempo-tapper toggle
* Time-Stretch     — reference BPM editor, buffer fill, smoothing α
* Gesture Control  — live camera feed (flicker-free), hold bar
* Track Queue      — scrollable list, inline BPM editor, reorder/load/remove
                     (persisted to tracklist.json automatically)
* System Log       — scrollable log drain

Changes from v0.5.0
--------------------
1. Audio I/O, gain, cam index, gesture map, pedal, tempo-tapper moved
   into a dedicated collapsible Settings panel (no longer split across
   two separate panels).
2. Track list is persistent via PersistentQueue / tracklist.json.
3. Camera feed no longer flashes:
   - Texture upload is rate-limited (max 30 fps via frame counter).
   - upload converts BGR→RGBA once and reuses a pre-allocated buffer.
   - Gesture-draw items are now updated with dpg.configure_item instead
     of delete/redraw every frame (eliminates the 1-frame blank flash).

Install:
    pip install dearpygui mutagen sounddevice
"""

import math
import os
import time
import numpy as np
import threading

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

import cv2

from config import CFG
from state  import PhantomState
from logger import Logger
from gesture_train_ui import GestureTrainUI

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

CAM_W, CAM_H   = 640, 480
_BLANK_TEXTURE  = [0.0] * (CAM_W * CAM_H * 4)   # pre-allocated RGBA float32


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(s: float) -> str:
    s = int(s)
    return f"{s//60}:{s%60:02d}"


def _sd_devices() -> tuple[list, list]:
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


def _read_audio_meta(path: str) -> tuple[float | None, float]:
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
#  PHANTOM UI
# ═══════════════════════════════════════════════════════════════════════════════

class PhantomUI(GestureTrainUI):
    WIN_W, WIN_H = 1420, 900   # slightly taller to accommodate Settings panel

    # How often (in render ticks) to push a new camera frame to the texture.
    # At ~60 fps this gives ~30 texture uploads / second — smooth but cheap.
    _CAM_UPLOAD_EVERY = 2

    def __init__(self, state: PhantomState, logger: Logger):
        self.state  = state
        self.logger = logger
        self._tick  = 0

        self._queue_sel: int        = -1
        self._last_queue_sig: tuple = (-1, -1, -1, -1)

        self._in_devices:  list = []
        self._out_devices: list = []
        self._in_sel:  int = 0
        self._out_sel: int = 0

        # Camera texture buffer — reused every frame (no GC churn)
        self._cam_buf = np.zeros((CAM_H, CAM_W, 4), dtype=np.float32)

        # Last gesture state — only update DPG items when they change
        self._last_gesture: str       = ""
        self._last_gesture_col: tuple = C["text_dim"]

    # ── theme helpers ─────────────────────────────────────────────────────────
    def _btn(self, fg, bg, bd):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Text,          fg)
                dpg.add_theme_color(dpg.mvThemeCol_Button,        bg)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, bg)
                dpg.add_theme_color(dpg.mvThemeCol_Border,        bd)
        return t

    # ── entry point ───────────────────────────────────────────────────────────
    def setup(self):
        dpg.create_context()

        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                width=CAM_W, height=CAM_H,
                default_value=_BLANK_TEXTURE,
                tag="cam_texture",
            )
        self._cam_w = CAM_W
        self._cam_h = CAM_H

        dpg.create_viewport(
            title="Phantom Conductor",
            width=self.WIN_W, height=self.WIN_H,
            min_width=1100, min_height=750,
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
        self._build_train_popup()
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
            self._update_train_ui()
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
    def _setup_file_dialog(self):
        with dpg.file_dialog(
            label="Add Track(s)", tag="file_dlg",
            width=740, height=540, show=False,
            callback=self._cb_file_dialog,
            cancel_callback=lambda s, a, u: None,
            file_count=100,
        ):
            dpg.add_file_extension(".*",    color=C["text_dim"])
            dpg.add_file_extension(".wav",  color=C["green"])
            dpg.add_file_extension(".mp3",  color=C["amber"])
            dpg.add_file_extension(".flac", color=C["blue"])
            dpg.add_file_extension(".ogg",  color=C["text"])

    def _cb_file_dialog(self, sender, app_data, user_data):
        self.logger.info(f"file_dlg app_data keys: {list(app_data.keys())}")
        current_path = app_data.get("current_path", "")
        selections   = app_data.get("selections", {})
        file_path    = app_data.get("file_path_name", "")

        candidates = []
        for display_name, sel_path in selections.items():
            if sel_path and os.path.sep in sel_path:
                candidates.append(sel_path)
            else:
                name = sel_path or display_name
                candidates.append(os.path.join(current_path, name))
        if file_path:
            candidates.append(file_path if os.path.isabs(file_path)
                              else os.path.join(current_path, file_path))

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
            # PersistentQueue.add() saves to JSON automatically
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
                with dpg.child_window(width=710, height=-138,
                                      border=False, tag="left_col", resizable_x=True):
                    self._build_bpm_module()
                    dpg.add_spacer(height=4)
                    self._build_waveform_module()
                    dpg.add_spacer(height=4)
                    self._build_transport_module()
                    dpg.add_spacer(height=4)
                    self._build_settings_module()   # ← replaces old I/O + controls
                    dpg.add_spacer(height=4)
                    self._build_stretch_module()

                dpg.add_spacer(width=4)

                with dpg.child_window(width=-1, height=-138,
                                      border=False, tag="right_col"):
                    self._build_gesture_module()
                    dpg.add_spacer(height=4)
                    self._build_queue_module()

            dpg.add_spacer(height=4)
            self._build_log_panel()

    def _build_header(self):
        with dpg.child_window(height=34, border=True, tag="hdr"):
            with dpg.group(horizontal=True):
                dpg.add_text("PHANTOM CONDUCTOR", color=C["amber"])
                dpg.add_text("  v0.5.1", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_text("●", tag="sys_led", color=C["green"])
                dpg.add_text("RUNNING", tag="sys_state", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_text("00:00:00", tag="sys_clock", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_button(label=" TRAIN GESTURES ",
                    tag="btn_train_open",
                    callback=self._cb_train_open,
                    width=140)
                dpg.bind_item_theme("btn_train_open", self._th_blue)

    def _build_bpm_module(self):
        with dpg.child_window(height=130, border=True, tag="bpm_panel"):
            dpg.add_text("BPM DETECTION", color=C["text_dim"])
            dpg.add_spacer(height=2)
            with dpg.group(horizontal=True):
                with dpg.group():
                    with dpg.drawlist(width=120, height=56, tag="bpm_draw"):
                        dpg.draw_text((0, 0), "---", tag="bpm_draw_text",
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
            with dpg.drawlist(width=688, height=12, tag="ratio_bar_draw"):
                dpg.draw_rectangle((0, 4), (688, 10), color=C["border"],
                                   fill=C["panel2"], tag="ratio_track")
                dpg.draw_line((340, 2), (340, 14), color=C["border2"],
                              tag="ratio_center")
                dpg.draw_rectangle((340, 4), (340, 10), color=C["amber"],
                                   fill=C["amber"], tag="ratio_fill")
            with dpg.group(horizontal=True):
                dpg.add_text("0.5×", color=C["text_dim"])
                dpg.add_spacer(width=289)
                dpg.add_text("1.0×", color=C["amber"])
                dpg.add_spacer(width=289)
                dpg.add_text("2.0×", color=C["text_dim"])

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

    def _build_transport_module(self):
        with dpg.child_window(height=118, border=True, tag="transport_panel"):
            dpg.add_text("BACKING TRACK", color=C["text_dim"])
            with dpg.group(horizontal=True):
                dpg.add_text("no file loaded", tag="file_name", color=C["amber"])
                dpg.add_spacer(width=10)
                dpg.add_text("—", tag="file_meta", color=C["text_dim"])
            dpg.add_spacer(height=4)
            with dpg.drawlist(width=688, height=20, tag="timeline_draw"):
                dpg.draw_rectangle((0, 0), (688, 20), color=C["border"],
                                   fill=C["panel2"], tag="tl_bg")
                dpg.draw_rectangle((0, 0), (0, 20), color=C["amber_dim"],
                                   fill=C["amber_faint"], tag="tl_fill")
                dpg.draw_line((0, 0), (0, 20), color=C["amber"], tag="tl_head")
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

    # ─────────────────────────────────────────────────────────────────────────
    #  SETTINGS PANEL  (replaces old Audio I/O + Controls modules)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_settings_module(self):
        in_names  = [n for _, n in self._in_devices]
        out_names = [n for _, n in self._out_devices]
 
        with dpg.child_window(height=240, border=True, tag="settings_panel"):
            dpg.add_text("SETTINGS", color=C["text_dim"])
            dpg.add_separator()
            dpg.add_spacer(height=4)
 
            # ── Row 1: Audio I/O devices ──────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("AUDIO", color=C["amber"])
                dpg.add_spacer(width=8)
                with dpg.group(width=255):
                    dpg.add_text("Mic Input", color=C["text_dim"])
                    dpg.add_combo(items=in_names, tag="io_in_combo",
                                  default_value=in_names[0] if in_names else "",
                                  width=248, callback=self._cb_io_in)
                dpg.add_spacer(width=6)
                with dpg.group(width=255):
                    dpg.add_text("Speaker Output", color=C["text_dim"])
                    dpg.add_combo(items=out_names, tag="io_out_combo",
                                  default_value=out_names[0] if out_names else "",
                                  width=248, callback=self._cb_io_out)
                dpg.add_spacer(width=6)
                with dpg.group():
                    dpg.add_spacer(height=17)
                    dpg.add_button(label=" APPLY ", tag="io_apply_btn",
                                   callback=self._cb_io_apply, width=88)
                    dpg.bind_item_theme("io_apply_btn", self._th_grn)
 
            dpg.add_spacer(height=4)
 
            # ── Row 2: Gain sliders ───────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("GAIN", color=C["amber"])
                dpg.add_spacer(width=8)
                with dpg.group(width=180):
                    dpg.add_text("Input (mic pre-gain)", color=C["text_dim"])
                    dpg.add_slider_float(tag="gain_input", width=170,
                                         default_value=CFG.input_gain,
                                         min_value=0.0, max_value=3.0,
                                         format="%.2f")
                dpg.add_spacer(width=10)
                with dpg.group(width=180):
                    dpg.add_text("Output (backing track)", color=C["text_dim"])
                    dpg.add_slider_float(tag="io_gain", width=170,
                                         default_value=CFG.output_gain,
                                         min_value=0.0, max_value=2.0,
                                         format="%.2f")
                dpg.add_spacer(width=10)
                with dpg.group(horizontal=True):
                    dpg.add_text("I/O:", color=C["text_dim"])
                    dpg.add_text("●", tag="io_led", color=C["text_dim"])
                    dpg.add_spacer(width=4)
                    dpg.add_text("not started", tag="io_status",
                                 color=C["text_dim"])
 
            dpg.add_spacer(height=6)
 
            # ── Row 3: Video device ───────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("VIDEO", color=C["amber"])
                dpg.add_spacer(width=8)
                with dpg.group(width=220):
                    dpg.add_text("Camera Index", color=C["text_dim"])
                    dpg.add_input_int(tag="cfg_cam_index",
                                      default_value=CFG.cam_index,
                                      min_value=0, max_value=15, width=90,
                                      callback=self._cb_cam_index)
                dpg.add_spacer(width=10)
                dpg.add_text("(restart required to take effect)",
                             color=C["text_dim"])
 
            dpg.add_spacer(height=6)
 
            # ── Row 4: Gesture command mapper ─────────────────────────────────
            _ALL_CMDS = ["play", "pause", "toggle", "next", "prev",
                         "loop_toggle", "loop_next", "loop_prev", "none"]
 
            with dpg.group(horizontal=True):
                dpg.add_text("GESTURES", color=C["amber"])
                dpg.add_spacer(width=8)
 
                with dpg.group(width=154):
                    dpg.add_text("Open Hand (5)", color=C["text_dim"])
                    dpg.add_combo(
                        items=_ALL_CMDS, tag="gmap_play_combo",
                        default_value=CFG.gesture_map.get("PLAY", "play"),
                        width=146,
                        callback=lambda s, a, u: self._cb_gesture_map("PLAY", a),
                    )
                dpg.add_spacer(width=4)
 
                with dpg.group(width=154):
                    dpg.add_text("Fist (0)", color=C["text_dim"])
                    dpg.add_combo(
                        items=_ALL_CMDS, tag="gmap_pause_combo",
                        default_value=CFG.gesture_map.get("PAUSE", "pause"),
                        width=146,
                        callback=lambda s, a, u: self._cb_gesture_map("PAUSE", a),
                    )
                dpg.add_spacer(width=4)
 
                with dpg.group(width=154):
                    dpg.add_text("Index / Point (1)", color=C["text_dim"])
                    dpg.add_combo(
                        items=_ALL_CMDS, tag="gmap_point_combo",
                        default_value=CFG.gesture_map.get("POINT", "next"),
                        width=146,
                        callback=lambda s, a, u: self._cb_gesture_map("POINT", a),
                    )
                dpg.add_spacer(width=4)
 
                with dpg.group(width=154):
                    dpg.add_text("Peace / V (2)", color=C["text_dim"])
                    dpg.add_combo(
                        items=_ALL_CMDS, tag="gmap_peace_combo",
                        default_value=CFG.gesture_map.get("PEACE", "loop_toggle"),
                        width=146,
                        callback=lambda s, a, u: self._cb_gesture_map("PEACE", a),
                    )
                dpg.add_spacer(width=6)
 
                with dpg.group(width=130):
                    dpg.add_text("Hold frames", color=C["text_dim"])
                    dpg.add_input_int(tag="cfg_hold_frames",
                                      default_value=CFG.gesture_hold_frames,
                                      min_value=1, max_value=60, width=80,
                                      callback=self._cb_hold_frames)
                    dpg.add_checkbox(label=" Foot pedal",
                                     tag="cfg_use_pedal",
                                     default_value=CFG.use_pedal,
                                     callback=self._cb_use_pedal)
                    dpg.add_checkbox(label=" Tempo tapper",
                                     tag="cfg_use_tapper",
                                     default_value=CFG.use_tempo_tapper,
                                     callback=self._cb_use_tapper)
 

    def _build_stretch_module(self):
        """Reference BPM editor, buffer fill, smoothing α — kept separate."""
        with dpg.child_window(height=96, border=True, tag="ctrl_panel"):
            dpg.add_text("TIME-STRETCH / REFERENCE BPM", color=C["text_dim"])
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                with dpg.group(width=220):
                    dpg.add_text("Track Ref BPM (editable)", color=C["text_dim"])
                    with dpg.group(horizontal=True):
                        dpg.add_input_float(tag="ctrl_bpm_input",
                                            default_value=120.0,
                                            min_value=20.0, max_value=300.0,
                                            step=0.5, step_fast=5.0,
                                            width=128, format="%.1f")
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
                                         default_value=CFG.smooth_alpha,
                                         min_value=0.0, max_value=1.0, width=90)

    # ─────────────────────────────────────────────────────────────────────────
    #  GESTURE / CAMERA PANEL  (flicker-free)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_gesture_module(self):
        with dpg.child_window(width=-1, height=270, border=True,
                              tag="gesture_panel"):
            dpg.add_text("GESTURE CONTROL", color=C["text_dim"])
            dpg.add_separator()
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                # Camera feed — image widget backed by dynamic texture
                dpg.add_image("cam_texture", width=213, height=160,
                              tag="cam_feed")
                dpg.add_spacer(width=10)
                with dpg.group():
                    # Static text items updated in-place (no delete/redraw)
                    dpg.add_text("[ ]",     tag="gest_icon",  color=C["text_dim"])
                    dpg.add_text("NO HAND", tag="gest_name",  color=C["text_dim"])
                    dpg.add_text("—",       tag="gest_conf",  color=C["text_dim"])
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
                    dpg.add_text("Open Hand = PLAY",  color=C["green"])
                    dpg.add_text("Fist       = PAUSE", color=C["blue"])
                    dpg.add_spacer(height=4)
                    dpg.add_button(label=" CLEAR CMD ", tag="btn_clear_cmd",
                                   callback=lambda: self.state.set_command(None),
                                   width=120)
                    dpg.bind_item_theme("btn_clear_cmd", self._th_red)

    def _build_queue_module(self):
        with dpg.child_window(width=-1, height=-1, border=True,
                              tag="queue_panel"):
            dpg.add_text("TRACK QUEUE", color=C["text_dim"])
            dpg.add_separator()
            dpg.add_spacer(height=4)
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
                dpg.add_spacer(width=8)
                dpg.add_text("💾 auto-saved", tag="queue_save_indicator",
                             color=C["text_dim"])

            dpg.add_spacer(height=6)
            with dpg.child_window(height=52, border=True, tag="bpm_edit_box"):
                dpg.add_text("TRACK BPM EDITOR", color=C["text_dim"])
                with dpg.group(horizontal=True):
                    dpg.add_text("BPM:", color=C["text_dim"])
                    dpg.add_input_float(tag="q_bpm_input", default_value=120.0,
                                        min_value=20.0, max_value=300.0,
                                        step=0.5, step_fast=5.0,
                                        width=112, format="%.1f")
                    dpg.add_button(label=" SET BPM ", tag="q_bpm_set",
                                   callback=self._cb_q_set_bpm, width=76)
                    dpg.bind_item_theme("q_bpm_set", self._th_amb)
                    dpg.add_spacer(width=6)
                    dpg.add_text("← select a row, then SET BPM",
                                 tag="q_bpm_hint", color=C["text_dim"])

            dpg.add_spacer(height=4)

            # Scrollable list area — table is rebuilt inside here each redraw
            with dpg.child_window(tag="queue_list_outer", height=-1, border=False,
                                  horizontal_scrollbar=False):
                dpg.add_text("— empty —", tag="queue_empty_label",
                             color=C["text_dim"])

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

    def _cb_io_in(self, sender, app_data, user_data):
        self._in_sel = next(
            (i for i, (_, n) in enumerate(self._in_devices) if n == app_data), 0)

    def _cb_io_out(self, sender, app_data, user_data):
        self._out_sel = next(
            (i for i, (_, n) in enumerate(self._out_devices) if n == app_data), 0)

    def _cb_io_apply(self):
        dev_in   = self._in_devices[self._in_sel][0]  if self._in_devices  else None
        dev_out  = self._out_devices[self._out_sel][0] if self._out_devices else None
        if isinstance(dev_in,  int) and dev_in  < 0: dev_in  = None
        if isinstance(dev_out, int) and dev_out < 0: dev_out = None
        in_gain  = dpg.get_value("gain_input")
        out_gain = dpg.get_value("io_gain")
        self.state.request_io_restart(dev_in, dev_out,
                                       input_gain=in_gain, output_gain=out_gain)
        in_lbl  = self._in_devices[self._in_sel][1]  if self._in_devices  else "default"
        out_lbl = self._out_devices[self._out_sel][1] if self._out_devices else "default"
        self.logger.ok(
            f"I/O apply: in=[{in_lbl}]  out=[{out_lbl}]  "
            f"in_gain={in_gain:.2f}  out_gain={out_gain:.2f}"
        )
        dpg.configure_item("io_led", color=C["amber"])
        dpg.set_value("io_status", "applying…")

    def _cb_cam_index(self, sender, app_data, user_data):
        CFG.set("cam_index", app_data)
        self.logger.info(f"cam index → {app_data}  (restart to apply)")

    def _cb_gesture_map(self, gesture_key: str, value: str):
        gmap = dict(CFG.gesture_map)
        gmap[gesture_key] = value
        CFG.set("gesture_map", gmap)
        self.logger.info(f"gesture map: {gesture_key} → {value}")

    def _cb_hold_frames(self, sender, app_data, user_data):
        CFG.set("gesture_hold_frames", app_data)
        with self.state._lock:
            self.state.gesture_hold_target = app_data
        self.logger.info(f"hold frames → {app_data}")

    def _cb_use_pedal(self, sender, app_data, user_data):
        CFG.set("use_pedal", app_data)
        self.logger.info(f"foot pedal: {'ON' if app_data else 'OFF'}")

    def _cb_use_tapper(self, sender, app_data, user_data):
        CFG.set("use_tempo_tapper", app_data)
        self.logger.info(f"tempo tapper: {'ON' if app_data else 'OFF'}")

    def _cb_set_ref_bpm(self):
        bpm = dpg.get_value("ctrl_bpm_input")
        if bpm and bpm > 0:
            self.state.set_bpm_original(bpm)
            if self._queue_sel >= 0:
                self.state.queue.set_bpm(self._queue_sel, bpm)
            self.logger.ok(f"ref BPM → {bpm:.1f}")
            self._force_queue_redraw()

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
            # PersistentQueue.remove() saves JSON automatically
            self.state.queue.remove(self._queue_sel)
            tracks2, _ = self.state.queue.snapshot()
            self._queue_sel = min(self._queue_sel, len(tracks2) - 1)
            self.logger.info(f"queue: removed {name}")
            self._force_queue_redraw()

    def _cb_q_clear(self):
        self.state.queue.clear()   # PersistentQueue.clear() saves JSON
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
        self.state.queue.set_bpm(self._queue_sel, bpm)   # auto-saves JSON
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
        out_gain = dpg.get_value("io_gain")
        with self.state._lock:
            self.state.gain = out_gain

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
                dpg.set_value("io_status",
                              f"active  in={dev_in}  out={dev_out}")
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
            dpg.draw_text((0, 0), bpm_str, parent="bpm_draw",
                          tag="bpm_draw_text", color=col, size=50)
        except Exception:
            pass

        dpg.set_value("bpm_source", snap["bpm_source"])
        dpg.set_value("ratio_val",  f"{ratio:.3f}")
        dpg.set_value("onset_val",  f"{onset:.3f}")
        dpg.set_value("ctrl_buf",   f"{int(snap['buffer_fill']*100)}%")
        if raw and corr:
            dpg.set_value("bpm_debug", f"raw: {raw:.1f}  corr: {corr:.1f}")
        dpg.configure_item("pill_sync",
                           color=C["green"] if synced else C["text_dim"])

        try:
            if not dpg.is_item_focused("ctrl_bpm_input"):
                dpg.set_value("ctrl_bpm_input", orig)
        except Exception:
            pass

        def rx(r):
            return 340 + (math.log2(max(0.5, min(2.0, r))) * 340)

        x  = rx(ratio)
        fc = C["green"] if synced else (C["amber"] if ratio >= 1.0 else C["blue_dim"])
        try:
            dpg.delete_item("ratio_fill")
            lo, hi = (340, x) if ratio >= 1.0 else (x, 340)
            dpg.draw_rectangle((lo, 4), (hi, 10), color=fc, fill=fc,
                               parent="ratio_bar_draw", tag="ratio_fill")
        except Exception:
            pass

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
        except Exception:
            pass
        try:
            bw = 590 // max(1, len(wave))
            with dpg.draw_node(parent="waveform_draw", tag="wave_bars"):
                for i, v in enumerate(wave):
                    h = max(1, int(v * 48))
                    x = i * bw
                    shade = (C["amber"] if v > 0.7
                             else C["amber_dim"] if v > 0.3
                             else C["amber_faint"])
                    dpg.draw_rectangle((x, 25 - h // 2),
                                       (x + max(1, bw - 1), 25 + h // 2),
                                       color=shade, fill=shade)
        except Exception:
            pass

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
            dpg.delete_item("tl_fill")
            dpg.delete_item("tl_head")
            dpg.draw_rectangle((0, 0), (x, 20), color=C["amber_dim"],
                               fill=C["amber_faint"],
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
                        mx = int((m / dur) * 688)
                        dpg.draw_line((mx, 0), (mx, 20),
                                      color=C["blue"], thickness=2)
            except Exception:
                pass

    # ── Camera / gesture update (flicker-free) ────────────────────────────────
    def _upload_camera_frame(self, frame_bgr: np.ndarray):
        """
        Convert a BGR numpy frame to an RGBA float32 texture upload.

        Key fix for the flashing bug:
        - We reuse self._cam_buf (no per-frame allocation).
        - cv2.resize + cv2.cvtColor write into the existing buffer in one shot.
        - dpg.set_value replaces the texture contents atomically — DPG never
          sees a blank frame because we never delete and re-create the texture.
        - Upload is skipped on ticks where _tick % _CAM_UPLOAD_EVERY != 0.
        """
        if self._tick % self._CAM_UPLOAD_EVERY != 0:
            return
        try:
            resized = cv2.resize(frame_bgr, (self._cam_w, self._cam_h))
            rgba    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
            # Write float32 values into pre-allocated buffer
            np.copyto(self._cam_buf,
                      rgba.astype(np.float32) * (1.0 / 255.0))
            dpg.set_value("cam_texture", self._cam_buf.flatten().tolist())
        except Exception:
            pass

    def _update_gesture(self, snap: dict):
        # ── Camera feed ───────────────────────────────────────────────────────
        with self.state._lock:
            frame = self.state.latest_frame
        if frame is not None:
            self._upload_camera_frame(frame)

        # ── Gesture text — only reconfigure when value actually changes ────────
        name   = snap["gesture_name"]
        conf   = snap["gesture_confidence"]
        hold   = snap["gesture_hold_frames"]
        target = snap["gesture_hold_target"]
        hands  = snap["hands_detected"]
        active = snap["camera_active"]
        cmd    = snap["last_command"]

        icon  = GESTURE_ICONS.get(name, "[ ]")
        gcol  = (C["green"]    if name == "PLAY"
                 else C["blue"] if name == "PAUSE"
                 else C["text_dim"])

        if name != self._last_gesture or gcol != self._last_gesture_col:
            # Update text items in-place — no delete/redraw, no flash
            dpg.configure_item("gest_icon", default_value=icon,   color=gcol)
            dpg.configure_item("gest_name", default_value=name,   color=gcol)
            dpg.configure_item("gest_conf",
                               default_value=f"{conf:.0%}" if active else "—",
                               color=C["text_dim"])
            self._last_gesture     = name
            self._last_gesture_col = gcol

        dpg.set_value("hold_bar", min(1.0, hold / max(1, target)))
        dpg.configure_item("hold_bar", overlay=f"{hold} / {target}")
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
        content_hash = hash(tuple(
            (t["name"], t.get("bpm"), t.get("duration", 0.0))
            for t in tracks
        ))
        sig = (len(tracks), cur, self._queue_sel, content_hash)
        if sig != self._last_queue_sig:
            self._last_queue_sig = sig
            self._refresh_queue_table(tracks, cur)
        try:
            dpg.set_value("q_bpm_hint",
                          "" if self._queue_sel >= 0
                          else "← select a row, then SET BPM")
        except Exception:
            pass

    def _force_queue_redraw(self):
        self._last_queue_sig = (-1, -1, -1, -1)

    def _refresh_queue_table(self, tracks, current_idx):
        # Tear down the old table (if any) and the empty label
        try:
            dpg.delete_item("queue_table")
        except Exception:
            pass
        try:
            if tracks: dpg.hide_item("queue_empty_label")
            else:      dpg.show_item("queue_empty_label")
        except Exception:
            pass

        if not tracks:
            return

        # Build a fresh table inside the scroll window
        with dpg.table(
            tag="queue_table",
            parent="queue_list_outer",
            header_row=True,
            borders_innerV=True,
            borders_outerV=False,
            borders_outerH=False,
            row_background=False,
            resizable=True,
            policy=dpg.mvTable_SizingFixedFit,
            width=-1,
        ):
            dpg.add_table_column(label="#",        init_width_or_weight=28)
            dpg.add_table_column(label="File",     init_width_or_weight=260,
                                 width_stretch=True)
            dpg.add_table_column(label="BPM",      init_width_or_weight=52)
            dpg.add_table_column(label="Duration", init_width_or_weight=60)

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
                prefix  = "▶" if is_cur else f"{i+1}"

                with dpg.table_row():
                    # Column 0 — index / playing indicator
                    dpg.add_text(prefix, color=row_col)

                    # Column 1 — filename as a selectable spanning the cell
                    sel = dpg.add_selectable(
                        label=t["name"],
                        default_value=is_sel,
                        callback=self._cb_q_row,
                        user_data=i,
                        span_columns=False,
                    )
                    with dpg.theme() as rt:
                        with dpg.theme_component(dpg.mvSelectable):
                            dpg.add_theme_color(dpg.mvThemeCol_Text,   row_col)
                            dpg.add_theme_color(dpg.mvThemeCol_Header, C["select"])
                    dpg.bind_item_theme(sel, rt)

                    # Column 2 — BPM
                    dpg.add_text(bpm_str, color=row_col)

                    # Column 3 — duration
                    dpg.add_text(dur_str, color=C["text_dim"])

    def _update_log(self):
        if self._tick % 6 != 0:
            return
        lines = self.logger.lines(80)
        dpg.set_value("log_text",
                      "\n".join(f"[{ts}] [{lvl.upper():4s}] {msg}"
                                for ts, lvl, msg in lines))