import math
import os
import threading
import time

import dearpygui.dearpygui as dpg
import numpy as np

from pedal_setup_ui import PedalSetupUI

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
from gesture_train_ui import GestureTrainUI
from logger import Logger
from state import PhantomState

#  PALETA DE COLORES

C = {
    "bg": (245, 245, 243, 255),
    "panel": (255, 255, 255, 255),
    "panel2": (250, 250, 248, 255),
    "border": (220, 220, 215, 255),
    "border2": (200, 200, 195, 255),
    "text": (30, 30, 30, 255),
    "text_dim": (130, 130, 125, 255),
    "amber": (210, 130, 20, 255),
    "amber_dim": (170, 100, 15, 255),
    "amber_faint": (255, 240, 210, 255),
    "green": (50, 160, 50, 255),
    "green_dim": (35, 120, 35, 255),
    "red": (200, 55, 55, 255),
    "red_faint": (255, 230, 230, 255),
    "blue": (45, 110, 200, 255),
    "blue_dim": (30, 80, 160, 255),
    "cyan": (30, 170, 170, 255),
    "select": (210, 230, 255, 255),
    "white": (255, 255, 255, 255),
}

GESTURE_ICONS = {"NO HAND": " - ", "PLAY": "[O]", "PAUSE": "[F]"}

CAM_W, CAM_H = 640, 480
_BLANK_TEXTURE = [0.0] * (CAM_W * CAM_H * 4)


#  UTILIDADES


def _fmt(s: float) -> str:
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def _sd_devices() -> tuple[list, list]:
    if not HAS_SD:
        stub = [(-1, "sounddevice no instalado")]
        return stub, stub
    inputs, outputs = [], []
    try:
        for i, d in enumerate(sd.query_devices()):
            lbl = f"{i}: {d['name']}"
            if d["max_input_channels"] > 0:
                inputs.append((i, lbl))
            if d["max_output_channels"] > 0:
                outputs.append((i, lbl))
    except Exception:
        pass
    if not inputs:
        inputs = [(-1, "No se encontró dispositivo de entrada")]
    if not outputs:
        outputs = [(-1, "No se encontró dispositivo de salida")]
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
            for key in ("TBPM", "bpm", "BPM", "TXXX:BPM", "----:com.apple.iTunes:BPM"):
                if key in tags:
                    raw = tags[key]
                    val = str(
                        raw[0]
                        if (hasattr(raw, "__iter__") and not isinstance(raw, str))
                        else raw
                    )
                    try:
                        bpm = float(val.strip())
                        break
                    except ValueError:
                        pass
    except Exception:
        pass
    return bpm, dur


#  PHANTOM UI


class PhantomUI(GestureTrainUI, PedalSetupUI):
    WIN_W, WIN_H = 1420, 900

    _CAM_UPLOAD_EVERY = 2

    def __init__(self, state, logger, pedal=None):
        self.state = state
        self.logger = logger
        self._tick = 0

        self._queue_sel: int = -1
        self._last_queue_sig: tuple = (-1, -1, -1, -1)

        self._in_devices: list = []
        self._out_devices: list = []
        self._in_sel: int = 0
        self._out_sel: int = 0

        self._cam_buf = np.zeros((CAM_H, CAM_W, 4), dtype=np.float32)
        self._last_gesture: str = ""
        self._last_gesture_col: tuple = C["text_dim"]
        self._last_bpm_source: str = ""

        self.pedal = pedal
        self._pedal_last_sig = ()

    # ── helpers de tema ───────────────────────────────────────────────────────
    def _btn(self, fg, bg, bd):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Text, fg)
                dpg.add_theme_color(dpg.mvThemeCol_Button, bg)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, bg)
                dpg.add_theme_color(dpg.mvThemeCol_Border, bd)
        return t

    # ── punto de entrada ──────────────────────────────────────────────────────
    def setup(self):
        dpg.create_context()
        print("STARTED UI THREAD")

        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                width=CAM_W,
                height=CAM_H,
                default_value=_BLANK_TEXTURE,
                tag="cam_texture",
            )
        self._cam_w = CAM_W
        self._cam_h = CAM_H

        dpg.create_viewport(
            title="Phantom Conductor",
            width=self.WIN_W,
            height=self.WIN_H,
            min_width=1100,
            min_height=750,
            resizable=True,
        )
        self._apply_theme()

        self._th_idle = self._btn(C["text"], C["panel2"], C["border2"])
        self._th_play = self._btn(C["green"], (21, 48, 16, 255), C["green_dim"])
        self._th_loop = self._btn(C["amber"], C["amber_faint"], C["amber_dim"])
        self._th_amb = self._btn(C["amber"], C["amber_faint"], C["amber_dim"])
        self._th_grn = self._btn(C["green"], (21, 48, 16, 255), C["green_dim"])
        self._th_red = self._btn(C["red"], C["red_faint"], (100, 40, 40, 255))
        self._th_blue = self._btn(C["blue"], (15, 30, 55, 255), C["blue_dim"])
        self._th_dim = self._btn(C["text_dim"], C["panel"], C["border"])

        self._in_devices, self._out_devices = _sd_devices()
        self._build_ui()
        self._setup_file_dialog()
        self._build_train_popup()
        self._build_pedal_popup()
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
            self._update_pedal_ui()
            self._update_queue_panel()
            self._update_io_status(snap)
            self._sync_gain()
            self._update_log()
            self._update_clock(snap)
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    # ── tema global ───────────────────────────────────────────────────────────
    def _apply_theme(self):
        with dpg.theme() as g:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, C["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, C["panel"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, C["panel2"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_Border, C["border"])
                dpg.add_theme_color(dpg.mvThemeCol_Text, C["text"])
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, C["panel"])
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, C["panel2"])
                dpg.add_theme_color(dpg.mvThemeCol_Button, C["panel2"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, C["border"])
                dpg.add_theme_color(dpg.mvThemeCol_Header, C["select"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, C["border"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, C["select"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, C["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, C["border2"])
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, C["panel2"])
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, C["amber"])
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, C["amber_dim"])
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 2)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
        dpg.bind_theme(g)

    # ── diálogo de archivos ───────────────────────────────────────────────────
    def _setup_file_dialog(self):
        with dpg.file_dialog(
            label="Agregar pista(s)",
            tag="file_dlg",
            width=740,
            height=540,
            show=False,
            callback=self._cb_file_dialog,
            cancel_callback=lambda s, a, u: None,
            file_count=100,
        ):
            dpg.add_file_extension(".*", color=C["text_dim"])
            dpg.add_file_extension(".wav", color=C["green"])
            dpg.add_file_extension(".mp3", color=C["amber"])
            dpg.add_file_extension(".flac", color=C["blue"])
            dpg.add_file_extension(".ogg", color=C["text"])

    def _cb_file_dialog(self, sender, app_data, user_data):
        self.logger.info(f"file_dlg claves app_data: {list(app_data.keys())}")
        current_path = app_data.get("current_path", "")
        selections = app_data.get("selections", {})
        file_path = app_data.get("file_path_name", "")

        candidates = []
        for display_name, sel_path in selections.items():
            if sel_path and os.path.sep in sel_path:
                candidates.append(sel_path)
            else:
                name = sel_path or display_name
                candidates.append(os.path.join(current_path, name))
        if file_path:
            candidates.append(
                file_path
                if os.path.isabs(file_path)
                else os.path.join(current_path, file_path)
            )

        seen, paths = set(), []
        for p in candidates:
            p = os.path.normpath(p)
            if p not in seen:
                seen.add(p)
                paths.append(p)

        added = 0
        for path in sorted(paths):
            if not os.path.isfile(path):
                self.logger.warn(f"omitiendo (no es un archivo): {path}")
                continue
            bpm, dur = _read_audio_meta(path)
            self.state.queue.add(path, bpm=bpm, duration=dur)
            flag = f"  BPM={bpm:.1f}" if bpm else "  BPM=? (ingresar manualmente)"
            self.logger.ok(f"agregado: {os.path.basename(path)}{flag}")
            added += 1

        if added == 0:
            self.logger.warn("no se encontraron archivos válidos en la selección")
        self._force_queue_redraw()

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

            # Título
            self._build_header()
            dpg.add_spacer(height=4)

            # =========================
            # Header superior fijo
            # =========================
            with dpg.child_window(
                tag="fixed_header",
                height=475,
                border=False,
                no_scrollbar=True,
            ):
                with dpg.group(horizontal=True):
                    # Columna izquierda
                    with dpg.child_window(
                        tag="header_left",
                        width=750,
                        height=-1,
                        border=False,
                        no_scrollbar=True,
                    ):
                        self._build_bpm_module()
                        dpg.add_spacer(height=4)

                    dpg.add_spacer(width=4)

                    # Columna derecha
                    with dpg.child_window(
                        tag="header_right",
                        width=-1,
                        height=-1,
                        border=False,
                        no_scrollbar=True,
                    ):
                        self._build_gesture_module()

            dpg.add_spacer(height=4)

            # =========================
            # Zona inferior scrollable
            # =========================
            with dpg.child_window(
                tag="scroll_body",
                height=300,
                border=False,
                horizontal_scrollbar=False,
            ):
                with dpg.group(horizontal=True):
                    # Stretch
                    with dpg.child_window(
                        tag="left_col",
                        width=600,
                        height=-1,
                        border=False,
                        no_scrollbar=True,
                    ):
                        self._build_stretch_module()

                    dpg.add_spacer(width=4)

                    # Cola
                    with dpg.child_window(
                        tag="right_col",
                        width=-1,
                        height=-1,
                        border=False,
                        no_scrollbar=True,
                    ):
                        self._build_queue_module()

            dpg.add_spacer(height=4)
            self._build_log_panel()
        self._build_settings_window()

    # ── header de título ──────────────────────────────────────────────────────
    def _build_header(self):
        with dpg.child_window(height=34, border=True, tag="hdr"):
            with dpg.group(horizontal=True):
                dpg.add_text("PHANTOM CONDUCTOR", color=C["amber"])
                dpg.add_text("  v0.5.3", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_text("●", tag="sys_led", color=C["green"])
                dpg.add_text("EN EJECUCIÓN", tag="sys_state", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_text("00:00:00", tag="sys_clock", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_button(
                    label=" CONFIGURACIÓN ",
                    callback=lambda: dpg.show_item("settings_window"),
                )
                dpg.add_button(
                    label=" ENTRENAR GESTOS ",
                    tag="btn_train_open",
                    callback=self._cb_train_open,
                    width=160,
                )
                dpg.add_button(
                    label=" PEDAL SETUP ",
                    tag="btn_pedal_open",
                    callback=self._cb_pedal_open,
                    width=140,
                )
                dpg.bind_item_theme("btn_train_open", self._th_blue)

    # ── header fijo: BPM ─────────────────────────────────────────────────────
    def _build_bpm_module(self):
        with dpg.child_window(height=-1, border=True, tag="bpm_panel", width=750):
            dpg.add_text("DETECCIÓN DE BPM", color=C["text_dim"])
            dpg.add_spacer(height=2)
            with dpg.group(horizontal=True):
                with dpg.group():
                    with dpg.drawlist(width=140, height=56, tag="bpm_draw"):
                        dpg.draw_text(
                            (0, 0),
                            "---",
                            tag="bpm_draw_text",
                            color=C["amber"],
                            size=50,
                        )
                dpg.add_spacer(width=16)
                with dpg.group():
                    dpg.add_text("FUENTE:", color=C["text_dim"])
                    dpg.add_text("entrada de audio", tag="bpm_source", color=C["text"])
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text("[AUDIO]", tag="pill_audio", color=C["green"])
                        dpg.add_spacer(width=6)
                        dpg.add_text("[TAP]", tag="pill_tap", color=C["text_dim"])
                        dpg.add_spacer(width=6)
                        dpg.add_text("[SYNC]", tag="pill_sync", color=C["text_dim"])
                    dpg.add_spacer(height=4)
                    dpg.add_text(
                        "bruto: ---  corr: ---", tag="bpm_debug", color=C["text_dim"]
                    )
                dpg.add_spacer(width=20)
                with dpg.group():
                    dpg.add_text("RATIO", color=C["text_dim"])
                    dpg.add_text("1.000", tag="ratio_val", color=C["amber"])
                    dpg.add_spacer(height=4)
                    dpg.add_text("ONSET", color=C["text_dim"])
                    dpg.add_text("0.000", tag="onset_val", color=C["text_dim"])

            dpg.add_spacer(height=4)
            with dpg.drawlist(width=688, height=12, tag="ratio_bar_draw"):
                dpg.draw_rectangle(
                    (0, 4),
                    (688, 10),
                    color=C["border"],
                    fill=C["panel2"],
                    tag="ratio_track",
                )
                dpg.draw_line(
                    (340, 2), (340, 14), color=C["border2"], tag="ratio_center"
                )
                dpg.draw_rectangle(
                    (340, 4),
                    (340, 10),
                    color=C["amber"],
                    fill=C["amber"],
                    tag="ratio_fill",
                )
            with dpg.group(horizontal=True):
                dpg.add_text("0.5×", color=C["text_dim"])
                dpg.add_spacer(width=289)
                dpg.add_text("1.0×", color=C["amber"])
                dpg.add_spacer(width=289)
                dpg.add_text("2.0×", color=C["text_dim"])

            dpg.add_text("NIVEL DE ENTRADA", color=C["text_dim"])
            with dpg.group(horizontal=True):
                with dpg.drawlist(width=590, height=50, tag="waveform_draw"):
                    pass
                dpg.add_spacer(width=6)
                with dpg.group():
                    dpg.add_text("RMS:", color=C["text_dim"])
                    dpg.add_text("0.000", tag="rms_val", color=C["green"])
                dpg.add_spacer(width=6)
            with dpg.group():
                dpg.add_text("PICO:", color=C["text_dim"])
                dpg.add_text("0.000", tag="peak_val", color=C["amber"])
                dpg.add_text("PISTA DE FONDO", color=C["text_dim"])
                with dpg.group(horizontal=True):
                    dpg.add_text(
                        "sin archivo cargado", tag="file_name", color=C["amber"]
                    )
                    dpg.add_spacer(width=10)
                    dpg.add_text("", tag="file_meta", color=C["text_dim"])
                dpg.add_spacer(height=4)
                with dpg.drawlist(width=688, height=20, tag="timeline_draw"):
                    dpg.draw_rectangle(
                        (0, 0),
                        (688, 20),
                        color=C["border"],
                        fill=C["panel2"],
                        tag="tl_bg",
                    )
                    dpg.draw_rectangle(
                        (0, 0),
                        (0, 20),
                        color=C["amber_dim"],
                        fill=C["amber_faint"],
                        tag="tl_fill",
                    )
                    dpg.draw_line((0, 0), (0, 20), color=C["amber"], tag="tl_head")
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label=" |<  ",
                        tag="btn_prev",
                        callback=self._cb_prev,
                        width=44,
                    )
                    dpg.bind_item_theme("btn_prev", self._th_dim)
                    dpg.add_button(
                        label=" REPRODUCIR  ",
                        tag="btn_play",
                        callback=self._cb_play,
                        width=100,
                    )
                    dpg.add_button(
                        label=" >|  ",
                        tag="btn_next",
                        callback=self._cb_next,
                        width=44,
                    )
                    dpg.bind_item_theme("btn_next", self._th_dim)
                    dpg.add_spacer(width=6)
                    dpg.add_button(
                        label=" LOOP ",
                        tag="btn_loop",
                        callback=self._cb_loop,
                        width=62,
                    )
                    dpg.add_spacer(width=6)
                    dpg.add_button(
                        label=" + MARCA ",
                        tag="btn_marker",
                        callback=self._cb_add_marker,
                        width=82,
                    )
                    dpg.bind_item_theme("btn_marker", self._th_dim)
                    dpg.add_spacer(width=80)
                    dpg.add_text("0:00 / 0:00", tag="time_display", color=C["text_dim"])

    # ─────────────────────────────────────────────────────────────────────────
    #  PANEL DE CONFIGURACIÓN  (columna izquierda, scrollable)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_settings_contents(self):
        in_names = [n for _, n in self._in_devices]
        out_names = [n for _, n in self._out_devices]

        dpg.add_text("CONFIGURACIÓN", color=C["text_dim"])
        dpg.add_separator()
        dpg.add_spacer(height=4)

        # Fila 1: dispositivos de audio I/O
        with dpg.group(horizontal=True):
            dpg.add_text("AUDIO", color=C["amber"])
            dpg.add_spacer(width=8)
            with dpg.group(width=255):
                dpg.add_text("Entrada (micrófono)", color=C["text_dim"])
                dpg.add_combo(
                    items=in_names,
                    tag="io_in_combo",
                    default_value=in_names[0] if in_names else "",
                    width=248,
                    callback=self._cb_io_in,
                )
            dpg.add_spacer(width=6)
            with dpg.group(width=255):
                dpg.add_text("Salida (altavoces)", color=C["text_dim"])
                dpg.add_combo(
                    items=out_names,
                    tag="io_out_combo",
                    default_value=out_names[0] if out_names else "",
                    width=248,
                    callback=self._cb_io_out,
                )
            dpg.add_spacer(width=6)
            with dpg.group():
                dpg.add_spacer(height=17)
                dpg.add_button(
                    label=" APLICAR ",
                    tag="io_apply_btn",
                    callback=self._cb_io_apply,
                    width=88,
                )
                dpg.bind_item_theme("io_apply_btn", self._th_grn)

        dpg.add_spacer(height=4)

        # Fila 2: sliders de ganancia
        with dpg.group(horizontal=True):
            dpg.add_text("GANANCIA", color=C["amber"])
            dpg.add_spacer(width=8)
            with dpg.group(width=180):
                dpg.add_text("Entrada (pre-ganancia mic)", color=C["text_dim"])
                dpg.add_slider_float(
                    tag="gain_input",
                    width=170,
                    default_value=CFG.input_gain,
                    min_value=0.0,
                    max_value=3.0,
                    format="%.2f",
                )
            dpg.add_spacer(width=10)
            with dpg.group(width=180):
                dpg.add_text("Salida (pista de fondo)", color=C["text_dim"])
                dpg.add_slider_float(
                    tag="io_gain",
                    width=170,
                    default_value=CFG.output_gain,
                    min_value=0.0,
                    max_value=2.0,
                    format="%.2f",
                )
            dpg.add_spacer(width=10)
            with dpg.group(horizontal=True):
                dpg.add_text("I/O:", color=C["text_dim"])
                dpg.add_text("●", tag="io_led", color=C["text_dim"])
                dpg.add_spacer(width=4)
                dpg.add_text("sin iniciar", tag="io_status", color=C["text_dim"])

        dpg.add_spacer(height=6)

        # Fila 3: dispositivo de video
        with dpg.group(horizontal=True):
            dpg.add_text("VIDEO", color=C["amber"])
            dpg.add_spacer(width=8)
            with dpg.group(width=220):
                dpg.add_text("Índice de cámara", color=C["text_dim"])
                dpg.add_input_int(
                    tag="cfg_cam_index",
                    default_value=CFG.cam_index,
                    min_value=0,
                    max_value=15,
                    width=90,
                    callback=self._cb_cam_index,
                )
            dpg.add_spacer(width=10)
            dpg.add_text("(requiere reinicio para aplicar)", color=C["text_dim"])

        dpg.add_spacer(height=6)
        dpg.add_checkbox(
            label="Omitir frames (inferencia)",
            callback=lambda s, a: CFG.set("inference_skip_enabled", a),
            default_value=CFG.inference_skip_enabled,
        )
        dpg.add_slider_int(
            label="Ejecutar cada N frames",
            min_value=1,
            max_value=6,
            default_value=CFG.inference_skip_frames,
            callback=lambda s, a: CFG.set("inference_skip_frames", a),
        )
        dpg.add_spacer(height=6)

        # Fila 4: mapeador de gestos
        _ALL_CMDS = [
            "play",
            "pause",
            "toggle",
            "next",
            "prev",
            "loop_toggle",
            "loop_next",
            "loop_prev",
            "none",
        ]

        with dpg.group(horizontal=True):
            dpg.add_text("GESTOS", color=C["amber"])
            dpg.add_spacer(width=8)
            with dpg.group(width=154):
                dpg.add_text("Mano abierta (5)", color=C["text_dim"])
                dpg.add_combo(
                    items=_ALL_CMDS,
                    tag="gmap_play_combo",
                    default_value=CFG.gesture_map.get("PLAY", "play"),
                    width=146,
                    callback=lambda s, a, u: self._cb_gesture_map("PLAY", a),
                )
            dpg.add_spacer(width=4)
            with dpg.group(width=154):
                dpg.add_text("Puño (0)", color=C["text_dim"])
                dpg.add_combo(
                    items=_ALL_CMDS,
                    tag="gmap_pause_combo",
                    default_value=CFG.gesture_map.get("PAUSE", "pause"),
                    width=146,
                    callback=lambda s, a, u: self._cb_gesture_map("PAUSE", a),
                )
            dpg.add_spacer(width=4)
            with dpg.group(width=154):
                dpg.add_text("Índice / Señalar (1)", color=C["text_dim"])
                dpg.add_combo(
                    items=_ALL_CMDS,
                    tag="gmap_point_combo",
                    default_value=CFG.gesture_map.get("POINT", "next"),
                    width=146,
                    callback=lambda s, a, u: self._cb_gesture_map("POINT", a),
                )
            dpg.add_spacer(width=4)
            with dpg.group(width=154):
                dpg.add_text("Paz / V (2)", color=C["text_dim"])
                dpg.add_combo(
                    items=_ALL_CMDS,
                    tag="gmap_peace_combo",
                    default_value=CFG.gesture_map.get("PEACE", "loop_toggle"),
                    width=146,
                    callback=lambda s, a, u: self._cb_gesture_map("PEACE", a),
                )
            dpg.add_spacer(width=6)
            with dpg.group(width=130):
                dpg.add_text("Frames de hold", color=C["text_dim"])
                dpg.add_input_int(
                    tag="cfg_hold_frames",
                    default_value=CFG.gesture_hold_frames,
                    min_value=1,
                    max_value=60,
                    width=80,
                    callback=self._cb_hold_frames,
                )
                dpg.add_checkbox(
                    label=" Pedal",
                    tag="cfg_use_pedal",
                    default_value=CFG.use_pedal,
                    callback=self._cb_use_pedal,
                )
                dpg.add_checkbox(
                    label=" Tempo tapper",
                    tag="cfg_use_tapper",
                    default_value=CFG.use_tempo_tapper,
                    callback=self._cb_use_tapper,
                )
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=6)

        with dpg.group(horizontal=True):
            dpg.add_spacer(width=560)
            dpg.add_button(
                label="Cerrar",
                width=100,
                callback=lambda: dpg.hide_item("settings_window"),
            )

    def _build_settings_window(self):
        with dpg.window(
            label="Configuración",
            tag="settings_window",
            show=False,
            width=950,
            height=500,
        ):
            self._build_settings_contents()

    # ── time-stretch (columna izquierda, scrollable) ──────────────────────────
    def _build_stretch_module(self):
        """Editor de BPM de referencia, buffer fill y suavizado α."""
        with dpg.child_window(height=-1, border=True, tag="ctrl_panel"):
            dpg.add_text("TIME-STRETCH / BPM DE REFERENCIA", color=C["text_dim"])
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                with dpg.group(width=100):
                    dpg.add_text("BPM ref de pista (editable)", color=C["text_dim"])
                    with dpg.group(horizontal=True):
                        dpg.add_input_float(
                            tag="ctrl_bpm_input",
                            default_value=120.0,
                            min_value=20.0,
                            max_value=300.0,
                            step=0.5,
                            step_fast=5.0,
                            width=75,
                            format="%.1f",
                        )
                        dpg.add_button(
                            label=" FIJAR ",
                            tag="ctrl_bpm_set",
                            callback=self._cb_set_ref_bpm,
                            width=10,
                        )
                        dpg.bind_item_theme("ctrl_bpm_set", self._th_amb)
                dpg.add_spacer(width=10)
                with dpg.group(width=110):
                    dpg.add_text("Stretch", color=C["text_dim"])
                    dpg.add_text("pyrubberband", color=C["text"])
                with dpg.group(width=110):
                    dpg.add_text("Buffer fill", color=C["text_dim"])
                    dpg.add_text("0%", tag="ctrl_buf", color=C["text"])
                with dpg.group(width=90):
                    dpg.add_text("SR", color=C["text_dim"])
                    dpg.add_text("44100 Hz", color=C["text"])
                with dpg.group(width=100):
                    dpg.add_text("Suavizado α", color=C["text_dim"])
                    dpg.add_slider_float(
                        tag="slider_alpha",
                        default_value=CFG.smooth_alpha,
                        min_value=0.0,
                        max_value=1.0,
                        width=90,
                    )

    # ─────────────────────────────────────────────────────────────────────────
    #  PANEL DE GESTOS / CÁMARA  (columna derecha, sin parpadeo)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_gesture_module(self):
        with dpg.child_window(width=-1, height=290, border=True, tag="gesture_panel"):
            dpg.add_text("CONTROL DE GESTOS", color=C["text_dim"])
            dpg.add_separator()
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_image("cam_texture", width=213, height=160, tag="cam_feed")
                dpg.add_spacer(width=10)
                with dpg.group():
                    dpg.add_text("[ ]", tag="gest_icon", color=C["text_dim"])
                    dpg.add_text("SIN MANO", tag="gest_name", color=C["text_dim"])
                    dpg.add_text("-", tag="gest_conf", color=C["text_dim"])
                dpg.add_spacer(width=10)
                with dpg.group():
                    dpg.add_text("HOLD", color=C["text_dim"])
                    dpg.add_progress_bar(
                        tag="hold_bar",
                        default_value=0.0,
                        width=130,
                        height=8,
                        overlay="0 / 8",
                    )
                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text("CAM:", color=C["text_dim"])
                        dpg.add_text("●", tag="cam_led", color=C["text_dim"])
                        dpg.add_text(
                            "DESCONECTADA", tag="cam_state", color=C["text_dim"]
                        )
                    dpg.add_text("MANOS: 0", tag="cam_hands", color=C["text"])
                    dpg.add_text("CMD:   -", tag="cam_lastcmd", color=C["text"])
                    dpg.add_spacer(height=4)
                    dpg.add_text("Mano abierta = REPRODUCIR", color=C["green"])
                    dpg.add_text("Puño         = PAUSAR", color=C["blue"])
                    dpg.add_spacer(height=4)
                    dpg.add_button(
                        label=" LIMPIAR CMD ",
                        tag="btn_clear_cmd",
                        callback=lambda: self.state.set_command(None),
                        width=120,
                    )
                    dpg.bind_item_theme("btn_clear_cmd", self._th_red)

    # ── cola de pistas (columna derecha, ocupa el resto) ─────────────────────
    def _build_queue_module(self):
        with dpg.child_window(width=-1, height=-1, border=True, tag="queue_panel"):
            dpg.add_text("COLA DE PISTAS", color=C["text_dim"])
            dpg.add_separator()
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label=" AGREGAR ",
                    tag="btn_add",
                    callback=lambda: dpg.show_item("file_dlg"),
                    width=80,
                )
                dpg.bind_item_theme("btn_add", self._th_grn)
                dpg.add_button(
                    label="subir", tag="btn_q_up", callback=self._cb_q_up, width=36
                )
                dpg.bind_item_theme("btn_q_up", self._th_dim)
                dpg.add_button(
                    label="bajar",
                    tag="btn_q_down",
                    callback=self._cb_q_down,
                    width=36,
                )
                dpg.bind_item_theme("btn_q_down", self._th_dim)
                dpg.add_button(
                    label=" CARGAR ",
                    tag="btn_q_load",
                    callback=self._cb_q_load,
                    width=84,
                )
                dpg.bind_item_theme("btn_q_load", self._th_amb)
                dpg.add_button(
                    label=" QUITAR ",
                    tag="btn_q_rem",
                    callback=self._cb_q_remove,
                    width=76,
                )
                dpg.bind_item_theme("btn_q_rem", self._th_red)
                dpg.add_button(
                    label=" LIMPIAR TODO ",
                    tag="btn_q_clear",
                    callback=self._cb_q_clear,
                    width=100,
                )
                dpg.bind_item_theme("btn_q_clear", self._th_red)
                dpg.add_spacer(width=8)
                dpg.add_text(
                    "guardado automático",
                    tag="queue_save_indicator",
                    color=C["text_dim"],
                )

            dpg.add_spacer(height=6)
            with dpg.child_window(height=75, border=True, tag="bpm_edit_box"):
                dpg.add_text("EDITOR DE BPM DE PISTA", color=C["text_dim"])
                with dpg.group(horizontal=True):
                    dpg.add_text("BPM:", color=C["text_dim"])
                    dpg.add_input_float(
                        tag="q_bpm_input",
                        default_value=120.0,
                        min_value=20.0,
                        max_value=300.0,
                        step=0.5,
                        step_fast=5.0,
                        width=112,
                        format="%.1f",
                    )
                    dpg.add_button(
                        label=" FIJAR BPM ",
                        tag="q_bpm_set",
                        callback=self._cb_q_set_bpm,
                        width=84,
                    )
                    dpg.bind_item_theme("q_bpm_set", self._th_amb)
                    dpg.add_spacer(width=6)
                    dpg.add_text(
                        "← selecciona una fila y luego FIJAR BPM",
                        tag="q_bpm_hint",
                        color=C["text_dim"],
                    )

            dpg.add_spacer(height=4)

            with dpg.child_window(
                tag="queue_list_outer",
                height=-1,
                border=False,
                horizontal_scrollbar=False,
            ):
                dpg.add_text(" vacío ", tag="queue_empty_label", color=C["text_dim"])

    # ── log ───────────────────────────────────────────────────────────────────
    def _build_log_panel(self):
        with dpg.child_window(height=130, border=True, tag="log_panel"):
            with dpg.group(horizontal=True):
                dpg.add_text("LOG DEL SISTEMA", color=C["text_dim"])
                dpg.add_spacer(width=20)
                dpg.add_button(
                    label=" LMP ",
                    tag="btn_clr_log",
                    callback=lambda: self.logger.clear(),
                    width=50,
                )
                dpg.bind_item_theme("btn_clr_log", self._th_amb)
            dpg.add_separator()
            with dpg.child_window(tag="log_scroll", height=-1, border=False):
                with dpg.theme() as _log_input_theme:
                    with dpg.theme_component(dpg.mvInputText):
                        dpg.add_theme_color(
                            dpg.mvThemeCol_FrameBg,
                            C["panel"],
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_color(
                            dpg.mvThemeCol_FrameBgHovered,
                            C["panel"],
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_color(
                            dpg.mvThemeCol_FrameBgActive,
                            C["panel"],
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_color(
                            dpg.mvThemeCol_Border,
                            (0, 0, 0, 0),
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_color(
                            dpg.mvThemeCol_Text,
                            C["text_dim"],
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_style(
                            dpg.mvStyleVar_FramePadding,
                            0,
                            0,
                            category=dpg.mvThemeCat_Core,
                        )
                dpg.add_input_text(
                    tag="log_text",
                    default_value="",
                    multiline=True,
                    readonly=True,
                    width=-1,
                    height=-1,
                    tab_input=False,
                )
                dpg.bind_item_theme("log_text", _log_input_theme)

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
                self.logger.info(f"carga automática: {track['name']}")
            return
        new_state = self.state.toggle()
        self.logger.log(
            f"transporte: {'REPRODUCIR' if new_state else 'PAUSAR'}",
            "ok" if new_state else "warn",
        )

    def _cb_prev(self):
        track = self.state.queue.prev_track()
        if track:
            with self.state._lock:
                self.state.load_new_track = track
            self.logger.info(f"transporte: anterior → {track['name']}")

    def _cb_next(self):
        track = self.state.queue.next_track()
        if track:
            with self.state._lock:
                self.state.load_new_track = track
            self.logger.info(f"transporte: siguiente → {track['name']}")

    def _cb_loop(self):
        with self.state._lock:
            self.state.is_looping = not self.state.is_looping
            looping = self.state.is_looping
        self.logger.log(
            f"loop: {'ACTIVADO' if looping else 'DESACTIVADO'}",
            "ok" if looping else "info",
        )

    def _cb_add_marker(self):
        with self.state._lock:
            pos = self.state.track_position
            self.state.markers.append(pos)
        self.logger.info(f"marca en {pos:.2f}s")

    def _cb_io_in(self, sender, app_data, user_data):
        self._in_sel = next(
            (i for i, (_, n) in enumerate(self._in_devices) if n == app_data), 0
        )

    def _cb_io_out(self, sender, app_data, user_data):
        self._out_sel = next(
            (i for i, (_, n) in enumerate(self._out_devices) if n == app_data), 0
        )

    def _cb_io_apply(self):
        dev_in = self._in_devices[self._in_sel][0] if self._in_devices else None
        dev_out = self._out_devices[self._out_sel][0] if self._out_devices else None
        if isinstance(dev_in, int) and dev_in < 0:
            dev_in = None
        if isinstance(dev_out, int) and dev_out < 0:
            dev_out = None
        in_gain = dpg.get_value("gain_input")
        out_gain = dpg.get_value("io_gain")
        self.state.request_io_restart(
            dev_in, dev_out, input_gain=in_gain, output_gain=out_gain
        )
        in_lbl = (
            self._in_devices[self._in_sel][1] if self._in_devices else "predeterminado"
        )
        out_lbl = (
            self._out_devices[self._out_sel][1]
            if self._out_devices
            else "predeterminado"
        )
        self.logger.ok(
            f"I/O aplicado: entrada=[{in_lbl}]  salida=[{out_lbl}]  "
            f"ganancia_entrada={in_gain:.2f}  ganancia_salida={out_gain:.2f}"
        )
        dpg.configure_item("io_led", color=C["amber"])
        dpg.set_value("io_status", "aplicando…")

    def _cb_cam_index(self, sender, app_data, user_data):
        CFG.set("cam_index", app_data)
        self.logger.info(f"índice de cámara → {app_data}  (reiniciar para aplicar)")

    def _cb_gesture_map(self, gesture_key: str, value: str):
        gmap = dict(CFG.gesture_map)
        gmap[gesture_key] = value
        CFG.set("gesture_map", gmap)
        self.logger.info(f"mapa de gestos: {gesture_key} → {value}")

    def _cb_hold_frames(self, sender, app_data, user_data):
        CFG.set("gesture_hold_frames", app_data)
        with self.state._lock:
            self.state.gesture_hold_target = app_data
        self.logger.info(f"frames de hold → {app_data}")

    def _cb_use_pedal(self, sender, app_data, user_data):
        CFG.set("use_pedal", app_data)
        self.logger.info(f"pedal de pie: {'ACTIVADO' if app_data else 'DESACTIVADO'}")

    def _cb_use_tapper(self, sender, app_data, user_data):
        CFG.set("use_tempo_tapper", app_data)
        self.logger.info(
            f"tempo tapper: {'ACTIVADO - escritura de BPM de audio pausada' if app_data else 'DESACTIVADO - BPM de audio reanudado'}"
        )

    def _cb_set_ref_bpm(self):
        bpm = dpg.get_value("ctrl_bpm_input")
        if bpm and bpm > 0:
            self.state.set_bpm_original(bpm)
            if self._queue_sel >= 0:
                self.state.queue.set_bpm(self._queue_sel, bpm)
            self.logger.ok(f"BPM ref → {bpm:.1f}")
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
        if self._queue_sel < 0:
            return
        tracks, _ = self.state.queue.snapshot()
        if self._queue_sel < len(tracks):
            name = tracks[self._queue_sel]["name"]
            self.state.queue.remove(self._queue_sel)
            tracks2, _ = self.state.queue.snapshot()
            self._queue_sel = min(self._queue_sel, len(tracks2) - 1)
            self.logger.info(f"cola: eliminado {name}")
            self._force_queue_redraw()

    def _cb_q_clear(self):
        self.state.queue.clear()
        self._queue_sel = -1
        self.logger.info("cola: limpiada")
        self._force_queue_redraw()

    def _cb_q_load(self):
        if self._queue_sel < 0:
            return
        track = self.state.queue.select(self._queue_sel)
        if track:
            with self.state._lock:
                self.state.load_new_track = track
            if track.get("bpm"):
                dpg.set_value("q_bpm_input", track["bpm"])
                dpg.set_value("ctrl_bpm_input", track["bpm"])
            self.logger.ok(f"cola: cargando → {track['name']}")

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
            self.logger.warn("selecciona primero una fila de pista")
            return
        bpm = dpg.get_value("q_bpm_input")
        if not bpm or bpm <= 0:
            return
        self.state.queue.set_bpm(self._queue_sel, bpm)
        _, current_idx = self.state.queue.snapshot()
        if self._queue_sel == current_idx:
            self.state.set_bpm_original(bpm)
            dpg.set_value("ctrl_bpm_input", bpm)
        tracks, _ = self.state.queue.snapshot()
        name = tracks[self._queue_sel]["name"] if self._queue_sel < len(tracks) else "?"
        self.logger.ok(f"BPM fijado: {name} → {bpm:.1f}")
        self._force_queue_redraw()

    # ═════════════════════════════════════════════════════════════════════════
    #  ACTUALIZACIONES POR FRAME
    # ═════════════════════════════════════════════════════════════════════════

    def _sync_gain(self):
        out_gain = dpg.get_value("io_gain")
        with self.state._lock:
            self.state.gain = out_gain

    def _update_io_status(self, snap: dict):
        if snap.get("io_restart_requested"):
            dpg.configure_item("io_led", color=C["amber"])
            dpg.set_value("io_status", "aplicando…")
        else:
            dev_in = snap.get("dev_in")
            dev_out = snap.get("dev_out")
            if dev_in is not None or dev_out is not None:
                dpg.configure_item("io_led", color=C["green"])
                dpg.set_value(
                    "io_status", f"activo  entrada={dev_in}  salida={dev_out}"
                )
            else:
                dpg.configure_item("io_led", color=C["text_dim"])
                dpg.set_value(
                    "io_status",
                    "sin iniciar - selecciona dispositivos y haz clic en APLICAR",
                )

    def _update_clock(self, snap):
        e = int(time.time() - snap["start_time"])
        dpg.set_value(
            "sys_clock", f"{e // 3600:02d}:{(e % 3600) // 60:02d}:{e % 60:02d}"
        )

    def _update_bpm(self, snap):
        bpm = snap["bpm_live"]
        orig = snap["bpm_original"]
        ratio = snap["stretch_ratio"]
        onset = snap["onset_max"]
        raw = snap["bpm_raw"]
        corr = snap["bpm_corrected"]
        source = snap.get("bpm_source", "audio")

        bpm_str = f"{bpm:.1f}" if bpm else "---"
        synced = bool(bpm and abs(ratio - 1.0) < 0.03)
        col = C["green"] if synced else C["amber"]

        try:
            dpg.delete_item("bpm_draw_text")
            dpg.draw_text(
                (0, 0),
                bpm_str,
                parent="bpm_draw",
                tag="bpm_draw_text",
                color=col,
                size=50,
            )
        except Exception:
            pass

        dpg.set_value(
            "bpm_source", "entrada tap" if source == "tap" else "entrada de audio"
        )
        dpg.set_value("ratio_val", f"{ratio:.3f}")
        dpg.set_value("onset_val", f"{onset:.3f}")
        dpg.set_value("ctrl_buf", f"{int(snap['buffer_fill'] * 100)}%")
        if raw and corr:
            dpg.set_value("bpm_debug", f"bruto: {raw:.1f}  corr: {corr:.1f}")
        dpg.configure_item("pill_sync", color=C["green"] if synced else C["text_dim"])

        if source != self._last_bpm_source:
            tapper_enabled = CFG.get("use_tempo_tapper", False)
            if source == "tap":
                audio_col, tap_col = C["text_dim"], C["cyan"]
            else:
                audio_col = C["green"]
                tap_col = C["amber_dim"] if tapper_enabled else C["text_dim"]
            dpg.configure_item("pill_audio", color=audio_col)
            dpg.configure_item("pill_tap", color=tap_col)
            self._last_bpm_source = source

        try:
            if not dpg.is_item_focused("ctrl_bpm_input"):
                dpg.set_value("ctrl_bpm_input", orig)
        except Exception:
            pass

        def rx(r):
            return 340 + (math.log2(max(0.5, min(2.0, r))) * 340)

        x = rx(ratio)
        fc = C["green"] if synced else (C["amber"] if ratio >= 1.0 else C["blue_dim"])
        try:
            dpg.delete_item("ratio_fill")
            lo, hi = (340, x) if ratio >= 1.0 else (x, 340)
            dpg.draw_rectangle(
                (lo, 4),
                (hi, 10),
                color=fc,
                fill=fc,
                parent="ratio_bar_draw",
                tag="ratio_fill",
            )
        except Exception:
            pass

    def _update_waveform(self, snap):
        wave = snap["waveform"][-64:]
        rms = snap["rms"]
        peak = snap["peak"]
        dpg.set_value("rms_val", f"{rms:.3f}")
        dpg.set_value("peak_val", f"{peak:.3f}")
        dpg.configure_item("rms_val", color=C["green"] if rms > 0.01 else C["text_dim"])
        dpg.configure_item("peak_val", color=C["red"] if peak > 0.9 else C["amber"])
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
                    shade = (
                        C["amber"]
                        if v > 0.7
                        else C["amber_dim"]
                        if v > 0.3
                        else C["amber_faint"]
                    )
                    dpg.draw_rectangle(
                        (x, 25 - h // 2),
                        (x + max(1, bw - 1), 25 + h // 2),
                        color=shade,
                        fill=shade,
                    )
        except Exception:
            pass

    def _update_transport(self, snap):
        playing = snap["is_playing"]
        looping = snap["is_looping"]
        pos = snap["track_position"]
        dur = snap["track_duration"]
        fname = snap["track_path"]
        orig = snap["bpm_original"]

        dpg.configure_item("btn_play", label=" PAUSAR " if playing else " REPRODUCIR  ")
        dpg.bind_item_theme("btn_play", self._th_play if playing else self._th_idle)
        dpg.bind_item_theme("btn_loop", self._th_loop if looping else self._th_idle)

        if fname:
            dpg.set_value("file_name", os.path.basename(fname))
            dpg.set_value("file_meta", f"BPM ref: {orig:.1f}  ·  {_fmt(dur)}")

        dpg.set_value("time_display", f"{_fmt(pos)} / {_fmt(dur)}")

        x = int((pos / dur) * 688) if dur > 0 else 0
        try:
            dpg.delete_item("tl_fill")
            dpg.delete_item("tl_head")
            dpg.draw_rectangle(
                (0, 0),
                (x, 20),
                color=C["amber_dim"],
                fill=C["amber_faint"],
                parent="timeline_draw",
                tag="tl_fill",
            )
            dpg.draw_line(
                (x, 0), (x, 20), color=C["amber"], parent="timeline_draw", tag="tl_head"
            )
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
                        dpg.draw_line((mx, 0), (mx, 20), color=C["blue"], thickness=2)
            except Exception:
                pass

    def _upload_camera_frame(self, frame_bgr: np.ndarray):
        """Convierte un frame BGR a textura RGBA float32, reutilizando el buffer."""
        if self._tick % self._CAM_UPLOAD_EVERY != 0:
            return
        try:
            resized = cv2.resize(frame_bgr, (self._cam_w, self._cam_h))
            rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
            np.copyto(self._cam_buf, rgba.astype(np.float32) * (1.0 / 255.0))
            dpg.set_value("cam_texture", self._cam_buf.flatten().tolist())
        except Exception:
            pass

    def _update_gesture(self, snap: dict):
        with self.state._lock:
            frame = self.state.latest_frame
        if frame is not None:
            self._upload_camera_frame(frame)

        name = snap["gesture_name"]
        conf = snap["gesture_confidence"]
        hold = snap["gesture_hold_frames"]
        target = snap["gesture_hold_target"]
        hands = snap["hands_detected"]
        active = snap["camera_active"]
        cmd = snap["last_command"]

        icon = GESTURE_ICONS.get(name, "[ ]")
        gcol = (
            C["green"]
            if name == "PLAY"
            else C["blue"]
            if name == "PAUSE"
            else C["text_dim"]
        )

        if name != self._last_gesture or gcol != self._last_gesture_col:
            dpg.configure_item("gest_icon", default_value=icon, color=gcol)
            dpg.configure_item("gest_name", default_value=name, color=gcol)
            dpg.configure_item(
                "gest_conf",
                default_value=f"{conf:.0%}" if active else "-",
                color=C["text_dim"],
            )
            self._last_gesture = name
            self._last_gesture_col = gcol

        dpg.set_value("hold_bar", min(1.0, hold / max(1, target)))
        dpg.configure_item("hold_bar", overlay=f"{hold} / {target}")
        dpg.configure_item("cam_led", color=C["green"] if active else C["text_dim"])
        dpg.set_value("cam_state", "ACTIVA" if active else "EN ESPERA")
        dpg.configure_item("cam_state", color=C["green"] if active else C["text_dim"])
        dpg.set_value("cam_hands", f"MANOS: {hands}")
        dpg.set_value("cam_lastcmd", f"CMD:   {cmd}" if cmd else "CMD:   -")
        dpg.configure_item("cam_lastcmd", color=C["green"] if cmd else C["text"])

    def _update_queue_panel(self):
        tracks, cur = self.state.queue.snapshot()
        content_hash = hash(
            tuple((t["name"], t.get("bpm"), t.get("duration", 0.0)) for t in tracks)
        )
        sig = (len(tracks), cur, self._queue_sel, content_hash)
        if sig != self._last_queue_sig:
            self._last_queue_sig = sig
            self._refresh_queue_table(tracks, cur)
        try:
            dpg.set_value(
                "q_bpm_hint",
                "" if self._queue_sel >= 0 else "selecciona una fila y luego FIJAR BPM",
            )
        except Exception:
            pass

    def _force_queue_redraw(self):
        self._last_queue_sig = (-1, -1, -1, -1)

    def _refresh_queue_table(self, tracks, current_idx):
        try:
            dpg.delete_item("queue_table")
        except Exception:
            pass
        try:
            if tracks:
                dpg.hide_item("queue_empty_label")
            else:
                dpg.show_item("queue_empty_label")
        except Exception:
            pass

        if not tracks:
            return

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
            dpg.add_table_column(label="#", init_width_or_weight=28)
            dpg.add_table_column(
                label="Archivo", init_width_or_weight=260, width_stretch=True
            )
            dpg.add_table_column(label="BPM", init_width_or_weight=52)
            dpg.add_table_column(label="Duración", init_width_or_weight=60)

            for i, t in enumerate(tracks):
                is_cur = i == current_idx
                is_sel = i == self._queue_sel
                bpm = t.get("bpm")
                bpm_str = f"{bpm:.0f}" if bpm is not None else "??"
                dur = t.get("duration", 0.0)
                dur_str = f"{int(dur) // 60}:{int(dur) % 60:02d}" if dur else "-"
                row_col = (
                    C["amber"] if is_cur else C["white"] if is_sel else C["text_dim"]
                )
                prefix = "->" if is_cur else f"{i + 1}"

                with dpg.table_row():
                    dpg.add_text(prefix, color=row_col)
                    sel = dpg.add_selectable(
                        label=t["name"],
                        default_value=is_sel,
                        callback=self._cb_q_row,
                        user_data=i,
                        span_columns=False,
                    )
                    with dpg.theme() as rt:
                        with dpg.theme_component(dpg.mvSelectable):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, row_col)
                            dpg.add_theme_color(dpg.mvThemeCol_Header, C["select"])
                    dpg.bind_item_theme(sel, rt)
                    dpg.add_text(bpm_str, color=row_col)
                    dpg.add_text(dur_str, color=C["text_dim"])

    def _update_log(self):
        if self._tick % 6 != 0:
            return
        lines = self.logger.lines(80)
        dpg.set_value(
            "log_text",
            "\n".join(f"[{ts}] [{lvl.upper():4s}] {msg}" for ts, lvl, msg in lines),
        )
