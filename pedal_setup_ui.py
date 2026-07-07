from itertools import combinations

import dearpygui.dearpygui as dpg

from config import CFG
from pedal import (
    DEFAULT_HOLD_THRESHOLD_S,
    DEFAULT_STUCK_TIMEOUT_S,
    N_BUTTONS,
    event_name,
    get_pedal_map,
    set_pedal_command,
)

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
    "add_marker",
]

_C = {
    "bg": (22, 22, 22, 255),
    "panel": (28, 28, 28, 255),
    "border": (42, 42, 42, 255),
    "text": (212, 207, 200, 255),
    "dim": (110, 106, 98, 255),
    "amber": (239, 159, 39, 255),
    "green": (99, 197, 71, 255),
    "red": (226, 75, 74, 255),
    "cyan": (50, 210, 210, 255),
}


# Display order: singles first (B1, B2, ... ), then combos by size.
# Mirrors pedal.ALL_EVENT_NAMES generation but kept separate here so the
# UI's row order is an explicit, readable list rather than relying on
# import-time generation order from the logic module.
def _ordered_event_names() -> list[str]:
    names = []
    for n in range(1, N_BUTTONS + 1):
        for combo in combinations(range(1, N_BUTTONS + 1), n):
            for kind in ("TAP", "HOLD"):
                names.append(event_name(frozenset(combo), kind))
    return names


class PedalSetupUI:
    def _build_pedal_popup(self):
        self._pedal_last_sig: tuple = ()

        with dpg.window(
            label="Configuración de Pedal",
            tag="pedal_win",
            width=620,
            height=560,
            show=False,
            pos=(460, 140),
        ):
            dpg.add_text("PEDAL — 3 BOTONES (TAP / HOLD / COMBO)", color=_C["amber"])
            dpg.add_separator()
            dpg.add_spacer(height=6)

            # Live button indicators
            with dpg.group(horizontal=True):
                dpg.add_text("ESTADO EN VIVO:", color=_C["dim"])
                dpg.add_spacer(width=10)
                for i in range(1, N_BUTTONS + 1):
                    dpg.add_text(f"[B{i}]", tag=f"pedal_live_b{i}", color=_C["dim"])
                    dpg.add_spacer(width=8)
                dpg.add_spacer(width=16)
                dpg.add_text(
                    "—" if self.pedal is not None else "sin controlador conectado",
                    tag="pedal_live_status",
                    color=_C["dim"],
                )

            dpg.add_spacer(height=10)

            # Timing knobs
            with dpg.group(horizontal=True):
                dpg.add_text("Umbral tap/hold (s)", color=_C["dim"])
                dpg.add_input_float(
                    tag="pedal_hold_threshold",
                    default_value=CFG.get(
                        "pedal_hold_threshold_s", DEFAULT_HOLD_THRESHOLD_S
                    ),
                    min_value=0.05,
                    max_value=3.0,
                    step=0.05,
                    width=90,
                    format="%.2f",
                    callback=self._cb_pedal_threshold,
                )
                dpg.add_spacer(width=16)
                dpg.add_text("Timeout de seguridad (s)", color=_C["dim"])
                dpg.add_input_float(
                    tag="pedal_stuck_timeout",
                    default_value=CFG.get(
                        "pedal_stuck_timeout_s", DEFAULT_STUCK_TIMEOUT_S
                    ),
                    min_value=1.0,
                    max_value=30.0,
                    step=0.5,
                    width=90,
                    format="%.1f",
                    callback=self._cb_pedal_stuck_timeout,
                )

            dpg.add_spacer(height=10)
            dpg.add_text(
                "Un botón solo = TAP (rápido) o HOLD (mantenido).",
                color=_C["dim"],
            )
            dpg.add_text(
                "Dos o más botones que se superponen en el tiempo = COMBO. "
                "Si dentro de un combo algunos sueltan como tap y otros como "
                "hold, el evento se descarta (deben coincidir todos).",
                color=_C["dim"],
                wrap=580,
            )
            dpg.add_spacer(height=8)
            dpg.add_separator()
            dpg.add_spacer(height=8)

            dpg.add_text("MAPEO DE COMANDOS", color=_C["amber"])
            dpg.add_spacer(height=4)

            with dpg.table(
                tag="pedal_map_table",
                header_row=True,
                borders_innerV=True,
                borders_outerV=False,
                borders_outerH=False,
                resizable=False,
                policy=dpg.mvTable_SizingFixedFit,
                width=-1,
            ):
                dpg.add_table_column(label="Evento", init_width_or_weight=160)
                dpg.add_table_column(label="Comando", init_width_or_weight=200)

                pedal_map = get_pedal_map()
                for name in _ordered_event_names():
                    with dpg.table_row():
                        dpg.add_text(name, color=_C["text"])
                        dpg.add_combo(
                            items=_ALL_CMDS,
                            tag=f"pedal_map_combo_{name}",
                            default_value=pedal_map.get(name, "none"),
                            width=190,
                            callback=self._cb_pedal_map_changed,
                            user_data=name,
                        )

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=480)
                dpg.add_button(
                    label="Cerrar",
                    width=100,
                    callback=lambda: dpg.hide_item("pedal_win"),
                )

    # ── abrir / cerrar ──────────────────────────────────────────────

    def _cb_pedal_open(self):
        dpg.show_item("pedal_win")

    # ── callbacks de configuración ───────────────────────────────────

    def _cb_pedal_threshold(self, sender, app_data, user_data):
        CFG.set("pedal_hold_threshold_s", app_data)
        self.logger.info(f"pedal: umbral tap/hold -> {app_data:.2f}s")

    def _cb_pedal_stuck_timeout(self, sender, app_data, user_data):
        CFG.set("pedal_stuck_timeout_s", app_data)
        self.logger.info(f"pedal: timeout de seguridad -> {app_data:.1f}s")

    def _cb_pedal_map_changed(self, sender, app_data, user_data):
        event_name = user_data
        set_pedal_command(event_name, app_data)
        self.logger.info(f"pedal: {event_name} -> {app_data}")

    # ── actualización por frame ──────────────────────────────────────

    def _update_pedal_ui(self):
        if not dpg.is_item_shown("pedal_win"):
            return

        if self.pedal is None:
            return

        down = self.pedal.live_down_buttons()
        sig = frozenset(down)
        if sig == self._pedal_last_sig:
            return
        self._pedal_last_sig = sig

        for i in range(1, N_BUTTONS + 1):
            is_down = i in down
            dpg.configure_item(
                f"pedal_live_b{i}",
                color=_C["green"] if is_down else _C["dim"],
            )

        if down:
            label = "+".join(f"B{i}" for i in sorted(down))
            dpg.set_value("pedal_live_status", f"presionado: {label}")
            dpg.configure_item("pedal_live_status", color=_C["cyan"])
        else:
            dpg.set_value("pedal_live_status", "—")
            dpg.configure_item("pedal_live_status", color=_C["dim"])
