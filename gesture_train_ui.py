"""
Phantom Conductor — Gesture Trainer UI
========================================
A floating Dear PyGui popup window for recording, reviewing, and deleting
custom gesture templates.

Usage
-----
Mix GestureTrainUI into PhantomUI (or instantiate standalone):

    from gesture_train_ui import GestureTrainUI

    class PhantomUI(GestureTrainUI, ...):
        ...

    # In _build_ui / toolbar, add the toggle button:
    #   dpg.add_button(label=" TRAIN ", callback=self._cb_train_open)

    # In run() loop, call once per frame:
    #   self._update_train_ui()

Window layout
-------------
┌─ GESTURE TRAINER ──────────────────────────────────────────────────┐
│  [Name ________________]  [Command ▾]  [▶ START]  [✕ CANCEL]      │
│                                                                      │
│  ████████░░  4 / 10 samples  — hold hand still, click Capture       │
│  [  CAPTURE SAMPLE  ]                                                │
│                                                                      │
│  ─ Saved gestures ──────────────────────────────────────────────────│
│  NAME         CMD          SAMPLES   [DELETE]                       │
│  MY_WAVE      loop_next    10        [DELETE]                       │
│  THUMBS_UP    play          7        [DELETE]  ← partial            │
└────────────────────────────────────────────────────────────────────┘
"""

import dearpygui.dearpygui as dpg
from gesture_trainer import TRAINER, SAMPLES_NEEDED

_ALL_CMDS = [
    "play", "pause", "toggle",
    "next", "prev",
    "loop_toggle", "loop_next", "loop_prev",
    "none",
]

_C = {
    "bg":       (22,  22,  22,  255),
    "panel":    (28,  28,  28,  255),
    "border":   (42,  42,  42,  255),
    "text":     (212, 207, 200, 255),
    "dim":      (110, 106,  98, 255),
    "amber":    (239, 159,  39,  255),
    "green":    (99,  197,  71,  255),
    "red":      (226,  75,  74,  255),
    "cyan":     (50,  210, 210,  255),
    "red_faint":(45,   16,  16,  255),
}


class GestureTrainUI:
    """
    Mixin for PhantomUI.

    Requires self.state (PhantomState) and self.logger (Logger) to exist.
    Call _build_train_popup() once during setup, _update_train_ui() every frame.
    """

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_train_popup(self):
        """Create the hidden popup window.  Call once during UI setup."""
        self._train_last_sig: tuple = ()   # (recording_name, sample_count, n_gestures)

        with dpg.window(
            label="Gesture Trainer",
            tag="train_win",
            width=560, height=420,
            show=False,
            no_close=False,
            on_close=self._cb_train_close,
            pos=(420, 200),
        ):
            # ── Header row ────────────────────────────────────────────────────
            dpg.add_text("NEW GESTURE", color=_C["amber"])
            dpg.add_separator()
            dpg.add_spacer(height=4)

            with dpg.group(horizontal=True):
                with dpg.group(width=180):
                    dpg.add_text("Name (UPPER_SNAKE)", color=_C["dim"])
                    dpg.add_input_text(tag="train_name_input",
                                       hint="e.g. THUMBS_UP",
                                       width=172)
                dpg.add_spacer(width=8)
                with dpg.group(width=160):
                    dpg.add_text("Mapped command", color=_C["dim"])
                    dpg.add_combo(items=_ALL_CMDS,
                                  tag="train_cmd_combo",
                                  default_value="none",
                                  width=152)
                dpg.add_spacer(width=8)
                with dpg.group():
                    dpg.add_spacer(height=17)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label=" START ",
                                       tag="train_start_btn",
                                       callback=self._cb_train_start,
                                       width=72)
                        dpg.add_spacer(width=4)
                        dpg.add_button(label=" CANCEL ",
                                       tag="train_cancel_btn",
                                       callback=self._cb_train_cancel,
                                       width=76)

            dpg.add_spacer(height=10)

            # ── Progress area ─────────────────────────────────────────────────
            dpg.add_text("Hold your hand still in front of the camera,",
                         color=_C["dim"])
            dpg.add_text("then click Capture Sample for each of the 25 samples.",
                         color=_C["dim"])
            dpg.add_spacer(height=6)

            dpg.add_progress_bar(
                tag="train_progress",
                default_value=0.0,
                width=-1, height=14,
                overlay="0 / 25",
            )
            dpg.add_spacer(height=4)
            dpg.add_text("— not recording —",
                         tag="train_status_text",
                         color=_C["dim"])
            dpg.add_spacer(height=6)

            dpg.add_button(
                label="  CAPTURE SAMPLE  ",
                tag="train_capture_btn",
                callback=self._cb_train_capture,
                width=-1, height=36,
                enabled=False,
            )
            dpg.add_spacer(height=4)
            dpg.add_button(
                label="  SAVE GESTURE  ",
                tag="train_save_btn",
                callback=self._cb_train_save,
                width=-1, height=30,
                enabled=False,
            )

            dpg.add_spacer(height=10)
            dpg.add_separator()

            # ── Saved gesture list ────────────────────────────────────────────
            dpg.add_text("SAVED GESTURES", color=_C["amber"])
            dpg.add_spacer(height=4)
            with dpg.child_window(tag="train_list_outer", height=-1, border=False):
                dpg.add_text("— none saved yet —",
                             tag="train_empty_label",
                             color=_C["dim"])

        self._train_refresh_list()

    # ── Open / close ──────────────────────────────────────────────────────────

    def _cb_train_open(self):
        dpg.show_item("train_win")
        self._train_refresh_list()

    def _cb_train_close(self):
        if TRAINER.is_recording():
            TRAINER.cancel_recording()
            self.logger.info("trainer: cancelled (window closed)")
        self._train_reset_ui()

    # ── Recording controls ────────────────────────────────────────────────────

    def _cb_train_start(self):
        name = dpg.get_value("train_name_input").strip().upper()
        if not name:
            dpg.set_value("train_status_text", "⚠  Enter a gesture name first")
            dpg.configure_item("train_status_text", color=_C["red"])
            return
        if TRAINER.is_recording():
            TRAINER.cancel_recording()
        TRAINER.start_recording(name)
        dpg.configure_item("train_capture_btn", enabled=True)
        dpg.configure_item("train_save_btn",    enabled=False)
        dpg.set_value("train_status_text",
                      f"Recording '{name}'  —  0 / {SAMPLES_NEEDED} samples")
        dpg.configure_item("train_status_text", color=_C["cyan"])
        dpg.set_value("train_progress", 0.0)
        dpg.configure_item("train_progress", overlay=f"0 / {SAMPLES_NEEDED}")
        self.logger.info(f"trainer: started recording '{name}'")

    def _cb_train_cancel(self):
        if TRAINER.is_recording():
            name = TRAINER.recording_name() or "?"
            TRAINER.cancel_recording()
            self.logger.info(f"trainer: cancelled '{name}'")
        self._train_reset_ui()

    def _cb_train_capture(self):
        """Request one landmark snapshot from the vision thread."""
        if not TRAINER.is_recording():
            return
        # Set flag; vision thread will consume it on its next frame
        self.state.capture_sample_requested = True

    def _cb_train_save(self):
        if not TRAINER.is_recording():
            return
        if TRAINER.sample_count() == 0:
            dpg.set_value("train_status_text", "⚠  No samples captured yet")
            dpg.configure_item("train_status_text", color=_C["red"])
            return
        cmd  = dpg.get_value("train_cmd_combo")
        name = TRAINER.finish(command=cmd)
        self.logger.ok(f"trainer: saved '{name}' -> {cmd}")
        dpg.set_value("train_status_text",
                      f"✓  Saved '{name}'  ({cmd})")
        dpg.configure_item("train_status_text", color=_C["green"])
        self._train_reset_ui()
        self._train_refresh_list()

    # ── Per-frame update ──────────────────────────────────────────────────────

    def _update_train_ui(self):
        """Call once per render frame from PhantomUI.run()."""
        if not dpg.is_item_shown("train_win"):
            return

        rec   = TRAINER.is_recording()
        count = TRAINER.sample_count() if rec else 0
        glist = TRAINER.list_gestures()
        sig   = (TRAINER.recording_name(), count, len(glist))

        if sig == self._train_last_sig:
            return                    # nothing changed
        self._train_last_sig = sig

        if rec:
            frac = count / SAMPLES_NEEDED
            dpg.set_value("train_progress", frac)
            dpg.configure_item("train_progress",
                               overlay=f"{count} / {SAMPLES_NEEDED}")
            dpg.set_value("train_status_text",
                          f"Recording '{TRAINER.recording_name()}'"
                          f"  —  {count} / {SAMPLES_NEEDED} samples")
            # Enable Save once we have at least 3 samples (usable minimum)
            can_save = count >= 3
            dpg.configure_item("train_save_btn", enabled=can_save)

            if count >= SAMPLES_NEEDED:
                dpg.configure_item("train_capture_btn", enabled=False)
                dpg.set_value("train_status_text",
                              f"10 samples ready — click SAVE GESTURE")
                dpg.configure_item("train_status_text", color=_C["green"])

        self._train_refresh_list()

    # ── Gesture list ──────────────────────────────────────────────────────────

    def _train_refresh_list(self):
        """Rebuild the saved-gesture table inside the popup."""
        try:
            dpg.delete_item("train_gesture_table")
        except Exception:
            pass

        gestures = TRAINER.list_gestures()
        try:
            if gestures:
                dpg.hide_item("train_empty_label")
            else:
                dpg.show_item("train_empty_label")
        except Exception:
            pass

        if not gestures:
            return

        with dpg.table(
            tag="train_gesture_table",
            parent="train_list_outer",
            header_row=True,
            borders_innerV=True,
            borders_outerV=False,
            borders_outerH=False,
            resizable=True,
            policy=dpg.mvTable_SizingFixedFit,
            width=-1,
        ):
            dpg.add_table_column(label="Name",    init_width_or_weight=160)
            dpg.add_table_column(label="Command", init_width_or_weight=120)
            dpg.add_table_column(label="Samples", init_width_or_weight=70)
            dpg.add_table_column(label="",        init_width_or_weight=80)

            for g in gestures:
                name    = g["name"]
                cmd     = g["command"]
                samples = g["samples"]
                complete = samples >= SAMPLES_NEEDED

                with dpg.table_row():
                    dpg.add_text(name,
                                 color=_C["cyan"] if complete else _C["amber"])
                    dpg.add_text(cmd, color=_C["text"])
                    dpg.add_text(
                        f"{samples} / {SAMPLES_NEEDED}",
                        color=_C["green"] if complete else _C["amber"],
                    )
                    dpg.add_button(
                        label=" DEL ",
                        callback=self._cb_train_delete,
                        user_data=name,
                        width=56,
                    )

    def _cb_train_delete(self, sender, app_data, user_data):
        name = user_data
        TRAINER.delete(name)
        self.logger.info(f"trainer: deleted '{name}'")
        self._train_refresh_list()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _train_reset_ui(self):
        dpg.set_value("train_progress", 0.0)
        dpg.configure_item("train_progress", overlay="0 / 10")
        dpg.set_value("train_status_text", "— not recording —")
        dpg.configure_item("train_status_text", color=_C["dim"])
        dpg.configure_item("train_capture_btn", enabled=False)
        dpg.configure_item("train_save_btn",    enabled=False)