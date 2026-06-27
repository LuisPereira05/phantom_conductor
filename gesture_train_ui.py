import dearpygui.dearpygui as dpg

from gesture_trainer import MIN_CLIPS, TRAINER

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
    "red_faint": (45, 16, 16, 255),
}

# Cuantos fotogramas la barra de progreso "requiere" (cosmético)
_FRAMES_TARGET = 300  # 5 clips × ~60 frames each


class GestureTrainUI:
    def _build_train_popup(self):
        self._train_last_sig: tuple = ()

        

        with dpg.window(
            label="Entrenar Gestos",
            tag="train_win",
            width=580,
            height=480,
            show=False,
            no_close=False,
            on_close=self._cb_train_close,
            pos=(420, 180),
        ):
            # Header
            dpg.add_text("NUEVO GESTO", color=_C["amber"])
            dpg.add_separator()
            dpg.add_spacer(height=4)

            with dpg.group(horizontal=True):
                with dpg.group(width=180):
                    dpg.add_text("Nombre (UPPER_SNAKE)", color=_C["dim"])
                    dpg.add_input_text(
                        tag="train_name_input", hint="ej. THUMBS_UP", width=172
                    )
                dpg.add_spacer(width=8)
                with dpg.group(width=160):
                    dpg.add_text("Comando mapeado", color=_C["dim"])
                    dpg.add_combo(
                        items=_ALL_CMDS,
                        tag="train_cmd_combo",
                        default_value="none",
                        width=152,
                    )
                dpg.add_spacer(width=8)
                with dpg.group():
                    dpg.add_spacer(height=17)
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label=" COMENZAR SESIÓN ",
                            tag="train_start_btn",
                            callback=self._cb_train_start,
                            width=110,
                        )
                        dpg.add_spacer(width=4)
                        dpg.add_button(
                            label=" CANCELAR ",
                            tag="train_cancel_btn",
                            callback=self._cb_train_cancel,
                            width=76,
                        )

            dpg.add_spacer(height=10)

            # Instrucciones
            dpg.add_text(
                "Grabe varios clips desde diferentes ángulos y distancias ",
                color=_C["dim"],
            )
            dpg.add_text(
                f"Necesitas al menos {MIN_CLIPS} clips antes que puedas guardar",
                color=_C["dim"],
            )
            dpg.add_spacer(height=6)

            # Contador de clips y progreso
            dpg.add_text("— no grabando —", tag="train_status_text", color=_C["dim"])
            dpg.add_spacer(height=4)
            dpg.add_progress_bar(
                tag="train_progress",
                default_value=0.0,
                width=-1,
                height=14,
                overlay=f"0 clips  |  0 fotogramas",
            )
            dpg.add_spacer(height=6)

            # Botones
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="  GRABAR  ",
                    tag="train_rec_btn",
                    callback=self._cb_train_rec,
                    width=120,
                    height=40,
                    enabled=False,
                )
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="  PARAR CLIP  ",
                    tag="train_stop_btn",
                    callback=self._cb_train_stop,
                    width=140,
                    height=40,
                    enabled=False,
                )

            dpg.add_spacer(height=6)

            # Guardar
            dpg.add_button(
                label="  GUARDAR GESTO  ",
                tag="train_save_btn",
                callback=self._cb_train_save,
                width=-1,
                height=32,
                enabled=False,
            )

            dpg.add_spacer(height=10)
            dpg.add_separator()

            # Lista de gestos guardados
            dpg.add_text("GESTOS GUARDADOS", color=_C["amber"])
            dpg.add_spacer(height=4)
            with dpg.child_window(tag="train_list_outer", height=-1, border=False):
                dpg.add_text(
                    "- ningún gesto guardado -",
                    tag="train_empty_label",
                    color=_C["dim"],
                )

        self._train_refresh_list()

    # Abrir / Cerrar

    def _cb_train_open(self):
        dpg.show_item("train_win")
        self._train_refresh_list()

    def _cb_train_close(self):
        if TRAINER.is_recording():
            TRAINER.cancel_recording()
            self.logger.info("trainer: cancelado (ventana cerrada)")
        self.state.clip_recording_active = False
        self._train_reset_ui()

    # Controles de sesión

    def _cb_train_start(self):
        name = dpg.get_value("train_name_input").strip().upper()
        if not name:
            self._set_status("!! Digite un nombre para el gesto primero", _C["red"])
            return
        if TRAINER.is_recording():
            TRAINER.cancel_recording()
        TRAINER.start_recording(name)
        self.state.clip_recording_active = False

        dpg.configure_item("train_rec_btn", enabled=True)
        dpg.configure_item("train_stop_btn", enabled=False)
        dpg.configure_item("train_save_btn", enabled=False)
        self._set_status(
            f"Session open: '{name}'  - presione GRABAR para empezar un clip",
            _C["cyan"],
        )
        self.logger.info(f"trainer: sesión iniciada para '{name}'")

    def _cb_train_cancel(self):
        if TRAINER.is_recording():
            name = TRAINER.recording_name() or "?"
            TRAINER.cancel_recording()
            self.logger.info(f"trainer: cancelado '{name}'")
        self.state.clip_recording_active = False
        self._train_reset_ui()

    # Controles de clip

    def _cb_train_rec(self):
        if not TRAINER.is_recording():
            return
        if TRAINER.is_clip_active():
            return
        TRAINER.start_clip()
        self.state.clip_recording_active = True
        dpg.configure_item("train_rec_btn", enabled=False)
        dpg.configure_item("train_stop_btn", enabled=True)
        self._set_status(
            f"Grabando clip {TRAINER.clip_count() + 1}  - mueva su mano naturalmente",
            _C["red"],
        )

    def _cb_train_stop(self):
        if not TRAINER.is_clip_active():
            return
        self.state.clip_recording_active = False
        TRAINER.stop_clip()

        clips = TRAINER.clip_count()
        frames = TRAINER.frame_count()
        can_save = clips >= MIN_CLIPS

        dpg.configure_item("train_rec_btn", enabled=True)
        dpg.configure_item("train_stop_btn", enabled=False)
        dpg.configure_item("train_save_btn", enabled=can_save)

        if can_save:
            self._set_status(
                f"{clips} clips ({frames} frames)  - listo para guardar o añadir más clips",
                _C["green"],
            )
        else:
            remaining = MIN_CLIPS - clips
            self._set_status(
                f"Clip {clips} guardado ({frames} frames total)  "
                f"grabe {remaining} clips más desde un ángulo distinto",
                _C["cyan"],
            )
        self.logger.info(f"trainer: clip {clips} guardado ({frames} frames total)")

    def _cb_train_save(self):
        if not TRAINER.is_recording():
            return
        clips = TRAINER.clip_count()
        if clips < MIN_CLIPS:
            self._set_status(
                f"!! Se necesitan {MIN_CLIPS} clips; solo {clips} grabados", _C["red"]
            )
            return
        cmd = dpg.get_value("train_cmd_combo")
        name = TRAINER.finish(command=cmd)
        self.logger.ok(f"trainer: guardado '{name}' → {cmd}")
        self._set_status(f" Guardado '{name}'  ({cmd})", _C["green"])
        self._train_reset_ui()
        self._train_refresh_list()

    # Actualización

    def _update_train_ui(self):
        if not dpg.is_item_shown("train_win"):
            return

        rec = TRAINER.is_recording()
        clips = TRAINER.clip_count() if rec else 0
        frames = TRAINER.frame_count() if rec else 0
        active = TRAINER.is_clip_active() if rec else False
        n_g = len(TRAINER.list_gestures())

        sig = (TRAINER.recording_name(), clips, frames, active, n_g)
        if sig == self._train_last_sig:
            return
        self._train_last_sig = sig

        if rec:
            frac = min(frames / max(_FRAMES_TARGET, 1), 1.0)
            dpg.set_value("train_progress", frac)
            dpg.configure_item(
                "train_progress",
                overlay=f"{clips} clip{'s' if clips != 1 else ''}  |  {frames} frames",
            )
            if active:
                self._set_status(
                    f"Grabando clip {clips + 1}  "
                    f"—  {len(TRAINER._clip_vectors) if hasattr(TRAINER, '_clip_vectors') else '?'} frames",
                    _C["red"],
                )

        self._train_refresh_list()

    # Lista de gestos

    def _train_refresh_list(self):
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
            dpg.add_table_column(label="Name", init_width_or_weight=160)
            dpg.add_table_column(label="Command", init_width_or_weight=120)
            dpg.add_table_column(label="Clips", init_width_or_weight=50)
            dpg.add_table_column(label="Frames", init_width_or_weight=70)
            dpg.add_table_column(label="", init_width_or_weight=80)

            for g in gestures:
                name = g["name"]
                cmd = g["command"]
                clips = g.get("clips", 1)
                frames = g["frames"]
                ready = clips >= MIN_CLIPS

                with dpg.table_row():
                    dpg.add_text(name, color=_C["cyan"] if ready else _C["amber"])
                    dpg.add_text(cmd, color=_C["text"])
                    dpg.add_text(
                        str(clips), color=_C["green"] if ready else _C["amber"]
                    )
                    dpg.add_text(str(frames), color=_C["text"])
                    dpg.add_button(
                        label=" DEL ",
                        callback=self._cb_train_delete,
                        user_data=name,
                        width=56,
                    )

    def _cb_train_delete(self, sender, app_data, user_data):
        TRAINER.delete(user_data)
        self.logger.info(f"trainer: eliminado '{user_data}'")
        self._train_refresh_list()

    # Helpers

    def _set_status(self, text: str, color: tuple):
        dpg.set_value("train_status_text", text)
        dpg.configure_item("train_status_text", color=color)

    def _train_reset_ui(self):
        dpg.set_value("train_progress", 0.0)
        dpg.configure_item("train_progress", overlay=f"0 clips  |  0 frames")
        self._set_status("— not recording —", _C["dim"])
        dpg.configure_item("train_rec_btn", enabled=False)
        dpg.configure_item("train_stop_btn", enabled=False)
        dpg.configure_item("train_save_btn", enabled=False)
