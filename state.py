import collections
import threading
import time

from config import CFG
from tracklist import PersistentQueue

TAP_OVERRIDE_WINDOW_S = 4.0


class PhantomState:
    """Contenedor thread-safe — los hilos de trabajo escriben, la UI lee cada frame."""

    def __init__(self):
        self._lock = threading.Lock()

        self.bpm_live: float | None = None
        self.bpm_original: float = 120.0
        self.bpm_raw: float | None = None
        self.bpm_corrected: float | None = None
        self.stretch_ratio: float = 1.0
        self.bpm_source: str = "audio"
        self.onset_max: float = 0.0
        self.last_bpm_dbg: dict = {}
        self.last_bpm_analysis_dbg: dict = {}
        self._tap_override_until: float = 0.0
        self.recent_beat_times: collections.deque[float] = collections.deque(maxlen=4)

        self.is_playing: bool = False
        self.is_looping: bool = False
        self.track_path: str = ""
        self.track_duration: float = 0.0
        self.track_position: float = 0.0

        self.seek_request: float | None = None
        self.markers: list[float] = []
        self.loop_section_index: int = -1

        self.load_new_track: dict | None = None
        self.skip_to_next: bool = False
        self.skip_to_prev: bool = False

        self.dev_in: int | None = CFG.dev_in
        self.dev_out: int | None = CFG.dev_out
        self.gain: float = CFG.output_gain
        self.input_gain: float = CFG.input_gain
        self.io_restart_requested: bool = False
        self.latest_frame = None

        self.rms: float = 0.0
        self.peak: float = 0.0
        self.waveform: list[float] = [0.0] * 64
        self.buffer_fill: float = 0.0

        self.gesture_name: str = "NO HAND"
        self.gesture_confidence: float = 0.0
        self.gesture_hold_frames: int = 0
        self.gesture_hold_target: int = CFG.gesture_hold_frames
        self.pending_command: str | None = None
        self.last_command: str | None = None
        self.hands_detected: int = 0
        self.camera_active: bool = False

        self.tap_connected: bool = False

        self.running: bool = True
        self.start_time: float = time.time()

        self.queue: PersistentQueue = PersistentQueue()

    # Reproducción
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

    # BPM
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
        Setter incondicional — comportamiento sin cambios para los callers
        existentes (HUD de gestos, overrides manuales de UI, etc.). No toca
        bpm_source ni la ventana de prioridad de tap; los dos hilos escritores
        deben usar apply_tap_bpm() / apply_audio_bpm() para no pisarse.
        """
        with self._lock:
            self.bpm_live = bpm
            self.bpm_raw = raw if raw is not None else bpm
            self.bpm_corrected = corrected if corrected is not None else bpm
            self.stretch_ratio = bpm / self.bpm_original if self.bpm_original else 1.0
            self.onset_max = onset_max

    def apply_tap_bpm(self, bpm: float):
        """
        Llamado por tempo_tapper_thread cuando llega una línea BPM válida.
        Siempre gana de inmediato y abre una ventana durante la cual
        apply_audio_bpm() se negará a sobreescribirlo.
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
        Llamado por bpm_analysis_thread en lugar de set_bpm() directamente.
        Retorna False (no-op) mientras un tap manual reciente esté dentro
        de su ventana de prioridad. Retorna True si escribió el valor.
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

    # Marcadores y secciones de loop
    def add_marker(self, position: float | None = None) -> float:
        """
        Añade un marcador en `position` (o en track_position si no se
        especifica). Ignora duplicados dentro de un epsilon pequeño para
        que pulsar el pedal dos veces por accidente en el mismo instante
        no genere dos marcadores casi idénticos.
        """
        EPS = 0.05  # segundos
        with self._lock:
            pos = self.track_position if position is None else position
            if any(abs(pos - m) < EPS for m in self.markers):
                return pos
            self.markers.append(pos)
        return pos

    def clear_markers(self):
        """
        Borra todos los marcadores de la pista actual. Si el loop estaba en
        modo sección (loop_section_index >= 0), no queda ninguna sección
        válida que apuntar, así que se resetea a -1 (pista completa) en
        lugar de dejar un índice apuntando a nada. `is_looping` se deja
        intacto: si el usuario quiere loop de pista completa, seguirá activo.
        """
        with self._lock:
            self.markers.clear()
            self.loop_section_index = -1

    def request_seek(self, position: float):
        with self._lock:
            self.seek_request = position

    def get_loop_sections(self) -> list[tuple[float, float]]:
        with self._lock:
            pts = sorted(set(self.markers))
            dur = self.track_duration
        if not pts:
            return []
        bounds = [0.0] + pts + [dur]
        return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    def get_active_loop_section(self) -> tuple[float, float] | None:
        sections = self.get_loop_sections()
        with self._lock:
            idx = self.loop_section_index
        if idx < 0 or not sections:
            return None
        return sections[idx % len(sections)]

    def step_loop_section(self, delta: int):
        """Avanza (+1) o retrocede (-1) la sección de loop activa."""
        sections = self.get_loop_sections()
        if not sections:
            return
        with self._lock:
            if self.loop_section_index < 0:
                self.loop_section_index = 0 if delta > 0 else len(sections) - 1
            else:
                self.loop_section_index = (self.loop_section_index + delta) % len(
                    sections
                )
            sec = sections[self.loop_section_index]
            self.track_position = sec[0]
            self.seek_request = sec[0]

    # Dispatcher de comandos
    def dispatch_command(self, cmd: str, logger=None) -> bool:
        """
        Ejecuta un comando de transporte por nombre. Retorna True si fue manejado.

        Comandos soportados

        play          — reanudar / iniciar reproducción
        pause          — pausar reproducción
        toggle        — alternar play/pause
        next          — cargar la siguiente pista de la cola
        prev          — cargar la pista anterior de la cola
        loop_toggle   — activar/desactivar loop; si hay marcadores, activa el modo de sección (sección 0) al primer activar
        loop_next     — avanzar a la siguiente sección de loop (activa loop si estaba apagado)
        loop_prev     — retroceder a la sección anterior (activa loop si estaba apagado)
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
                if not self.is_looping:
                    self.loop_section_index = -1

        elif cmd in ("loop_next", "loop_prev"):
            delta = 1 if cmd == "loop_next" else -1
            with self._lock:
                self.is_looping = True
            self.step_loop_section(delta)
        elif cmd == "add_marker":
            self.add_marker()
        else:
            return False

        with self._lock:
            self.last_command = cmd
            self.pending_command = cmd
        if logger:
            logger.info(f"cmd: {cmd}")
        return True

    # Gestos
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

    # Audio
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

    # I/O
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

    # Sistema
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
            d["recent_beat_times"] = list(self.recent_beat_times)
            return d
