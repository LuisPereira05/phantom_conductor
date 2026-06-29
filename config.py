import json
import os
import threading

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "phantom_config.json")

_DEFAULTS: dict = {
    # AUDIO
    "dev_in": None,  # Índice de dispositivo sounddevice (None = predeterminado de sistema)
    "dev_out": None,
    "input_gain": 1.0,  # multiplicador de volumen de entrada
    "output_gain": 0.85,  # multiplicador de volumen de salida
    # VIDEO
    "cam_index": 0,  # Índice de cámara de OpenCV
    # GESTO (Controlado por gesture_pool.meta.json)
    "gesture_map": {
        "PLAY": "play",
        "PAUSE": "pause",
    },
    "gesture_hold_frames": 8,  # fotogramas en que un gesto debe ser mantenido para que se ejecute la acción
    # INFERENCIA
    "inference_skip_enabled": False,  # para equipos con menos poder de hardware, ejecuta la inferencia cada N fotogramas
    "inference_skip_frames": 2,  # cantidad de fotogramas no procesados por MediaPipe
    # PEDAL
    "use_pedal": False,
    "pedal_key": "space",  # tecla de simulación del pedal
    # TEMPO TAPPER
    "use_tempo_tapper": False,
    # ANÁLISIS DE BPM
    "smooth_alpha": 0.3,  # razón de suavizado
    "min_bpm": 60,
    "max_bpm": 200,
    "analyze_every": 0.25,  # tiempo de espera para análisis.
    "bpm_median_window": 4,  # cantidad de beats para análisis estadístico
    "rms_threshold": 0.015,  # umbral de detección de transientes (picos)
}


class Config:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict = dict(_DEFAULTS)
        self.load()

    # Persistencia
    def load(self):
        """Carga configuración desde el JSON, y usando valores por defecto desde _DEFAULTS"""
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            with self._lock:
                for k, v in saved.items():
                    if k in self._data:
                        self._data[k] = v
        except Exception as e:
            print(f"[config] carga fallida: {e}")

    def save(self):
        try:
            with self._lock:
                snapshot = dict(self._data)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            print(f"[config] save failed: {e}")

    # ACCESO
    def __getattr__(self, name: str):
        # Solo se llama cuadno el acceso de atributos normal falla
        with object.__getattribute__(self, "_lock"):
            data = object.__getattribute__(self, "_data")
            if name in data:
                return data[name]
        raise AttributeError(f"Config no tiene campo '{name}'")

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value, autosave: bool = True):
        with self._lock:
            self._data[key] = value
        if autosave:
            self.save()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def reset_to_defaults(self):
        with self._lock:
            self._data = dict(_DEFAULTS)
        self.save()


# Singleton
CFG = Config()
