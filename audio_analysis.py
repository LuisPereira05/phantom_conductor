import collections
import time

import librosa
import numpy as np
from scipy.signal import butter, lfilter

from buffers import SR, audio_buffer
from config import CFG
from logger import Logger
from state import PhantomState

# ── Constants --------------------------------------------------------------
HOP_LENGTH = 256
BANDPASS = (40, 8000)  # Filtro "high-pass" para exponer transientes m


#  DSP HELPERS
def _butter_bandpass(lo: float, hi: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, [max(1e-3, lo / nyq), min(0.999, hi / nyq)], btype="band")
    return b, a


def _apply_bandpass(y: np.ndarray, fs: int, lo: float, hi: float) -> np.ndarray:
    b, a = _butter_bandpass(lo, hi, fs)
    return lfilter(b, a, y)


def _pick_peak(ac: np.ndarray, lag_min: int, lag_max: int) -> float | None:
    """Interpolación parabólica al rededor del pico más fuerte de autocorrelación."""
    seg = ac[lag_min : lag_max + 1]
    if not seg.size:
        return None
    i = lag_min + int(np.argmax(seg))
    if 0 < i < len(ac) - 1:
        y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
        d = y0 - 2 * y1 + y2
        if d:
            return i + 0.5 * (y0 - y2) / d
    return float(i)


def _octave_correct(bpm: float, bpm_original: float | None) -> float:
    """
    Elige la octava de BPM (1/2, normal, o x2) más cercana al BPM del backing track.
    La variable "bpm_original" es el BPM de referencia del backing track.
    Si el bpm_original es nulo, no cambia el bpm actual.
    """
    if bpm_original is None or not np.isfinite(bpm_original) or bpm_original <= 0:
        return bpm
    candidates = [bpm / 2, bpm, bpm * 2]
    return min(candidates, key=lambda x: abs(x - bpm_original))


# ESTIMADOR PRINCIPAL


def estimate_bpm(
    y: np.ndarray,
    sr: int = SR,
    bpm_original: float | None = None,
    logger: Logger | None = None,
) -> tuple[float | None, dict]:
    """
    Estima el BPM de un buffer de audio
    Returns
    -------
    (bpm_corr, debug_dict)
    bpm_corr es la estimación de una ventana (con correción de octava) antes del filtro de mediana.
    Retorna None cuando el buffer es demasiado corto, demasiado silencioso, o la autocorrelación produce un pico inválido.
    El suavizado (Filtro de mediana) es abordado en bpm_analysis_thread, para que la ventana de tiempo persista entre llamadas.
    """
    min_bpm = CFG.min_bpm
    max_bpm = CFG.max_bpm

    if len(y) < sr * 2:
        return None, {"msg": "buffer demasiado corto"}

    y = librosa.util.normalize(y.astype(np.float32))
    y = _apply_bandpass(y, sr, *BANDPASS)

    _, y_p = librosa.effects.hpss(y)

    onset = librosa.onset.onset_strength(
        y=y_p, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median
    )

    if onset.size < 8 or np.max(onset) < 1e-3:
        return None, {"msg": "ataque débil"}

    # Slice from centre of full correlogram — correct for even-length arrays.
    ac_full = np.correlate(onset, onset, mode="full")
    ac = ac_full[ac_full.size // 2 :]

    lag_min = max(2, int(np.floor(60.0 * sr / (max_bpm * HOP_LENGTH))))
    lag_max = min(int(np.ceil(60.0 * sr / (min_bpm * HOP_LENGTH))), len(ac) - 1)

    if lag_min >= lag_max:
        return None, {"msg": "rango inválido"}

    lag = _pick_peak(ac, lag_min, lag_max)
    if lag is None or not np.isfinite(lag) or lag <= 0:
        return None, {"msg": "lag inválido"}

    bpm_raw = 60.0 * sr / (HOP_LENGTH * lag)
    bpm_corr = float(np.clip(_octave_correct(bpm_raw, bpm_original), min_bpm, max_bpm))

    if logger is not None:
        logger.debug(
            f"bpm_raw={bpm_raw:.1f}  bpm_corr={bpm_corr:.1f}  "
            f"onset_max={float(np.max(onset)):.3f}"
        )

    return bpm_corr, {
        "bpm_raw": float(bpm_raw),
        "bpm_corr": bpm_corr,
        "onset_max": float(np.max(onset)),
    }


# THREAD DE ANÁLISIS


def bpm_analysis_thread(state: PhantomState, logger: Logger):

    last = 0.0
    tapper_was_active = False

    # Ventana rotativa de estimaciones.
    bpm_window: collections.deque[float] = collections.deque(
        maxlen=CFG.get("bpm_median_window", 8)
    )

    while state.alive():
        analyze_every = CFG.analyze_every
        now = time.time()

        # Para mantener el tamaño de la ventana en sincronía con la configuración sin recrear el deque (en buffers.py)
        new_window_size = CFG.get("bpm_median_window", 8)
        if new_window_size != bpm_window.maxlen:
            bpm_window = collections.deque(bpm_window, maxlen=new_window_size)

        if now - last >= analyze_every and len(audio_buffer) >= SR * 2:
            last = now
            y = np.array(audio_buffer, dtype=np.float32)

            # RMS gate — Umbral de silencio
            rms = float(np.sqrt(np.mean(y**2)))
            rms_threshold = CFG.get("rms_threshold", 0.01)
            if rms < rms_threshold:
                # logger.debug(f"bpm-analysis: silent (rms={rms:.4f} < {rms_threshold})")
                with state._lock:
                    state.buffer_fill = min(1.0, len(audio_buffer) / (SR * 10))
                continue

            # Estimación
            bpm_new, dbg = estimate_bpm(
                y,
                bpm_original=state.bpm_original,
                logger=logger,
            )

            tapper_active = CFG.get("use_tempo_tapper", False)

            if tapper_active and not tapper_was_active:
                logger.info(
                    "bpm-analysis: tempo tapper activado — Escritura de BPM de audio pausada"
                )
            elif tapper_was_active and not tapper_active:
                logger.info(
                    "bpm-analysis: tempo tapper desactivado — Reanudando escritura de BPM de audio"
                )
            tapper_was_active = tapper_active

            if bpm_new is not None and np.isfinite(bpm_new):
                state.last_bpm_analysis_dbg = dbg

                # Filtro de Mediana
                bpm_window.append(bpm_new)
                bpm_smooth = float(np.median(bpm_window))

                if not tapper_active:
                    wrote = state.apply_audio_bpm(
                        bpm_smooth,
                        raw=dbg.get("bpm_raw"),
                        corrected=dbg.get("bpm_corr"),
                        onset_max=dbg.get("onset_max", 0.0),
                    )
                    if wrote:
                        state.last_bpm_dbg = dbg

            with state._lock:
                state.buffer_fill = min(1.0, len(audio_buffer) / (SR * 10))

        time.sleep(analyze_every / 4)
