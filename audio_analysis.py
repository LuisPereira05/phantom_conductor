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
MIN_ONSET_GAP_S = 60.0 / CFG.max_bpm  # ou um valor fixo tipo 0.12s

# Ventana usada para el gate de silencio — SOLO la cola más reciente del
# buffer, no todo audio_buffer (que puede tener hasta BUFFER_SEC segundos
# de historia). Ver nota larga en bpm_analysis_thread.
SILENCE_GATE_WINDOW_S = 0.5


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
    onset_wait_s: float | None = None,  # NOVO
    onset_delta: float = 0.1,  # NOVO — era hardcoded/implícito antes
) -> tuple[float | None, dict]:
    """
    Estima el BPM de un buffer de audio
    Returns
    -------
    (bpm_corr, debug_dict)
    bpm_corr es la estimación de una ventana (con correción de octava) antes del filtro de mediana.
    Retorna None cuando el buffer es demasiado corto, demasiado silencioso, o la autocorrelación produce un pico inválido.
    El suavizado (Filtro de mediana) es abordado en bpm_analysis_thread, para que la ventana de tiempo persista entre llamadas.

    debug_dict también incluye "onset_frame_times": tiempos (en segundos,
    relativos al INICIO de `y`) de los picos individuales de onset detectados
    en esta ventana — ver _detect_onset_times(). bpm_analysis_thread los
    traduce a tiempo de reloj absoluto antes de escribirlos en
    state.recent_beat_times; estimate_bpm() en sí no conoce el reloj de
    pared, solo el buffer que se le pasó.
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

    # Picos de onset individuales dentro de esta ventana — independientes
    # del cálculo de BPM por arriba (que usa la envolvente completa), pero
    # reutiliza la misma envolvente "onset" para no correr onset_strength
    # dos veces. units="frames" + frames_to_time evita perder precisión por
    # redondeo a milisegundos en cada paso.
    wait_frames = max(
        1, int(round((onset_wait_s or MIN_ONSET_GAP_S) * sr / HOP_LENGTH))
    )
    wait_frames = int(wait_frames)
    delta = float(onset_delta)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="frames",
        wait=wait_frames,
        delta=delta,
    )
    onset_frame_times = librosa.frames_to_time(
        onset_frames, sr=sr, hop_length=HOP_LENGTH
    ).tolist()

    if logger is not None:
        logger.debug(
            f"bpm_raw={bpm_raw:.1f}  bpm_corr={bpm_corr:.1f}  "
            f"onset_max={float(np.max(onset)):.3f}"
        )

    return bpm_corr, {
        "bpm_raw": float(bpm_raw),
        "bpm_corr": bpm_corr,
        "onset_max": float(np.max(onset)),
        "onset_frame_times": onset_frame_times,
    }


# THREAD DE ANÁLISIS


def bpm_analysis_thread(state: PhantomState, logger: Logger):
    print("STARTED BPM THREAD")
    last = 0.0
    tapper_was_active = False

    # Ventana rotativa de estimaciones.
    bpm_window: collections.deque[float] = collections.deque(
        maxlen=CFG.get("bpm_median_window", 8)
    )

    # Marca de tiempo (reloj de pared) del onset más reciente que ya fue
    # empujado a state.recent_beat_times. Cada ventana de análisis se
    # superpone con la anterior (el buffer es un deque rotativo de hasta
    # BUFFER_SEC segundos, no un clip aislado), así que sin este guardia
    # el mismo onset físico se reenviaría en cada pase mientras siga
    # dentro del buffer — re-detectado, no un beat nuevo.
    last_emitted_onset_wall_time: float | None = None

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
            #
            # OJO: audio_buffer es una ventana rotativa de hasta BUFFER_SEC
            # (10s) segundos, no un clip aislado. Medir el RMS sobre TODA
            # esa ventana hace que, apenas dejás de tocar, el gate tarde
            # hasta ~10s en activarse — diluido por el audio fuerte de
            # segundos atrás que todavía sigue dentro del buffer.
            #
            # Mientras tanto la ventana usada para estimar BPM (más abajo)
            # sigue corriendo con esa mezcla de "silencio reciente" +
            # "música vieja", y como los onsets reales se van espaciando
            # cada vez más a medida que se acerca el silencio, la
            # autocorrelación converge a un lag cada vez más largo — un
            # BPM cada vez más bajo — durante varios segundos, hasta que
            # el promedio de TODO el buffer finalmente cae por debajo del
            # umbral. Por eso el BPM caía "de a poco" en vez de
            # simplemente congelarse apenas se dejaba de tocar.
            #
            # El fix: medir el RMS solo de la cola más reciente del
            # buffer (SILENCE_GATE_WINDOW_S ≈ 0.5s), que reacciona casi
            # de inmediato en vez de esperar a que toda la ventana de
            # 10s se "limpie".
            recent_n = min(len(y), int(SR * SILENCE_GATE_WINDOW_S))
            rms_recent = float(np.sqrt(np.mean(y[-recent_n:] ** 2)))
            rms_threshold = CFG.get("rms_threshold", 0.01)
            if rms_recent < rms_threshold:
                # logger.debug(f"bpm-analysis: silent (rms={rms_recent:.4f} < {rms_threshold})")
                with state._lock:
                    state.buffer_fill = min(1.0, len(audio_buffer) / (SR * 10))
                continue

            # Estimación
            bpm_new, dbg = estimate_bpm(
                y,
                bpm_original=state.bpm_original,
                logger=logger,
                onset_wait_s=0.15,
                onset_delta=0.405,
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

            # ── Emitir onsets nuevos a state.recent_beat_times ──────────
            # Esto alimenta el PLL en audio_processing.backing_track_thread
            # (vía phase_lock.PhaseLock.update()), que de otro modo nunca
            # recibe un solo beat — recent_beat_times no se escribía en
            # ningún lado antes de este parche.
            #
            # y[] fue tomado de audio_buffer en el momento `now`, así que
            # su última muestra corresponde a wall-clock `now` y su primera
            # muestra a `now - len(y)/SR`. onset_frame_times está en
            # segundos relativos al INICIO de y[], así que se traduce
            # sumando ese offset.
            frame_times = dbg.get("onset_frame_times") if bpm_new is not None else None
            if frame_times:
                window_start_wall = now - (len(y) / SR)
                for ft in frame_times:
                    onset_wall_time = window_start_wall + ft
                    if (
                        last_emitted_onset_wall_time is None
                        or onset_wall_time - last_emitted_onset_wall_time
                        >= MIN_ONSET_GAP_S
                    ):
                        state.recent_beat_times.append(onset_wall_time)
                        last_emitted_onset_wall_time = onset_wall_time

            with state._lock:
                state.buffer_fill = min(1.0, len(audio_buffer) / (SR * 10))

        time.sleep(analyze_every / 4)
