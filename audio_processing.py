import os
import time

import librosa
import numpy as np

from buffers import SR, audio_queue
from logger import Logger
from state import PhantomState

try:
    from mutagen import File as MutagenFile

    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

try:
    import pyrubberband as pyrb

    HAS_PYRB = True
except ImportError:
    HAS_PYRB = False
    print("[warn] pyrubberband no fue encontrado — time-stretch deshabilitado")

MIN_BPM = 60
MAX_BPM = 200


# LECTOR DE BPM TAG


def _read_bpm_tag(path: str) -> float | None:
    """Prueba todos los tags de BPM conocidos (requiere mutagen)"""
    if not HAS_MUTAGEN:
        return None
    try:
        af = MutagenFile(path)
        tags = af.tags if af else None
        if not tags:
            return None
        for key in ("TBPM", "bpm", "BPM", "TXXX:BPM", "----:com.apple.iTunes:BPM"):
            if key in tags:
                raw = tags[key]
                val = str(
                    raw[0]
                    if (hasattr(raw, "__iter__") and not isinstance(raw, str))
                    else raw
                )
                return float(val.strip())
    except Exception:
        pass
    return None


# ESTIMADOR DE BPM (en caso el archivo no tenga tag BPM)


def _estimate_bpm_from_file(y: np.ndarray, sr: int, logger: Logger) -> float | None:
    """Usa el tracker de beats como "plan b"."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__len__"):
            tempo = float(tempo[0]) if len(tempo) else None
        else:
            tempo = float(tempo)
        if tempo and MIN_BPM <= tempo <= MAX_BPM:
            return tempo
        for candidate in (tempo * 2, tempo / 2):
            if candidate and MIN_BPM <= candidate <= MAX_BPM:
                return float(candidate)
    except Exception as e:
        logger.warn(f"librosa beat_track falló: {e}")
    return None


# TRACK LOADER


def load_track(
    path: str, state: PhantomState, logger: Logger
) -> tuple[np.ndarray, float, float]:
    """
    Carga cualquier archivo de audio soportado por librosa.

    Retorna:
    (samples_float32, bpm_original, duration_seconds)
    """
    logger.info(f"loading: {os.path.basename(path)}")
    y, _ = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / SR

    with state._lock:
        ui_bpm = state.bpm_original

    tag_bpm = _read_bpm_tag(path)
    if tag_bpm:
        logger.ok(f"Tag BPM encontrado: {tag_bpm:.1f}")

    if tag_bpm is None:
        logger.info("Sin tag BPM, ejecutando estimación de BPM...")
        tag_bpm = _estimate_bpm_from_file(y, SR, logger)
        if tag_bpm:
            logger.ok(f"BPM estimado: {tag_bpm:.1f}")
        else:
            logger.warn(f"Estimación fallida, usando referencia manual: {ui_bpm:.1f}")

    bpm_orig = tag_bpm if tag_bpm else ui_bpm
    logger.ok(f"track listo: {dur:.1f}s  bpm_ref={bpm_orig:.1f}")
    return y.astype(np.float32), bpm_orig, dur


# THREAD DE BACKING TRACK


def _consume_skip_flags(state: PhantomState) -> str | None:
    """
    Automaticamente leer y eliminar flags de "next" (state.skip_to_next) o "prev" (skip_to_prev).

    Retorna "next", "prev", o None.
    Si ambas flags son seteadas en el mismo tick, se prioriza "next".
    """
    with state._lock:
        nxt = state.skip_to_next
        prv = state.skip_to_prev
        state.skip_to_next = False
        state.skip_to_prev = False
    if nxt:
        return "next"
    if prv:
        return "prev"
    return None


def backing_track_thread(state: PhantomState, logger: Logger):
    """
    Espera una señal de track de state (state.load_new_track), lo carga al buffer y reproduce con time-stretching.
    Respeta señales del usuario.
    All terminar un track, pasa al siguiente o repite si is_looping == true.

    Phase-lock loop (PLL)

    En cada beat, nuevos marcadores de tiempo son registrados en una instancia de PhaseLock (phase_lock.py).
    Compara cada marcador detectado con una grilla extrapolada basada en detecciones anteriores.
    Se eliminan variaciones altas (más de +- media duración de un beat extrapolado).
    Se promedian las últimas n variaciones y retorna un multiplicador que mueve la velocidad de reproducción gradualmente durante 10 beats aprox.
    El cálculo del multiplicador final es:

        rate = (bpm_live / bpm_orig) * pll.rate_correction

    Se resetea el PLL cada vez que se carga un nuevo track para eliminar error de fase del track anterior.
    """
    from phase_lock import PhaseLock

    y_full = None
    bpm_orig = 120.0
    pos = 0
    t_next = time.time()

    filtered_rate = 1.0

    pll = PhaseLock(logger=logger)
    last_seen_beat: float | None = None

    while state.alive():
        # Requesiciones del usuario
        skip = _consume_skip_flags(state)
        if skip:
            next_t = (
                state.queue.next_track() if skip == "next" else state.queue.prev_track()
            )
            if next_t:
                with state._lock:
                    state.load_new_track = next_t
                logger.info(f"skip: {skip} → {next_t['name']}")
            else:
                logger.info(f"skip: {skip} requested but queue has no track")

        # Checkear nuevos tracks que cargar
        with state._lock:
            new_track = state.load_new_track
            if new_track:
                state.load_new_track = None

        if new_track:
            if new_track.get("bpm"):
                state.set_bpm_original(float(new_track["bpm"]))

            try:
                y_full, bpm_orig, dur = load_track(new_track["path"], state, logger)

                if new_track.get("bpm"):
                    bpm_orig = float(new_track["bpm"])

                with state._lock:
                    state.bpm_original = bpm_orig
                    state.bpm_live = bpm_orig
                    state.stretch_ratio = 1.0
                    state.track_path = new_track["path"]
                    state.track_duration = dur
                    state.track_position = 0.0
                    state.markers = []

                state.queue.set_bpm(state.queue._index, bpm_orig)
                pos = 0
                t_next = time.time()
                state.play()
                logger.ok(f"playing: {new_track['name']}  BPM={bpm_orig:.1f}")

                # Resetea el PLL
                pll.reset()
                last_seen_beat = None
                filtered_rate = 1.0

            except Exception as e:
                logger.err(f"failed to load track: {e}")
                y_full = None
                new_track = None

        # Sin cambios
        if y_full is None:
            time.sleep(0.05)
            continue

        # Pausado
        if not state.playing():
            time.sleep(0.05)
            t_next = time.time()
            continue

        # Track finalizado
        if pos >= len(y_full):
            logger.ok("track finished")
            with state._lock:
                looping = state.is_looping
            if looping:
                pos = 0
                t_next = time.time()
                pll.reset()
                last_seen_beat = None
                logger.info("loop: restarting")
            else:
                next_t = state.queue.next_track()
                if next_t:
                    with state._lock:
                        state.load_new_track = next_t
                    y_full = None
                    pos = 0
                else:
                    state.pause()
                    with state._lock:
                        state.track_position = state.track_duration
                    y_full = None
                    logger.info("queue empty — stopped")
            continue

        # Construcción de bloque de audio

        safe_orig = max(1.0, bpm_orig)
        bpm_live = state.get_bpm() or safe_orig
        rate = (bpm_live / safe_orig) * pll.rate_correction

        # How many OUTPUT samples per beat?
        out_beat_size = int(60.0 / max(1.0, bpm_live) * SR)

        # How many INPUT samples needed to produce out_beat_size output?
        in_beat_size = int(out_beat_size * rate)

        end = min(pos + in_beat_size, len(y_full))
        block = y_full[pos:end] * state.gain

        # Time-stretch this chunk
        if HAS_PYRB and len(block) > 512 and abs(rate - 1.0) > 0.005:
            try:
                block = pyrb.time_stretch(block, SR, rate)
            except Exception as e:
                logger.warn(f"time-stretch: {e}")

        # Wait and play
        wait = t_next - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            audio_queue.put_nowait(block.astype(np.float32))
        except Exception:
            pass

        pos += in_beat_size  # advance in INPUT space
        t_next += len(block) / SR  # advance by actual playback duration
        state.set_position(pos / SR)  # position tracking (approximate)
