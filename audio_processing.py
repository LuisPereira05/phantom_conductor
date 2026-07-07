import os
import struct
import time

import librosa
import numpy as np

from buffers import (
    MARKER_BUFFER_SEC,
    SR,
    audio_queue,
    marker_buffers,
    marker_buffers_lock,
)
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


# IMPORTADOR DE MARCADORES DESDE WAV (chunk RIFF 'cue ')


def _read_wav_cue_points(path: str, logger: Logger) -> list[float]:
    """
    Lee los cue points nativos de un WAV (chunk RIFF 'cue ') y los
    retorna como lista de posiciones en segundos, ordenadas.

    Este es un mecanismo estándar de RIFF/WAV, no algo propietario:
    FL Studio (entre otros DAWs) puede embeber los marcadores de la
    línea de tiempo directamente ahí al exportar/guardar un wav (p. ej.
    "Save the audio again to embed the markers inside the wav file",
    como documenta el propio manual de FL Studio para el flujo de
    cue points en Slicex/Visualizer).

    MP3 no tiene un chunk equivalente — es un stream, no un contenedor
    RIFF — así que para mp3 esta función retorna [] sin intentar nada.

    Nunca lanza: cualquier archivo sin chunk 'cue ', corrupto, o que no
    sea wav simplemente resulta en lista vacía, para no tumbar la carga
    de la pista por metadata inesperada.
    """
    if not path.lower().endswith(".wav"):
        return []

    try:
        with open(path, "rb") as f:
            riff_id, _riff_size, wave_id = struct.unpack("<4sI4s", f.read(12))
            if riff_id != b"RIFF" or wave_id != b"WAVE":
                return []

            native_sr: int | None = None
            cue_positions: list[int] = []

            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                chunk_id, chunk_size = struct.unpack("<4sI", header)
                chunk_bytes = f.read(chunk_size)
                if len(chunk_bytes) < chunk_size:
                    break  # archivo truncado

                if chunk_id == b"fmt " and len(chunk_bytes) >= 8:
                    # wFormatTag(H) nChannels(H) nSamplesPerSec(I) ...
                    _, _, native_sr = struct.unpack("<HHI", chunk_bytes[:8])

                elif chunk_id == b"cue " and len(chunk_bytes) >= 4:
                    (n_cues,) = struct.unpack("<I", chunk_bytes[:4])
                    for i in range(n_cues):
                        off = 4 + i * 24
                        entry = chunk_bytes[off : off + 24]
                        if len(entry) < 24:
                            break
                        # ID(4s) Position(I) DataChunkID(4s) ChunkStart(I) BlockStart(I) SampleOffset(I)
                        _, position, _, _, _, _ = struct.unpack("<4sI4sIII", entry)
                        cue_positions.append(position)

                # los chunks RIFF están alineados a 2 bytes
                if chunk_size % 2 == 1:
                    f.read(1)

            if not cue_positions or not native_sr:
                return []

            return sorted(pos / native_sr for pos in cue_positions)

    except Exception as e:
        logger.warn(f"wav cue points: no se pudieron leer ({e})")
        return []


# TRACK LOADER


def load_track(
    path: str, state: PhantomState, logger: Logger
) -> tuple[np.ndarray, float, float, list[float]]:
    """
    Carga cualquier archivo de audio soportado por librosa.

    Retorna:
    (samples_float32, bpm_original, duration_seconds, markers_seconds)

    markers_seconds viene de los cue points nativos del wav si el
    archivo los tiene (p. ej. exportado desde FL Studio con los
    marcadores embebidos); para mp3, o un wav sin chunk 'cue ', es [].
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

    markers = _read_wav_cue_points(path, logger)
    # descarta cue points fuera de rango (p. ej. si el wav fue recortado
    # después de embeber los marcadores)
    markers = [m for m in markers if 0.0 <= m <= dur]
    if markers:
        logger.ok(f"{len(markers)} marcador(es) importado(s) del wav (cue points)")

    logger.ok(f"track listo: {dur:.1f}s  bpm_ref={bpm_orig:.1f}")
    return y.astype(np.float32), bpm_orig, dur, markers


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


def _consume_seek_request(state: PhantomState) -> float | None:
    """
    Lee y limpia state.seek_request (segundos, dominio del audio
    original). Puesto por state.request_seek() / step_loop_section() —
    lo único que hace jump_to_marker/loop_next/loop_prev "sonar" de
    verdad, en vez de solo actualizar el dato que muestra la UI.
    """
    with state._lock:
        pos = state.seek_request
        state.seek_request = None
    return pos


def _flush_audio_queue(logger: Logger, reason: str) -> int:
    """
    Vacía audio_queue por completo (get_nowait hasta que quede vacía),
    descartando cualquier bloque ya encolado pero todavía no reproducido.

    Se llama justo ANTES de saltar al inicio de una sección de loop.
    Sin esto, un bloque que ya estaba en la cola — encolado en la
    iteración anterior, antes de detectar que se cruzó sec_end — seguiría
    reproduciéndose después del salto, delante del buffer de preview del
    marcador de inicio. Eso rompe el "salto instantáneo": en vez de oírse
    el inicio de la sección de inmediato, se oye primero la cola de lo
    que ya estaba encolado.

    Trade-off a tener en cuenta: si la thread de audio output está
    consumiendo la cola en el mismo instante en que esta hace el flush,
    puede quedar momentáneamente sin datos (una ventana de silencio muy
    breve) hasta que el put_nowait() del buffer de marcador la vuelva a
    llenar. En la práctica esa ventana es del orden de un callback de
    audio (unos pocos ms), pero es un costo real a cambio de eliminar la
    repetición de contenido viejo.

    Retorna cuántos bloques se descartaron (para logging).
    """
    discarded = 0
    while True:
        try:
            audio_queue.get_nowait()
            discarded += 1
        except Exception:
            break
    if discarded:
        logger.info(f"audio_queue flush: {discarded} bloque(s) descartado(s) ({reason})")
    return discarded


def _sync_marker_buffers(
    state: PhantomState, y_full: np.ndarray, rate: float, logger: Logger
) -> None:
    """
    Mantiene buffers.marker_buffers sincronizado con state.markers:

      - crea un buffer crudo (MARKER_BUFFER_SEC segundos, tomado
        directamente de y_full, SIN estirar) para cualquier marcador
        nuevo que aparezca en state.markers — sea añadido a mano vía
        el pedal o importado del wav al cargar la pista.
      - re-estira cada buffer existente si el rate actual difiere lo
        suficiente del usado la última vez (umbral de 0.01 para no
        recalcular en cada micro-fluctuación del PLL).
      - elimina buffers de marcadores que ya no existen (tras
        clear_markers, o al cargar una pista nueva).

    Se llama en cada iteración del loop de reproducción mientras haya
    una pista cargada, así el buffer estirado de cada marcador siempre
    refleja el rate que se está usando en ese instante — sin importar
    cuánto cambie el BPM en vivo — y queda listo para un salto
    instantáneo sin silencio ni glitch de time-stretch.
    """
    with state._lock:
        current_markers = list(state.markers)

    current_set = set(current_markers)
    buf_len = int(MARKER_BUFFER_SEC * SR)

    with marker_buffers_lock:
        # Eliminar buffers huérfanos (marcador borrado o pista cambiada)
        for pos in list(marker_buffers.keys()):
            if pos not in current_set:
                del marker_buffers[pos]

        # Crear buffers crudos para marcadores nuevos
        for pos in current_markers:
            if pos in marker_buffers:
                continue
            start = int(pos * SR)
            if start >= len(y_full):
                continue
            end = min(start + buf_len, len(y_full))
            raw = y_full[start:end].copy()
            marker_buffers[pos] = {"raw": raw, "stretched": raw, "rate": 1.0}

        # Re-estirar buffers cuyo rate quedó desactualizado
        for pos, entry in marker_buffers.items():
            if abs(entry["rate"] - rate) < 0.01:
                continue
            raw = entry["raw"]
            if HAS_PYRB and len(raw) > 512 and abs(rate - 1.0) > 0.005:
                try:
                    entry["stretched"] = pyrb.time_stretch(raw, SR, rate)
                except Exception as e:
                    logger.warn(f"marker preview stretch: {e}")
                    entry["stretched"] = raw
            else:
                entry["stretched"] = raw
            entry["rate"] = rate


def _jump_to_position(
    seek_sec: float,
    y_full: np.ndarray,
    logger: Logger,
) -> tuple[int, float]:
    """
    Salta a `seek_sec` (segundos, dominio del audio original).

    Si hay un buffer de marcador ya precalculado y estirado para esa
    posición exacta, lo encola de inmediato: nada de silencio mientras
    pyrubberband procesaría un bloque fresco, y el fragmento ya está a
    la velocidad de reproducción actual. `pos` avanza más allá de ese
    fragmento para que el siguiente bloque normal continúe justo
    después, sin solaparse ni repetirlo.

    Si no hay buffer (salto a un punto que no es un marcador registrado)
    simplemente reposiciona `pos` y deja que el loop normal construya
    el siguiente bloque — con el pequeño costo de un time-stretch al
    vuelo, igual que cualquier bloque normal.

    Retorna (nuevo pos, nuevo t_next).

    Siempre vacía audio_queue justo antes de encolar el contenido nuevo
    (ver _flush_audio_queue). Esto vive ACÁ adentro, y no en cada call
    site, a propósito: _jump_to_position es el único punto por el que
    pasa cualquier salto (marcador manual vía pedal/UI, cambio de
    sección de loop, wraparound automático de fin de sección, futura UI
    "ir a"). Si el flush quedara afuera, cada nuevo tipo de salto que se
    agregue en el futuro tendría que acordarse de llamarlo — y olvidarlo
    reproduce exactamente el bug original: hasta un beat completo de
    contenido viejo sonando antes del salto.
    """
    target_sample = max(0, min(int(seek_sec * SR), len(y_full)))

    with marker_buffers_lock:
        entry = marker_buffers.get(seek_sec)

    _flush_audio_queue(logger, f"seek a {seek_sec:.2f}s")

    if entry is not None:
        try:
            audio_queue.put_nowait(entry["stretched"].astype(np.float32))
            logger.info(f"seek: salto a marcador @ {seek_sec:.2f}s (preview buffer)")
            return target_sample + len(entry["raw"]), time.time() + len(
                entry["stretched"]
            ) / SR
        except Exception:
            pass  # cola llena — cae al camino normal de abajo

    logger.info(f"seek: salto a {seek_sec:.2f}s (sin preview buffer)")
    return target_sample, time.time()


def backing_track_thread(state: PhantomState, logger: Logger):
    print("STARTED BackingTrack THREAD")
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
                y_full, bpm_orig, dur, markers = load_track(
                    new_track["path"], state, logger
                )

                if new_track.get("bpm"):
                    bpm_orig = float(new_track["bpm"])

                with state._lock:
                    state.bpm_original = bpm_orig
                    state.bpm_live = bpm_orig
                    state.stretch_ratio = 1.0
                    state.track_path = new_track["path"]
                    state.track_duration = dur
                    state.track_position = 0.0
                    # importados del wav (cue points) si los tenía; [] para
                    # mp3 u otro archivo sin chunk 'cue '
                    state.markers = markers
                    # las secciones son por-pista: no heredar el índice de
                    # sección de la pista anterior (que puede no tener
                    # sentido con el nuevo set de marcadores, o directamente
                    # no existir). is_looping se deja intacto a propósito:
                    # si el usuario tenía "repetir pista completa" activado,
                    # sigue aplicando a la pista nueva.
                    state.loop_section_index = -1

                # Pista nueva → y_full cambió, cualquier buffer de
                # marcador previo (de la pista anterior) queda obsoleto.
                with marker_buffers_lock:
                    marker_buffers.clear()

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

        # Salto pedido (marcador / sección de loop / futura UI "ir a")
        seek_pos = _consume_seek_request(state)
        if seek_pos is not None:
            pos, t_next = _jump_to_position(seek_pos, y_full, logger)
            state.set_position(pos / SR)

        # Pausado
        if not state.playing():
            time.sleep(0.05)
            t_next = time.time()
            continue

        # Sección de loop activa: si la posición ya pasó su fin, saltamos
        # de vuelta a su inicio. Se revisa ANTES que "track finalizado" a
        # propósito: cuando la sección activa es la última de la pista
        # (su fin coincide exactamente con track_duration — lo cual pasa
        # siempre que se entra a una sección con loop_prev desde el
        # estado inicial, ya que arranca en la última sección), ambas
        # condiciones se cumplen a la vez. Si "track finalizado" se
        # revisara primero, reiniciaría la pista entera desde 0 en cada
        # vuelta en lugar de repetir solo la sección — eso era lo que
        # causaba quedar "trancado" repitiendo de más.
        active_section = state.get_active_loop_section()
        if active_section is not None:
            sec_start, sec_end = active_section
            if pos / SR >= sec_end:
                # El flush de audio_queue ocurre adentro de
                # _jump_to_position (ver docstring): cualquier bloque
                # que ya estuviera encolado se descarta ahí mismo, justo
                # antes de encolar el buffer de preview de sec_start.
                pos, t_next = _jump_to_position(sec_start, y_full, logger)
                state.set_position(pos / SR)
                # el salto rompe la continuidad de fase que asume el PLL
                pll.reset()
                last_seen_beat = None
                logger.info(f"loop section: repitiendo {sec_start:.2f}s–{sec_end:.2f}s")
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

        # Mantener listos los previews de marcadores a la velocidad actual
        _sync_marker_buffers(state, y_full, rate, logger)

        # How many OUTPUT samples per beat?
        out_beat_size = int(60.0 / max(1.0, bpm_live) * SR)

        # How many INPUT samples needed to produce out_beat_size output?
        in_beat_size = int(out_beat_size * rate)

        end = min(pos + in_beat_size, len(y_full))

        # Acotar el bloque al fin de la sección de loop activa (si hay
        # una). Sin esto, el bloque se construye completo (típicamente
        # un beat entero) sin mirar el límite de la sección, y recién en
        # la iteración siguiente el chequeo de arriba nota que ya nos
        # pasamos — para entonces ese pedazo de la sección siguiente ya
        # fue encolado y sonó. Acotando acá, el chequeo de arriba salta
        # de vuelta apenas se agota exactamente esta sección, nunca un
        # beat después.
        if active_section is not None:
            _, sec_end = active_section
            sec_end_sample = int(sec_end * SR)
            end = min(end, sec_end_sample)

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