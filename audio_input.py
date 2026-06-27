import threading
import time
from queue import Empty

import numpy as np
import sounddevice as sd

from buffers import SR, audio_buffer, audio_queue
from logger import Logger
from state import PhantomState

HOP_STREAM_SEC = 0.05


#  LLAMADA DE CAPTURA DE AUDIO


def make_audio_callback(state: PhantomState, logger: Logger):
    """Retorna una llamada de sounddevice ligada al estado (PhantomState) y logger."""

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warn(f"audio input: {status}")
        mono = np.mean(indata, axis=1).astype(np.float32)
        audio_buffer.extend(mono)
        rms = float(np.sqrt(np.mean(mono**2)))
        state.push_waveform(rms)

    return audio_callback


# THREAD DE ENTRADA


def input_thread(dev_in: int | None, state: PhantomState, logger: Logger):
    """
    Abre un InputStream en dev_in.
    Alimenta el audio_buffer a través de la llamada.
    Se apaga cuando STATE.io_restart_requested o STATE no está ejecutándose.
    """
    blocksize = int(SR * HOP_STREAM_SEC)
    callback = make_audio_callback(state, logger)
    try:
        stream = sd.InputStream(
            device=dev_in,
            channels=1,
            samplerate=SR,
            blocksize=blocksize,
            dtype="float32",
            callback=callback,
        )
        stream.start()
        logger.ok(f"input stream started: dev={dev_in}  SR={SR}")
    except Exception as e:
        logger.err(f"input stream no se pudo abrir: {e}")
        return

    while state.alive() and not state.io_restart_requested:
        time.sleep(0.1)

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    logger.info("input stream cerrado")


# THREAD DE REPRODUCCIÓN


def playback_thread(dev_out: int | None, state: PhantomState, logger: Logger):
    """
    Abre un OutputStream en dev_out, vacía audio_queue en él.
    Se apaga cuando STATE.io_restart_requested o cuando STATE no está ejecutándose.
    """
    try:
        stream = sd.OutputStream(
            device=dev_out,
            samplerate=SR,
            channels=1,
            dtype="float32",
        )
        stream.start()
        logger.ok(f"output stream started: dev={dev_out}  SR={SR}")
    except Exception as e:
        logger.err(f"output stream no se pudo abrir: {e}")
        return

    while state.alive() and not state.io_restart_requested:
        if not state.playing():
            time.sleep(0.02)
            continue
        try:
            block = audio_queue.get(timeout=0.1).astype(np.float32)
            block = np.clip(block, -1.0, 1.0)
            stream.write(block)
        except Empty:
            time.sleep(0.01)
        except Exception as e:
            logger.err(f"playback write error: {e}")
            time.sleep(0.05)

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    logger.info("output stream cerrado")


# THREAD GERENTE DE I/O


def io_manager_thread(state: PhantomState, logger: Logger):
    """
    Abre los Streams de audio (Entrada y Salida) usando los dispositivos por defecto.
    Cuando el usuario presiona "APPLY" en la interfaz, para los Streams actuales y abre los nuevos según la configuración.
    """
    dev_in, dev_out = None, None

    def _start():
        nonlocal dev_in, dev_out
        t_in = threading.Thread(
            target=input_thread,
            args=(dev_in, state, logger),
            daemon=True,
            name="audio-in",
        )
        t_out = threading.Thread(
            target=playback_thread,
            args=(dev_out, state, logger),
            daemon=True,
            name="audio-out",
        )
        t_in.start()
        t_out.start()
        return t_in, t_out

    t_in, t_out = _start()

    while state.alive():
        time.sleep(0.2)

        if state.io_restart_requested:
            logger.info("REINICIANDO I/O, rotando Streams")
            t_in.join(timeout=2.0)
            t_out.join(timeout=2.0)

            # Drain stale audio queue
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except Exception:
                    break

            dev_in, dev_out = state.consume_io_restart()
            t_in, t_out = _start()
            logger.ok(f"I/O REINICIADO: in={dev_in}  out={dev_out}")

    t_in.join(timeout=1.0)
    t_out.join(timeout=1.0)
