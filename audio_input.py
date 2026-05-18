"""
Phantom Conductor — Audio Input & I/O Manager
=============================================
Responsibilities
----------------
* audio_callback   : sounddevice calls this on its own OS thread;
                     pushes mono samples into the shared ring buffer
                     and updates the RMS waveform on PhantomState.
* input_thread     : opens sounddevice.InputStream; exits when
                     STATE.io_restart_requested is set.
* playback_thread  : opens sounddevice.OutputStream; drains audio_queue.
* io_manager_thread: watches for I/O restart requests from the UI and
                     cycles both stream threads onto new devices.

This module owns NO analysis and NO track loading — it is pure I/O.
"""

import time
import threading
import numpy as np
import sounddevice as sd
from queue import Empty

from buffers import audio_buffer, audio_queue, SR
from state   import PhantomState
from logger  import Logger

HOP_STREAM_SEC = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
#  MIC CAPTURE CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def make_audio_callback(state: PhantomState, logger: Logger):
    """Return a sounddevice callback bound to the given state & logger."""

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warn(f"audio input: {status}")
        mono = np.mean(indata, axis=1).astype(np.float32)
        audio_buffer.extend(mono)
        rms = float(np.sqrt(np.mean(mono ** 2)))
        state.push_waveform(rms)

    return audio_callback


# ═══════════════════════════════════════════════════════════════════════════════
#  INPUT STREAM THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def input_thread(dev_in: int | None, state: PhantomState, logger: Logger):
    """
    Opens an InputStream on dev_in, feeds samples to audio_buffer via callback.
    Exits when STATE.io_restart_requested or STATE is no longer alive.
    """
    blocksize = int(SR * HOP_STREAM_SEC)
    callback  = make_audio_callback(state, logger)
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
        logger.ok(f"input  stream started: dev={dev_in}  SR={SR}")
    except Exception as e:
        logger.err(f"input stream failed to open: {e}")
        return

    while state.alive() and not state.io_restart_requested:
        time.sleep(0.1)

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    logger.info("input stream closed")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLAYBACK STREAM THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def playback_thread(dev_out: int | None, state: PhantomState, logger: Logger):
    """
    Opens an OutputStream on dev_out, drains audio_queue into it.
    Exits when STATE.io_restart_requested is set or STATE is no longer alive.
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
        logger.err(f"output stream failed to open: {e}")
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
    logger.info("output stream closed")


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O MANAGER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def io_manager_thread(state: PhantomState, logger: Logger):
    """
    Starts the initial audio I/O streams using default devices, then
    monitors STATE.io_restart_requested.  When the UI clicks APPLY it
    stops the current streams and opens new ones on the chosen devices.
    """
    dev_in, dev_out = None, None   # start with system defaults

    def _start():
        nonlocal dev_in, dev_out
        t_in  = threading.Thread(
            target=input_thread,
            args=(dev_in, state, logger),
            daemon=True, name="audio-in",
        )
        t_out = threading.Thread(
            target=playback_thread,
            args=(dev_out, state, logger),
            daemon=True, name="audio-out",
        )
        t_in.start()
        t_out.start()
        return t_in, t_out

    t_in, t_out = _start()

    while state.alive():
        time.sleep(0.2)

        if state.io_restart_requested:
            logger.info("I/O restart requested — cycling streams…")
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
            logger.ok(f"I/O restarted: in={dev_in}  out={dev_out}")

    t_in.join(timeout=1.0)
    t_out.join(timeout=1.0)
