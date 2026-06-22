"""
Phantom Conductor — Entry Point
================================
Starts all daemon threads and then hands the main thread to Dear PyGui.

Thread map
----------
  bpm-analysis    → audio_analysis.bpm_analysis_thread
  backing-track   → audio_processing.backing_track_thread
  io-manager      → audio_input.io_manager_thread
                      ├─ audio-in   → audio_input.input_thread
                      └─ audio-out  → audio_input.playback_thread
  gesture-vision  → gesture_recognition.gesture_vision_thread
                      (internally uses video_input helpers)
  tempo-tapper    → tempo_tapper.tempo_tapper_thread
                      (reads BPM taps from the Arduino piezo over serial)

Main thread → ui.PhantomUI.run()  (Dear PyGui must run on the main thread)

Changes from v0.5.0
--------------------
* Camera index now read from CFG (phantom_config.json) by default.
  Command-line override still works: `python main.py 2`
* Introduces config.py and tracklist.py as new dependencies.

Changes from v0.5.2 (tempo tapper)
-----------------------------------
* Added the tempo-tapper daemon thread. It always starts — it does NOT
  exit early if CFG.use_tempo_tapper is False, because the thread now
  checks that flag live on every serial line (see tempo_tapper.py) so
  the Settings checkbox can enable/disable it at runtime without a
  restart. If no Arduino is plugged in, the thread just logs a
  "no serial port found — retrying…" warning periodically and is
  otherwise harmless to leave running.
"""

import threading

from audio_analysis import bpm_analysis_thread
from audio_input import io_manager_thread
from audio_processing import backing_track_thread
from config import CFG
from gesture_recognition import gesture_vision_thread
from logger import Logger
from state import PhantomState
from tempo_tapper import tempo_tapper_thread
from ui import PhantomUI


def main():
    print("=" * 58)
    print("  PHANTOM CONDUCTOR v0.5.3")
    print("=" * 58)

    # Camera index: CLI arg overrides saved config
    cam_prompt = f"  Camera index to use? (Enter = {CFG.cam_index}): "
    print(cam_prompt, end="", flush=True)
    cam_str = input().strip()
    if cam_str.isdigit():
        cam_idx = int(cam_str)
        CFG.set("cam_index", cam_idx)
    else:
        cam_idx = CFG.cam_index

    print()
    print("  → Add tracks via the Queue panel (+ADD)")
    print("  → Configure audio/video in the SETTINGS panel, then click APPLY")
    print("  → Open hand = PLAY  |  Fist = PAUSE  |  Space = toggle  |  Q = quit")
    print("  → Tempo tapper: tap the piezo, then enable it in SETTINGS")
    print("=" * 58 + "\n")

    state = PhantomState()
    logger = Logger()

    # ── Worker threads ─────────────────────────────────────────────────────────
    threading.Thread(
        target=bpm_analysis_thread,
        args=(state, logger),
        daemon=True,
        name="bpm-analysis",
    ).start()

    threading.Thread(
        target=backing_track_thread,
        args=(state, logger),
        daemon=True,
        name="backing-track",
    ).start()

    threading.Thread(
        target=io_manager_thread,
        args=(state, logger),
        daemon=True,
        name="io-manager",
    ).start()

    threading.Thread(
        target=gesture_vision_thread,
        args=(cam_idx, state, logger),
        daemon=True,
        name="gesture-vision",
    ).start()

    threading.Thread(
        target=tempo_tapper_thread,
        args=(state, logger),
        daemon=True,
        name="tempo-tapper",
    ).start()

    # ── UI — must run on the main thread ───────────────────────────────────────
    ui = PhantomUI(state, logger)
    try:
        ui.run()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop()
        print("\nPhantom Conductor stopped.")


if __name__ == "__main__":
    main()
