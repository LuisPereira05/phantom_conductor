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

Main thread → ui.PhantomUI.run()  (Dear PyGui must run on the main thread)
"""

import threading

from state               import PhantomState
from logger              import Logger
from audio_analysis      import bpm_analysis_thread
from audio_processing    import backing_track_thread
from audio_input         import io_manager_thread
from gesture_recognition import gesture_vision_thread
from ui                  import PhantomUI


def main():
    print("=" * 58)
    print("  PHANTOM CONDUCTOR v0.5.0")
    print("=" * 58)
    print("  Camera index to use? (Enter = 0):", end=" ", flush=True)
    cam_str = input().strip()
    cam_idx = int(cam_str) if cam_str.isdigit() else 0
    print()
    print("  → Add tracks via the Queue panel (+ADD)")
    print("  → Select I/O devices in the AUDIO I/O panel, then click APPLY")
    print("  → Open hand = PLAY  |  Fist = PAUSE  |  Space = toggle  |  Q = quit")
    print("=" * 58 + "\n")

    state  = PhantomState()
    logger = Logger()

    # ── Worker threads ─────────────────────────────────────────────────────────
    threading.Thread(
        target=bpm_analysis_thread,
        args=(state, logger),
        daemon=True, name="bpm-analysis",
    ).start()

    threading.Thread(
        target=backing_track_thread,
        args=(state, logger),
        daemon=True, name="backing-track",
    ).start()

    threading.Thread(
        target=io_manager_thread,
        args=(state, logger),
        daemon=True, name="io-manager",
    ).start()

    threading.Thread(
        target=gesture_vision_thread,
        args=(cam_idx, state, logger),
        daemon=True, name="gesture-vision",
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
