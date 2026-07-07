import threading

from audio_analysis import bpm_analysis_thread
from audio_input import io_manager_thread
from audio_processing import backing_track_thread
from config import CFG
from gesture_recognition import gesture_vision_thread
from logger import Logger
from pedal import make_pedal_dispatch
from state import PhantomState
from tempo_tapper import tempo_tapper_thread
from ui import PhantomUI


def main():
    print("=" * 58)
    print("  PHANTOM CONDUCTOR v0.5.3")
    print("=" * 58)

    cam_idx = CFG.cam_index

    print()
    print("  -> Añade tracks mediante el panel de la derecha")
    print(
        "  -> Configura el audio/video en el panel de configuración (PRESIONE 'APLICAR' PARA NOTAR LOS CAMBIOS)"
    )
    print("  -> Mano abierta = PLAY  |  Puño = PAUSE  |  Espacio = toggle  |  Q = quit")
    print("  -> Tempo tapper: golpee el piezo, luego habilítelo en Configuración")
    print("=" * 58 + "\n")

    state = PhantomState()
    logger = Logger()

    # Threads
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

    pedal = make_pedal_dispatch(state, logger)

    threading.Thread(
        target=tempo_tapper_thread,
        args=(state, logger, pedal),
        daemon=True,
        name="tempo-tapper",
    ).start()

    # Interfaz
    ui = PhantomUI(state, logger, pedal=pedal)
    try:
        ui.run()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop()
        print("\nPhantom Conductor stopped.")


if __name__ == "__main__":
    main()
