import ctypes
import os
import queue
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np

from config import CFG
from gesture_trainer import TRAINER
from logger import Logger
from state import PhantomState
from video_input import download_model, make_landmarker, open_camera, read_frame

# Índices de Landmarks (MediaPipe)
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]

PLAY_COLOR = (50, 200, 50)
PAUSE_COLOR = (50, 50, 220)
POINT_COLOR = (220, 160, 50)
PEACE_COLOR = (160, 80, 220)
CUSTOM_COLOR = (50, 210, 210)
NONE_COLOR = (180, 180, 180)

BUILTIN_COLORS = {
    "PLAY": PLAY_COLOR,
    "PAUSE": PAUSE_COLOR,
    "POINT": POINT_COLOR,
    "PEACE": PEACE_COLOR,
}

MIN_HOLD = 8

# Buffer de landmarks (evita una alocación de 21x3 cada fotograma)
_LM_BUF = np.empty((21, 3), dtype=np.float32)


# PRIORIZADOR DE THREADS (Linux, prioriza audio antes que video)


def _try_set_nice(level: int = -10) -> None:
    try:
        os.nice(level)
    except (OSError, AttributeError):
        pass


def _try_set_realtime(priority: int = 10) -> None:
    import sys

    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(),
                2,  # Prioridad más alta
            )
        except Exception:
            pass
    else:
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)

            class _SchedParam(ctypes.Structure):
                _fields_ = [("sched_priority", ctypes.c_int)]

            param = _SchedParam(priority)
            SCHED_FIFO = 1
            ret = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
            if ret != 0:
                _try_set_nice(-10)
        except Exception:
            _try_set_nice(-10)


# CLASIFICADOR BASADO EN REGLAS (Gestos por defecto)


def _lm_arr(lm_list) -> np.ndarray:
    # Escribe las posiciones de landmarks al buffer y lo retorna.
    for i, p in enumerate(lm_list):
        _LM_BUF[i, 0] = p.x
        _LM_BUF[i, 1] = p.y
        _LM_BUF[i, 2] = p.z
    return _LM_BUF


def _finger_up(lm: np.ndarray, tip: int, pip: int) -> bool:
    return lm[tip][1] < lm[pip][1]


def _thumb_up(lm: np.ndarray, handedness: str) -> bool:
    return (
        lm[THUMB_TIP][0] < lm[THUMB_IP][0]
        if handedness == "Right"
        else lm[THUMB_TIP][0] > lm[THUMB_IP][0]
    )


def classify_gesture(lm: np.ndarray, handedness: str) -> str | None:
    """
    Clasificador basado en reglas para los gestos por defecto, para cuando TRAINER.match() no encuentra gestos entrenados por el usuario.
    Retorna "PLAY" / "PAUSE" / "POINT" / "PEACE" / None.
    """
    thumb = _thumb_up(lm, handedness)
    index = _finger_up(lm, INDEX_TIP, INDEX_PIP)
    middle = _finger_up(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring = _finger_up(lm, RING_TIP, RING_PIP)
    pinky = _finger_up(lm, PINKY_TIP, PINKY_PIP)
    n = sum([thumb, index, middle, ring, pinky])

    if n == 5:
        return "PLAY"
    if n == 0:
        return "PAUSE"
    if index and not middle and not ring and not pinky:
        return "POINT"
    if index and middle and not ring and not pinky:
        return "PEACE"
    return None


# ESTABILIZADOR


class GestureStabilizer:
    def __init__(self, window: int = 10):
        self._h = deque(maxlen=window)

    def update(self, gesture: str | None) -> str | None:
        self._h.append(gesture)
        return Counter(self._h).most_common(1)[0][0]


# CLASIFICADOR


def classify_with_trainer(
    lm: np.ndarray, handedness: str
) -> tuple[str | None, bool, float]:
    """
    Se ejecuta la plantilla definida por el usuario (Entrenada)

    Retorna
    (gesture_name, is_custom, match_distance)
    """
    custom_name, dist = TRAINER.match(lm)
    if custom_name is not None:
        return custom_name, True, dist

    builtin = classify_gesture(lm, handedness)
    return builtin, False, 0.0


# DIBUJAR HUD


def _draw_rect_alpha(img, x1, y1, x2, y2, color, alpha=0.65):
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1] - 1), min(y2, img.shape[0] - 1)
    if x2 <= x1 or y2 <= y1:
        return
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _shadow_text(img, text, pos, font, scale, color, thick=2):
    x, y = pos
    cv2.putText(
        img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA
    )
    cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)


def _draw_hand(frame, lm_px: list[tuple[int, int]], color):
    tips = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP}
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, lm_px[a], lm_px[b], color, 2, cv2.LINE_AA)
    for i, pt in enumerate(lm_px):
        r = 7 if i in tips else 4
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_hud(
    frame,
    gesture_raw: str | None,
    is_custom: bool,
    fps: float,
    state: PhantomState,
    recording: bool,
    sample_count: int,
):
    h, w = frame.shape[:2]
    F = cv2.FONT_HERSHEY_DUPLEX
    FS = cv2.FONT_HERSHEY_SIMPLEX

    playing = state.playing()
    bpm_live = state.get_bpm()
    with state._lock:
        bpm_orig = state.bpm_original
        dbg = state.last_bpm_dbg

    # Barra superior
    _draw_rect_alpha(frame, 0, 0, w, 52, (10, 10, 10))
    _shadow_text(frame, "Phantom Conductor", (12, 34), F, 0.75, (220, 220, 220))
    fps_str = f"FPS: {fps:.0f}"
    tw = cv2.getTextSize(fps_str, FS, 0.5, 1)[0][0]
    _shadow_text(frame, fps_str, (w - tw - 12, 34), FS, 0.5, (80, 255, 120))

    if recording:
        banner = f"● REC  {sample_count} fotogramas capturados"
        _draw_rect_alpha(frame, 0, 52, w, 90, (60, 10, 10))
        btw = cv2.getTextSize(banner, F, 0.7, 2)[0][0]
        _shadow_text(frame, banner, ((w - btw) // 2, 78), F, 0.7, (80, 80, 255), 2)

    px, py = 12, h - 160
    _draw_rect_alpha(frame, px, py, px + 360, py + 145, (10, 10, 10))
    state_str = "PLAY" if playing else "PAUSE"
    state_col = PLAY_COLOR if playing else PAUSE_COLOR
    cv2.rectangle(frame, (px, py), (px + 6, py + 145), state_col, -1)
    _shadow_text(frame, state_str, (px + 16, py + 40), F, 1.1, state_col, 2)

    if bpm_live:
        ratio = bpm_live / bpm_orig if bpm_orig else 1.0
        ratio_col = (50, 200, 50) if abs(ratio - 1.0) < 0.05 else (50, 200, 255)
        _shadow_text(
            frame,
            f"BPM live: {bpm_live:.1f}",
            (px + 16, py + 80),
            FS,
            0.65,
            (200, 200, 200),
        )
        _shadow_text(
            frame,
            f"ratio: {ratio:.3f}  ref: {bpm_orig:.1f}",
            (px + 16, py + 108),
            FS,
            0.52,
            ratio_col,
        )
        state.set_bpm(
            bpm_live,
            raw=dbg.get("bpm_raw"),
            corrected=dbg.get("bpm_corr"),
            onset_max=dbg.get("onset_max", 0.0),
        )
    else:
        _shadow_text(
            frame, "Detectando BPM...", (px + 16, py + 80), FS, 0.6, (140, 140, 140)
        )

    if is_custom:
        g_col = CUSTOM_COLOR
        g_label = f"[custom] {gesture_raw}"
    else:
        g_col = BUILTIN_COLORS.get(gesture_raw, NONE_COLOR)
        LABELS = {
            "PLAY": "Open Hand -> PLAY",
            "PAUSE": "Fist -> PAUSE",
            "POINT": "Index -> POINT",
            "PEACE": "Peace -> PEACE",
            None: "Gesture: --",
        }
        g_label = LABELS.get(gesture_raw, "Gesture: --")

    _shadow_text(frame, g_label, (px + 16, py + 135), FS, 0.48, g_col, 1)
    _shadow_text(
        frame,
        "5=PLAY  0=PAUSE  1=POINT  2=PEACE  [custom]  Space=toggle  Q=quit",
        (12, h - 8),
        FS,
        0.40,
        (100, 100, 100),
        1,
    )


# PIPELINE DE VIDEO


def _capture_thread(
    cap: cv2.VideoCapture,
    frame_q: queue.Queue,
    state: PhantomState,
    logger: Logger,
) -> None:
    """
    Fase 1: Captura de video
    """
    while state.alive():
        frame, mp_img = read_frame(cap)
        if frame is None:
            logger.err("capture: lectura de fotograma falló, abortando")
            state.stop()
            break

        # Expone el último fotograma redimensionado
        with state._lock:
            state.latest_frame = cv2.resize(
                frame, (320, 240), interpolation=cv2.INTER_NEAREST
            )

        try:
            frame_q.put_nowait((frame, mp_img))
        except queue.Full:
            pass  # Inferencia aún ocupada, saltando ciclo


def _inference_thread(
    frame_q: queue.Queue,
    result_q: queue.Queue,
    state: PhantomState,
    logger: Logger,
) -> None:
    logger.ok("HandLandmarker cargado")
    skip_counter = 0
    with make_landmarker() as detector:
        while state.alive():
            try:
                frame, mp_img = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if CFG.inference_skip_enabled:
                skip_counter += 1
                if skip_counter % CFG.inference_skip_frames != 0:
                    continue  # Saltar fotograma

            result = detector.detect(mp_img)

            try:
                result_q.put_nowait((frame, result))
            except queue.Full:
                pass


# ENDPOINT PÚBLICO


def gesture_vision_thread(cam_idx: int, state: PhantomState, logger: Logger):
    print("STARTED VISION THREAD")
    # Orquestración del Pipeline, luego clasifica, dibuja el HUD y muestra la imagen en esta thread.
    download_model(logger)

    try:
        cap = open_camera(cam_idx, logger)
    except RuntimeError as e:
        logger.err(f"vision: {e}")
        return

    frame_q: queue.Queue = queue.Queue(maxsize=1)
    result_q: queue.Queue = queue.Queue(maxsize=1)

    t_capture = threading.Thread(
        target=_capture_thread,
        args=(cap, frame_q, state, logger),
        name="gesture-capture",
        daemon=True,
    )
    t_inference = threading.Thread(
        target=_inference_thread,
        args=(frame_q, result_q, state, logger),
        name="gesture-inference",
        daemon=True,
    )
    t_capture.start()
    t_inference.start()

    # Clasificación
    stab = GestureStabilizer(window=10)
    prev_stable = None
    prev_time = time.time()
    hold_count = 0
    pending = None
    last_lm = None

    while state.alive():
        # Carga el último resultado de la inferencia
        try:
            frame, result = result_q.get(timeout=0.05)
        except queue.Empty:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.warn("vision: quit by user (Q)")
                state.stop()
                break
            elif key == ord(" "):
                new_state = state.toggle()
                logger.info(f"space: {'PLAY' if new_state else 'PAUSE'}")
            continue

        h_f, w_f = frame.shape[:2]  # tamaños del fotograma
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # Integración con gesture_trainer.py
        if getattr(state, "clip_recording_active", False):
            if last_lm is not None and TRAINER.is_recording():
                TRAINER.capture_frame(last_lm.copy())

        # Clasificación
        raw_gesture = None
        is_custom = False

        if result.hand_landmarks:
            hand_lm = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name

            lm = _lm_arr(hand_lm)
            lm_px = [(int(p.x * w_f), int(p.y * h_f)) for p in hand_lm]

            last_lm = lm.copy()  # copia para trainer

            raw_gesture, is_custom, _dist = classify_with_trainer(lm, handedness)
            stable = stab.update(raw_gesture)

            if stable == pending:
                hold_count += 1
            else:
                pending = stable
                hold_count = 1

            state.set_gesture(
                stable or "NO HAND", confidence=0.0, hold_frames=hold_count, hands=1
            )

            # Ejecución de señal
            min_hold = CFG.gesture_hold_frames
            if hold_count >= min_hold and stable is not None and stable != prev_stable:
                cmd = (
                    TRAINER.command_for(stable)
                    if is_custom
                    else CFG.gesture_map.get(stable, "none")
                )
                if cmd and cmd != "none":
                    state.dispatch_command(cmd, logger)
                    src = "custom" if is_custom else "builtin"
                    logger.ok(f"gesture [{src}] {stable} -> {cmd}  (hold={hold_count})")
                else:
                    logger.info(f"gesture {stable} mapped to 'none' — skipped")

                prev_stable = stable
                hold_count = 0
            elif stable and stable != prev_stable:
                logger.info(f"gesture: {stable}  hold={hold_count}/{min_hold}")

            hand_color = (
                CUSTOM_COLOR if is_custom else BUILTIN_COLORS.get(stable, NONE_COLOR)
            )
            _draw_hand(frame, lm_px, hand_color)

            wx, wy = lm_px[WRIST]
            label = stable or "..."
            lbl_col = (
                CUSTOM_COLOR
                if is_custom
                else BUILTIN_COLORS.get(stable, (200, 200, 200))
            )
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0][0]
            _draw_rect_alpha(
                frame,
                wx - tw // 2 - 12,
                max(wy - 55, 5),
                wx + tw // 2 + 12,
                max(wy - 10, 50),
                lbl_col,
                0.45,
            )
            _shadow_text(
                frame,
                label,
                (wx - tw // 2, max(wy - 15, 45)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        else:
            last_lm = None
            if pending is not None:
                logger.info("gesture: no hand detected")
            pending = None
            hold_count = 0
            state.set_gesture("NO HAND", confidence=0.0, hold_frames=0, hands=0)

        _draw_hud(
            frame,
            raw_gesture,
            is_custom,
            fps,
            state,
            recording=TRAINER.is_clip_active(),
            sample_count=TRAINER.frame_count(),
        )

        cv2.imshow("Phantom Conductor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.warn("vision: quit by user (Q)")
            state.stop()
            break
        elif key == ord(" "):
            new_state = state.toggle()
            logger.info(f"space: {'PLAY' if new_state else 'PAUSE'}")

    cap.release()
    # cv2.destroyAllWindows()
    logger.warn("gesture_vision_thread abortada")
    state.stop()
