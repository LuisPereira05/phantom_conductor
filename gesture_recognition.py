"""
Phantom Conductor — Gesture Recognition & HUD
=============================================
Responsibilities
----------------
* classify_gesture      : maps 21 MediaPipe landmarks → PLAY / PAUSE / None
* GestureStabilizer     : majority-vote window (default 10 frames)
* _draw_hand / _draw_hud: OpenCV drawing helpers for the camera window
* gesture_vision_thread : main loop — reads frames via video_input,
                          classifies gestures, writes to PhantomState,
                          and displays the annotated camera window.

Signal execution (PLAY / PAUSE) is performed here as soon as a gesture
is held for MIN_HOLD consecutive frames.
"""

import cv2
import math
import time
import numpy as np
from collections import deque, Counter

from state       import PhantomState
from logger      import Logger
from video_input import download_model, open_camera, read_frame, make_landmarker

# ── Landmark indices ───────────────────────────────────────────────────────────
WRIST                               = 0
THUMB_IP,    THUMB_TIP              = 3,  4
INDEX_PIP,   INDEX_TIP              = 6,  8
MIDDLE_MCP,  MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP,    RING_TIP               = 14, 16
PINKY_PIP,   PINKY_TIP              = 18, 20

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

PLAY_COLOR  = (50,  200, 50)
PAUSE_COLOR = (50,   50, 220)
NONE_COLOR  = (180, 180, 180)

MIN_HOLD = 8


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def _lm_arr(lm_list) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in lm_list])


def _finger_up(lm: np.ndarray, tip: int, pip: int) -> bool:
    return lm[tip][1] < lm[pip][1]


def _thumb_up(lm: np.ndarray, handedness: str) -> bool:
    return (lm[THUMB_TIP][0] < lm[THUMB_IP][0]
            if handedness == "Right"
            else lm[THUMB_TIP][0] > lm[THUMB_IP][0])


def classify_gesture(lm: np.ndarray, handedness: str) -> str | None:
    """
    Returns
    -------
    "PLAY"  — open hand (5 fingers extended)
    "PAUSE" — fist (0 fingers extended)
    None    — transitional / ambiguous
    """
    fingers = [
        _thumb_up(lm, handedness),
        _finger_up(lm, INDEX_TIP,  INDEX_PIP),
        _finger_up(lm, MIDDLE_TIP, MIDDLE_PIP),
        _finger_up(lm, RING_TIP,   RING_PIP),
        _finger_up(lm, PINKY_TIP,  PINKY_PIP),
    ]
    n = sum(fingers)
    if n == 5: return "PLAY"
    if n == 0: return "PAUSE"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE STABILIZER
# ═══════════════════════════════════════════════════════════════════════════════

class GestureStabilizer:
    """Majority-vote smoothing over a sliding window of recent gestures."""

    def __init__(self, window: int = 10):
        self._h = deque(maxlen=window)

    def update(self, gesture: str | None) -> str | None:
        self._h.append(gesture)
        winner = Counter(self._h).most_common(1)[0][0]
        return winner   # may be None


# ═══════════════════════════════════════════════════════════════════════════════
#  HUD DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_rect_alpha(img, x1, y1, x2, y2, color, alpha=0.65):
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]-1), min(y2, img.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def _shadow_text(img, text, pos, font, scale, color, thick=2):
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), font, scale, (0,0,0), thick+1, cv2.LINE_AA)
    cv2.putText(img, text, pos,        font, scale, color,   thick,   cv2.LINE_AA)


def _draw_hand(frame, lm_px: list[tuple[int, int]], color):
    tips = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP}
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, lm_px[a], lm_px[b], color, 2, cv2.LINE_AA)
    for i, pt in enumerate(lm_px):
        r = 7 if i in tips else 4
        cv2.circle(frame, pt, r, color,       -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (255,255,255), 1, cv2.LINE_AA)


def _draw_hud(frame, gesture_raw: str | None, fps: float,
              state: PhantomState):
    """Render the status overlay onto the camera frame (in-place)."""
    h, w = frame.shape[:2]
    F  = cv2.FONT_HERSHEY_DUPLEX
    FS = cv2.FONT_HERSHEY_SIMPLEX

    playing  = state.playing()
    bpm_live = state.get_bpm()
    with state._lock:
        bpm_orig = state.bpm_original
        dbg      = state.last_bpm_dbg

    # Top bar
    _draw_rect_alpha(frame, 0, 0, w, 52, (10,10,10))
    _shadow_text(frame, "Phantom Conductor", (12, 34), F, 0.75, (220,220,220))
    fps_str = f"FPS: {fps:.0f}"
    tw = cv2.getTextSize(fps_str, FS, 0.5, 1)[0][0]
    _shadow_text(frame, fps_str, (w-tw-12, 34), FS, 0.5, (80,255,120))

    # Status panel bottom-left
    px, py = 12, h - 160
    _draw_rect_alpha(frame, px, py, px+340, py+145, (10,10,10))
    state_str = "▶  PLAY" if playing else "⏸  PAUSE"
    state_col = PLAY_COLOR if playing else PAUSE_COLOR
    cv2.rectangle(frame, (px, py), (px+6, py+145), state_col, -1)
    _shadow_text(frame, state_str, (px+16, py+40), F, 1.1, state_col, 2)

    if bpm_live:
        ratio     = bpm_live / bpm_orig if bpm_orig else 1.0
        ratio_col = (50,200,50) if abs(ratio-1.0) < 0.05 else (50,200,255)
        _shadow_text(frame, f"BPM live: {bpm_live:.1f}",
                     (px+16, py+80), FS, 0.65, (200,200,200))
        _shadow_text(frame, f"ratio: {ratio:.3f}  ref: {bpm_orig:.1f}",
                     (px+16, py+108), FS, 0.52, ratio_col)
        state.set_bpm(bpm_live,
                      raw=dbg.get("bpm_raw"),
                      corrected=dbg.get("bpm_corr"),
                      onset_max=dbg.get("onset_max", 0.0))
    else:
        _shadow_text(frame, "Detecting BPM…",
                     (px+16, py+80), FS, 0.6, (140,140,140))

    g_col   = (PLAY_COLOR  if gesture_raw == "PLAY"
               else PAUSE_COLOR if gesture_raw == "PAUSE"
               else NONE_COLOR)
    g_label = {"PLAY":  "Gesture: Open Hand → PLAY",
               "PAUSE": "Gesture: Fist      → PAUSE",
               None:    "Gesture: --"}.get(gesture_raw, "Gesture: --")
    _shadow_text(frame, g_label, (px+16, py+135), FS, 0.48, g_col, 1)
    _shadow_text(frame,
                 "Open Hand=PLAY   Fist=PAUSE   Space=toggle   Q=quit",
                 (12, h-8), FS, 0.42, (100,100,100), 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE VISION THREAD  (main loop)
# ═══════════════════════════════════════════════════════════════════════════════

def gesture_vision_thread(cam_idx: int, state: PhantomState, logger: Logger):
    """
    Full gesture pipeline running in a daemon thread:
      1. download model if needed
      2. open camera
      3. per frame: detect → classify → stabilize → hold-count → act
      4. draw hand skeleton + HUD onto camera frame
      5. show in cv2 window; handle Q / Space keys

    Writes to state: gesture fields, PLAY/PAUSE command.
    Exits by calling state.stop() when the user presses Q.
    """
    download_model(logger)

    try:
        cap = open_camera(cam_idx, logger)
    except RuntimeError as e:
        logger.err(f"vision: {e}")
        return

    logger.ok("HandLandmarker loaded")

    stab        = GestureStabilizer(window=10)
    prev_stable = None
    prev_time   = time.time()
    hold_count  = 0
    pending     = None

    with make_landmarker() as detector:
        while state.alive():
            frame, mp_img = read_frame(cap)
            if frame is None:
                logger.err("vision: frame read failed")
                break

            h_f, w_f = frame.shape[:2]
            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            result      = detector.detect(mp_img)
            raw_gesture = None

            if result.hand_landmarks:
                hand_lm    = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name
                lm    = _lm_arr(hand_lm)
                lm_px = [(int(p.x * w_f), int(p.y * h_f)) for p in hand_lm]

                raw_gesture = classify_gesture(lm, handedness)
                stable      = stab.update(raw_gesture)

                if stable == pending:
                    hold_count += 1
                else:
                    pending    = stable
                    hold_count = 1

                state.set_gesture(stable or "NO HAND",
                                  confidence=0.0,
                                  hold_frames=hold_count,
                                  hands=1)

                # ── Signal execution ──────────────────────────────────────────
                if (hold_count >= MIN_HOLD
                        and stable != prev_stable
                        and stable in ("PLAY", "PAUSE")):
                    if stable == "PLAY":
                        state.play()
                        logger.ok(f"gesture confirmed: PLAY  (hold={hold_count})")
                    else:
                        state.pause()
                        logger.ok(f"gesture confirmed: PAUSE (hold={hold_count})")
                    state.set_command(stable)
                    prev_stable = stable
                    hold_count  = 0
                elif stable and stable != prev_stable:
                    logger.info(f"gesture: {stable}  hold={hold_count}/{MIN_HOLD}")

                hand_color = (PLAY_COLOR  if stable == "PLAY"
                              else PAUSE_COLOR if stable == "PAUSE"
                              else (180,180,180))
                _draw_hand(frame, lm_px, hand_color)

                # Wrist label
                wx, wy  = lm_px[WRIST]
                label   = stable or "..."
                lbl_col = (PLAY_COLOR  if stable == "PLAY"
                           else PAUSE_COLOR if stable == "PAUSE"
                           else (200,200,200))
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0][0]
                _draw_rect_alpha(frame,
                                 wx-tw//2-12, max(wy-55, 5),
                                 wx+tw//2+12, max(wy-10, 50),
                                 lbl_col, 0.45)
                _shadow_text(frame, label, (wx-tw//2, max(wy-15, 45)),
                             cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)
            else:
                if pending is not None:
                    logger.info("gesture: no hand detected")
                pending    = None
                hold_count = 0
                state.set_gesture("NO HAND", confidence=0.0,
                                  hold_frames=0, hands=0)

            _draw_hud(frame, raw_gesture, fps, state)
            cv2.imshow("Phantom Conductor — Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.warn("vision: quit by user (Q)")
                state.stop()
                break
            elif key == ord(" "):
                new_state = state.toggle()
                logger.info(f"space: {'PLAY' if new_state else 'PAUSE'}")

    cap.release()
    cv2.destroyAllWindows()
    logger.warn("gesture_vision_thread exited")
    state.stop()
