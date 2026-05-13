"""
Phantom Conductor — Integração BPM + Controle Gestual
======================================================
Arquitetura multithreaded:
  - Captura: áudio (sounddevice) + vídeo (OpenCV)
  - Processamento: detecção de BPM + reconhecimento gestual (MediaPipe)
  - Reprodução: backing track com time-stretch beat-a-beat (pyrubberband)
  - UI: Dear PyGui com fila de tracks e file manager

Gestos:
  - Mão Aberta (5 dedos) → PLAY
  - Punho      (0 dedos) → PAUSE

Instalação:
    pip install mediapipe opencv-python numpy sounddevice librosa
                scipy mutagen pyrubberband dearpygui
"""

import cv2
import numpy as np
import sounddevice as sd
import librosa
import threading
import time
import urllib.request
import os
from collections import deque, Counter
from queue import Queue, Empty
from scipy.signal import butter, lfilter
from mutagen.mp3 import MP3

try:
    import pyrubberband as pyrb
    HAS_PYRB = True
except ImportError:
    HAS_PYRB = False
    print("[warn] pyrubberband não encontrado — sem time-stretch")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

from phantom_ui import PhantomState, Logger, PhantomUI

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════
SR             = 44100          # sample rate — 44.1 kHz has wider device support
BUFFER_SEC     = 10
HOP_STREAM_SEC = 0.05
ANALYZE_EVERY  = 0.25
HOP_LENGTH     = 256
MIN_BPM        = 70
MAX_BPM        = 190
SMOOTH_ALPHA   = 0.3
BANDPASS       = (40, 3000)

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Landmark indices
WRIST                         = 0
THUMB_IP,    THUMB_TIP        = 3,  4
INDEX_PIP,   INDEX_TIP        = 6,  8
MIDDLE_MCP,  MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP,    RING_TIP         = 14, 16
PINKY_PIP,   PINKY_TIP        = 18, 20

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE + AUDIO BUFFERS
# ═══════════════════════════════════════════════════════════════════════════════
STATE        = PhantomState()
audio_buffer = deque(maxlen=int(SR * BUFFER_SEC))  # ring buffer for BPM analysis
audio_queue  = Queue(maxsize=60)                    # blocks ready for playback
LOG          = Logger()

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS
# ═══════════════════════════════════════════════════════════════════════════════
def download_model():
    if os.path.exists(MODEL_PATH):
        return
    LOG.info(f"Baixando modelo MediaPipe (~8 MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        LOG.ok("Modelo baixado!")
    except Exception as e:
        LOG.err(f"Erro ao baixar modelo: {e}")
        raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  BPM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def butter_bandpass(lo, hi, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order,
                  [max(1e-3, lo / nyq), min(0.999, hi / nyq)],
                  btype='band')
    return b, a

def apply_bandpass(y, fs, lo, hi):
    b, a = butter_bandpass(lo, hi, fs)
    return lfilter(b, a, y)

def pick_peak_autocorr(ac, lag_min, lag_max):
    seg = ac[lag_min:lag_max + 1]
    if not seg.size:
        return None
    i = lag_min + int(np.argmax(seg))
    if 0 < i < len(ac) - 1:
        y0, y1, y2 = ac[i-1], ac[i], ac[i+1]
        d = y0 - 2*y1 + y2
        if d:
            return i + 0.5 * (y0 - y2) / d
    return float(i)

def octave_correct(bpm, prev):
    if prev is None or not np.isfinite(prev):
        return bpm
    return min([bpm/2, bpm, bpm*2], key=lambda x: abs(x - prev))

def estimate_bpm(y, sr=SR, bpm_prev=None):
    if len(y) < sr * 2:
        return None, {"msg": "buffer curto"}
    y = librosa.util.normalize(y.astype(np.float32))
    y = apply_bandpass(y, sr, *BANDPASS)
    _, y_p = librosa.effects.hpss(y)
    onset = librosa.onset.onset_strength(
        y=y_p, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median)
    if onset.size < 8 or np.max(onset) < 1e-3:
        return None, {"msg": "onset fraco"}
    ac = np.correlate(onset, onset, mode='full')[len(onset) - 1:]
    lag_min = max(2, int(np.floor(60 * sr / (MAX_BPM * HOP_LENGTH))))
    lag_max = min(int(np.ceil(60 * sr / (MIN_BPM * HOP_LENGTH))), len(ac) - 1)
    if lag_min >= lag_max:
        return None, {"msg": "faixa inválida"}
    lag = pick_peak_autocorr(ac, lag_min, lag_max)
    if lag is None or not np.isfinite(lag) or lag <= 0:
        return None, {"msg": "lag inválido"}
    bpm_raw  = 60.0 * sr / (HOP_LENGTH * lag)
    bpm_corr = np.clip(octave_correct(bpm_raw, bpm_prev), MIN_BPM, MAX_BPM)
    if bpm_prev is None or not np.isfinite(bpm_prev):
        bpm_s = bpm_corr
    elif abs(bpm_corr - bpm_prev) < 5.0:
        bpm_s = SMOOTH_ALPHA * bpm_corr + (1 - SMOOTH_ALPHA) * bpm_prev
    else:
        bpm_s = bpm_corr
    return bpm_s, {
        "bpm_raw": float(bpm_raw),
        "bpm_corr": float(bpm_corr),
        "onset_max": float(np.max(onset)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════
class GestureStabilizer:
    def __init__(self, window=10):
        self.history = deque(maxlen=window)

    def update(self, g):
        self.history.append(g)
        return Counter(self.history).most_common(1)[0][0]

def lm_to_array(lm_list):
    return np.array([[p.x, p.y, p.z] for p in lm_list])

def is_finger_up(lm, tip, pip):
    return lm[tip][1] < lm[pip][1]

def is_thumb_up(lm, handedness):
    return (lm[THUMB_TIP][0] < lm[THUMB_IP][0]
            if handedness == "Right"
            else lm[THUMB_TIP][0] > lm[THUMB_IP][0])

def classify_gesture(lm, handedness):
    """
    Returns 'PLAY' (open hand), 'PAUSE' (fist), or None (transition).
    """
    f = [
        is_thumb_up(lm, handedness),
        is_finger_up(lm, INDEX_TIP,  INDEX_PIP),
        is_finger_up(lm, MIDDLE_TIP, MIDDLE_PIP),
        is_finger_up(lm, RING_TIP,   RING_PIP),
        is_finger_up(lm, PINKY_TIP,  PINKY_PIP),
    ]
    count = sum(f)
    if count == 5:
        return "PLAY"
    if count == 0:
        return "PAUSE"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Audio capture callback
# ═══════════════════════════════════════════════════════════════════════════════
def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice on each audio block."""
    if status:
        LOG.warn(f"áudio: {status}")
    mono = np.mean(indata, axis=1).astype(np.float32)
    audio_buffer.extend(mono)
    # Push RMS to waveform display
    rms = float(np.sqrt(np.mean(mono ** 2)))
    STATE.push_waveform(rms)


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — BPM analysis
# ═══════════════════════════════════════════════════════════════════════════════
def bpm_analysis_thread():
    last_analysis = 0.0
    while STATE.alive():
        now = time.time()
        if (now - last_analysis >= ANALYZE_EVERY
                and len(audio_buffer) >= SR * 2):
            last_analysis = now
            y = np.array(audio_buffer, dtype=np.float32)
            bpm_new, dbg = estimate_bpm(y, bpm_prev=STATE.get_bpm())
            if bpm_new and np.isfinite(bpm_new):
                STATE.set_bpm(
                    bpm_new,
                    raw=dbg.get("bpm_raw"),
                    corrected=dbg.get("bpm_corr"),
                    onset_max=dbg.get("onset_max", 0.0),
                )
                STATE.last_bpm_dbg = dbg
            # update buffer fill indicator
            with STATE._lock:
                STATE.buffer_fill = min(1.0, len(audio_buffer) / (SR * BUFFER_SEC))
        time.sleep(ANALYZE_EVERY / 4)


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Backing track playback (time-stretch beat-by-beat)
# ═══════════════════════════════════════════════════════════════════════════════
def _load_track(path: str) -> tuple[np.ndarray, float, float]:
    """Load an audio file; return (samples, bpm_orig, duration)."""
    LOG.info(f"carregando: {os.path.basename(path)}")
    y, _ = librosa.load(path, sr=SR, mono=True)
    dur  = len(y) / SR

    # Try to read BPM tag
    bpm_orig = None
    try:
        tags = MP3(path).tags
        if tags and "TBPM" in tags:
            bpm_orig = float(tags["TBPM"].text[0])
            LOG.ok(f"BPM do arquivo: {bpm_orig:.1f}")
    except Exception:
        pass

    if bpm_orig is None:
        with STATE._lock:
            bpm_orig = STATE.bpm_original or 120.0

    LOG.ok(f"track pronta: {len(y)/SR:.1f}s  bpm_ref={bpm_orig:.1f}")
    return y, bpm_orig, dur


def backing_track_thread():
    """
    Main playback loop.
    - Waits for a track to be loaded (via STATE.load_new_track or queue).
    - Plays it beat-by-beat with live time-stretching to match detected BPM.
    - Responds to PLAY/PAUSE gestures.
    - When a track finishes, advances to the next in the queue.
    """
    y_full    = None
    bpm_orig  = 120.0
    pos       = 0
    next_beat = time.time()
    gain      = 0.85

    while STATE.alive():
        # ── Check for a new track signal ──────────────────────────────────────
        with STATE._lock:
            new_track = STATE.load_new_track
            if new_track:
                STATE.load_new_track = None

        if new_track:
            try:
                y_full, bpm_orig, dur = _load_track(new_track["path"])
                # If the track dict already has a BPM from tags/UI, prefer it
                if new_track.get("bpm"):
                    bpm_orig = float(new_track["bpm"])
                with STATE._lock:
                    STATE.bpm_original   = bpm_orig
                    STATE.bpm_live       = bpm_orig   # seed BPM estimator
                    STATE.stretch_ratio  = 1.0
                    STATE.track_path     = new_track["path"]
                    STATE.track_duration = dur
                    STATE.track_position = 0.0
                    STATE.markers        = []
                pos       = 0
                next_beat = time.time()
                LOG.ok(f"playing: {new_track['name']}")
            except Exception as e:
                LOG.err(f"erro ao carregar track: {e}")
                y_full = None
                new_track = None

        # ── If nothing loaded, idle ───────────────────────────────────────────
        if y_full is None:
            time.sleep(0.05)
            continue

        # ── PAUSE — wait without advancing ───────────────────────────────────
        if not STATE.playing():
            time.sleep(0.05)
            next_beat = time.time()   # reset timing on resume
            continue

        # ── Track finished ────────────────────────────────────────────────────
        if pos >= len(y_full):
            LOG.ok("track finalizada")
            with STATE._lock:
                is_looping = STATE.is_looping
            if is_looping:
                pos = 0
                next_beat = time.time()
                LOG.info("loop: reiniciando")
            else:
                # Try next track in queue
                next_t = STATE.queue.next_track()
                if next_t:
                    with STATE._lock:
                        STATE.load_new_track = next_t
                    y_full = None
                    pos    = 0
                else:
                    # No more tracks — stop
                    STATE.pause()
                    with STATE._lock:
                        STATE.track_position = STATE.track_duration
                    y_full = None
                    LOG.info("fila vazia — parado")
            continue

        # ── Build one beat block ──────────────────────────────────────────────
        bpm_orig_safe = max(1.0, bpm_orig)
        beat_size = int(60.0 / bpm_orig_safe * SR)
        end       = min(pos + beat_size, len(y_full))
        block     = y_full[pos:end].astype(np.float32) * gain

        bpm_live = STATE.get_bpm() or bpm_orig_safe
        rate     = bpm_live / bpm_orig_safe

        # Time-stretch to match live BPM
        if HAS_PYRB and len(block) > 256 and abs(rate - 1.0) > 0.01:
            try:
                block = pyrb.time_stretch(block, SR, rate)
            except Exception as e:
                LOG.warn(f"time-stretch error: {e}")

        # Wait for the beat's scheduled time
        wait = next_beat - time.time()
        if wait > 0:
            time.sleep(wait)

        # Enqueue for playback (non-blocking: drop if full)
        try:
            audio_queue.put_nowait(block)
        except Exception:
            pass  # buffer full — skip block to stay in sync

        # Advance position
        pos       += beat_size
        next_beat += 60.0 / max(1.0, bpm_live)

        # Update playback position in state (every beat)
        STATE.set_position(pos / SR)


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Output playback
# ═══════════════════════════════════════════════════════════════════════════════
def playback_thread():
    """
    Pulls beat blocks from audio_queue and writes to the output device.
    Runs independently from the backing_track_thread so output is smooth.
    """
    with sd.OutputStream(samplerate=SR, channels=1, dtype='float32') as out:
        LOG.ok(f"playback stream aberto: SR={SR}")
        while STATE.alive():
            if not STATE.playing():
                time.sleep(0.02)
                continue
            try:
                block = audio_queue.get(timeout=0.1).astype(np.float32)
                # Clip to prevent distortion
                block = np.clip(block, -1.0, 1.0)
                out.write(block)
            except Empty:
                time.sleep(0.01)
            except Exception as e:
                LOG.err(f"playback error: {e}")
                time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Gesture vision
# ═══════════════════════════════════════════════════════════════════════════════
PLAY_COLOR  = (50,  200, 50)
PAUSE_COLOR = (50,   50, 220)
NONE_COLOR  = (180, 180, 180)

def draw_rect_alpha(img, x1, y1, x2, y2, color, alpha=0.65):
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]-1), min(y2, img.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

def shadow_text(img, text, pos, font, scale, color, thick=2):
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), font, scale, (0,0,0), thick+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)

def draw_hand(frame, lm_px, color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, lm_px[a], lm_px[b], color, 2, cv2.LINE_AA)
    tips = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP}
    for i, pt in enumerate(lm_px):
        r = 7 if i in tips else 4
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (255,255,255), 1, cv2.LINE_AA)

def draw_hud(frame, gesture_raw, fps):
    h, w = frame.shape[:2]
    F  = cv2.FONT_HERSHEY_DUPLEX
    FS = cv2.FONT_HERSHEY_SIMPLEX

    playing  = STATE.playing()
    bpm_live = STATE.get_bpm()
    with STATE._lock:
        bpm_orig = STATE.bpm_original
        dbg      = STATE.last_bpm_dbg

    # Top bar
    draw_rect_alpha(frame, 0, 0, w, 52, (10,10,10))
    shadow_text(frame, "Phantom Conductor", (12, 34), F, 0.75, (220,220,220))
    fps_str = f"FPS: {fps:.0f}"
    tw = cv2.getTextSize(fps_str, FS, 0.5, 1)[0][0]
    shadow_text(frame, fps_str, (w-tw-12, 34), FS, 0.5, (80,255,120))

    # Status panel
    px, py = 12, h - 160
    draw_rect_alpha(frame, px, py, px+340, py+145, (10,10,10))

    state_str = "▶  PLAY" if playing else "⏸  PAUSE"
    state_col = PLAY_COLOR if playing else PAUSE_COLOR
    cv2.rectangle(frame, (px, py), (px+6, py+145), state_col, -1)
    shadow_text(frame, state_str, (px+16, py+40), F, 1.1, state_col, 2)

    if bpm_live:
        ratio     = bpm_live / bpm_orig if bpm_orig else 1.0
        ratio_col = (50, 200, 50) if abs(ratio-1.0) < 0.05 else (50, 200, 255)
        shadow_text(frame, f"BPM live: {bpm_live:.1f}",
                    (px+16, py+80), FS, 0.65, (200,200,200))
        shadow_text(frame, f"ratio: {ratio:.3f}  ref: {bpm_orig:.1f}",
                    (px+16, py+108), FS, 0.52, ratio_col)
        # Sync to PhantomState so the DPG UI stays updated
        STATE.set_bpm(
            bpm_live,
            raw=dbg.get("bpm_raw"),
            corrected=dbg.get("bpm_corr"),
            onset_max=dbg.get("onset_max", 0.0),
        )
    else:
        shadow_text(frame, "Detectando BPM...", (px+16, py+80), FS, 0.6, (140,140,140))

    g_col = (PLAY_COLOR  if gesture_raw == "PLAY"
             else PAUSE_COLOR if gesture_raw == "PAUSE"
             else NONE_COLOR)
    g_label = {
        "PLAY":  "Gesto: Mao Aberta  ->  PLAY",
        "PAUSE": "Gesto: Punho       ->  PAUSE",
        None:    "Gesto: --",
    }.get(gesture_raw, "Gesto: --")
    shadow_text(frame, g_label, (px+16, py+135), FS, 0.48, g_col, 1)

    shadow_text(frame, "Mao Aberta=PLAY   Punho=PAUSE   Q=sair   Espaco=toggle",
                (12, h-8), FS, 0.42, (100,100,100), 1)


def gesture_vision_thread(cam_idx=0):
    """Captures camera frames, detects hand gestures, controls play/pause."""
    download_model()

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        LOG.err(f"visão: não foi possível abrir câmera {cam_idx}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    LOG.ok(f"câmera {cam_idx} aberta: 1280×720")

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.55,
    )
    LOG.ok("HandLandmarker carregado")

    stabilizer      = GestureStabilizer(window=10)
    prev_stable     = None
    prev_time       = time.time()
    MIN_HOLD        = 8
    hold_count      = 0
    pending_gesture = None

    with HandLandmarker.create_from_options(options) as detector:
        while STATE.alive():
            ret, frame = cap.read()
            if not ret:
                LOG.err("visão: falha ao ler frame")
                break
            frame = cv2.flip(frame, 1)
            h_f, w_f = frame.shape[:2]

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_img)

            raw_gesture = None

            if result.hand_landmarks:
                hand_lm    = result.hand_landmarks[0]
                hand_info  = result.handedness[0]
                handedness = hand_info[0].category_name

                lm    = lm_to_array(hand_lm)
                lm_px = [(int(p.x * w_f), int(p.y * h_f)) for p in hand_lm]

                raw_gesture = classify_gesture(lm, handedness)
                stable      = stabilizer.update(raw_gesture or "None")
                stable      = None if stable == "None" else stable

                # Debounce
                if stable == pending_gesture:
                    hold_count += 1
                else:
                    pending_gesture = stable
                    hold_count      = 1

                STATE.set_gesture(
                    stable or "NO HAND",
                    confidence=0.0,
                    hold_frames=hold_count,
                    hands=1,
                )

                if (hold_count >= MIN_HOLD
                        and stable != prev_stable
                        and stable in ("PLAY", "PAUSE")):
                    if stable == "PLAY":
                        STATE.play()
                        LOG.ok(f"gesto confirmado: PLAY  (hold={hold_count})")
                    else:
                        STATE.pause()
                        LOG.ok(f"gesto confirmado: PAUSE  (hold={hold_count})")
                    STATE.set_command(stable)
                    prev_stable = stable
                    hold_count  = 0
                elif stable and stable != prev_stable:
                    LOG.info(f"gesto: {stable}  hold={hold_count}/{MIN_HOLD}")

                hand_color = (PLAY_COLOR  if stable == "PLAY"
                              else PAUSE_COLOR if stable == "PAUSE"
                              else (180,180,180))
                draw_hand(frame, lm_px, hand_color)

                wx, wy  = lm_px[WRIST]
                label   = stable or "..."
                lbl_col = (PLAY_COLOR  if stable == "PLAY"
                           else PAUSE_COLOR if stable == "PAUSE"
                           else (200,200,200))
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0][0]
                draw_rect_alpha(frame,
                                wx-tw//2-12, max(wy-55, 5),
                                wx+tw//2+12, max(wy-10, 50),
                                lbl_col, 0.45)
                shadow_text(frame, label, (wx-tw//2, max(wy-15, 45)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)

            else:
                if pending_gesture is not None:
                    LOG.info("gesto: sem mão detectada")
                pending_gesture = None
                hold_count      = 0
                STATE.set_gesture("NO HAND", confidence=0.0, hold_frames=0, hands=0)

            draw_hud(frame, raw_gesture, fps)
            cv2.imshow("Phantom Conductor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                LOG.warn("visão: encerrado pelo usuário (Q)")
                STATE.stop()
                break
            elif key == ord(' '):
                new_state = STATE.toggle()
                LOG.info(f"tecla espaço: {'PLAY' if new_state else 'PAUSE'}")

    cap.release()
    cv2.destroyAllWindows()
    LOG.warn("gesture_vision_thread encerrada")
    STATE.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS — Device listing
# ═══════════════════════════════════════════════════════════════════════════════
def list_input_devices():
    print("\n=== Dispositivos de ENTRADA ===")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i:>3}  {d['name']}")
    print("================================")

def list_output_devices():
    print("\n=== Dispositivos de SAÍDA ===")
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            print(f"  {i:>3}  {d['name']}")
    print("================================")

def pick_output_device(requested_idx: int | None = None) -> int | None:
    """
    Return a valid output device index.
    Tries requested_idx first; falls back to system default.
    Returns None if we should let sounddevice choose automatically.
    """
    if requested_idx is None:
        return None
    try:
        d = sd.query_devices(requested_idx)
        if d["max_output_channels"] > 0:
            return requested_idx
        print(f"[warn] dispositivo {requested_idx} não tem saída — usando padrão")
    except Exception:
        print(f"[warn] dispositivo {requested_idx} inválido — usando padrão")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 56)
    print("  PHANTOM CONDUCTOR — v0.4.0")
    print("=" * 56)

    # ── Input device ─────────────────────────────────────────────────────────
    list_input_devices()
    dev_in_str = input("Índice do microfone (Enter=padrão): ").strip()
    dev_in = int(dev_in_str) if dev_in_str else None

    # ── Output device ─────────────────────────────────────────────────────────
    list_output_devices()
    dev_out_str = input("Índice da saída de áudio (Enter=padrão): ").strip()
    dev_out = pick_output_device(int(dev_out_str) if dev_out_str else None)

    # ── Camera ────────────────────────────────────────────────────────────────
    cam_str = input("Índice da câmera [0]: ").strip()
    cam_idx = int(cam_str) if cam_str else 0

    # ── Optional initial track ────────────────────────────────────────────────
    mp3_path = input("Backing track inicial (.mp3/.wav, Enter=pular): ").strip()
    if mp3_path and os.path.isfile(mp3_path):
        bpm_tag = None
        try:
            tags = MP3(mp3_path).tags
            if tags and "TBPM" in tags:
                bpm_tag = float(tags["TBPM"].text[0])
        except Exception:
            pass
        bpm_q = bpm_tag
        if bpm_q is None:
            bpm_q_str = input("BPM original (manual, Enter=120): ").strip()
            bpm_q = float(bpm_q_str) if bpm_q_str else 120.0
        STATE.bpm_original = bpm_q
        # Add to queue and signal load
        STATE.queue.add(mp3_path, bpm=bpm_q)
        with STATE._lock:
            STATE.load_new_track = {
                "path": mp3_path,
                "name": os.path.basename(mp3_path),
                "bpm":  bpm_q,
            }
        LOG.ok(f"initial track: {os.path.basename(mp3_path)}  BPM={bpm_q:.1f}")
    else:
        LOG.info("sem track inicial — use o painel Queue para adicionar")

    print("\n" + "=" * 56)
    print("  🖐  Mão Aberta  →  PLAY")
    print("  ✊  Punho       →  PAUSE")
    print("  [Espaço]        →  Toggle  |  [Q] → Sair")
    print("=" * 56 + "\n")

    blocksize = int(SR * HOP_STREAM_SEC)

    # ── Start worker threads ───────────────────────────────────────────────────
    threading.Thread(target=bpm_analysis_thread, daemon=True,
                     name="bpm-analysis").start()
    threading.Thread(target=backing_track_thread, daemon=True,
                     name="backing-track").start()

    # Playback thread — pass output device
    def _playback_thread_with_dev():
        with sd.OutputStream(
            device=dev_out,
            samplerate=SR,
            channels=1,
            dtype='float32',
        ) as out:
            LOG.ok(f"playback stream aberto: SR={SR}  dev={dev_out}")
            while STATE.alive():
                if not STATE.playing():
                    time.sleep(0.02)
                    continue
                try:
                    block = audio_queue.get(timeout=0.1).astype(np.float32)
                    block = np.clip(block, -1.0, 1.0)
                    out.write(block)
                except Empty:
                    time.sleep(0.01)
                except Exception as e:
                    LOG.err(f"playback error: {e}")
                    time.sleep(0.05)

    threading.Thread(target=_playback_thread_with_dev, daemon=True,
                     name="playback").start()

    # ── Audio input stream ────────────────────────────────────────────────────
    with sd.InputStream(
        device=dev_in,
        channels=1,
        samplerate=SR,
        blocksize=blocksize,
        callback=audio_callback,
    ):
        LOG.ok(f"input stream aberto: SR={SR}  dev={dev_in}")

        # Gesture vision in a daemon thread (has its own OpenCV window)
        threading.Thread(target=gesture_vision_thread, args=(cam_idx,),
                         daemon=True, name="gesture-vision").start()

        # Run the DPG UI on the main thread (DearPyGui requires this)
        ui = PhantomUI(STATE, LOG)
        try:
            ui.run()
        except KeyboardInterrupt:
            pass
        finally:
            STATE.stop()
            print("\n🛑 Encerrado.")


if __name__ == "__main__":
    main()