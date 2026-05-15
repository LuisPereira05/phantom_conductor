"""
Phantom Conductor — Integração BPM + Controle Gestual
======================================================
Arquitetura multithreaded:
  - Captura: áudio (sounddevice) + vídeo (OpenCV)
  - Processamento: detecção de BPM + reconhecimento gestual (MediaPipe)
  - Reprodução: backing track com time-stretch beat-a-beat (pyrubberband)
  - UI: Dear PyGui — fila de tracks, editor de BPM, painel I/O

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

# mutagen — supports MP3, WAV, FLAC, OGG …
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
    print("[warn] pyrubberband not found — time-stretch disabled")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

from phantom_ui import PhantomState, Logger, PhantomUI

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
SR             = 44100       # 44.1 kHz — widest device compatibility
BUFFER_SEC     = 10
HOP_STREAM_SEC = 0.05
ANALYZE_EVERY  = 0.25
HOP_LENGTH     = 256
MIN_BPM        = 60
MAX_BPM        = 200
SMOOTH_ALPHA   = 0.3
BANDPASS       = (40, 3000)

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# MediaPipe landmark indices
WRIST                             = 0
THUMB_IP,    THUMB_TIP            = 3,  4
INDEX_PIP,   INDEX_TIP            = 6,  8
MIDDLE_MCP,  MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP,    RING_TIP             = 14, 16
PINKY_PIP,   PINKY_TIP            = 18, 20

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE + RING BUFFERS  (module-level so threads can share them)
# ═══════════════════════════════════════════════════════════════════════════════
STATE        = PhantomState()
audio_buffer = deque(maxlen=int(SR * BUFFER_SEC))  # mic samples for BPM analysis
audio_queue  = Queue(maxsize=80)                    # beat blocks ready to play
LOG          = Logger()

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS — MediaPipe model download
# ═══════════════════════════════════════════════════════════════════════════════
def download_model():
    if os.path.exists(MODEL_PATH):
        return
    LOG.info("Baixando modelo MediaPipe (~8 MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        LOG.ok("Modelo baixado com sucesso.")
    except Exception as e:
        LOG.err(f"Falha ao baixar modelo: {e}")
        raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  BPM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def _butter_bandpass(lo, hi, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order,
                  [max(1e-3, lo / nyq), min(0.999, hi / nyq)],
                  btype="band")
    return b, a

def _apply_bandpass(y, fs, lo, hi):
    b, a = _butter_bandpass(lo, hi, fs)
    return lfilter(b, a, y)

def _pick_peak(ac, lag_min, lag_max):
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

def _octave_correct(bpm, prev):
    if prev is None or not np.isfinite(prev):
        return bpm
    return min([bpm/2, bpm, bpm*2], key=lambda x: abs(x - prev))

def estimate_bpm(y, sr=SR, bpm_prev=None):
    if len(y) < sr * 2:
        return None, {"msg": "buffer too short"}
    y = librosa.util.normalize(y.astype(np.float32))
    y = _apply_bandpass(y, sr, *BANDPASS)
    _, y_p = librosa.effects.hpss(y)
    onset = librosa.onset.onset_strength(
        y=y_p, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median)
    if onset.size < 8 or np.max(onset) < 1e-3:
        return None, {"msg": "weak onset"}
    ac      = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    lag_min = max(2, int(np.floor(60 * sr / (MAX_BPM * HOP_LENGTH))))
    lag_max = min(int(np.ceil(60 * sr / (MIN_BPM * HOP_LENGTH))), len(ac) - 1)
    if lag_min >= lag_max:
        return None, {"msg": "invalid range"}
    lag = _pick_peak(ac, lag_min, lag_max)
    if lag is None or not np.isfinite(lag) or lag <= 0:
        return None, {"msg": "invalid lag"}
    bpm_raw  = 60.0 * sr / (HOP_LENGTH * lag)
    bpm_corr = float(np.clip(_octave_correct(bpm_raw, bpm_prev), MIN_BPM, MAX_BPM))
    if bpm_prev is None or not np.isfinite(bpm_prev):
        bpm_s = bpm_corr
    elif abs(bpm_corr - bpm_prev) < 5.0:
        bpm_s = SMOOTH_ALPHA * bpm_corr + (1 - SMOOTH_ALPHA) * bpm_prev
    else:
        bpm_s = bpm_corr
    return bpm_s, {
        "bpm_raw":   float(bpm_raw),
        "bpm_corr":  bpm_corr,
        "onset_max": float(np.max(onset)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════
class GestureStabilizer:
    def __init__(self, window=10):
        self._h = deque(maxlen=window)

    def update(self, g):
        self._h.append(g)
        return Counter(self._h).most_common(1)[0][0]

def _lm_arr(lm_list):
    return np.array([[p.x, p.y, p.z] for p in lm_list])

def _finger_up(lm, tip, pip):
    return lm[tip][1] < lm[pip][1]

def _thumb_up(lm, handedness):
    return (lm[THUMB_TIP][0] < lm[THUMB_IP][0]
            if handedness == "Right"
            else lm[THUMB_TIP][0] > lm[THUMB_IP][0])

def classify_gesture(lm, handedness):
    """PLAY = open hand (5 fingers), PAUSE = fist (0 fingers), None = transition."""
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
#  AUDIO CAPTURE CALLBACK  (sounddevice calls this on its own thread)
# ═══════════════════════════════════════════════════════════════════════════════
def audio_callback(indata, frames, time_info, status):
    if status:
        LOG.warn(f"audio input: {status}")
    mono = np.mean(indata, axis=1).astype(np.float32)
    audio_buffer.extend(mono)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    STATE.push_waveform(rms)


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — BPM analysis
# ═══════════════════════════════════════════════════════════════════════════════
def bpm_analysis_thread():
    last = 0.0
    while STATE.alive():
        now = time.time()
        if now - last >= ANALYZE_EVERY and len(audio_buffer) >= SR * 2:
            last = now
            y = np.array(audio_buffer, dtype=np.float32)
            bpm_new, dbg = estimate_bpm(y, bpm_prev=STATE.get_bpm())
            if bpm_new and np.isfinite(bpm_new):
                STATE.set_bpm(bpm_new,
                              raw=dbg.get("bpm_raw"),
                              corrected=dbg.get("bpm_corr"),
                              onset_max=dbg.get("onset_max", 0.0))
                STATE.last_bpm_dbg = dbg
            with STATE._lock:
                STATE.buffer_fill = min(
                    1.0, len(audio_buffer) / (SR * BUFFER_SEC))
        time.sleep(ANALYZE_EVERY / 4)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRACK LOADER  — supports MP3, WAV, FLAC, OGG, etc.
# ═══════════════════════════════════════════════════════════════════════════════
def _read_bpm_tag(path: str) -> float | None:
    """Try every known BPM tag field across formats."""
    if not HAS_MUTAGEN:
        return None
    try:
        af   = MutagenFile(path)
        tags = af.tags if af else None
        if not tags:
            return None
        for key in ("TBPM", "bpm", "BPM",
                    "TXXX:BPM", "----:com.apple.iTunes:BPM"):
            if key in tags:
                raw = tags[key]
                val = str(raw[0] if (hasattr(raw, "__iter__")
                                     and not isinstance(raw, str))
                          else raw)
                return float(val.strip())
    except Exception:
        pass
    return None


def _estimate_bpm_from_file(y: np.ndarray, sr: int) -> float | None:
    """Use librosa beat tracker as a fallback BPM estimator for untagged files."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # beat_track may return an array in newer librosa
        if hasattr(tempo, "__len__"):
            tempo = float(tempo[0]) if len(tempo) else None
        else:
            tempo = float(tempo)
        if tempo and MIN_BPM <= tempo <= MAX_BPM:
            return tempo
        # try half / double
        for candidate in (tempo * 2, tempo / 2):
            if candidate and MIN_BPM <= candidate <= MAX_BPM:
                return float(candidate)
    except Exception as e:
        LOG.warn(f"librosa beat_track failed: {e}")
    return None


def load_track(path: str) -> tuple[np.ndarray, float, float]:
    """
    Load any supported audio file.
    Returns (samples_float32, bpm_original, duration_seconds).

    BPM priority:
      1. Tag already stored in STATE.bpm_original (set by queue BPM editor)
      2. Metadata tag in the file (TBPM / bpm / …)
      3. librosa beat_track estimation
      4. Fall back to whatever STATE.bpm_original already holds (default 120)
    """
    LOG.info(f"loading: {os.path.basename(path)}")
    y, _ = librosa.load(path, sr=SR, mono=True)
    dur  = len(y) / SR

    # 1. Check if the user already set a BPM for this track via the UI
    with STATE._lock:
        ui_bpm = STATE.bpm_original  # seeded from queue entry when load was requested

    # 2. Try file tag
    tag_bpm = _read_bpm_tag(path)
    if tag_bpm:
        LOG.ok(f"BPM from tag: {tag_bpm:.1f}")

    # 3. Fall back to beat tracking
    if tag_bpm is None:
        LOG.info("no BPM tag — running beat tracker…")
        tag_bpm = _estimate_bpm_from_file(y, SR)
        if tag_bpm:
            LOG.ok(f"BPM from beat_track: {tag_bpm:.1f}")
        else:
            LOG.warn(f"beat_track failed — using ref BPM {ui_bpm:.1f}")

    bpm_orig = tag_bpm if tag_bpm else ui_bpm
    LOG.ok(f"track ready: {dur:.1f}s  bpm_ref={bpm_orig:.1f}")
    return y.astype(np.float32), bpm_orig, dur


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Backing track playback (beat-by-beat, time-stretch)
# ═══════════════════════════════════════════════════════════════════════════════
def backing_track_thread():
    """
    Waits for a track signal (STATE.load_new_track), loads it, then plays
    beat-by-beat with live time-stretching.  Respects PLAY/PAUSE.
    When a track ends, auto-advances the queue.
    """
    y_full   = None
    bpm_orig = 120.0
    pos      = 0
    t_next   = time.time()

    while STATE.alive():

        # ── Check for a new track to load ────────────────────────────────────
        with STATE._lock:
            new_track = STATE.load_new_track
            if new_track:
                STATE.load_new_track = None

        if new_track:
            # If user set a BPM in queue editor, honour it by seeding STATE first
            if new_track.get("bpm"):
                STATE.set_bpm_original(float(new_track["bpm"]))

            try:
                y_full, bpm_orig, dur = load_track(new_track["path"])

                # If the track had no tag but user specified one via queue, keep that
                if new_track.get("bpm"):
                    bpm_orig = float(new_track["bpm"])

                with STATE._lock:
                    STATE.bpm_original   = bpm_orig
                    STATE.bpm_live       = bpm_orig   # seed BPM smoothing
                    STATE.stretch_ratio  = 1.0
                    STATE.track_path     = new_track["path"]
                    STATE.track_duration = dur
                    STATE.track_position = 0.0
                    STATE.markers        = []
                    # If BPM was unknown and we estimated it, push it back to the queue
                    cur_idx = STATE.queue._index
                # Propagate estimated BPM back to queue entry so UI shows it
                STATE.queue.set_bpm(STATE.queue._index, bpm_orig)

                pos    = 0
                t_next = time.time()
                # Auto-start playback when a track is loaded
                STATE.play()
                LOG.ok(f"playing: {new_track['name']}  BPM={bpm_orig:.1f}")
            except Exception as e:
                LOG.err(f"failed to load track: {e}")
                y_full    = None
                new_track = None

        # ── Idle — nothing loaded ─────────────────────────────────────────────
        if y_full is None:
            time.sleep(0.05)
            continue

        # ── Paused — wait without advancing ──────────────────────────────────
        if not STATE.playing():
            time.sleep(0.05)
            t_next = time.time()
            continue

        # ── Track finished ────────────────────────────────────────────────────
        if pos >= len(y_full):
            LOG.ok("track finished")
            with STATE._lock:
                looping = STATE.is_looping
            if looping:
                pos    = 0
                t_next = time.time()
                LOG.info("loop: restarting")
            else:
                next_t = STATE.queue.next_track()
                if next_t:
                    with STATE._lock:
                        STATE.load_new_track = next_t
                    y_full = None
                    pos    = 0
                else:
                    STATE.pause()
                    with STATE._lock:
                        STATE.track_position = STATE.track_duration
                    y_full = None
                    LOG.info("queue empty — stopped")
            continue

        # ── Build one beat block ──────────────────────────────────────────────
        safe_orig = max(1.0, bpm_orig)
        beat_size = int(60.0 / safe_orig * SR)
        end       = min(pos + beat_size, len(y_full))
        with STATE._lock:
            gain = STATE.gain
        block = y_full[pos:end] * gain

        bpm_live = STATE.get_bpm() or safe_orig
        rate     = bpm_live / safe_orig

        # Time-stretch
        if HAS_PYRB and len(block) > 512 and abs(rate - 1.0) > 0.005:
            try:
                block = pyrb.time_stretch(block, SR, rate)
            except Exception as e:
                LOG.warn(f"time-stretch: {e}")

        # Wait for scheduled beat time
        wait = t_next - time.time()
        if wait > 0:
            time.sleep(wait)

        # Push to playback queue (drop if full to stay in sync)
        try:
            audio_queue.put_nowait(block.astype(np.float32))
        except Exception:
            pass

        pos    += beat_size
        t_next += 60.0 / max(1.0, bpm_live)
        STATE.set_position(pos / SR)


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Audio output (restartable)
# ═══════════════════════════════════════════════════════════════════════════════
def playback_thread(dev_out: int | None):
    """
    Opens an OutputStream on dev_out, drains audio_queue into it.
    Exits when STATE.io_restart_requested is set (caller restarts with new device)
    or when STATE is no longer alive.
    """
    try:
        stream = sd.OutputStream(
            device=dev_out,
            samplerate=SR,
            channels=1,
            dtype="float32",
        )
        stream.start()
        LOG.ok(f"output stream started: dev={dev_out}  SR={SR}")
    except Exception as e:
        LOG.err(f"output stream failed to open: {e}")
        return

    while STATE.alive() and not STATE.io_restart_requested:
        if not STATE.playing():
            time.sleep(0.02)
            continue
        try:
            block = audio_queue.get(timeout=0.1).astype(np.float32)
            block = np.clip(block, -1.0, 1.0)
            stream.write(block)
        except Empty:
            time.sleep(0.01)
        except Exception as e:
            LOG.err(f"playback write error: {e}")
            time.sleep(0.05)

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    LOG.info("output stream closed")


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Audio input (restartable)
# ═══════════════════════════════════════════════════════════════════════════════
def input_thread(dev_in: int | None):
    """
    Opens an InputStream on dev_in, feeds samples to audio_buffer via callback.
    Exits when STATE.io_restart_requested or STATE is no longer alive.
    """
    blocksize = int(SR * HOP_STREAM_SEC)
    try:
        stream = sd.InputStream(
            device=dev_in,
            channels=1,
            samplerate=SR,
            blocksize=blocksize,
            dtype="float32",
            callback=audio_callback,
        )
        stream.start()
        LOG.ok(f"input  stream started: dev={dev_in}  SR={SR}")
    except Exception as e:
        LOG.err(f"input stream failed to open: {e}")
        return

    while STATE.alive() and not STATE.io_restart_requested:
        time.sleep(0.1)

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    LOG.info("input stream closed")


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O MANAGER THREAD  — watches for restart requests from the UI
# ═══════════════════════════════════════════════════════════════════════════════
def io_manager_thread():
    """
    Starts the initial audio I/O streams using default devices, then
    monitors STATE.io_restart_requested.  When the UI clicks APPLY it
    stops the current streams and opens new ones on the chosen devices.
    """
    dev_in, dev_out = None, None   # start with system defaults

    def _start():
        nonlocal dev_in, dev_out
        t_in  = threading.Thread(target=input_thread,   args=(dev_in,),
                                 daemon=True, name="audio-in")
        t_out = threading.Thread(target=playback_thread, args=(dev_out,),
                                 daemon=True, name="audio-out")
        t_in.start()
        t_out.start()
        return t_in, t_out

    t_in, t_out = _start()

    while STATE.alive():
        time.sleep(0.2)

        if STATE.io_restart_requested:
            LOG.info("I/O restart requested — cycling streams…")
            # Signal threads to exit by letting io_restart_requested stay True
            # They check it in their loops and exit
            t_in.join(timeout=2.0)
            t_out.join(timeout=2.0)

            # Drain stale audio queue
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except Exception:
                    break

            # Now consume the new device indices and clear the flag
            dev_in, dev_out = STATE.consume_io_restart()

            t_in, t_out = _start()
            LOG.ok(f"I/O restarted: in={dev_in}  out={dev_out}")

    # Cleanup on shutdown
    t_in.join(timeout=1.0)
    t_out.join(timeout=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE VISION  — OpenCV window + MediaPipe
# ═══════════════════════════════════════════════════════════════════════════════
PLAY_COLOR  = (50,  200, 50)
PAUSE_COLOR = (50,   50, 220)
NONE_COLOR  = (180, 180, 180)

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
    cv2.putText(img, text,  pos,       font, scale, color,   thick,   cv2.LINE_AA)

def _draw_hand(frame, lm_px, color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, lm_px[a], lm_px[b], color, 2, cv2.LINE_AA)
    tips = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP}
    for i, pt in enumerate(lm_px):
        r = 7 if i in tips else 4
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (255,255,255), 1, cv2.LINE_AA)

def _draw_hud(frame, gesture_raw, fps):
    h, w = frame.shape[:2]
    F  = cv2.FONT_HERSHEY_DUPLEX
    FS = cv2.FONT_HERSHEY_SIMPLEX

    playing  = STATE.playing()
    bpm_live = STATE.get_bpm()
    with STATE._lock:
        bpm_orig = STATE.bpm_original
        dbg      = STATE.last_bpm_dbg

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
        STATE.set_bpm(bpm_live,
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


def gesture_vision_thread(cam_idx=0):
    download_model()

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        LOG.err(f"vision: cannot open camera {cam_idx}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    LOG.ok(f"camera {cam_idx} open: 1280×720")

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.55,
    )
    LOG.ok("HandLandmarker loaded")

    stab        = GestureStabilizer(window=10)
    prev_stable = None
    prev_time   = time.time()
    MIN_HOLD    = 8
    hold_count  = 0
    pending     = None

    with HandLandmarker.create_from_options(options) as detector:
        while STATE.alive():
            ret, frame = cap.read()
            if not ret:
                LOG.err("vision: frame read failed")
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
                handedness = result.handedness[0][0].category_name
                lm    = _lm_arr(hand_lm)
                lm_px = [(int(p.x * w_f), int(p.y * h_f)) for p in hand_lm]

                raw_gesture = classify_gesture(lm, handedness)
                stable      = stab.update(raw_gesture or "None")
                stable      = None if stable == "None" else stable

                if stable == pending:
                    hold_count += 1
                else:
                    pending    = stable
                    hold_count = 1

                STATE.set_gesture(stable or "NO HAND",
                                  confidence=0.0,
                                  hold_frames=hold_count,
                                  hands=1)

                if (hold_count >= MIN_HOLD
                        and stable != prev_stable
                        and stable in ("PLAY", "PAUSE")):
                    if stable == "PLAY":
                        STATE.play()
                        LOG.ok(f"gesture confirmed: PLAY  (hold={hold_count})")
                    else:
                        STATE.pause()
                        LOG.ok(f"gesture confirmed: PAUSE (hold={hold_count})")
                    STATE.set_command(stable)
                    prev_stable = stable
                    hold_count  = 0
                elif stable and stable != prev_stable:
                    LOG.info(f"gesture: {stable}  hold={hold_count}/{MIN_HOLD}")

                hand_color = (PLAY_COLOR  if stable == "PLAY"
                              else PAUSE_COLOR if stable == "PAUSE"
                              else (180,180,180))
                _draw_hand(frame, lm_px, hand_color)

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
                    LOG.info("gesture: no hand detected")
                pending    = None
                hold_count = 0
                STATE.set_gesture("NO HAND", confidence=0.0,
                                  hold_frames=0, hands=0)

            _draw_hud(frame, raw_gesture, fps)
            cv2.imshow("Phantom Conductor — Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                LOG.warn("vision: quit by user (Q)")
                STATE.stop()
                break
            elif key == ord(" "):
                new_state = STATE.toggle()
                LOG.info(f"space: {'PLAY' if new_state else 'PAUSE'}")

    cap.release()
    cv2.destroyAllWindows()
    LOG.warn("gesture_vision_thread exited")
    STATE.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 58)
    print("  PHANTOM CONDUCTOR v0.5.0")
    print("=" * 58)
    print("  Camera index to use? (Enter = 0):", end=" ", flush=True)
    cam_str = input().strip()
    cam_idx = int(cam_str) if cam_str.isdigit() else 0
    print()
    print("  Audio devices and camera are configured inside the UI.")
    print("  → Add tracks via the Queue panel (+ADD)")
    print("  → Select I/O devices in the AUDIO I/O panel, then click APPLY")
    print("  → Open hand = PLAY  |  Fist = PAUSE  |  Space = toggle  |  Q = quit")
    print("=" * 58 + "\n")

    # ── Worker threads ─────────────────────────────────────────────────────────
    threading.Thread(target=bpm_analysis_thread, daemon=True,
                     name="bpm-analysis").start()
    threading.Thread(target=backing_track_thread, daemon=True,
                     name="backing-track").start()
    threading.Thread(target=io_manager_thread, daemon=True,
                     name="io-manager").start()
    threading.Thread(target=gesture_vision_thread, args=(cam_idx,), daemon=True,
                     name="gesture-vision").start()

    # ── DearPyGui UI — must run on the main thread ─────────────────────────────
    ui = PhantomUI(STATE, LOG)
    try:
        ui.run()
    except KeyboardInterrupt:
        pass
    finally:
        STATE.stop()
        print("\n🛑 Phantom Conductor stopped.")


if __name__ == "__main__":
    main()