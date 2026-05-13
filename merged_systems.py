"""
Phantom Conductor — Integração BPM + Controle Gestual
======================================================
Arquitetura multithreaded com camadas:
  - Captura: áudio (sounddevice) + vídeo (OpenCV)
  - Processamento: detecção de BPM + reconhecimento gestual (MediaPipe)
  - Apresentação: reprodução de áudio sincronizada via fila thread-safe

Gestos de controle:
  - Mão Aberta  (5 dedos) → PLAY
  - Punho       (0 dedos) → PAUSE

Dependências:
    pip install mediapipe opencv-python numpy sounddevice librosa
                scipy mido mutagen pyrubberband
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
import mido
from mido import Message
import mutagen
from mutagen.mp3 import MP3
import pyrubberband as pyrb

# ── MediaPipe Tasks API (0.10+) ───────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════
SR            = 48000
BUFFER_SEC    = 10
HOP_STREAM_SEC= 0.05
ANALYZE_EVERY = 0.25
HOP_LENGTH    = 256
MIN_BPM, MAX_BPM = 70, 190
SMOOTH_ALPHA  = 0.3
BANDPASS      = (40, 3000)
ONSET_AGG     = np.median
BLOCK_SEC     = 1.0
OVERLAP_SEC   = 0.05

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Índices landmarks
WRIST      = 0
THUMB_IP, THUMB_TIP   = 3, 4
INDEX_PIP, INDEX_TIP  = 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP,  RING_TIP   = 14, 16
PINKY_PIP, PINKY_TIP  = 18, 20

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  ESTADO COMPARTILHADO (thread-safe)
# ═══════════════════════════════════════════════════════════════════════════════
class SharedState:
    def __init__(self):
        self._lock        = threading.Lock()
        self.bpm_smooth   = None
        self.bpm_original = None
        self.is_playing   = False       # PLAY/PAUSE controlado por gesto
        self.running      = True        # sinal de shutdown geral
        self.last_gesture = "Nenhum"
        self.last_bpm_dbg = {}

    # ── BPM ──────────────────────────────────────────────────────────────────
    def get_bpm(self):
        with self._lock:
            return self.bpm_smooth

    def set_bpm(self, v):
        with self._lock:
            self.bpm_smooth = v

    # ── Playback ─────────────────────────────────────────────────────────────
    def play(self):
        with self._lock:
            self.is_playing = True

    def pause(self):
        with self._lock:
            self.is_playing = False

    def toggle(self):
        with self._lock:
            self.is_playing = not self.is_playing
            return self.is_playing

    def playing(self):
        with self._lock:
            return self.is_playing

    def alive(self):
        with self._lock:
            return self.running

    def stop(self):
        with self._lock:
            self.running = False

    # ── Gesto ────────────────────────────────────────────────────────────────
    def set_gesture(self, g):
        with self._lock:
            self.last_gesture = g

    def get_gesture(self):
        with self._lock:
            return self.last_gesture


STATE = SharedState()
audio_buffer = deque(maxlen=int(SR * BUFFER_SEC))
audio_queue  = Queue(maxsize=40)


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS — MODELO
# ═══════════════════════════════════════════════════════════════════════════════
def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Baixando modelo MediaPipe (~8 MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo baixado!")
    except Exception as e:
        print(f"Erro ao baixar modelo: {e}\nURL: {MODEL_URL}")
        raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMADA DE PROCESSAMENTO — BPM
# ═══════════════════════════════════════════════════════════════════════════════
def butter_bandpass(lo, hi, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [max(1e-3, lo/nyq), min(0.999, hi/nyq)], btype='band')
    return b, a

def apply_bandpass(y, fs, lo, hi):
    b, a = butter_bandpass(lo, hi, fs)
    return lfilter(b, a, y)

def pick_peak_autocorr(ac, lag_min, lag_max):
    seg = ac[lag_min:lag_max+1]
    if not seg.size:
        return None
    i = lag_min + int(np.argmax(seg))
    if 0 < i < len(ac)-1:
        y0, y1, y2 = ac[i-1], ac[i], ac[i+1]
        d = y0 - 2*y1 + y2
        if d:
            return i + 0.5*(y0-y2)/d
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
    onset = librosa.onset.onset_strength(y=y_p, sr=sr, hop_length=HOP_LENGTH,
                                         aggregate=ONSET_AGG)
    if onset.size < 8 or np.max(onset) < 1e-3:
        return None, {"msg": "onset fraco"}
    ac = np.correlate(onset, onset, mode='full')[len(onset)-1:]
    lag_min = max(2, int(np.floor(60*sr / (MAX_BPM * HOP_LENGTH))))
    lag_max = min(int(np.ceil(60*sr / (MIN_BPM * HOP_LENGTH))), len(ac)-1)
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
        bpm_s = SMOOTH_ALPHA * bpm_corr + (1-SMOOTH_ALPHA) * bpm_prev
    else:
        bpm_s = bpm_corr
    return bpm_s, {"bpm_raw": bpm_raw, "bpm_corr": bpm_corr,
                   "onset_max": float(np.max(onset))}


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMADA DE PROCESSAMENTO — GESTO
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
    return lm[THUMB_TIP][0] < lm[THUMB_IP][0] if handedness == "Right" \
           else lm[THUMB_TIP][0] > lm[THUMB_IP][0]

def classify_gesture(lm, handedness):
    """
    Retorna 'PLAY', 'PAUSE' ou None (gesto ambíguo/intermediário).
    PLAY  = Mão Aberta (todos os 5 dedos levantados)
    PAUSE = Punho     (nenhum dedo levantado)
    """
    f = [
        is_thumb_up(lm, handedness),
        is_finger_up(lm, INDEX_TIP, INDEX_PIP),
        is_finger_up(lm, MIDDLE_TIP, MIDDLE_PIP),
        is_finger_up(lm, RING_TIP, RING_PIP),
        is_finger_up(lm, PINKY_TIP, PINKY_PIP),
    ]
    count = sum(f)
    if count == 5:
        return "PLAY"       # 🖐 Mão aberta
    if count == 0:
        return "PAUSE"      # ✊ Punho
    return None             # gesto de transição, ignorado


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Captura de áudio (callback)
# ═══════════════════════════════════════════════════════════════════════════════
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[áudio] {status}")
    audio_buffer.extend(np.mean(indata, axis=1))


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Análise de BPM
# ═══════════════════════════════════════════════════════════════════════════════
def bpm_analysis_thread():
    last_analysis = 0.0
    while STATE.alive():
        now = time.time()
        if now - last_analysis >= ANALYZE_EVERY and len(audio_buffer) >= SR * 2:
            last_analysis = now
            y = np.array(audio_buffer, dtype=np.float32)
            bpm_new, dbg = estimate_bpm(y, bpm_prev=STATE.get_bpm())
            if bpm_new and np.isfinite(bpm_new):
                STATE.set_bpm(bpm_new)
                STATE.last_bpm_dbg = dbg
        time.sleep(ANALYZE_EVERY / 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Processamento do backing track (time-stretch beat-a-beat)
# ═══════════════════════════════════════════════════════════════════════════════
def backing_track_thread(mp3_file):
    y_full, _ = librosa.load(mp3_file, sr=SR, mono=True)
    bpm_orig   = STATE.bpm_original
    beat_size  = int(60.0 / bpm_orig * SR)
    pos        = 0
    next_beat  = time.time()

    while pos < len(y_full) and STATE.alive():
        # Se em PAUSE, aguarda sem avançar
        if not STATE.playing():
            time.sleep(0.05)
            next_beat = time.time()   # reseta tempo ao retomar
            continue

        end   = min(pos + beat_size, len(y_full))
        block = y_full[pos:end].astype(np.float32)

        bpm_live = STATE.get_bpm() or bpm_orig
        rate     = bpm_live / bpm_orig

        if len(block) > 1:
            block = pyrb.time_stretch(block, SR, rate=rate)

        # Aguarda slot de tempo
        wait = next_beat - time.time()
        if wait > 0:
            time.sleep(wait)

        # Coloca na fila de reprodução
        while STATE.alive():
            try:
                audio_queue.put(block, timeout=0.1)
                break
            except:
                pass

        pos       += beat_size
        next_beat += 60.0 / bpm_live

    print("\n[backing] Fim do arquivo.")


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Reprodução dos blocos da fila
# ═══════════════════════════════════════════════════════════════════════════════
def playback_thread():
    with sd.OutputStream(samplerate=SR, channels=1, dtype='float32') as out:
        while STATE.alive():
            if not STATE.playing():
                time.sleep(0.02)
                continue
            try:
                block = audio_queue.get(timeout=0.1).astype(np.float32)
                out.write(block)
            except Empty:
                time.sleep(0.01)


# ═══════════════════════════════════════════════════════════════════════════════
#  THREAD — Visão: captura + reconhecimento gestual + exibição
# ═══════════════════════════════════════════════════════════════════════════════
# ── Helpers visuais ──────────────────────────────────────────────────────────
PLAY_COLOR  = (50,  200,  50)   # verde
PAUSE_COLOR = (50,   50, 220)   # azul
NONE_COLOR  = (180, 180, 180)   # cinza

def draw_rect_alpha(img, x1, y1, x2, y2, color, alpha=0.65):
    x1,y1 = max(x1,0), max(y1,0)
    x2,y2 = min(x2,img.shape[1]-1), min(y2,img.shape[0]-1)
    if x2<=x1 or y2<=y1: return
    ov = img.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

def shadow_text(img, text, pos, font, scale, color, thick=2):
    x,y = pos
    cv2.putText(img,text,(x+2,y+2),font,scale,(0,0,0),thick+1,cv2.LINE_AA)
    cv2.putText(img,text,pos,font,scale,color,thick,cv2.LINE_AA)

def draw_hand(frame, lm_px, color):
    for a,b in HAND_CONNECTIONS:
        cv2.line(frame, lm_px[a], lm_px[b], color, 2, cv2.LINE_AA)
    tips = {THUMB_TIP,INDEX_TIP,MIDDLE_TIP,RING_TIP,PINKY_TIP}
    for i,pt in enumerate(lm_px):
        r = 7 if i in tips else 4
        cv2.circle(frame,pt,r,color,-1,cv2.LINE_AA)
        cv2.circle(frame,pt,r,(255,255,255),1,cv2.LINE_AA)


def draw_hud(frame, gesture_raw, fps):
    """Desenha HUD minimalista focado no estado play/pause e BPM."""
    h, w = frame.shape[:2]
    F  = cv2.FONT_HERSHEY_DUPLEX
    FS = cv2.FONT_HERSHEY_SIMPLEX

    playing    = STATE.playing()
    bpm_live   = STATE.get_bpm()
    bpm_orig   = STATE.bpm_original
    dbg        = STATE.last_bpm_dbg

    # ── Barra superior ───────────────────────────────────────────────────────
    draw_rect_alpha(frame, 0, 0, w, 52, (10,10,10))
    shadow_text(frame, "Phantom Conductor", (12, 34), F, 0.75, (220,220,220))
    fps_str = f"FPS: {fps:.0f}"
    tw = cv2.getTextSize(fps_str, FS, 0.5, 1)[0][0]
    shadow_text(frame, fps_str, (w-tw-12, 34), FS, 0.5, (80,255,120))

    # ── Painel de status (centro-baixo) ──────────────────────────────────────
    px, py = 12, h - 160
    draw_rect_alpha(frame, px, py, px+340, py+145, (10,10,10))

    # Status PLAY / PAUSE
    state_str  = "▶  PLAY" if playing else "⏸  PAUSE"
    state_col  = PLAY_COLOR if playing else PAUSE_COLOR
    # Barra lateral colorida
    cv2.rectangle(frame, (px, py), (px+6, py+145), state_col, -1)
    shadow_text(frame, state_str, (px+16, py+40), F, 1.1, state_col, 2)

    # BPM
    if bpm_live:
        ratio = bpm_live / bpm_orig if bpm_orig else 1.0
        bpm_str = f"BPM live: {bpm_live:.1f}"
        shadow_text(frame, bpm_str, (px+16, py+80), FS, 0.65, (200,200,200))
        ratio_col = (50,200,50) if abs(ratio-1.0)<0.05 else (50,200,255)
        shadow_text(frame, f"ratio: {ratio:.3f}  ref: {bpm_orig:.1f}",
                    (px+16, py+108), FS, 0.52, ratio_col)
    else:
        shadow_text(frame, "Detectando BPM…", (px+16, py+80), FS, 0.6, (140,140,140))

    # Gesto detectado
    g_col = PLAY_COLOR if gesture_raw=="PLAY" else \
            PAUSE_COLOR if gesture_raw=="PAUSE" else NONE_COLOR
    g_label = {
        "PLAY":  "Gesto: Mao Aberta  →  PLAY",
        "PAUSE": "Gesto: Punho       →  PAUSE",
        None:    "Gesto: —",
    }.get(gesture_raw, "Gesto: —")
    shadow_text(frame, g_label, (px+16, py+135), FS, 0.48, g_col, 1)

    # ── Legenda ──────────────────────────────────────────────────────────────
    shadow_text(frame, "Mao Aberta=PLAY   Punho=PAUSE   Q=sair",
                (12, h-8), FS, 0.42, (100,100,100), 1)


def gesture_vision_thread(cam_idx=0):
    """Thread de visão: captura câmera, detecta gestos, aplica play/pause."""
    download_model()

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[visão] Não foi possível abrir câmera {cam_idx}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.55,
    )

    stabilizer     = GestureStabilizer(window=10)
    prev_stable    = None   # último gesto confirmado (evita repetição)
    prev_time      = time.time()

    # Debounce: só muda estado após MIN_HOLD frames consecutivos do mesmo gesto
    MIN_HOLD = 8
    hold_count = 0
    pending_gesture = None

    with HandLandmarker.create_from_options(options) as detector:
        while STATE.alive():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h_f, w_f = frame.shape[:2]

            # FPS
            now  = time.time()
            fps  = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            # Detecção
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_img)

            raw_gesture = None

            if result.hand_landmarks:
                hand_lm   = result.hand_landmarks[0]
                hand_info = result.handedness[0]
                handedness = hand_info[0].category_name

                lm    = lm_to_array(hand_lm)
                lm_px = [(int(p.x*w_f), int(p.y*h_f)) for p in hand_lm]

                raw_gesture = classify_gesture(lm, handedness)
                stable      = stabilizer.update(raw_gesture or "None")
                stable      = None if stable == "None" else stable

                # ── Debounce: aplica gesto só após MIN_HOLD frames ───────────
                if stable == pending_gesture:
                    hold_count += 1
                else:
                    pending_gesture = stable
                    hold_count = 1

                if hold_count >= MIN_HOLD and stable != prev_stable and stable in ("PLAY","PAUSE"):
                    if stable == "PLAY":
                        STATE.play()
                    else:
                        STATE.pause()
                    STATE.set_gesture(stable)
                    prev_stable = stable
                    hold_count  = 0

                # Cor do esqueleto reflete o gesto atual
                hand_color = PLAY_COLOR if stable == "PLAY" else \
                             PAUSE_COLOR if stable == "PAUSE" else (180,180,180)
                draw_hand(frame, lm_px, hand_color)

                # Label flutuante
                wx, wy = lm_px[WRIST]
                label = stable or "…"
                lbl_col = PLAY_COLOR if stable == "PLAY" else \
                          PAUSE_COLOR if stable == "PAUSE" else (200,200,200)
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0][0]
                draw_rect_alpha(frame, wx-tw//2-12, max(wy-55,5),
                                wx+tw//2+12, max(wy-10,50), lbl_col, 0.45)
                shadow_text(frame, label, (wx-tw//2, max(wy-15,45)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)

            else:
                pending_gesture = None
                hold_count      = 0

            draw_hud(frame, raw_gesture, fps)
            cv2.imshow("Phantom Conductor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                STATE.stop()
                break
            elif key == ord(' '):   # barra de espaço como fallback
                STATE.toggle()

    cap.release()
    cv2.destroyAllWindows()
    STATE.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS — Dispositivos
# ═══════════════════════════════════════════════════════════════════════════════
def list_input_devices():
    print("\n=== Dispositivos de ENTRADA ===")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i:>3}  {d['name']}")
    print("================================")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Backing track ─────────────────────────────────────────────────────────
    mp3_file = input("Caminho do backing track (.mp3): ").strip()

    bpm_orig = None
    try:
        audio_meta = MP3(mp3_file)
        tags = audio_meta.tags
        if tags and "TBPM" in tags:
            bpm_orig = float(tags["TBPM"].text[0])
            print(f"BPM do arquivo: {bpm_orig:.2f}")
    except Exception as e:
        print(f"Não foi possível ler BPM do arquivo: {e}")

    if bpm_orig is None:
        bpm_orig = float(input("BPM original (manual): ").strip())

    STATE.bpm_original = bpm_orig
    STATE.set_bpm(bpm_orig)   # seed inicial

    # ── Dispositivo de entrada ────────────────────────────────────────────────
    list_input_devices()
    dev_idx = int(input("Índice do dispositivo de ENTRADA: ").strip())
    dev = sd.query_devices(dev_idx)
    print(f"\n🎤 Microfone: {dev['name']}  |  SR={SR} Hz")

    # ── Câmera ────────────────────────────────────────────────────────────────
    cam_idx_str = input("Índice da câmera [0]: ").strip()
    cam_idx = int(cam_idx_str) if cam_idx_str else 0

    print("\n" + "="*54)
    print("  PHANTOM CONDUCTOR — Controle Gestual")
    print("="*54)
    print("  🖐  Mão Aberta  →  PLAY")
    print("  ✊  Punho       →  PAUSE")
    print("  [Espaço]        →  Toggle manual")
    print("  [Q]             →  Sair")
    print("="*54 + "\n")

    blocksize = int(SR * HOP_STREAM_SEC)

    # ── Inicia threads ────────────────────────────────────────────────────────
    threading.Thread(target=bpm_analysis_thread, daemon=True).start()
    threading.Thread(target=backing_track_thread, args=(mp3_file,), daemon=True).start()
    threading.Thread(target=playback_thread, daemon=True).start()

    # Captura de áudio (thread interna do sounddevice)
    with sd.InputStream(device=dev_idx, channels=1, samplerate=SR,
                        blocksize=blocksize, callback=audio_callback):

        # Thread de visão roda no thread principal (OpenCV exige)
        gesture_vision_thread(cam_idx)

        STATE.stop()
        print("\n🛑 Encerrado.")


if __name__ == "__main__":
    main()