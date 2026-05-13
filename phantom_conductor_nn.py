"""
Phantom Conductor NN — BPM + Controle Gestual com classificador treinado em HaGRID
====================================================================================
Diferença do phantom_conductor.py:
  - detect_gesture() é substituído por um MLP treinado em landmarks extraídos
    das imagens do dataset HaGRID (https://github.com/hukenovs/hagrid)
  - O classificador recebe 63 features (21 landmarks × x,y,z normalizados)
  - Treinamento é feito uma vez via train_gesture_classifier.py

Pipeline:
  frame → MediaPipe landmarks (21 pts) → normalização → MLP → classe → comando

Dependências EXTRAS (além das do phantom_conductor.py):
    pip install torch scikit-learn tqdm

Uso:
  1. Treine o classificador:     python train_gesture_classifier.py
  2. Execute o conductor:        python phantom_conductor_nn.py
"""

import cv2, numpy as np, sounddevice as sd, librosa
import threading, time, urllib.request, os
from collections import deque
from queue import Queue, Empty
from scipy.signal import butter, lfilter
from mutagen.mp3 import MP3
import pyrubberband as pyrb
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import pickle
import numpy as np

# ── Classifier import (graceful fallback to rule-based if not trained yet) ────
CLASSIFIER_PATH = "gesture_classifier.pkl"

try:
    with open(CLASSIFIER_PATH, "rb") as f:
        clf_data = pickle.load(f)
    CLASSIFIER   = clf_data["model"]
    LABEL_NAMES  = clf_data["labels"]     # list of gesture name strings
    SCALER       = clf_data.get("scaler") # optional StandardScaler
    print(f"✅ Classificador carregado: {len(LABEL_NAMES)} gestos → {LABEL_NAMES}")
    USE_NN = True
except FileNotFoundError:
    print("⚠️  gesture_classifier.pkl não encontrado.")
    print("   Execute train_gesture_classifier.py primeiro.")
    print("   Usando fallback: regras geométricas.\n")
    USE_NN = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  (idêntico ao phantom_conductor.py)
# ═══════════════════════════════════════════════════════════════════════════════
SR             = 48000
BUFFER_SEC     = 10
HOP_STREAM_SEC = 0.05
ANALYZE_EVERY  = 0.25
HOP_LENGTH     = 256
MIN_BPM, MAX_BPM = 70, 190
SMOOTH_ALPHA   = 0.3
BANDPASS       = (40, 3000)
ONSET_AGG      = np.median
CONSECUTIVE_BEATS = 2

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")

WRIST = 0
THUMB_IP,  THUMB_TIP  = 3,  4
INDEX_PIP, INDEX_TIP  = 6,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP,  RING_TIP   = 14, 16
PINKY_PIP, PINKY_TIP  = 18, 20

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

PLAY_COLOR  = (50, 200,  50)
PAUSE_COLOR = (50,  50, 220)
NONE_COLOR  = (180,180, 180)

# HaGRID class names that map to our commands
# Adjust these to match whatever classes you trained on
# HaGRID 500k dataset uses train_val_ prefix
HAGRID_PLAY_CLASSES  = {"train_val_five", "train_val_palm",
                         "train_val_four", "train_val_three2"}
HAGRID_PAUSE_CLASSES = {"train_val_fist", "train_val_stop",
                         "train_val_stop_inverted", "train_val_mute"}


# ═══════════════════════════════════════════════════════════════════════════════
#  ESTADO COMPARTILHADO  (idêntico)
# ═══════════════════════════════════════════════════════════════════════════════
class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.bpm_smooth   = None
        self.bpm_original = None
        self.is_playing   = False
        self.running      = True
        self.last_gesture = "Nenhum"
        self.last_bpm_dbg = {}

    def get_bpm(self):
        with self._lock: return self.bpm_smooth
    def set_bpm(self, v):
        with self._lock: self.bpm_smooth = v
    def play(self):
        with self._lock: self.is_playing = True
    def pause(self):
        with self._lock: self.is_playing = False
    def toggle(self):
        with self._lock:
            self.is_playing = not self.is_playing
            return self.is_playing
    def playing(self):
        with self._lock: return self.is_playing
    def alive(self):
        with self._lock: return self.running
    def stop(self):
        with self._lock: self.running = False
    def set_gesture(self, g):
        with self._lock: self.last_gesture = g

STATE        = SharedState()
audio_buffer = deque(maxlen=int(SR * BUFFER_SEC))
audio_queue  = Queue(maxsize=40)


# ═══════════════════════════════════════════════════════════════════════════════
#  LANDMARK FEATURES — normalização invariante a escala e posição
# ═══════════════════════════════════════════════════════════════════════════════
def landmarks_to_features(lm_list):
    """
    Converte 21 landmarks em vetor de 63 features normalizado.

    Normalização:
      1. Translação: subtrai o pulso (landmark 0) → posição relativa
      2. Escala: divide pela distância pulso→middle_mcp → invariante a distância
      3. Flatten: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]

    Isso permite que o classificador generalize para diferentes tamanhos de mão
    e distâncias da câmera — exatamente o que faz o treinamento em HaGRID valer.
    """
    lm = np.array([[p.x, p.y, p.z] for p in lm_list])

    # 1. Centraliza no pulso
    lm = lm - lm[WRIST]

    # 2. Normaliza pela escala da mão
    scale = np.linalg.norm(lm[MIDDLE_MCP])
    if scale > 1e-6:
        lm = lm / scale

    return lm.flatten()   # shape: (63,)


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE DETECTION — NN ou fallback geométrico
# ═══════════════════════════════════════════════════════════════════════════════
def is_palm_facing_camera(lm):
    return lm[WRIST][2] < lm[MIDDLE_MCP][2]

def is_finger_up(lm, tip, pip):
    return lm[tip][1] < lm[pip][1]

def is_thumb_up(lm, handedness):
    palm_facing = is_palm_facing_camera(lm)
    if handedness == "Right":
        return lm[THUMB_TIP][0] < lm[THUMB_IP][0] if not palm_facing \
               else lm[THUMB_TIP][0] > lm[THUMB_IP][0]
    else:
        return lm[THUMB_TIP][0] > lm[THUMB_IP][0] if not palm_facing \
               else lm[THUMB_TIP][0] < lm[THUMB_IP][0]

def detect_gesture_rules(lm_arr, handedness):
    """Fallback geométrico (mesmo do phantom_conductor.py)."""
    f = [
        is_thumb_up(lm_arr, handedness),
        is_finger_up(lm_arr, INDEX_TIP,  INDEX_PIP),
        is_finger_up(lm_arr, MIDDLE_TIP, MIDDLE_PIP),
        is_finger_up(lm_arr, RING_TIP,   RING_PIP),
        is_finger_up(lm_arr, PINKY_TIP,  PINKY_PIP),
    ]
    count = sum(f)
    thumb, index, middle, ring, pinky = f
    if count == 0: return "Punho",      (60,  60, 220)
    if count == 5: return "Mao Aberta", (50, 200,  50)
    if thumb and not index and not middle and not ring and not pinky:
        return "Joinha!",   (0,   200, 255)
    if not thumb and index and middle and not ring and not pinky:
        return "Paz",       (255, 200,   0)
    if not thumb and index and not middle and not ring and not pinky:
        return "Apontando", (200, 100, 255)
    return f"{count} dedos",(180, 180, 180)

def detect_gesture_nn(lm_list, handedness):
    """
    Classificador treinado em HaGRID.
    Retorna (nome_classe_hagrid, cor_bgr).
    """
    features = landmarks_to_features(lm_list).reshape(1, -1)
    if SCALER is not None:
        features = SCALER.transform(features)
    pred    = CLASSIFIER.predict(features)[0]
    proba   = CLASSIFIER.predict_proba(features)[0]
    conf    = proba.max()
    # Always resolve to string regardless of whether pred is int or already a string
    label   = LABEL_NAMES[int(pred)] if isinstance(pred, (int, np.integer))               else (LABEL_NAMES[LABEL_NAMES.index(pred)] if pred in LABEL_NAMES else str(pred))

    # Cor baseada no comando
    if label in HAGRID_PLAY_CLASSES:   color = PLAY_COLOR
    elif label in HAGRID_PAUSE_CLASSES: color = PAUSE_COLOR
    else:                               color = NONE_COLOR

    # Exibe confiança no label
    display = f"{label} ({conf:.0%})"
    return display, label, color

def detect_gesture(lm_list, lm_arr, handedness):
    """
    Ponto de entrada unificado.
    Retorna (display_name, raw_label, color_bgr).
    """
    if USE_NN:
        display, label, color = detect_gesture_nn(lm_list, handedness)
        return display, label, color
    else:
        name, color = detect_gesture_rules(lm_arr, handedness)
        return name, name, color

def gesture_to_command(raw_label):
    """Mapeia label HaGRID (ou geométrico) para comando."""
    if USE_NN:
        # Resolve int index → string name if needed
        label = LABEL_NAMES[raw_label] if isinstance(raw_label, (int, np.integer))                 else str(raw_label)
        if label in HAGRID_PLAY_CLASSES:   return "PLAY"
        if label in HAGRID_PAUSE_CLASSES:  return "PAUSE"
        return None
    else:
        if raw_label == "Mao Aberta": return "PLAY"
        if raw_label == "Punho":      return "PAUSE"
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  BPM  (idêntico)
# ═══════════════════════════════════════════════════════════════════════════════
def butter_bandpass(lo, hi, fs, order=4):
    nyq = 0.5*fs
    b,a = butter(order,[max(1e-3,lo/nyq),min(0.999,hi/nyq)],btype='band')
    return b,a

def apply_bandpass(y,fs,lo,hi):
    return lfilter(*butter_bandpass(lo,hi,fs),y)

def pick_peak_autocorr(ac,lag_min,lag_max):
    seg=ac[lag_min:lag_max+1]
    if not seg.size: return None
    i=lag_min+int(np.argmax(seg))
    if 0<i<len(ac)-1:
        y0,y1,y2=ac[i-1],ac[i],ac[i+1]; d=y0-2*y1+y2
        if d: return i+0.5*(y0-y2)/d
    return float(i)

def octave_correct(bpm,prev):
    if prev is None or not np.isfinite(prev): return bpm
    return min([bpm/2,bpm,bpm*2],key=lambda x:abs(x-prev))

def estimate_bpm(y,sr=SR,bpm_prev=None):
    if len(y)<sr*2: return None,{"msg":"buffer curto"}
    y=librosa.util.normalize(y.astype(np.float32))
    y=apply_bandpass(y,sr,*BANDPASS)
    _,y_p=librosa.effects.hpss(y)
    onset=librosa.onset.onset_strength(y=y_p,sr=sr,hop_length=HOP_LENGTH,aggregate=ONSET_AGG)
    if onset.size<8 or np.max(onset)<1e-3: return None,{"msg":"onset fraco"}
    ac=np.correlate(onset,onset,mode='full')[len(onset)-1:]
    lag_min=max(2,int(np.floor(60*sr/(MAX_BPM*HOP_LENGTH))))
    lag_max=min(int(np.ceil(60*sr/(MIN_BPM*HOP_LENGTH))),len(ac)-1)
    if lag_min>=lag_max: return None,{"msg":"faixa inválida"}
    lag=pick_peak_autocorr(ac,lag_min,lag_max)
    if lag is None or not np.isfinite(lag) or lag<=0: return None,{"msg":"lag inválido"}
    bpm_raw=60.0*sr/(HOP_LENGTH*lag)
    bpm_corr=np.clip(octave_correct(bpm_raw,bpm_prev),MIN_BPM,MAX_BPM)
    if bpm_prev is None or not np.isfinite(bpm_prev): bpm_s=bpm_corr
    elif abs(bpm_corr-bpm_prev)<5.0: bpm_s=SMOOTH_ALPHA*bpm_corr+(1-SMOOTH_ALPHA)*bpm_prev
    else: bpm_s=bpm_corr
    return bpm_s,{"bpm_raw":bpm_raw,"bpm_corr":bpm_corr,"onset_max":float(np.max(onset))}


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUAL HELPERS  (idêntico)
# ═══════════════════════════════════════════════════════════════════════════════
def draw_rect_alpha(img,x1,y1,x2,y2,color,alpha=0.65):
    x1,y1=max(x1,0),max(y1,0); x2,y2=min(x2,img.shape[1]-1),min(y2,img.shape[0]-1)
    if x2<=x1 or y2<=y1: return
    ov=img.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

def shadow_text(img,text,pos,font,scale,color,thick=2):
    x,y=pos
    cv2.putText(img,text,(x+2,y+2),font,scale,(0,0,0),thick+1,cv2.LINE_AA)
    cv2.putText(img,text,pos,font,scale,color,thick,cv2.LINE_AA)

def draw_hand(frame,lm_px,color):
    for a,b in HAND_CONNECTIONS:
        cv2.line(frame,lm_px[a],lm_px[b],color,2,cv2.LINE_AA)
    tips={THUMB_TIP,INDEX_TIP,MIDDLE_TIP,RING_TIP,PINKY_TIP}
    for i,pt in enumerate(lm_px):
        r=7 if i in tips else 4
        cv2.circle(frame,pt,r,color,-1,cv2.LINE_AA)
        cv2.circle(frame,pt,r,(255,255,255),1,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO THREADS  (idêntico)
# ═══════════════════════════════════════════════════════════════════════════════
def audio_callback(indata,frames,time_info,status):
    if status: print(f"[áudio] {status}")
    audio_buffer.extend(np.mean(indata,axis=1))

def bpm_analysis_thread():
    last=0.0
    while STATE.alive():
        now=time.time()
        if now-last>=ANALYZE_EVERY and len(audio_buffer)>=SR*2:
            last=now; y=np.array(audio_buffer,dtype=np.float32)
            bpm_new,dbg=estimate_bpm(y,bpm_prev=STATE.get_bpm())
            if bpm_new and np.isfinite(bpm_new): STATE.set_bpm(bpm_new); STATE.last_bpm_dbg=dbg
        time.sleep(ANALYZE_EVERY/2)

def backing_track_thread(mp3_file):
    y_full,_=librosa.load(mp3_file,sr=SR,mono=True)
    bpm_orig=STATE.bpm_original; beat_size=int(60.0/bpm_orig*SR)
    pos=0; next_beat=time.time()
    while pos<len(y_full) and STATE.alive():
        bpm_live=STATE.get_bpm() or bpm_orig
        wait=next_beat-time.time()
        if wait>0: time.sleep(wait)
        next_beat+=60.0/bpm_live
        if not STATE.playing(): continue
        end=min(pos+beat_size,len(y_full)); block=y_full[pos:end].astype(np.float32)
        if len(block)>1: block=pyrb.time_stretch(block,SR,rate=bpm_live/bpm_orig)
        while STATE.alive():
            try: audio_queue.put(block,timeout=0.1); break
            except: pass
        pos+=beat_size
    print("\n[backing] Fim do arquivo.")

def playback_thread():
    with sd.OutputStream(samplerate=SR,channels=1,dtype='float32') as out:
        while STATE.alive():
            if not STATE.playing(): time.sleep(0.02); continue
            try: out.write(audio_queue.get(timeout=0.1).astype(np.float32))
            except Empty: time.sleep(0.01)


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE VISION THREAD  (mesma arquitetura, detect_gesture() agora é NN)
# ═══════════════════════════════════════════════════════════════════════════════
def gesture_vision_thread(cam_idx=0):
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo MediaPipe (~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    cap=cv2.VideoCapture(cam_idx,cv2.CAP_V4L2)
    if not cap.isOpened(): cap=cv2.VideoCapture(cam_idx,cv2.CAP_ANY)
    if not cap.isOpened(): print(f"[gesto] Câmera {cam_idx} não encontrada."); STATE.stop(); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,  1)

    last_result=[None]; last_result_lock=threading.Lock(); frame_ts=0

    def on_result(result,_img,_ts):
        with last_result_lock: last_result[0]=result

    options=HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=on_result, num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    pending_cmd=None; consecutive=0; confirmed_cmd=None
    beat_interval=60.0/STATE.bpm_original; next_beat_check=time.time()+beat_interval
    F=cv2.FONT_HERSHEY_DUPLEX; FS=cv2.FONT_HERSHEY_SIMPLEX; prev_time=time.time()
    mode_label = "NN" if USE_NN else "Regras"

    with HandLandmarker.create_from_options(options) as detector:
        while STATE.alive():
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.flip(frame,1); h_f,w_f=frame.shape[:2]
            now=time.time(); fps=1.0/max(now-prev_time,1e-9); prev_time=now

            frame_ts+=33
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            detector.detect_async(mp_img,frame_ts)

            with last_result_lock: result=last_result[0]

            display_name=None; raw_label=None; cmd=None

            if result and result.hand_landmarks:
                hand_lm   =result.hand_landmarks[0]
                hand_info =result.handedness[0]
                handedness=hand_info[0].category_name

                lm_list = hand_lm
                lm_arr  = np.array([[p.x,p.y,p.z] for p in hand_lm])
                lm_px   = [(int(p.x*w_f),int(p.y*h_f)) for p in hand_lm]

                display_name, raw_label, g_color = detect_gesture(lm_list, lm_arr, handedness)
                cmd = gesture_to_command(raw_label)

                draw_hand(frame,lm_px,g_color)
                wx,wy=lm_px[WRIST]; label_y=max(wy-35,65)
                tw=cv2.getTextSize(display_name,F,0.7,2)[0][0]
                draw_rect_alpha(frame,wx-tw//2-12,label_y-30,wx+tw//2+12,label_y+8,g_color,0.5)
                shadow_text(frame,display_name,(wx-tw//2,label_y),F,0.7,(255,255,255),2)

            if now>=next_beat_check:
                bpm_live=STATE.get_bpm() or STATE.bpm_original
                beat_interval=60.0/bpm_live; next_beat_check=now+beat_interval
                if cmd is not None:
                    if cmd==pending_cmd: consecutive+=1
                    else: pending_cmd=cmd; consecutive=1
                    if consecutive>=CONSECUTIVE_BEATS and cmd!=confirmed_cmd:
                        STATE.play() if cmd=="PLAY" else STATE.pause()
                        STATE.set_gesture(cmd); confirmed_cmd=cmd; consecutive=0
                else:
                    pending_cmd=None; consecutive=0

            # HUD
            playing=STATE.playing(); bpm_live=STATE.get_bpm(); bpm_orig=STATE.bpm_original
            draw_rect_alpha(frame,0,0,w_f,48,(10,10,10))
            shadow_text(frame,f"Phantom Conductor [{mode_label}]",(10,32),F,0.65,(220,220,220))
            shadow_text(frame,f"FPS:{fps:.0f}",(w_f-80,32),FS,0.5,(80,255,120))
            state_col=PLAY_COLOR if playing else PAUSE_COLOR
            draw_rect_alpha(frame,0,h_f-65,w_f,h_f,(10,10,10))
            shadow_text(frame,"PLAY" if playing else "PAUSE",(10,h_f-32),F,0.9,state_col,2)
            if bpm_live and bpm_orig:
                ratio=bpm_live/bpm_orig; r_col=(50,200,50) if abs(ratio-1.0)<0.05 else (50,200,255)
                shadow_text(frame,f"BPM:{bpm_live:.1f}  ratio:{ratio:.3f}  ref:{bpm_orig:.1f}",
                            (120,h_f-32),FS,0.5,r_col)
            g_col=PLAY_COLOR if cmd=="PLAY" else PAUSE_COLOR if cmd=="PAUSE" else NONE_COLOR
            prog=f" ({consecutive}/{CONSECUTIVE_BEATS})" if pending_cmd else ""
            shadow_text(frame,f"Gesto: {display_name or '—'}{prog}",(10,h_f-10),FS,0.48,g_col,1)
            shadow_text(frame,"Mao Aberta=PLAY  Punho=PAUSE  Espaco=toggle  Q=sair",
                        (w_f//2-210,h_f-10),FS,0.38,(100,100,100),1)

            cv2.imshow("Phantom Conductor NN",frame)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): STATE.stop(); break
            elif key==ord(' '): STATE.toggle()

    cap.release(); cv2.destroyAllWindows(); STATE.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    mp3_file=input("Caminho do backing track (.mp3): ").strip()
    bpm_orig=None
    try:
        tags=MP3(mp3_file).tags
        if tags and "TBPM" in tags: bpm_orig=float(tags["TBPM"].text[0]); print(f"BPM: {bpm_orig:.2f}")
    except: pass
    if bpm_orig is None: bpm_orig=float(input("BPM original: ").strip())

    STATE.bpm_original=bpm_orig; STATE.set_bpm(bpm_orig)

    print("\n=== Dispositivos de ENTRADA ===")
    for i,d in enumerate(sd.query_devices()):
        if d["max_input_channels"]>0: print(f"  {i:>3}  {d['name']}")
    dev_idx=int(input("Índice de ENTRADA: ").strip())
    cam_idx_str=input("Índice da câmera [0]: ").strip()
    cam_idx=int(cam_idx_str) if cam_idx_str else 0

    blocksize=int(SR*HOP_STREAM_SEC)
    threading.Thread(target=bpm_analysis_thread,                    daemon=True).start()
    threading.Thread(target=backing_track_thread, args=(mp3_file,), daemon=True).start()
    threading.Thread(target=playback_thread,                        daemon=True).start()

    with sd.InputStream(device=dev_idx,channels=1,samplerate=SR,
                        blocksize=blocksize,callback=audio_callback):
        gesture_vision_thread(cam_idx)

    STATE.stop(); print("\n🛑 Encerrado.")

if __name__=="__main__":
    main()