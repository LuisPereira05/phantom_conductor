"""
Reconhecimento de Gestos da Mão com MediaPipe 0.10+
====================================================
Dependências:
    pip install mediapipe opencv-python numpy

Uso:
    python reconhecimento_gestos.py

O modelo será baixado automaticamente na primeira execução (~8MB).
"""

import cv2
import numpy as np
import time
import os
import urllib.request
from collections import deque, Counter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions


# ─── Download automático do modelo ────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Baixando modelo MediaPipe (~8MB)...")
    print(f"URL: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo baixado com sucesso!")
    except Exception as e:
        print(f"\nErro ao baixar automaticamente: {e}")
        print("Por favor baixe manualmente e coloque na mesma pasta do script:")
        print(MODEL_URL)
        raise SystemExit(1)


# ─── Índices dos landmarks ─────────────────────────────────────────────────────
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20
MIDDLE_MCP = 9

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ─── Lógica de gestos ──────────────────────────────────────────────────────────
def lm_to_array(landmark_list):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list])

def is_finger_up(lm, tip, pip):
    return lm[tip][1] < lm[pip][1]

def is_thumb_up(lm, handedness):
    return lm[THUMB_TIP][0] < lm[THUMB_IP][0] if handedness == "Right" \
           else lm[THUMB_TIP][0] > lm[THUMB_IP][0]

def fingers_status(lm, handedness):
    return [
        is_thumb_up(lm, handedness),
        is_finger_up(lm, INDEX_TIP, INDEX_PIP),
        is_finger_up(lm, MIDDLE_TIP, MIDDLE_PIP),
        is_finger_up(lm, RING_TIP, RING_PIP),
        is_finger_up(lm, PINKY_TIP, PINKY_PIP),
    ]

def dist(lm, a, b):
    return np.linalg.norm(lm[a] - lm[b])

def detect_gesture(lm, handedness):
    """Retorna (nome_gesto, cor_bgr)."""
    f = fingers_status(lm, handedness)
    thumb, index, middle, ring, pinky = f
    count = sum(f)

    if count == 0:
        return "Punho", (60, 60, 220)
    if count == 5:
        return "Mao Aberta", (50, 200, 50)
    if thumb and not index and not middle and not ring and not pinky:
        return "Joinha!", (0, 200, 255)
    if not thumb and index and middle and not ring and not pinky:
        return "Paz / Vitoria", (255, 200, 0)
    if not thumb and index and not middle and not ring and not pinky:
        return "Apontando", (200, 100, 255)
    if not thumb and index and not middle and not ring and pinky:
        return "Rock!", (0, 50, 255)
    if thumb and index and not middle and not ring and pinky:
        return "Spider-Man!", (200, 0, 0)
    if thumb and not index and not middle and not ring and pinky:
        return "Shaka!", (50, 210, 210)

    palm = dist(lm, WRIST, MIDDLE_MCP)
    if palm > 0 and dist(lm, THUMB_TIP, INDEX_TIP) / palm < 0.22 and middle and ring and pinky:
        return "OK!", (0, 255, 150)

    if not thumb and index and middle and ring and pinky:
        return "Quatro", (255, 100, 100)
    if not thumb and index and middle and ring and not pinky:
        return "Tres", (255, 150, 0)
    if thumb and index and not middle and not ring and not pinky:
        return "L", (100, 255, 255)

    return f"{count} dedos", (180, 180, 180)


# ─── Estabilizador de gestos ───────────────────────────────────────────────────
class GestureStabilizer:
    def __init__(self, window=8):
        self.history = deque(maxlen=window)

    def update(self, gesture):
        self.history.append(gesture)
        return Counter(self.history).most_common(1)[0][0]


# ─── Utilitários visuais ───────────────────────────────────────────────────────
def draw_rect_alpha(img, x1, y1, x2, y2, color, alpha=0.65):
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]-1), min(y2, img.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

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

def draw_hud(frame, hands_data, fps):
    h, w = frame.shape[:2]
    F  = cv2.FONT_HERSHEY_DUPLEX
    FS = cv2.FONT_HERSHEY_SIMPLEX

    # Barra superior
    draw_rect_alpha(frame, 0, 0, w, 50, (15, 15, 15))
    shadow_text(frame, "Reconhecimento de Gestos | MediaPipe 0.10+",
                (10, 33), F, 0.6, (200, 200, 200))
    fps_str = f"FPS: {fps:.0f}"
    tw = cv2.getTextSize(fps_str, FS, 0.55, 1)[0][0]
    shadow_text(frame, fps_str, (w - tw - 12, 33), FS, 0.55, (80, 255, 120))

    # Painel por mão detectada
    for i, (gesture, color, handedness) in enumerate(hands_data):
        px, py = 10 + i * 290, h - 115
        draw_rect_alpha(frame, px, py, px + 270, py + 100, (15, 15, 15))
        cv2.rectangle(frame, (px, py), (px + 270, py + 5), color, -1)
        lado = "Mao Direita" if handedness == "Right" else "Mao Esquerda"
        shadow_text(frame, lado,    (px + 8, py + 28), FS, 0.50, (180,180,180), 1)
        shadow_text(frame, gesture, (px + 8, py + 68), F,  0.85, color, 2)

    if not hands_data:
        msg = "Mostre sua mao para a camera..."
        tw = cv2.getTextSize(msg, FS, 0.75, 2)[0][0]
        shadow_text(frame, msg, ((w-tw)//2, h//2), FS, 0.75, (140,140,140), 2)

    shadow_text(frame, "Q: sair   S: screenshot",
                (10, h - 8), FS, 0.42, (100,100,100), 1)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    download_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: nao foi possivel abrir a camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    stabilizers = {}
    prev_time = time.time()

    print("=" * 52)
    print("  Reconhecimento de Gestos — MediaPipe 0.10+")
    print("=" * 52)
    print("  ✊ Punho       🖐  Mao Aberta    👍 Joinha")
    print("  ✌  Paz/Vitoria  ☝  Apontando   🤘 Rock")
    print("  🕷  Spider-Man  🤙 Shaka         👌 OK")
    print("=" * 52)
    print("  Q = sair  |  S = screenshot")
    print("=" * 52)

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h_frame, w_frame = frame.shape[:2]

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result = detector.detect(mp_image)

            hands_data = []
            active_ids = set()

            if result.hand_landmarks:
                for idx, (hand_lm, hand_info) in enumerate(
                    zip(result.hand_landmarks, result.handedness)
                ):
                    handedness = hand_info[0].category_name  # "Left" ou "Right"
                    active_ids.add(idx)

                    lm = lm_to_array(hand_lm)
                    lm_px = [(int(p.x * w_frame), int(p.y * h_frame)) for p in hand_lm]

                    gesture, color = detect_gesture(lm, handedness)

                    if idx not in stabilizers:
                        stabilizers[idx] = GestureStabilizer()
                    gesture = stabilizers[idx].update(gesture)

                    # Recalcula cor para o gesto estabilizado
                    _, color = detect_gesture(lm, handedness)

                    draw_hand(frame, lm_px, color)

                    # Label flutuante
                    wx, wy = lm_px[WRIST]
                    label_y = max(wy - 35, 65)
                    tw = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0][0]
                    draw_rect_alpha(frame,
                                    wx - tw//2 - 12, label_y - 30,
                                    wx + tw//2 + 12, label_y + 8,
                                    color, alpha=0.5)
                    shadow_text(frame, gesture,
                                (wx - tw//2, label_y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                (255, 255, 255), 2)

                    hands_data.append((gesture, color, handedness))

            # Remove estabilizadores de mãos que saíram
            for k in list(stabilizers.keys()):
                if k not in active_ids:
                    del stabilizers[k]

            draw_hud(frame, hands_data, fps)
            cv2.imshow("Reconhecimento de Gestos", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Encerrando...")
                break
            elif key == ord("s"):
                fname = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(fname, frame)
                print(f"Screenshot salvo: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()