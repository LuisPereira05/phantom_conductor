"""
Teste do mecanismo de snapshot sob demanda.
Pressione ESPAÇO para tirar um snapshot manual (simulando o downbeat).
Pressione Q para sair.
"""

import cv2
import threading
import time
import numpy as np

# ── Evento que simula o sinal de downbeat ─────────────────────────────────────
snapshot_event = threading.Event()

# ── Thread simulando o metrônomo (dispara a cada 1s) ─────────────────────────
def metronome_thread():
    print("[metro] Metrônomo iniciado — 1 beat/segundo")
    while True:
        time.sleep(1.0)
        print("[metro] BEAT → disparando snapshot_event")
        snapshot_event.set()

threading.Thread(target=metronome_thread, daemon=True).start()

# ── Câmera ────────────────────────────────────────────────────────────────────
cam_idx = int(input("Índice da câmera: ").strip() or "0")

cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
if not cap.isOpened():
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)
if not cap.isOpened():
    print(f"Camara {cam_idx} nao encontrada.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ── Câmera contínua em background ─────────────────────────────────────────────
# A câmera nunca para — uma thread lê frames o mais rápido possível e
# sempre sobrescreve `latest_frame`. O snapshot lê essa variável: zero lag.
latest_frame = None
frame_lock   = threading.Lock()
cam_running  = True

def camera_capture_thread():
    """Lê frames continuamente, mantendo sempre o mais recente disponível."""
    global latest_frame, cam_running
    while cam_running:
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)
            with frame_lock:
                latest_frame = frame
        # sem sleep: roda tão rápido quanto a câmera permite

threading.Thread(target=camera_capture_thread, daemon=True).start()

# Aguarda o primeiro frame antes de continuar
print("Camera aberta. Aguardando primeiro frame...")
while True:
    with frame_lock:
        ready = latest_frame is not None
    if ready:
        break
    time.sleep(0.01)

print("Pronto!")
print("  ESPACO = snapshot manual   Q = sair")

# ── Tela inicial ──────────────────────────────────────────────────────────────
display = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(display, "Aguardando primeiro beat...",
            (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
cv2.namedWindow("Teste Snapshot", cv2.WINDOW_NORMAL)
cv2.imshow("Teste Snapshot", display)

snap_count = 0

while True:
    # Teclado sempre responsivo
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        print("[manual] Snapshot manual disparado")
        snapshot_event.set()

    # Aguarda beat com timeout curto (nao bloqueia o waitKey)
    fired = snapshot_event.wait(timeout=0.05)
    snapshot_event.clear()

    if not fired:
        cv2.imshow("Teste Snapshot", display)
        continue

    # ── GOT BEAT: lê o frame mais recente — já está pronto, sem espera ───────
    t0 = time.perf_counter()

    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    t_grab = time.perf_counter() - t0

    if frame is None:
        print("[snap] ERRO: nenhum frame disponivel!")
        continue

    snap_count += 1
    print(f"[snap #{snap_count}]  leitura={t_grab*1000:.2f}ms  shape={frame.shape}")

    # Anota o frame
    cv2.putText(frame, f"Snapshot #{snap_count}", (10, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 100), 2)
    cv2.putText(frame, f"leitura: {t_grab*1000:.2f}ms  (tempo real)",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, "ESPACO=snapshot manual  Q=sair",
                (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    # Flash amarelo indica beat
    cv2.circle(frame, (620, 20), 14, (0, 220, 255), -1)

    display = frame.copy()
    cv2.imshow("Teste Snapshot", display)

cam_running = False
cap.release()
cv2.destroyAllWindows()
print(f"\nTotal de snapshots: {snap_count}")