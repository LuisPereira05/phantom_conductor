"""
Diagnóstico de câmera — Phantom Conductor
Execute antes do programa principal para identificar o índice correto.
"""
import cv2
import sys

print("=" * 50)
print("  Diagnóstico de câmera")
print("=" * 50)

encontradas = []

for i in range(10):
    # cv2.CAP_V4L2 força backend V4L2 no Linux
    for backend, nome in [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "AUTO")]:
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✅ Índice {i} [{nome}]: {w}x{h} — OK")
                encontradas.append((i, backend, nome))
            else:
                print(f"  ⚠️  Índice {i} [{nome}]: abre mas não lê frame")
            cap.release()
            break  # achou com esse índice, não precisa tentar outro backend

if not encontradas:
    print("\n  ❌ Nenhuma câmera encontrada!")
    print("  Verifique se a câmera está conectada e com permissão de acesso.")
    print("  No Linux tente:  sudo usermod -aG video $USER  e reinicie a sessão.")
    sys.exit(1)

print()
print(f"  {len(encontradas)} câmera(s) encontrada(s).")
idx, backend, _ = encontradas[0]
print(f"  Use o índice {idx} no programa principal.")

# Mostra preview da câmera escolhida
print(f"\nAbrindo preview da câmera {idx}... (pressione Q para fechar)")
cap = cv2.VideoCapture(idx, backend)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao ler frame.")
        break
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Camera {idx} — OK  (Q para sair)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Diagnostico Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Diagnóstico concluído.")