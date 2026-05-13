"""
train_gesture_classifier.py — Treina MLP em landmarks extraídos do HaGRID
==========================================================================
Processo:
  1. Para cada imagem do HaGRID: extrai 21 landmarks com MediaPipe
  2. Normaliza: centraliza no pulso, escala pela distância pulso→middle_mcp
  3. Treina MLP scikit-learn nos 63 features (21 × x,y,z)
  4. Salva gesture_classifier.pkl

Estrutura esperada do HaGRID:
  hagrid/
    call/         ← nome da classe = nome da pasta
      img001.jpg
      img002.jpg
    fist/
      ...
    five/
    ...

HaGRID classes relevantes para play/pause:
  PLAY:  five, five_inverted, three2, four
  PAUSE: fist, fist_inverted, stop, stop_inverted

Baixe o subset do HaGRID em:
  https://github.com/hukenovs/hagrid  (subsample disponível ~1GB)
  ou via kaggle: hukenovs/hagrid-dataset

Dependências:
    pip install mediapipe opencv-python scikit-learn tqdm numpy
"""

import os, sys, pickle, argparse
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "hand_landmarker.task"
MODEL_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
OUTPUT_PATH  = "gesture_classifier.pkl"

# Quais classes do HaGRID usar (None = todas as pastas encontradas)
CLASSES_TO_USE = [
    "train_val_fist",
    "train_val_five",
    "train_val_stop",
    "train_val_stop_inverted",
    "train_val_four",
    "train_val_three",
    "train_val_three2",
    "train_val_two_up",
    "train_val_two_up_inverted",
    "train_val_one",
    "train_val_peace",
    "train_val_peace_inverted",
    "train_val_like",
    "train_val_dislike",
    "train_val_ok",
    "train_val_palm",
    "train_val_call",
    "train_val_rock",
    "train_val_mute",
]

MAX_IMAGES_PER_CLASS = 2000   # limite por classe para velocidade
WRIST      = 0
MIDDLE_MCP = 9


# ── Download modelo se necessário ─────────────────────────────────────────────
def download_model():
    if os.path.exists(MODEL_PATH): return
    print("Baixando hand_landmarker.task (~8MB)...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo baixado!")


# ── Normalização de landmarks ─────────────────────────────────────────────────
def landmarks_to_features(lm_list):
    lm = np.array([[p.x, p.y, p.z] for p in lm_list])
    lm = lm - lm[WRIST]
    scale = np.linalg.norm(lm[MIDDLE_MCP])
    if scale > 1e-6: lm = lm / scale
    return lm.flatten()   # (63,)


# ── Extração de features de um diretório de imagens ───────────────────────────
def extract_features_from_dir(class_dir, class_name, detector, max_images):
    """Processa até max_images imagens, retorna lista de feature vectors."""
    features = []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [f for f in os.listdir(class_dir)
             if os.path.splitext(f)[1].lower() in exts][:max_images]

    for fname in tqdm(files, desc=f"  {class_name}", leave=False, ncols=70):
        path = os.path.join(class_dir, fname)
        img  = cv2.imread(path)
        if img is None: continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result  = detector.detect(mp_img)

        if result.hand_landmarks:
            feat = landmarks_to_features(result.hand_landmarks[0])
            features.append(feat)

    return features


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hagrid_dir", help="Caminho para a pasta raiz do HaGRID")
    parser.add_argument("--max",  type=int, default=MAX_IMAGES_PER_CLASS,
                        help="Máximo de imagens por classe")
    parser.add_argument("--out",  default=OUTPUT_PATH,
                        help="Arquivo de saída do classificador")
    args = parser.parse_args()

    download_model()

    # Descobre classes disponíveis
    all_dirs = [d for d in os.listdir(args.hagrid_dir)
                if os.path.isdir(os.path.join(args.hagrid_dir, d))]
    classes  = [c for c in CLASSES_TO_USE if c in all_dirs] if CLASSES_TO_USE \
               else all_dirs
    print(f"\nClasses encontradas ({len(classes)}): {classes}\n")

    # Configuração MediaPipe IMAGE mode (estático, adequado para dataset)
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    X, y = [], []
    with HandLandmarker.create_from_options(options) as detector:
        for cls in classes:
            cls_dir  = os.path.join(args.hagrid_dir, cls)
            feats    = extract_features_from_dir(cls_dir, cls, detector, args.max)
            X.extend(feats)
            y.extend([cls] * len(feats))
            print(f"  {cls}: {len(feats)} landmarks extraídos")

    if not X:
        print("Nenhum landmark extraído. Verifique o caminho do HaGRID.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)
    print(f"\nTotal: {len(X)} amostras, {X.shape[1]} features, {len(set(y))} classes")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # Normalização
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Treina MLP
    print("\nTreinando MLP...")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True,
    )
    clf.fit(X_train, y_train)

    # Avaliação
    y_pred = clf.predict(X_test)
    print("\n" + "="*60)
    print(classification_report(y_test, y_pred,
                                  target_names=le.classes_))

    acc = (y_pred == y_test).mean()
    print(f"Acurácia no teste: {acc:.1%}")

    # Salva
    data = {
        "model":  clf,
        "scaler": scaler,
        "labels": list(le.classes_),
        "label_encoder": le,
    }
    with open(args.out, "wb") as f:
        pickle.dump(data, f)
    print(f"\n✅ Classificador salvo em: {args.out}")
    print(f"   Classes: {list(le.classes_)}")
    print("\nAgora execute: python phantom_conductor_nn.py")


if __name__ == "__main__":
    main()