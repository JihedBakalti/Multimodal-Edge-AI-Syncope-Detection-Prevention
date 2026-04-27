# test_vscode.py
# pip install ultralytics tensorflow opencv-python

# ── Supprime les logs lents TF AVANT tout import ────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"   # silence TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # désactive oneDNN (+ rapide)
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"  # force CPU direct (pas de scan GPU)

import cv2
import numpy as np
from collections import deque
import threading

# ── Config ──────────────────────────────────────────────────
MODEL_LSTM = "fall_detector.keras"
SEQ_LEN    = 30
CONF       = 0.4
SOURCE     = 0          # 0 = webcam  |  "video.mp4" = fichier
PRED_EVERY = 5          # prédire toutes les N frames

LABELS = ["No-Fall", "FALL !!"]
COLORS = [(0, 200, 0), (0, 0, 255)]

# ── État partagé entre le thread chargement et la boucle ────
models_ready = threading.Event()
yolo_model   = [None]
lstm_model   = [None]
predict_fn   = [None]

def load_models():
    """Chargement + warm-up dans un thread séparé."""
    from ultralytics import YOLO
    from tensorflow.keras.models import load_model  # import tardif = plus rapide

    y = YOLO("yolo11n-pose.pt")
    l = load_model(MODEL_LSTM)

    # Warm-up des deux modèles
    _dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    y(_dummy_frame, conf=CONF, verbose=False)

    _dummy_seq = np.zeros((1, SEQ_LEN, 34), dtype=np.float32)
    l.predict(_dummy_seq, verbose=0)

    yolo_model[0]  = y
    lstm_model[0]  = l
    predict_fn[0]  = l.__call__   # appel direct, sans overhead .predict()
    models_ready.set()
    print("✅ Modèles prêts !")

# ─── Lancer le chargement en arrière-plan ───────────────────
t = threading.Thread(target=load_models, daemon=True)
t.start()

# ─── Ouvrir la caméra IMMÉDIATEMENT ─────────────────────────
if isinstance(SOURCE, int):
    cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW)   # DirectShow = rapide sur Windows
else:
    cap = cv2.VideoCapture(SOURCE)

buffer    = deque(maxlen=SEQ_LEN)
label_txt = "Upload  models..."
color     = (180, 180, 180)
frame_idx = 0

# ── Boucle principale ────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_idx += 1

    if models_ready.is_set():
        # ── Modèles disponibles : inférence normale ──────────
        results  = yolo_model[0](frame, conf=CONF, verbose=False)
        kp_vec   = np.zeros(34)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kp = results[0].keypoints.xy[0].cpu().numpy()
            kp[:, 0] /= w
            kp[:, 1] /= h
            kp_vec = kp.flatten()

        buffer.append(kp_vec)

        if len(buffer) == SEQ_LEN and frame_idx % PRED_EVERY == 0:
            seq   = np.array(buffer, dtype=np.float32)[np.newaxis, ...]
            proba = predict_fn[0](seq, training=False).numpy()[0]
            cls   = int(np.argmax(proba))
            label_txt = f"{LABELS[cls]}  {proba[cls]:.2f}"
            color     = COLORS[cls]
        elif len(buffer) < SEQ_LEN:
            label_txt = f"buffering {len(buffer)}/{SEQ_LEN}"
            color     = (180, 180, 180)

        annotated = results[0].plot()
    else:
        # ── Modèles pas encore prêts : affiche juste la frame ─
        annotated = frame.copy()

    # ── Affichage ────────────────────────────────────────────
    cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(annotated, label_txt, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Fall Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
