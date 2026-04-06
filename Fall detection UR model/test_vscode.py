# test_vscode.py
# pip install ultralytics tensorflow opencv-python

import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque

# ── Config ──────────────────────────────────────────────────
MODEL_LSTM = "fall_detector.keras"   # téléchargé depuis Kaggle Output
SEQ_LEN    = 30
CONF       = 0.4
SOURCE     = 0        # 0 = webcam  |  "video.mp4" = fichier

LABELS = ["No-Fall", "FALL !!"]
COLORS = [(0, 200, 0), (0, 0, 255)]

# ── Load ────────────────────────────────────────────────────
yolo = YOLO("yolo11n-pose.pt")
lstm = load_model(MODEL_LSTM)
print("Modèles chargés")

# ── Buffer ──────────────────────────────────────────────────
buffer = deque(maxlen=SEQ_LEN)

# ── Loop ────────────────────────────────────────────────────
cap = cv2.VideoCapture(SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Extraction keypoints
    results = yolo(frame, conf=CONF, verbose=False)
    kp_vec  = np.zeros(34)

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kp = results[0].keypoints.xy[0].cpu().numpy()
        kp[:, 0] /= w
        kp[:, 1] /= h
        kp_vec = kp.flatten()

    buffer.append(kp_vec)

    # Prédiction
    label_txt = "buffering..."
    color     = (180, 180, 180)

    if len(buffer) == SEQ_LEN:
        seq   = np.array(buffer)[np.newaxis, ...]   # (1, 30, 34)
        proba = lstm.predict(seq, verbose=0)[0]
        cls   = int(np.argmax(proba))
        label_txt = f"{LABELS[cls]}  {proba[cls]:.2f}"
        color     = COLORS[cls]

    # Affichage
    annotated = results[0].plot()
    cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(annotated, label_txt, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Fall Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
