import sys
import pathlib
import re

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUTO-PATCH MEDIAPIPE  (safe to re-run, skips if already patched)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _patch_mediapipe():
    venv    = pathlib.Path(sys.executable).parent.parent
    mp_root = venv / "Lib" / "site-packages" / "mediapipe"
    if not mp_root.exists():
        return

    # Patch 1 — FieldDescriptor.label in solution_base.py
    sol = mp_root / "python" / "solution_base.py"
    if sol.exists():
        text = sol.read_text(encoding="utf-8", errors="ignore")
        if "PATCHED_LABEL" not in text:
            new_text = re.sub(
                r"(options_field_list\[field_name\])\.label",
                r"getattr(\1, 'label', None)  # PATCHED_LABEL",
                text
            )
            if new_text == text:
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if "].label" in line and "field_name" in line:
                        lines[i] = line.replace("].label", "]  # PATCHED_LABEL")
                new_text = "\n".join(lines)
            sol.write_text(new_text, encoding="utf-8")
            print("MediaPipe patch 1 applied (FieldDescriptor.label)")

    # Patch 2 — MessageFactory.GetPrototype across all .py files
    for f in mp_root.rglob("*.py"):
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if "GetPrototype" in t and "PATCHED_PROTO" not in t:
                fixed = t.replace(
                    "factory.GetPrototype(desc)",
                    "(factory.GetMessageClass(desc) "
                    "if hasattr(factory, 'GetMessageClass') "
                    "else factory.GetPrototype(desc))  # PATCHED_PROTO"
                )
                if fixed != t:
                    f.write_text(fixed, encoding="utf-8")
                    print(f"MediaPipe patch 2 applied ({f.name})")
        except Exception:
            pass

_patch_mediapipe()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_PATH       = "syncope_lstm_v1.h5"
SEQ_LEN          = 20      # must match training
EAR_THRESHOLD    = 0.25    # blink detection cutoff
LSTM_THRESHOLD   = 0.3     # risk score cutoff (tuned for high recall)
FRAME_THRESHOLD  = 45      # frames of sustained warning → CRITICAL (~1.5s)
CALIBRATION_TIME = 5       # seconds to measure baseline at startup

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOAD MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load {MODEL_PATH}\n{e}")
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MEDIAPIPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

frame_buffer = deque(maxlen=SEQ_LEN)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FEATURE EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]


def get_features(landmarks):
    def calc_ear(pts):
        p = [np.array([landmarks[i].x, landmarks[i].y]) for i in pts]
        return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2.0 * np.linalg.norm(p[0]-p[3]))

    l_ear   = calc_ear(L_EYE)
    r_ear   = calc_ear(R_EYE)
    avg_ear = (l_ear + r_ear) / 2.0

    upper = np.linalg.norm(np.array([landmarks[1].x,  landmarks[1].y])  - np.array([landmarks[10].x,  landmarks[10].y]))
    lower = np.linalg.norm(np.array([landmarks[1].x,  landmarks[1].y])  - np.array([landmarks[152].x, landmarks[152].y]))
    pitch = upper / lower

    mar = np.linalg.norm(
        np.array([landmarks[13].x, landmarks[13].y]) -
        np.array([landmarks[14].x, landmarks[14].y])
    )
    return l_ear, r_ear, avg_ear, mar, pitch


def get_lstm_prob():
    if len(frame_buffer) < SEQ_LEN:
        return None
    window = list(frame_buffer)
    ears   = [f[0] for f in window]
    blinks = sum(1 for j in range(1, len(ears))
                 if ears[j-1] >= EAR_THRESHOLD and ears[j] < EAR_THRESHOLD)
    seq = np.array([f + [blinks] for f in window], dtype=np.float32)[np.newaxis, ...]
    return float(model.predict(seq, verbose=0)[0][0])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CAMERA LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter              = 0
start_time           = time.time()
calibrated           = False
base_ear, base_pitch = 0.28, 1.0
prob                 = 0.0

print("System online. Sit upright and look at the camera — calibrating for 5 seconds...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = face_mesh.process(rgb_frame)

    status, color = "Monitoring", (0, 200, 0)
    debug_info    = ""

    # ── A. FACE DETECTED ─────────────────────────────────────────────────────
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        l_ear, r_ear, avg_ear, mar, pitch = get_features(landmarks)

        if not calibrated:
            elapsed   = time.time() - start_time
            remaining = max(0, int(CALIBRATION_TIME - elapsed))
            status, color = f"Calibrating: {remaining}s", (255, 220, 0)
            if elapsed >= CALIBRATION_TIME:
                base_ear, base_pitch = avg_ear, pitch
                calibrated = True
                print(f"Calibration done — baseline EAR={base_ear:.3f}  pitch={base_pitch:.3f}")
        else:
            frame_buffer.append([avg_ear, mar, pitch])
            prob = get_lstm_prob()

            if prob is None:
                status = f"Buffering {len(frame_buffer)}/{SEQ_LEN}"
                color  = (255, 220, 0)
            else:
                eyes_closed  = (l_ear < base_ear * 0.75) and (r_ear < base_ear * 0.75)
                head_slumped = (pitch > base_pitch * 1.15)

                if eyes_closed and (head_slumped or prob > LSTM_THRESHOLD):
                    counter += 1
                else:
                    counter = max(0, counter - 5)

                debug_info = f"EAR {avg_ear:.3f}  MAR {mar:.3f}  Pitch {pitch:.3f}  Risk {prob:.0%}"

    # ── B. FACE LOST ──────────────────────────────────────────────────────────
    else:
        if calibrated:
            if counter > 5:
                counter += 1
                status, color = "TRACKING LOST — possible slump", (0, 140, 255)
            else:
                counter = max(0, counter - 1)
                status, color = "Searching for face...", (180, 180, 180)

    # ── C. SEVERITY OVERRIDE ─────────────────────────────────────────────────
    if counter >= FRAME_THRESHOLD:
        status, color = "CRITICAL — SYNCOPE DETECTED", (0, 0, 220)
    elif counter > 10 and "TRACKING" not in status:
        status, color = "WARNING — signs detected", (0, 140, 255)

    # ── D. DRAW ───────────────────────────────────────────────────────────────
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 55), color, -1)
    cv2.putText(frame, status, (14, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    if calibrated and prob is not None:
        bar_w = int((w - 28) * min(prob, 1.0))
        cv2.rectangle(frame, (14, 62), (w - 14, 76), (50, 50, 50), -1)
        cv2.rectangle(frame, (14, 62), (14 + bar_w, 76), color, -1)

    if debug_info:
        cv2.putText(frame, debug_info, (14, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    cv2.putText(frame, f"Counter: {counter}/{FRAME_THRESHOLD}",
                (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Syncope Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()