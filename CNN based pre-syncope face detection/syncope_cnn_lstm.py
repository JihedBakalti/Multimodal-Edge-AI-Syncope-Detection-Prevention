"""
Syncope (pre-fainting) detection via webcam.
Uses MediaPipe FaceLandmarker (Tasks API, mediapipe >= 0.10) + optional .h5 LSTM.
Falls back to pure rule-based detection if no .h5 model is present.

Setup (one-time):
    pip install -r requirements.txt

Run:
    python syncope_detection.py
Press 'q' to quit.
"""

import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"]   = "3"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import warnings
warnings.filterwarnings("ignore")

import time
SCRIPT_START = time.perf_counter()          # before heavy imports so startup time is accurate

import cv2
import numpy as np
import h5py
import threading
import urllib.request
import pathlib
from collections import deque

# ─── PATHS ───────────────────────────────────────────────────────────────────

SCRIPT_DIR       = pathlib.Path(__file__).parent.resolve()
_FACE_MODEL_NAME = "face_landmarker.task"
_FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# Check repo folder first, then .cache/, then download.
# Put face_landmarker.task next to this script and cloners never wait.
_candidates = [
    SCRIPT_DIR / _FACE_MODEL_NAME,
    SCRIPT_DIR / ".cache" / _FACE_MODEL_NAME,
]
FACE_MODEL_PATH = next((p for p in _candidates if p.exists()), None)

if FACE_MODEL_PATH is None:
    FACE_MODEL_PATH = SCRIPT_DIR / ".cache" / _FACE_MODEL_NAME
    FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"face_landmarker.task not found. Downloading to {FACE_MODEL_PATH} ...")
    _tmp = FACE_MODEL_PATH.with_suffix(".part")
    def _dl_progress(block, block_size, total):
        pct = min(block * block_size / total * 100, 100) if total else 0
        print(f"\r  {pct:.1f}%", end="", flush=True)
    urllib.request.urlretrieve(_FACE_MODEL_URL, _tmp, reporthook=_dl_progress)
    print("\r  100.0% - saved.")
    _tmp.replace(FACE_MODEL_PATH)

LSTM_MODEL_PATH_H5 = SCRIPT_DIR / "syncope_lstm_v1.h5"

# ─── CONFIG ──────────────────────────────────────────────────────────────────

SEQ_LEN           = 20
EAR_THRESHOLD     = 0.25
LSTM_THRESHOLD    = 0.30
FRAME_THRESHOLD   = 90      # frames of sustained warning to trigger CRITICAL
WARNING_THRESHOLD = 20      # counter must exceed this for face-loss alarm
RISK_EMA_ALPHA    = 0.08
RISK_MAX_STEP     = 0.06
NORMAL_POSE_CAP   = 0.35
HIGH_RISK_LEVEL   = 0.99
HIGH_RISK_SECONDS = 3.0
BASE_PITCH        = 1.00

L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]

# ─── OPTIONAL LSTM ───────────────────────────────────────────────────────────
# Loads the trained .h5 weights directly with h5py + NumPy, so TensorFlow is
# not required at runtime.

model = None    # Native NumPy weight bundle
LSTM_MODEL_TYPE = None  # "native"

# Background init state
models_ready = False
init_error = None

_lstm_ready = False
_landmarker_ready = False
_init_lock = threading.Lock()
_lstm_load_seconds = None
_landmarker_load_seconds = None
_model_init_start = None


def _read_dataset(file_handle, path):
    return np.asarray(file_handle[path][()], dtype=np.float32)


def _load_native_lstm_weights():
    with h5py.File(LSTM_MODEL_PATH_H5, "r") as file_handle:
        return {
            "lstm1_kernel": _read_dataset(file_handle, "model_weights/lstm/sequential/lstm/lstm_cell/kernel"),
            "lstm1_recurrent_kernel": _read_dataset(file_handle, "model_weights/lstm/sequential/lstm/lstm_cell/recurrent_kernel"),
            "lstm1_bias": _read_dataset(file_handle, "model_weights/lstm/sequential/lstm/lstm_cell/bias"),
            "lstm2_kernel": _read_dataset(file_handle, "model_weights/lstm_1/sequential/lstm_1/lstm_cell/kernel"),
            "lstm2_recurrent_kernel": _read_dataset(file_handle, "model_weights/lstm_1/sequential/lstm_1/lstm_cell/recurrent_kernel"),
            "lstm2_bias": _read_dataset(file_handle, "model_weights/lstm_1/sequential/lstm_1/lstm_cell/bias"),
            "dense_kernel": _read_dataset(file_handle, "model_weights/dense/sequential/dense/kernel"),
            "dense_bias": _read_dataset(file_handle, "model_weights/dense/sequential/dense/bias"),
            "out_kernel": _read_dataset(file_handle, "model_weights/dense_1/sequential/dense_1/kernel"),
            "out_bias": _read_dataset(file_handle, "model_weights/dense_1/sequential/dense_1/bias"),
        }


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_layer_forward(inputs, kernel, recurrent_kernel, bias, return_sequences):
    units = recurrent_kernel.shape[0]
    hidden = np.zeros(units, dtype=np.float32)
    cell = np.zeros(units, dtype=np.float32)
    outputs = []

    for step in inputs:
        z = np.dot(step, kernel) + np.dot(hidden, recurrent_kernel) + bias
        i, f, c_bar, o = np.split(z, 4)
        i = _sigmoid(i)
        f = _sigmoid(f)
        c_bar = np.tanh(c_bar)
        o = _sigmoid(o)
        cell = f * cell + i * c_bar
        hidden = o * np.tanh(cell)
        if return_sequences:
            outputs.append(hidden.copy())

    if return_sequences:
        return np.asarray(outputs, dtype=np.float32)
    return hidden


def init_lstm_model():
    """Load the optional LSTM model directly from HDF5 weights."""
    global model, _lstm_ready, _lstm_load_seconds, LSTM_MODEL_TYPE
    start = time.perf_counter()
    try:
        if LSTM_MODEL_PATH_H5.exists():
            print("Loading native LSTM weights ...", end=" ", flush=True)
            model = _load_native_lstm_weights()
            LSTM_MODEL_TYPE = "native"
            print("done.")
        else:
            print("No LSTM model found. Using rule-based detection.")

        _lstm_ready = True
        _lstm_load_seconds = time.perf_counter() - start
        print(f"LSTM loaded in {_lstm_load_seconds:.2f}s")
    except Exception as e:
        init_error = e
        print(f"LSTM init failed: {e}")


def init_face_landmarker():
    """Load MediaPipe and create the face landmarker independently."""
    global mp, face_landmarker, models_ready, init_error, _landmarker_ready, _landmarker_load_seconds
    start = time.perf_counter()
    try:
        import mediapipe as mp
        print("Initialising face landmarker ...", end=" ", flush=True)
        _opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(_opts)
        print("done.")
        _landmarker_ready = True
        _landmarker_load_seconds = time.perf_counter() - start
        print(f"Face landmarker loaded in {_landmarker_load_seconds:.2f}s")
    except Exception as e:
        init_error = e
        print(f"Face landmarker init failed: {e}")


def mark_models_ready_if_done():
    """Print the combined model init time once both background loads finish."""
    global models_ready
    with _init_lock:
        if not models_ready and _lstm_ready and _landmarker_ready:
            models_ready = True
            if _model_init_start is not None:
                print(f"All models ready in {time.perf_counter() - _model_init_start:.2f}s")

# MediaPipe and model initialization will run in background thread (init_models)

# ─── CAMERA ──────────────────────────────────────────────────────────────────

def open_camera():
    """Open the webcam with a short warm-up so the first usable frame arrives faster."""
    print("Opening camera ...", end=" ", flush=True)
    start = time.perf_counter()

    candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [None]
    for backend in candidates:
        cap_obj = cv2.VideoCapture(0, backend) if backend is not None else cv2.VideoCapture(0)
        if not cap_obj.isOpened():
            cap_obj.release()
            continue

        cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_obj.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        warmup_deadline = time.perf_counter() + 2.0
        while time.perf_counter() < warmup_deadline:
            cap_obj.grab()
            ok, _ = cap_obj.read()
            if ok:
                print("done.")
                print(f"Camera ready in {time.perf_counter() - start:.2f}s")
                return cap_obj

        cap_obj.release()

    sys.exit("\nERROR: Could not open camera.")


# Start face-landmarker initialization as early as possible so cold-start time
# overlaps with camera setup.
print("Starting background initialization of models...", end=" ", flush=True)
_model_init_start = time.perf_counter()
t_face = threading.Thread(target=init_face_landmarker, daemon=True)
t_face.start()

cap = open_camera()

t_lstm = threading.Thread(target=init_lstm_model, daemon=True)
t_lstm.start()
print("started.")

# ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────

def calc_ear(lm, pts):
    p = [np.array([lm[i].x, lm[i].y]) for i in pts]
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (
        2.0 * np.linalg.norm(p[0] - p[3])
    )

def get_features(lm):
    l_ear   = calc_ear(lm, L_EYE)
    r_ear   = calc_ear(lm, R_EYE)
    avg_ear = (l_ear + r_ear) / 2.0
    mar = np.linalg.norm(
        np.array([lm[13].x, lm[13].y]) - np.array([lm[14].x, lm[14].y])
    )
    upper = np.linalg.norm(np.array([lm[1].x, lm[1].y]) - np.array([lm[10].x,  lm[10].y]))
    lower = np.linalg.norm(np.array([lm[1].x, lm[1].y]) - np.array([lm[152].x, lm[152].y]))
    pitch = (upper / lower) if lower > 1e-6 else 1.0
    return l_ear, r_ear, avg_ear, mar, pitch

def lstm_predict(buf):
    if model is None:
        return None
    if len(buf) < SEQ_LEN:
        return None
    window = list(buf)
    ears   = [f[0] for f in window]
    blink_count = sum(
        1 for j in range(1, len(ears))
        if ears[j - 1] >= EAR_THRESHOLD and ears[j] < EAR_THRESHOLD
    )
    blink_rate = blink_count / max(1, len(ears) - 1)
    seq = np.array([f + [blink_rate] for f in window], dtype=np.float32)
    
    if LSTM_MODEL_TYPE == "native":
        hidden_1 = _lstm_layer_forward(
            seq,
            model["lstm1_kernel"],
            model["lstm1_recurrent_kernel"],
            model["lstm1_bias"],
            return_sequences=True,
        )
        hidden_2 = _lstm_layer_forward(
            hidden_1,
            model["lstm2_kernel"],
            model["lstm2_recurrent_kernel"],
            model["lstm2_bias"],
            return_sequences=False,
        )
        dense = np.maximum(0.0, np.dot(hidden_2, model["dense_kernel"]) + model["dense_bias"])
        return float(_sigmoid(np.dot(dense, model["out_kernel"]) + model["out_bias"])[0])

    return None

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

LOOP_START       = time.perf_counter()   # anchor for MediaPipe timestamps
frame_buffer     = deque(maxlen=SEQ_LEN)
risk_ema         = None
high_risk_start  = None
counter          = 0
face_was_visible = False     # latched True once a face is seen
face_loss_alarm  = False     # latched True when face lost mid-warning
first_frame      = True

print("\nMonitoring started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Print startup time once, on the very first rendered frame
    if first_frame:
        first_frame = False
        print(f"Ready in {time.perf_counter() - SCRIPT_START:.2f}s")

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = []
    # Start detection as soon as the face landmarker is ready.
    # LSTM can finish later in background and will be used when available.
    mark_models_ready_if_done()
    if _landmarker_ready and globals().get("face_landmarker") is not None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int((time.perf_counter() - LOOP_START) * 1000)
        result   = face_landmarker.detect_for_video(mp_image, ts_ms)
        faces    = result.face_landmarks if result and result.face_landmarks else []
    else:
        cv2.putText(frame, "Initializing face landmarker...", (14, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    status     = "Monitoring"
    color      = (0, 200, 0)
    debug_info = ""
    risk_prob  = None

    # ── FACE VISIBLE ─────────────────────────────────────────────────────
    if faces:
        face_was_visible = True
        face_loss_alarm  = False        # face returned, cancel latch

        lm = faces[0]
        l_ear, r_ear, avg_ear, mar, pitch = get_features(lm)

        eyes_closed  = avg_ear < EAR_THRESHOLD
        head_slumped = pitch > BASE_PITCH * 1.15

        frame_buffer.append([avg_ear, mar, pitch])
        raw_prob = lstm_predict(frame_buffer)

        if raw_prob is not None:
            if risk_ema is None:
                # A single blink can be the first time the model emits a score.
                # Start conservatively so one blink does not appear as an instant spike.
                risk_ema = min(raw_prob, NORMAL_POSE_CAP) if not head_slumped else raw_prob
            else:
                target   = RISK_EMA_ALPHA * raw_prob + (1 - RISK_EMA_ALPHA) * risk_ema
                delta    = float(np.clip(target - risk_ema, -RISK_MAX_STEP, RISK_MAX_STEP))
                risk_ema = float(np.clip(risk_ema + delta, 0.0, 1.0))
            risk_prob = risk_ema

            if not eyes_closed and not head_slumped:
                risk_prob = min(risk_prob, NORMAL_POSE_CAP)
                if risk_ema > risk_prob:
                    risk_ema = risk_prob

        now = time.perf_counter()
        if risk_prob is not None and risk_prob >= HIGH_RISK_LEVEL:
            high_risk_start = high_risk_start or now
        else:
            high_risk_start = None

        sustained = (
            high_risk_start is not None
            and (now - high_risk_start) >= HIGH_RISK_SECONDS
        )

        if sustained:
            counter = FRAME_THRESHOLD
        elif eyes_closed and (head_slumped or (risk_prob is not None and risk_prob > LSTM_THRESHOLD)):
            counter += 1
        else:
            counter = max(0, counter - 5)

        r_disp = f"  Risk {risk_prob:.0%}" if risk_prob is not None else ""
        debug_info = f"EAR {avg_ear:.3f}  MAR {mar:.3f}  Pitch {pitch:.3f}{r_disp}"

    # ── FACE LOST ────────────────────────────────────────────────────────
    else:
        # If subject was showing warning signs and then vanished, assume collapse
        if face_was_visible and counter > WARNING_THRESHOLD and not face_loss_alarm:
            face_loss_alarm = True
            counter = FRAME_THRESHOLD

        if not face_loss_alarm:
            counter = max(0, counter - 1)

        if not face_was_visible:
            status, color = "No face detected", (180, 180, 180)
        elif face_loss_alarm:
            status, color = "CRITICAL - SYNCOPE DETECTED", (0, 0, 220)
        else:
            status, color = "Searching for face...", (180, 180, 180)

    # ── SEVERITY OVERRIDE ────────────────────────────────────────────────
    if counter >= FRAME_THRESHOLD:
        status, color = "CRITICAL - SYNCOPE DETECTED", (0, 0, 220)
    elif counter > WARNING_THRESHOLD:
        status, color = "WARNING - signs detected", (0, 140, 255)

    # ── DRAW ─────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 55), color, -1)
    cv2.putText(frame, status, (14, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    if risk_prob is not None:
        bar_w = int((w - 28) * min(risk_prob, 1.0))
        cv2.rectangle(frame, (14, 62), (w - 14, 76), (50, 50, 50), -1)
        cv2.rectangle(frame, (14, 62), (14 + bar_w, 76), color, -1)

    if debug_info:
        cv2.putText(frame, debug_info, (14, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    if high_risk_start is not None and risk_prob is not None:
        elapsed = time.perf_counter() - high_risk_start
        cv2.putText(frame, f"High risk: {elapsed:.1f}s / {HIGH_RISK_SECONDS:.1f}s",
                    (14, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    cv2.putText(frame, f"Counter: {counter}/{FRAME_THRESHOLD}",
                (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Syncope Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if globals().get("face_landmarker") is not None:
    try:
        face_landmarker.close()
    except Exception:
        pass
cv2.destroyAllWindows()