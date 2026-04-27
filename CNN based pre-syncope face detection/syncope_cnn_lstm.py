import sys
import pathlib
import re
import threading
import types
import time
import urllib.request
import os
import warnings

# Quiet noisy third-party startup logs (TensorFlow / MediaPipe / TFLite).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

QUIET_CONSOLE = True
SHOW_STARTUP_METRICS = False
STARTUP_TIME_PRINTED = False


def _log_info(message):
    if not QUIET_CONSOLE:
        print(message)


_native_stderr_saved_fd = None
_native_stderr_quiet = False
_startup_workers_remaining = 2
_startup_quiet_lock = threading.Lock()


def _begin_native_stderr_quiet():
    global _native_stderr_saved_fd, _native_stderr_quiet
    if _native_stderr_quiet:
        return
    sys.stderr.flush()
    _native_stderr_saved_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    _native_stderr_quiet = True


def _end_native_stderr_quiet():
    global _native_stderr_saved_fd, _native_stderr_quiet
    if not _native_stderr_quiet:
        return
    sys.stderr.flush()
    os.dup2(_native_stderr_saved_fd, 2)
    os.close(_native_stderr_saved_fd)
    _native_stderr_saved_fd = None
    _native_stderr_quiet = False


def _release_startup_quiet():
    global _startup_workers_remaining
    with _startup_quiet_lock:
        _startup_workers_remaining -= 1
        if _startup_workers_remaining <= 0:
            _end_native_stderr_quiet()


if QUIET_CONSOLE:
    _begin_native_stderr_quiet()

APP_START = time.perf_counter()
START_TS = {"app_start": APP_START}


def _mark_ts(name):
    START_TS[name] = time.perf_counter()


def _elapsed_from_start(name):
    if name not in START_TS:
        return None
    return START_TS[name] - APP_START


def _elapsed_between(start_key, end_key):
    if start_key not in START_TS or end_key not in START_TS:
        return None
    return START_TS[end_key] - START_TS[start_key]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUTO-PATCH MEDIAPIPE  (safe to re-run, skips if already patched)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _patch_mediapipe():
    venv    = pathlib.Path(sys.executable).parent.parent
    mp_root = venv / "Lib" / "site-packages" / "mediapipe"
    if not mp_root.exists():
        return

    sentinel = mp_root / ".copilot_patch_done"
    if sentinel.exists():
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
            _log_info("MediaPipe patch 1 applied (FieldDescriptor.label)")

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
                    _log_info(f"MediaPipe patch 2 applied ({f.name})")
        except Exception:
            pass

    try:
        sentinel.write_text("patched\n", encoding="utf-8")
        _log_info("MediaPipe patch sentinel written.")
    except Exception as e:
        print(f"MediaPipe patch warning: could not write sentinel ({e})")

_patch_mediapipe()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_mark_ts("before_heavy_imports")

import cv2
import numpy as np
from collections import deque

mp = None
load_model = None

_mark_ts("imports_ready")

try:
    MP_FACE_MESH = mp.tasks.vision.FaceLandmarker
    MP_FACE_LANDMARKER_OPTIONS = mp.tasks.vision.FaceLandmarkerOptions
    MP_RUNNING_MODE = mp.tasks.vision.RunningMode
    MP_BASE_OPTIONS = mp.tasks.BaseOptions
except Exception:
    MP_FACE_MESH = None
    MP_FACE_LANDMARKER_OPTIONS = None
    MP_RUNNING_MODE = None
    MP_BASE_OPTIONS = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_PATH       = "syncope_lstm_v1.h5"
FACE_MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_MODEL_PATH  = pathlib.Path(".cache") / "face_landmarker.task"
SEQ_LEN          = 20      # must match training
EAR_THRESHOLD    = 0.25    # blink detection cutoff
LSTM_THRESHOLD   = 0.3     # risk score cutoff (tuned for high recall)
FRAME_THRESHOLD  = 45      # frames of sustained warning → CRITICAL (~1.5s)
RISK_EMA_ALPHA    = 0.08    # lower = steadier, less jumpy risk score
RISK_MAX_STEP     = 0.06    # cap per-update movement so blinks do not spike the score
NORMAL_POSE_RISK_CAP = 0.35 # keep normal-looking faces from reading as near-critical
HIGH_RISK_LEVEL   = 0.99    # if sustained for this long, trigger fainting
HIGH_RISK_SECONDS = 3.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RUNTIME STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

model = None
model_ready = False
model_error = None
model_lock = threading.Lock()

mp_face_mesh = None
face_mesh = None
facemesh_ready = False
facemesh_error = None
facemesh_lock = threading.Lock()

frame_buffer = deque(maxlen=SEQ_LEN)
risk_score_ema = None
high_risk_start = None
face_seen_once = False
startup_no_face_prior_applied = False

startup_metrics_printed = False


def _print_startup_metrics_once():
    global startup_metrics_printed, STARTUP_TIME_PRINTED
    if not SHOW_STARTUP_METRICS:
        return
    if startup_metrics_printed:
        return
    if not (model_ready and facemesh_ready):
        return

    _mark_ts("detection_ready")
    startup_metrics_printed = True

    import_s = _elapsed_between("before_heavy_imports", "imports_ready")
    model_load_s = _elapsed_between("model_load_start", "model_load_done")
    facemesh_init_s = _elapsed_between("facemesh_init_start", "facemesh_init_done")
    camera_init_s = _elapsed_between("camera_init_start", "camera_init_done")
    first_frame_s = _elapsed_from_start("first_frame_shown")
    ready_s = _elapsed_from_start("detection_ready")

    def _fmt(v):
        return "n/a" if v is None else f"{v:.3f}s"

    print(
        "Startup metrics | "
        f"import_s={_fmt(import_s)} "
        f"model_load_s={_fmt(model_load_s)} "
        f"facemesh_init_s={_fmt(facemesh_init_s)} "
        f"camera_init_s={_fmt(camera_init_s)} "
        f"first_frame_s={_fmt(first_frame_s)} "
        f"ready_s={_fmt(ready_s)}"
    )


def _print_startup_time_once():
    global STARTUP_TIME_PRINTED
    if STARTUP_TIME_PRINTED:
        return
    first_frame_s = _elapsed_from_start("first_frame_shown")
    if first_frame_s is None:
        return

    STARTUP_TIME_PRINTED = True
    print(f"Launch time: {first_frame_s:.3f}s")


def _init_model_worker():
    global model, model_ready, model_error, load_model
    _mark_ts("model_load_start")
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass

        load_model = keras_load_model
        loaded_model = keras_load_model(MODEL_PATH)
        with model_lock:
            model = loaded_model
            model_ready = True
        _log_info("Model loaded successfully.")
    except Exception as e:
        model_error = str(e)
        _log_info(f"ERROR: Could not load {MODEL_PATH}\n{e}")
    finally:
        _mark_ts("model_load_done")
        if QUIET_CONSOLE:
            _release_startup_quiet()


def _init_facemesh_worker():
    global face_mesh, facemesh_ready, facemesh_error, mp, mp_face_mesh, MP_FACE_MESH, MP_FACE_LANDMARKER_OPTIONS, MP_RUNNING_MODE, MP_BASE_OPTIONS
    _mark_ts("facemesh_init_start")
    try:
        import mediapipe as mp_module
        mp = mp_module
        MP_FACE_MESH = mp_module.tasks.vision.FaceLandmarker
        MP_FACE_LANDMARKER_OPTIONS = mp_module.tasks.vision.FaceLandmarkerOptions
        MP_RUNNING_MODE = mp_module.tasks.vision.RunningMode
        MP_BASE_OPTIONS = mp_module.tasks.BaseOptions
        mp_face_mesh = MP_FACE_MESH

        if MP_FACE_MESH is None or MP_FACE_LANDMARKER_OPTIONS is None or MP_RUNNING_MODE is None or MP_BASE_OPTIONS is None:
            raise RuntimeError("MediaPipe FaceLandmarker API is unavailable in this environment.")

        FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not FACE_MODEL_PATH.exists():
            urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)

        options = MP_FACE_LANDMARKER_OPTIONS(
            base_options=MP_BASE_OPTIONS(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=MP_RUNNING_MODE.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        mesh = MP_FACE_MESH.create_from_options(options)
        with facemesh_lock:
            face_mesh = mesh
            facemesh_ready = True
        _log_info("FaceLandmarker initialized successfully.")
    except Exception as e:
        facemesh_error = str(e)
        _log_info(f"ERROR: FaceLandmarker initialization failed\n{e}")
    finally:
        _mark_ts("facemesh_init_done")
        if QUIET_CONSOLE:
            _release_startup_quiet()

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
    if (not model_ready) or len(frame_buffer) < SEQ_LEN:
        return None

    window = list(frame_buffer)
    ears   = [f[0] for f in window]
    blinks = sum(1 for j in range(1, len(ears))
                 if ears[j-1] >= EAR_THRESHOLD and ears[j] < EAR_THRESHOLD)
    seq = np.array([f + [blinks] for f in window], dtype=np.float32)[np.newaxis, ...]

    with model_lock:
        if model is None:
            return None
        return float(model.predict(seq, verbose=0)[0][0])


def update_risk_score(raw_prob):
    global risk_score_ema
    if raw_prob is None:
        return None

    if risk_score_ema is None:
        risk_score_ema = raw_prob
    else:
        target = (RISK_EMA_ALPHA * raw_prob) + ((1.0 - RISK_EMA_ALPHA) * risk_score_ema)
        delta = target - risk_score_ema
        if delta > RISK_MAX_STEP:
            delta = RISK_MAX_STEP
        elif delta < -RISK_MAX_STEP:
            delta = -RISK_MAX_STEP
        risk_score_ema += delta

    if risk_score_ema < 0.0:
        risk_score_ema = 0.0
    elif risk_score_ema > 1.0:
        risk_score_ema = 1.0

    return risk_score_ema


def _wrap_face_landmarker_result(result):
    if result is None or not getattr(result, "face_landmarks", None):
        return None
    face_landmarks = [types.SimpleNamespace(landmark=lm) for lm in result.face_landmarks]
    return types.SimpleNamespace(multi_face_landmarks=face_landmarks)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CAMERA LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

model_thread = threading.Thread(target=_init_model_worker, daemon=True)
facemesh_thread = threading.Thread(target=_init_facemesh_worker, daemon=True)
model_thread.start()
facemesh_thread.start()

_mark_ts("camera_init_start")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
_mark_ts("camera_init_done")

if not cap.isOpened():
    _log_info("ERROR: Could not open camera.")
    sys.exit(1)

counter              = 0
calibrated           = True
base_ear, base_pitch = 0.28, 1.0
raw_prob             = None
risk_prob            = None
first_frame_recorded = False

_log_info("System online. Monitoring starts immediately with default baseline values.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    results = None

    status, color = "Monitoring", (0, 200, 0)
    debug_info    = ""

    # Runtime readiness and error states.
    if facemesh_error or model_error:
        status, color = "ERROR - startup failed", (0, 0, 220)

    elif not facemesh_ready:
        status, color = "Initializing landmarks...", (255, 220, 0)

    else:
        with facemesh_lock:
            mesh = face_mesh
        if mesh is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.perf_counter() - APP_START) * 1000)
            results = _wrap_face_landmarker_result(mesh.detect_for_video(mp_image, timestamp_ms))

        if not model_ready:
            status, color = "Loading model...", (255, 220, 0)

    _print_startup_metrics_once()

    # ── A. FACE DETECTED ─────────────────────────────────────────────────────
    if results is not None and results.multi_face_landmarks:
        face_seen_once = True
        landmarks = results.multi_face_landmarks[0].landmark
        l_ear, r_ear, avg_ear, mar, pitch = get_features(landmarks)
        eyes_closed  = (l_ear < base_ear * 0.75) and (r_ear < base_ear * 0.75)
        head_slumped = (pitch > base_pitch * 1.15)

        frame_buffer.append([avg_ear, mar, pitch])

        if model_ready and not model_error:
            raw_prob = get_lstm_prob()
            risk_prob = update_risk_score(raw_prob)

            if risk_prob is None:
                status = f"Buffering {len(frame_buffer)}/{SEQ_LEN}"
                color  = (255, 220, 0)
            else:
                if not eyes_closed and not head_slumped:
                    risk_prob = min(risk_prob, NORMAL_POSE_RISK_CAP)
                    if risk_score_ema is not None and risk_score_ema > risk_prob:
                        risk_score_ema = risk_prob

                now = time.perf_counter()
                if risk_prob >= HIGH_RISK_LEVEL:
                    if high_risk_start is None:
                        high_risk_start = now
                else:
                    high_risk_start = None

                high_risk_elapsed = 0.0 if high_risk_start is None else (now - high_risk_start)
                sustained_high_risk = risk_prob >= HIGH_RISK_LEVEL and high_risk_elapsed >= HIGH_RISK_SECONDS

                if sustained_high_risk:
                    counter = FRAME_THRESHOLD
                elif eyes_closed and (head_slumped or risk_prob > LSTM_THRESHOLD):
                    counter += 1
                else:
                    counter = max(0, counter - 5)

                debug_info = f"EAR {avg_ear:.3f}  MAR {mar:.3f}  Pitch {pitch:.3f}  Risk {risk_prob:.0%}"
        else:
            debug_info = f"EAR {avg_ear:.3f}  MAR {mar:.3f}  Pitch {pitch:.3f}"

    # ── B. FACE LOST ──────────────────────────────────────────────────────────
    else:
        if calibrated:
            if not face_seen_once:
                if not startup_no_face_prior_applied:
                    risk_prob = 0.50
                    startup_no_face_prior_applied = True
                status, color = "No face detected - subject may have already fainted", (255, 220, 0)
            elif model_ready:
                high_risk_now = (risk_prob is not None and risk_prob >= HIGH_RISK_LEVEL) or (
                    risk_score_ema is not None and risk_score_ema >= HIGH_RISK_LEVEL
                )

                if high_risk_now:
                    counter = FRAME_THRESHOLD
                    status, color = "CRITICAL — SYNCOPE DETECTED", (0, 0, 220)
                else:
                    if counter > 5:
                        counter += 1
                        status, color = "TRACKING LOST - possible slump", (0, 140, 255)
                    else:
                        counter = max(0, counter - 1)
                        status, color = "Searching for face...", (180, 180, 180)
            else:
                status, color = "Loading model...", (255, 220, 0)

    # ── C. SEVERITY OVERRIDE ─────────────────────────────────────────────────
    if model_ready and (counter >= FRAME_THRESHOLD):
        status, color = "CRITICAL — SYNCOPE DETECTED", (0, 0, 220)
    elif model_ready and counter > 10 and "TRACKING" not in status:
        status, color = "WARNING — signs detected", (0, 140, 255)

    # ── D. DRAW ───────────────────────────────────────────────────────────────
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 55), color, -1)
    cv2.putText(frame, status, (14, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    if calibrated and risk_prob is not None:
        bar_w = int((w - 28) * min(risk_prob, 1.0))
        cv2.rectangle(frame, (14, 62), (w - 14, 76), (50, 50, 50), -1)
        cv2.rectangle(frame, (14, 62), (14 + bar_w, 76), color, -1)

    if debug_info:
        cv2.putText(frame, debug_info, (14, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    if risk_prob is not None and high_risk_start is not None:
        elapsed = time.perf_counter() - high_risk_start
        cv2.putText(frame, f"High risk: {elapsed:.1f}s / {HIGH_RISK_SECONDS:.1f}s",
                    (14, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    cv2.putText(frame, f"Counter: {counter}/{FRAME_THRESHOLD}",
                (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Syncope Detection", frame)

    if not first_frame_recorded:
        first_frame_recorded = True
        _mark_ts("first_frame_shown")
        _print_startup_time_once()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

with facemesh_lock:
    if face_mesh is not None:
        face_mesh.close()

cv2.destroyAllWindows()