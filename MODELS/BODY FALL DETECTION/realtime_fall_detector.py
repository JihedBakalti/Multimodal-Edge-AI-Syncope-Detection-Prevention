"""
realtime_fall_detector.py
=========================
Test en temps réel du modèle de détection de chute.

STRUCTURE DU DOSSIER:
    ton_dossier/
    ├── realtime_fall_detector.py   ← ce fichier
    ├── fall_lstm_traced.pt         ← depuis Kaggle outputs
    ├── norm_mean.npy               ← depuis Kaggle outputs
    └── norm_std.npy                ← depuis Kaggle outputs

INSTALLATION (terminal VSCode):
    pip install torch torchvision mediapipe opencv-python numpy

    Python 3.13+: Mediapipe fournit PoseLandmarker (Tasks), pas mp.solutions.
    Un fichier .task est téléchargé automatiquement dans ./mp_models/ au 1er lancement.

LANCER:
    python realtime_fall_detector.py
"""

import argparse
import cv2
import json
import math
import urllib.request
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from pathlib import Path
import time
import sys
import os

# Windows terminals often default to cp1252; this script prints Unicode (box chars, icons).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — ajuste ces valeurs si trop de false positives
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = _SCRIPT_DIR / "fall_lstm_traced.pt"
MEAN_PATH = _SCRIPT_DIR / "norm_mean.npy"
STD_PATH = _SCRIPT_DIR / "norm_std.npy"

WINDOW         = 30     # doit correspondre au notebook (WINDOW=30)
INPUT_DIM      = 99     # 33 joints × 3 coords

FALL_THRESHOLD = 0.85   # prob LSTM mini pour signal (↑ = moins de faux positifs)
N_CONFIRM      = 10      # frames consécutives avant alarme (↑ = moins de FP)
ALARM_COOLDOWN = 4.0    # secondes entre deux alarmes
FLOOR_Y_RATIO  = 0.72   # hanches au-dessus de 72% hauteur image = sol

# Beaucoup de faux positifs viennent de l’heuristique "au sol" (squat, chaise, caméra basse).
# False = seul le LSTM peut faire monter le compteur d’alarme (recommandé webcam PC portable).
# True  = comportement ancien : sol OU LSTM déclenche la confirmation.
FLOOR_TRIGGERS_ALARM = False

# Si FLOOR_TRIGGERS_ALARM=True : exiger un score géométrique plus haut (0–1) pour compter.
FLOOR_MIN_SCORE_FOR_ALARM = 0.78

# Distance normalisée (x,y) poing–genou au-dessous de ça ⇒ posture "penché, mains sur genoux"
# (cas typique de faux positif pour l’heuristique "au sol" + séquence confuse pour le LSTM).
SQUAT_WRIST_KNEE_MAX_DIST = 0.22

# Empêcher le LSTM seul de monter jusqu’à l’alarme quand cette posture détectée.
SUPPRESS_LSTM_WHEN_HANDS_NEAR_KNEES = True

# Au-dessus de cette prob, on n’applique plus le filtre « mains près des genoux »
# (évite de bloquer une vraie chute où les poignets se projettent près des genoux).
LSTM_SUPPRESS_BYPASS_PROB = 0.92

# MediaPipe Tasks (Python 3.13+ wheels): PoseLandmarker .task bundles (Pose model_complexity 0=lite 1=full 2=heavy).
POSE_LANDMARKER_URL_BY_COMPLEXITY = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def _coerce_xyz(lm_x, lm_y, lm_z):
    """Legacy Pose protobuf vs Tasks NormalizedLandmark (optional floats)."""
    x = float(0.0 if lm_x is None else lm_x)
    y = float(0.0 if lm_y is None else lm_y)
    z = float(0.0 if lm_z is None else lm_z)
    if math.isnan(x):
        x = 0.0
    if math.isnan(y):
        y = 0.0
    if math.isnan(z):
        z = 0.0
    return x, y, z


class _DrawingSpec:
    def __init__(self, *, color=(255, 255, 255), thickness=2, circle_radius=3):
        self.color = tuple(int(c) for c in color)
        self.thickness = int(thickness)
        self.circle_radius = int(circle_radius)


class _SolutionsPoseModel:
    def __init__(self, pose_module, pose):
        self._pose_module = pose_module
        self._pose = pose

    def process(self, rgb_u8_hwc, **_kwargs):
        return self._pose.process(rgb_u8_hwc)

    def draw_landmarks(self, frame_bgr, landmarks_ns, *,
                       landmark_drawing_spec, connection_drawing_spec):
        from mediapipe import solutions as mp_solutions

        if isinstance(landmark_drawing_spec, _DrawingSpec):
            landmark_drawing_spec = mp_solutions.drawing_utils.DrawingSpec(
                color=tuple(int(c) for c in landmark_drawing_spec.color),
                thickness=int(landmark_drawing_spec.thickness),
                circle_radius=int(landmark_drawing_spec.circle_radius),
            )

        if isinstance(connection_drawing_spec, _DrawingSpec):
            connection_drawing_spec = mp_solutions.drawing_utils.DrawingSpec(
                color=tuple(int(c) for c in connection_drawing_spec.color),
                thickness=int(connection_drawing_spec.thickness),
                circle_radius=int(connection_drawing_spec.circle_radius),
            )

        mp_solutions.drawing_utils.draw_landmarks(
            frame_bgr,
            landmarks_ns,
            self._pose_module.POSE_CONNECTIONS,
            landmark_drawing_spec,
            connection_drawing_spec,
        )

    def close(self):
        self._pose.close()


class _NormalizedLandmarksAdapter:
    def __init__(self, landmarks):
        self.landmark = landmarks


class _NormalizedLandmarksResult:
    def __init__(self, pose_landmarks_ns):
        self.pose_landmarks = pose_landmarks_ns


class _TasksPoseModel:
    def __init__(self, pose_landmarker):
        import importlib

        mp_image_lib = importlib.import_module("mediapipe.tasks.python.vision.core.image")

        self._landmarker = pose_landmarker
        self._Image = mp_image_lib.Image
        self._ImageFormat = mp_image_lib.ImageFormat

    def process(self, rgb_u8_hwc, *, timestamp_ms: int):
        rgb = rgb_u8_hwc
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)

        mp_image = self._Image(self._ImageFormat.SRGB, rgb)
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        if not result.pose_landmarks:
            return _NormalizedLandmarksResult(None)

        lms = result.pose_landmarks[0]
        if len(lms) != 33:
            return _NormalizedLandmarksResult(None)

        return _NormalizedLandmarksResult(_NormalizedLandmarksAdapter(lms))

    def draw_landmarks(self, frame_bgr, landmarks_ns, *,
                       landmark_drawing_spec, connection_drawing_spec):
        import importlib

        plm_mod = importlib.import_module("mediapipe.tasks.python.vision.pose_landmarker")

        h, w = frame_bgr.shape[:2]

        def _pt_xy(lm):
            return int(round(float(lm.x) * w)), int(round(float(lm.y) * h))

        if landmarks_ns is None:
            return

        lms = landmarks_ns.landmark
        pts = [_pt_xy(lm) for lm in lms]

        for conn in plm_mod.PoseLandmarksConnections.POSE_LANDMARKS:
            cv2.line(
                frame_bgr,
                pts[int(conn.start)],
                pts[int(conn.end)],
                tuple(int(c) for c in connection_drawing_spec.color),
                thickness=int(connection_drawing_spec.thickness),
            )

        for idx, _lm in enumerate(lms):
            x, y = pts[idx]
            cv2.circle(
                frame_bgr,
                (x, y),
                radius=int(landmark_drawing_spec.circle_radius),
                color=tuple(int(c) for c in landmark_drawing_spec.color),
                thickness=-1,
            )

    def close(self):
        self._landmarker.close()


def _ensure_pose_landmarker_model_file(model_complexity: int, dest_dir: str) -> str:
    url = POSE_LANDMARKER_URL_BY_COMPLEXITY[int(model_complexity)]
    fname = url.rsplit("/", 1)[-1]

    explicit = os.environ.get("MP_POSE_TASK_MODEL_PATH", "").strip()
    if explicit and os.path.exists(explicit):
        return explicit

    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, fname)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    tmp_path = dest_path + ".partial"
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except OSError:
        pass

    print(f"[mediapipe] Downloading PoseLandmarker model to: {dest_path}")
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, dest_path)
    return dest_path


def create_pose_backend(*, model_complexity: int,
                      min_detection_confidence: float,
                      min_tracking_confidence: float):
    """
    Use legacy mp.solutions.pose when available (e.g. Python 3.11 + mediapipe 0.10.14).
    Otherwise use PoseLandmarker (Tasks), required on Python 3.13 Mediapipe wheels.
    """
    if hasattr(mp, "solutions"):
        mp_pose_module = mp.solutions.pose

        pose = mp_pose_module.Pose(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

        print("   Pose backend: mp.solutions.pose (classic)")
        return _SolutionsPoseModel(mp_pose_module, pose)

    import importlib

    try:
        mp_tasks_python = importlib.import_module("mediapipe.tasks.python")
        plm_pkg = importlib.import_module("mediapipe.tasks.python.vision.pose_landmarker")
        VisionTaskRunningMode = importlib.import_module(
            "mediapipe.tasks.python.vision.core.vision_task_running_mode"
        ).VisionTaskRunningMode
        PoseLandmarker = plm_pkg.PoseLandmarker
        PoseLandmarkerOptions = plm_pkg.PoseLandmarkerOptions
        BaseOptions = mp_tasks_python.BaseOptions
    except Exception as exc:
        raise RuntimeError(
            "Mediapipe has no mp.solutions and Tasks PoseLandmarker could not be loaded. "
            "Use Python 3.11 + mediapipe with solutions, or install a Tasks-capable Mediapipe for this Python."
        ) from exc

    model_path = _ensure_pose_landmarker_model_file(
        int(model_complexity),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_models"),
    )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=float(min_detection_confidence),
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=float(min_tracking_confidence),
        output_segmentation_masks=False,
    )

    landmarker = PoseLandmarker.create_from_options(options)
    print("   Pose backend: PoseLandmarker (Tasks, Python 3.13+)")
    return _TasksPoseModel(landmarker)


# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    for f in [MODEL_PATH, MEAN_PATH, STD_PATH]:
        if not f.is_file():
            msg = (
                f"\n❌  Fichier manquant: '{f}'\n"
                "    → Place ce fichier dans le même dossier que realtime_fall_detector.py\n"
                "    → Télécharge-le depuis Kaggle: Output > fall_lstm_traced.pt / norm_*.npy"
            )
            print(msg)
            raise FileNotFoundError(str(f))

    model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
    model.eval()
    norm_mean = np.load(str(MEAN_PATH))   # (99,)
    norm_std = np.load(str(STD_PATH))  # (99,)

    # Warm-up
    dummy = torch.zeros(1, WINDOW, INPUT_DIM)
    with torch.no_grad():
        _ = model(dummy)

    print(f"✅  Modèle chargé  — {MODEL_PATH.name}")
    print(f"✅  Normalisation  — mean: [{norm_mean.min():.3f}, {norm_mean.max():.3f}]")
    return model, norm_mean, norm_std


# ══════════════════════════════════════════════════════════════════════════════
#  POSE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_keypoints(frame_bgr, pose_model, *, timestamp_ms: int):
    """Retourne (vecteur 99-dim, landmarks ou None)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if isinstance(pose_model, _TasksPoseModel):
        result = pose_model.process(rgb, timestamp_ms=int(timestamp_ms))
    else:
        result = pose_model.process(rgb)

    if result.pose_landmarks:
        outs = []
        for lm in result.pose_landmarks.landmark:
            x, y, z = _coerce_xyz(lm.x, lm.y, lm.z)
            outs.extend((x, y, z))

        kp = np.asarray(outs, dtype=np.float32)
        return kp, result.pose_landmarks
    return np.zeros(INPUT_DIM, dtype=np.float32), None


# ══════════════════════════════════════════════════════════════════════════════
#  DÉTECTION STATIQUE : PERSONNE DÉJÀ AU SOL
# ══════════════════════════════════════════════════════════════════════════════


def pose_looks_like_bent_over_hands_near_knees(landmarks) -> bool:
    """Standing / squat forward bend avec mains près des genoux (false positive au sol/LSTM)."""
    if landmarks is None:
        return False
    lm = landmarks.landmark
    lw, rw, lk, rk = 15, 16, 25, 26
    try:
        lx0, ly0, _ = _coerce_xyz(lm[lw].x, lm[lw].y, lm[lw].z)
        lx1, ly1, _ = _coerce_xyz(lm[lk].x, lm[lk].y, lm[lk].z)
        rx0, ry0, _ = _coerce_xyz(lm[rw].x, lm[rw].y, lm[rw].z)
        rx1, ry1, _ = _coerce_xyz(lm[rk].x, lm[rk].y, lm[rk].z)
        d_l = math.hypot(lx0 - lx1, ly0 - ly1)
        d_r = math.hypot(rx0 - rx1, ry0 - ry1)
    except (IndexError, AttributeError):
        return False
    return min(d_l, d_r) <= SQUAT_WRIST_KNEE_MAX_DIST


def is_on_floor(landmarks):
    """
    Heuristique géométrique pure — ne dépend pas du LSTM.
    Vérifie si la personne est allongée / effondrée.

    Retourne: (bool, score 0-1)
    """
    if landmarks is None:
        return False, 0.0

    if pose_looks_like_bent_over_hands_near_knees(landmarks):
        return False, 0.0

    lm = landmarks.landmark

    # Indices MediaPipe Pose
    NOSE        = 0
    L_SHOULDER  = 11;  R_SHOULDER  = 12
    L_HIP       = 23;  R_HIP       = 24
    L_ANKLE     = 27;  R_ANKLE     = 28
    L_KNEE      = 25;  R_KNEE      = 26

    # Coordonnées Y normalisées (0=haut, 1=bas)
    _, hy_l, _ = _coerce_xyz(lm[L_HIP].x, lm[L_HIP].y, lm[L_HIP].z)
    _, hy_r, _ = _coerce_xyz(lm[R_HIP].x, lm[R_HIP].y, lm[R_HIP].z)
    hip_y = (hy_l + hy_r) / 2
    _, ay_l, _ = _coerce_xyz(lm[L_ANKLE].x, lm[L_ANKLE].y, lm[L_ANKLE].z)
    _, ay_r, _ = _coerce_xyz(lm[R_ANKLE].x, lm[R_ANKLE].y, lm[R_ANKLE].z)
    ankle_y = (ay_l + ay_r) / 2
    _, sy_l, _ = _coerce_xyz(lm[L_SHOULDER].x, lm[L_SHOULDER].y, lm[L_SHOULDER].z)
    _, sy_r, _ = _coerce_xyz(lm[R_SHOULDER].x, lm[R_SHOULDER].y, lm[R_SHOULDER].z)
    shldr_y = (sy_l + sy_r) / 2
    _, ky_l, _ = _coerce_xyz(lm[L_KNEE].x, lm[L_KNEE].y, lm[L_KNEE].z)
    _, ky_r, _ = _coerce_xyz(lm[R_KNEE].x, lm[R_KNEE].y, lm[R_KNEE].z)
    knee_y = (ky_l + ky_r) / 2
    _, nose_y, _ = _coerce_xyz(lm[NOSE].x, lm[NOSE].y, lm[NOSE].z)

    # Critère 1 — hanches dans la partie basse de l'image
    hips_low  = hip_y > FLOOR_Y_RATIO

    # Critère 2 — corps horizontal (distance tête-chevilles compressée)
    body_span = abs(ankle_y - nose_y)
    body_flat = body_span < 0.42

    # Critère 3 — épaules et hanches au même niveau vertical
    vertical_gap = abs(shldr_y - hip_y)
    body_level   = vertical_gap < 0.18

    # Critère 4 — genoux et hanches proches du sol ensemble
    knees_low = knee_y > FLOOR_Y_RATIO - 0.05

    # Score de confiance (0-1)
    score = sum([
        float(hips_low)  * 0.35,
        float(body_flat) * 0.35,
        float(body_level)* 0.20,
        float(knees_low) * 0.10,
    ])

    # On déclare "au sol" si deux critères forts sont réunis
    on_floor = (hips_low and body_flat) or \
               (body_flat and body_level) or \
               (hips_low and body_level and knees_low)

    return on_floor, score


# ══════════════════════════════════════════════════════════════════════════════
#  INFÉRENCE LSTM
# ══════════════════════════════════════════════════════════════════════════════

def predict(model, frame_buffer, norm_mean, norm_std):
    """Normalise + inférence sur les 30 dernières frames."""
    seq      = np.array(list(frame_buffer), dtype=np.float32)   # (30, 99)
    seq_norm = (seq - norm_mean) / norm_std
    tensor   = torch.FloatTensor(seq_norm).unsqueeze(0)          # (1, 30, 99)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).numpy()[0]

    return float(probs[1])   # probabilité de chute


# ══════════════════════════════════════════════════════════════════════════════
#  DESSIN INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

FONT = cv2.FONT_HERSHEY_DUPLEX
C_GREEN  = (50, 210, 50)
C_ORANGE = (0, 165, 255)
C_RED    = (30, 30, 230)
C_WHITE  = (255, 255, 255)
C_DARK   = (18, 18, 18)
C_GRAY   = (160, 160, 160)


def draw_ui(frame, prob_fall, on_floor, floor_score,
            confirm_count, alarm_active, fps, buffer_filled, *,
            lstm_ok=False, lstm_hit=False):
    h, w = frame.shape[:2]

    # ── Barre de statut top ────────────────────────────────────────────────
    if alarm_active:
        bar_col  = C_RED
        status   = " Fall detected !"
    elif on_floor:
        bar_col  = C_ORANGE
        status   = " Person is on the floor"
    elif confirm_count > 0:
        bar_col  = C_ORANGE
        status   = f"  ATTENTION... ({confirm_count}/{N_CONFIRM})"
    elif lstm_ok and not lstm_hit:
        bar_col  = C_ORANGE
        status   = " LSTM high — pose filter (hands/knees)"
    elif buffer_filled and prob_fall > 0.45 and prob_fall <= FALL_THRESHOLD:
        bar_col  = C_ORANGE
        status   = f" LSTM rising {prob_fall:.2f} / {FALL_THRESHOLD}"
    else:
        bar_col  = C_GREEN
        status   = "  NORMAL"

    cv2.rectangle(frame, (0, 0), (w, 64), bar_col, -1)
    cv2.putText(frame, status, (10, 46), FONT, 1.3 if len(status) < 42 else 0.72, C_WHITE, 2)
    cv2.putText(frame, f"{fps:.0f} fps", (w - 100, 44), FONT, 0.8, C_WHITE, 1)

    if not buffer_filled:
        cv2.putText(frame, f"Chargement buffer... ({confirm_count}/{WINDOW})",
                    (10, 46), FONT, 0.85, C_WHITE, 2)

    # ── Panel bas ─────────────────────────────────────────────────────────
    panel_top = h - 120
    cv2.rectangle(frame, (0, panel_top), (w, h), C_DARK, -1)

    # Barre probabilité LSTM
    bar_area_w = w - 40
    fill_w     = int(bar_area_w * min(prob_fall, 1.0))
    bar_color  = C_RED if prob_fall > FALL_THRESHOLD else \
                 C_ORANGE if prob_fall > 0.45 else C_GREEN

    # Fond barre
    cv2.rectangle(frame, (20, panel_top + 10), (20 + bar_area_w, panel_top + 34),
                  (55, 55, 55), -1)
    # Remplissage
    if fill_w > 0:
        cv2.rectangle(frame, (20, panel_top + 10), (20 + fill_w, panel_top + 34),
                      bar_color, -1)
    # Contour
    cv2.rectangle(frame, (20, panel_top + 10), (20 + bar_area_w, panel_top + 34),
                  C_GRAY, 1)

    # Marqueur seuil
    thresh_x = 20 + int(bar_area_w * FALL_THRESHOLD)
    cv2.line(frame, (thresh_x, panel_top + 6), (thresh_x, panel_top + 38), C_RED, 2)
    cv2.putText(frame, f"seuil {FALL_THRESHOLD}",
                (thresh_x - 52, panel_top + 8), FONT, 0.38, C_RED, 1)

    # Label prob
    cv2.putText(frame, f"LSTM  {prob_fall:.2f}",
                (25, panel_top + 29), FONT, 0.6, C_WHITE, 1)

    # ── Ligne 2: métriques ────────────────────────────────────────────────
    y2 = panel_top + 62

    # Confirmation
    conf_col = C_RED if confirm_count >= N_CONFIRM else \
               C_ORANGE if confirm_count > 0 else C_GREEN
    cv2.putText(frame, f"Confirmation: {confirm_count}/{N_CONFIRM}",
                (20, y2), FONT, 0.65, conf_col, 1)

    # Sol
    floor_col = C_ORANGE if on_floor else C_GREEN
    floor_txt = f"Sol: {'OUI  {:.2f}'.format(floor_score) if on_floor else 'non'}"
    cv2.putText(frame, floor_txt, (w // 2, y2), FONT, 0.65, floor_col, 1)

    # ── Touches ───────────────────────────────────────────────────────────
    cv2.putText(frame, "[Q] Quitter    [R] Reset    [S] Screenshot",
                (20, h - 10), FONT, 0.42, C_GRAY, 1)

    # ── Flash rouge alarme ────────────────────────────────────────────────
    if alarm_active:
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(flash, 0.20, frame, 0.80, 0, frame)

    return frame


def _open_video_capture(camera_index: int):
    """Prefer DirectShow / MSMF on Windows (same pattern as other MODEL scripts)."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


def run_orchestrator_session(
    camera_index: int,
    duration_seconds: int,
    confirm_seconds: float,
    output_path: Path,
    *,
    no_gui: bool,
) -> dict:
    """
    Used by ``medical_assistant`` escalation: monitor up to ``duration_seconds``,
    require ``confirm_seconds`` of *continuous* fall signal before ``critical_detected``.
    Writes JSON summary to ``output_path``.
    """
    model, norm_mean, norm_std = load_model()
    pose_model = create_pose_backend(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_buffer = deque(maxlen=WINDOW)
    confirm_count = 0
    last_alarm_t = 0.0
    pose_t0_wall = None
    last_pose_ts_ms = None

    cap = _open_video_capture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    t_run = time.time()
    deadline = t_run + max(1, int(duration_seconds))
    sustained_start = None
    critical_detected = False
    max_fall_prob_seen = 0.0

    alarm_active = False
    prev_clock = time.time()
    fps_smooth = 0.0

    try:
        while True:
            now = time.time()
            if now >= deadline and not critical_detected:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if pose_t0_wall is None:
                pose_t0_wall = now
            elapsed_ms = int(max(0.0, (now - pose_t0_wall) * 1000.0))
            if last_pose_ts_ms is None:
                pose_ts_ms = elapsed_ms
            else:
                pose_ts_ms = max(last_pose_ts_ms + 1, elapsed_ms)
            last_pose_ts_ms = pose_ts_ms

            kp, landmarks = extract_keypoints(frame, pose_model, timestamp_ms=pose_ts_ms)
            frame_buffer.append(kp)

            if landmarks and not no_gui:
                pose_model.draw_landmarks(
                    frame,
                    landmarks,
                    landmark_drawing_spec=_DrawingSpec(
                        color=(0, 210, 255), thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=_DrawingSpec(
                        color=(0, 150, 200), thickness=2, circle_radius=2
                    ),
                )

            on_floor, floor_score = is_on_floor(landmarks)

            prob_fall = 0.0
            buffer_filled = len(frame_buffer) == WINDOW
            if buffer_filled:
                prob_fall = predict(model, frame_buffer, norm_mean, norm_std)
                max_fall_prob_seen = max(max_fall_prob_seen, prob_fall)

            bent_hands_knees = pose_looks_like_bent_over_hands_near_knees(landmarks)
            lstm_ok = buffer_filled and (prob_fall > FALL_THRESHOLD)
            suppress_lstm = (
                SUPPRESS_LSTM_WHEN_HANDS_NEAR_KNEES
                and bent_hands_knees
                and prob_fall < LSTM_SUPPRESS_BYPASS_PROB
            )
            lstm_hit = lstm_ok and (not suppress_lstm)
            floor_hit = (
                FLOOR_TRIGGERS_ALARM
                and on_floor
                and (floor_score >= FLOOR_MIN_SCORE_FOR_ALARM)
            )
            fall_signal = lstm_hit or floor_hit

            if fall_signal:
                confirm_count = min(confirm_count + 1, N_CONFIRM + 5)
            else:
                confirm_count = max(0, confirm_count - 1)

            if fall_signal:
                if sustained_start is None:
                    sustained_start = now
                elif now - sustained_start >= float(confirm_seconds):
                    critical_detected = True
                    break
            else:
                sustained_start = None

            now_alarm = (confirm_count >= N_CONFIRM) and (now - last_alarm_t > ALARM_COOLDOWN)
            if now_alarm:
                alarm_active = True
                last_alarm_t = now
            elif alarm_active and (now - last_alarm_t > 2.5):
                alarm_active = False

            if not no_gui:
                now_c = time.time()
                fps_smooth = 0.9 * fps_smooth + 0.1 / max(now_c - prev_clock, 1e-6)
                prev_clock = now_c
                frame = draw_ui(
                    frame,
                    prob_fall,
                    on_floor,
                    floor_score,
                    confirm_count if buffer_filled else len(frame_buffer),
                    alarm_active,
                    fps_smooth,
                    buffer_filled,
                    lstm_ok=lstm_ok,
                    lstm_hit=lstm_hit,
                )
                cv2.imshow("Fall Detector  —  [Q] Quit", frame)

            delay = cv2.waitKey(1) & 0xFF
            if not no_gui and delay in (ord("q"), 27):
                break
    finally:
        cap.release()
        if not no_gui:
            cv2.destroyAllWindows()
        pose_model.close()

    payload = {
        "critical_detected": critical_detected,
        "camera_indices_opened": [camera_index],
        "cameras_critical": [camera_index] if critical_detected else [],
        "max_risk_score": round(float(max_fall_prob_seen), 4),
        "max_fall_prob": round(float(max_fall_prob_seen), 4),
        "duration_seconds": int(duration_seconds),
        "camera_index": int(camera_index),
        "confirm_seconds": float(confirm_seconds),
        "detection_path": "body_fall",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return payload


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_interactive_demo():
    print("\n" + "═" * 58)
    print("   🛡️   FALL DETECTION — Temps Réel (VSCode)")
    print("═" * 58)

    try:
        model, norm_mean, norm_std = load_model()
    except FileNotFoundError:
        sys.exit(1)

    pose_model = create_pose_backend(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_buffer  = deque(maxlen=WINDOW)
    confirm_count = 0
    alarm_active  = False
    last_alarm_t  = 0.0
    prev_t        = time.time()
    fps           = 0.0
    shot_idx      = 0
    pose_t0_wall  = None
    last_pose_ts_ms = None

    # Ouvre la webcam (essaie 0, puis 1 si échec)
    cap = None
    for cam_idx in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            print(f"✅  Webcam ouverte  — index {cam_idx}")
            break
        cap.release()
    else:
        print("❌  Impossible d'ouvrir la webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(
        f"   WINDOW={WINDOW} | LSTM_SEUIL={FALL_THRESHOLD} | CONFIRM={N_CONFIRM} | "
        f"SOL_DECLENCHE_ALARME={'oui' if FLOOR_TRIGGERS_ALARM else 'non'}"
    )
    print("   [Q] Quitter  [R] Reset  [S] Screenshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[warn] Frame vide.")
            break

        frame = cv2.flip(frame, 1)   # effet miroir
        now   = time.time()
        fps   = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
        prev_t = now

        if pose_t0_wall is None:
            pose_t0_wall = now
        elapsed_ms = int(max(0.0, (now - pose_t0_wall) * 1000.0))
        if last_pose_ts_ms is None:
            pose_ts_ms = elapsed_ms
        else:
            pose_ts_ms = max(last_pose_ts_ms + 1, elapsed_ms)
        last_pose_ts_ms = pose_ts_ms

        # ── Pose ──────────────────────────────────────────────────────────
        kp, landmarks = extract_keypoints(frame, pose_model, timestamp_ms=pose_ts_ms)
        frame_buffer.append(kp)

        # Squelette
        if landmarks:
            pose_model.draw_landmarks(
                frame,
                landmarks,
                landmark_drawing_spec=_DrawingSpec(
                    color=(0, 210, 255), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=_DrawingSpec(
                    color=(0, 150, 200), thickness=2, circle_radius=2
                ),
            )

        # ── Détection statique sol ────────────────────────────────────────
        on_floor, floor_score = is_on_floor(landmarks)

        # ── Inférence LSTM ────────────────────────────────────────────────
        prob_fall     = 0.0
        buffer_filled = len(frame_buffer) == WINDOW

        if buffer_filled:
            prob_fall = predict(model, frame_buffer, norm_mean, norm_std)

        # ── Logique confirmation anti-false-positive ───────────────────────
        bent_hands_knees = pose_looks_like_bent_over_hands_near_knees(landmarks)
        lstm_ok = buffer_filled and (prob_fall > FALL_THRESHOLD)
        suppress_lstm = (
            SUPPRESS_LSTM_WHEN_HANDS_NEAR_KNEES
            and bent_hands_knees
            and prob_fall < LSTM_SUPPRESS_BYPASS_PROB
        )
        lstm_hit = lstm_ok and (not suppress_lstm)
        floor_hit = (
            FLOOR_TRIGGERS_ALARM
            and on_floor
            and (floor_score >= FLOOR_MIN_SCORE_FOR_ALARM)
        )
        fall_signal = lstm_hit or floor_hit

        if fall_signal:
            confirm_count = min(confirm_count + 1, N_CONFIRM + 5)
        else:
            # Décroissance progressive pour éviter reset brutal
            confirm_count = max(0, confirm_count - 1)

        # Déclenchement alarme
        now_alarm = (confirm_count >= N_CONFIRM) and \
                    (now - last_alarm_t > ALARM_COOLDOWN)

        if now_alarm:
            alarm_active = True
            last_alarm_t = now
            causes = []
            if lstm_hit:
                causes.append(f"LSTM={prob_fall:.2f}")
            if floor_hit:
                causes.append(f"sol(score={floor_score:.2f})")
            print(f"[{time.strftime('%H:%M:%S')}] 🚨 ALARME — {' + '.join(causes)}")
        elif alarm_active and (now - last_alarm_t > 2.5):
            alarm_active = False

        # ── Interface ─────────────────────────────────────────────────────
        frame = draw_ui(
            frame, prob_fall, on_floor, floor_score,
            confirm_count if buffer_filled else len(frame_buffer),
            alarm_active, fps, buffer_filled,
            lstm_ok=lstm_ok,
            lstm_hit=lstm_hit,
        )

        cv2.imshow("Fall Detector  —  [Q] Quit", frame)

        # ── Touches ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("\nArrêt.")
            break
        elif key == ord('r'):
            frame_buffer.clear()
            confirm_count = 0
            alarm_active  = False
            pose_t0_wall = time.time()
            last_pose_ts_ms = None
            print("[R] Reset effectué")
        elif key == ord('s'):
            fname = f"fall_screenshot_{shot_idx:04d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[S] Screenshot → {fname}")
            shot_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    pose_model.close()
    print("✅  Programme terminé.")


def main():
    parser = argparse.ArgumentParser(description="Body fall detection (_pose + LSTM).")
    parser.add_argument("--camera-index", type=int, default=None, help="Webcam index (orchestrator mode).")
    parser.add_argument("--duration", type=int, default=30, help="Monitoring window in seconds.")
    parser.add_argument(
        "--confirm-seconds",
        type=float,
        default=2.0,
        help="Continuous fall signal required before critical (orchestrator mode).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write JSON summary to this path (orchestrator mode). GUI on unless --no-gui.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="No OpenCV windows (paired with --output).",
    )
    args = parser.parse_args()

    if args.output:
        if args.camera_index is None:
            print("Orchestrator mode requires --camera-index.", file=sys.stderr)
            sys.exit(2)
        out = Path(args.output)
        try:
            run_orchestrator_session(
                camera_index=int(args.camera_index),
                duration_seconds=int(args.duration),
                confirm_seconds=float(args.confirm_seconds),
                output_path=out,
                no_gui=bool(args.no_gui),
            )
        except Exception as e:
            err = {"ok": False, "critical_detected": False, "message": str(e)}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2), encoding="utf-8")
            print(json.dumps(err, ensure_ascii=False))
            sys.exit(1)
        sys.exit(0)

    run_interactive_demo()


if __name__ == "__main__":
    main()
