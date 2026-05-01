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

LANCER:
    python realtime_fall_detector.py
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
import time
import sys
import os

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — ajuste ces valeurs si trop de false positives
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "fall_lstm_traced.pt"
MEAN_PATH  = "norm_mean.npy"
STD_PATH   = "norm_std.npy"

WINDOW         = 30     # doit correspondre au notebook (WINDOW=30)
INPUT_DIM      = 99     # 33 joints × 3 coords

FALL_THRESHOLD = 0.80   # prob minimum pour considérer une chute (↑ = moins de FP)
N_CONFIRM      = 6      # frames consécutives avant de déclencher l'alarme (↑ = moins de FP)
ALARM_COOLDOWN = 4.0    # secondes entre deux alarmes
FLOOR_Y_RATIO  = 0.72   # hanches au-dessus de 72% hauteur image = sol

# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    for f in [MODEL_PATH, MEAN_PATH, STD_PATH]:
        if not os.path.exists(f):
            print(f"\n❌  Fichier manquant: '{f}'")
            print("    → Place ce fichier dans le même dossier que realtime_fall_detector.py")
            print("    → Télécharge-le depuis Kaggle: Output > fall_lstm_traced.pt / norm_*.npy")
            sys.exit(1)

    model     = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    norm_mean = np.load(MEAN_PATH)   # (99,)
    norm_std  = np.load(STD_PATH)    # (99,)

    # Warm-up
    dummy = torch.zeros(1, WINDOW, INPUT_DIM)
    with torch.no_grad():
        _ = model(dummy)

    print(f"✅  Modèle chargé  — {MODEL_PATH}")
    print(f"✅  Normalisation  — mean: [{norm_mean.min():.3f}, {norm_mean.max():.3f}]")
    return model, norm_mean, norm_std


# ══════════════════════════════════════════════════════════════════════════════
#  POSE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_keypoints(frame_bgr, pose_model):
    """Retourne (vecteur 99-dim, landmarks ou None)."""
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb)
    if result.pose_landmarks:
        kp = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
            dtype=np.float32
        ).flatten()
        return kp, result.pose_landmarks
    return np.zeros(INPUT_DIM, dtype=np.float32), None


# ══════════════════════════════════════════════════════════════════════════════
#  DÉTECTION STATIQUE : PERSONNE DÉJÀ AU SOL
# ══════════════════════════════════════════════════════════════════════════════

def is_on_floor(landmarks):
    """
    Heuristique géométrique pure — ne dépend pas du LSTM.
    Vérifie si la personne est allongée / effondrée.

    Retourne: (bool, score 0-1)
    """
    if landmarks is None:
        return False, 0.0

    lm = landmarks.landmark

    # Indices MediaPipe Pose
    NOSE        = 0
    L_SHOULDER  = 11;  R_SHOULDER  = 12
    L_HIP       = 23;  R_HIP       = 24
    L_ANKLE     = 27;  R_ANKLE     = 28
    L_KNEE      = 25;  R_KNEE      = 26

    # Coordonnées Y normalisées (0=haut, 1=bas)
    hip_y    = (lm[L_HIP].y    + lm[R_HIP].y)    / 2
    ankle_y  = (lm[L_ANKLE].y  + lm[R_ANKLE].y)  / 2
    shldr_y  = (lm[L_SHOULDER].y + lm[R_SHOULDER].y) / 2
    knee_y   = (lm[L_KNEE].y   + lm[R_KNEE].y)   / 2
    nose_y   = lm[NOSE].y

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
            confirm_count, alarm_active, fps, buffer_filled):
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
    else:
        bar_col  = C_GREEN
        status   = "  NORMAL"

    cv2.rectangle(frame, (0, 0), (w, 64), bar_col, -1)
    cv2.putText(frame, status, (10, 46), FONT, 1.3, C_WHITE, 2)
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


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 58)
    print("   🛡️   FALL DETECTION — Temps Réel (VSCode)")
    print("═" * 58)

    model, norm_mean, norm_std = load_model()

    mp_pose_module = mp.solutions.pose
    mp_draw        = mp.solutions.drawing_utils
    pose_model     = mp_pose_module.Pose(
        static_image_mode         = False,
        model_complexity          = 1,
        min_detection_confidence  = 0.5,
        min_tracking_confidence   = 0.5,
    )

    frame_buffer  = deque(maxlen=WINDOW)
    confirm_count = 0
    alarm_active  = False
    last_alarm_t  = 0.0
    prev_t        = time.time()
    fps           = 0.0
    shot_idx      = 0

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

    print(f"   WINDOW={WINDOW} | THRESHOLD={FALL_THRESHOLD} | CONFIRM={N_CONFIRM}")
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

        # ── Pose ──────────────────────────────────────────────────────────
        kp, landmarks = extract_keypoints(frame, pose_model)
        frame_buffer.append(kp)

        # Squelette
        if landmarks:
            mp_draw.draw_landmarks(
                frame, landmarks,
                mp_pose_module.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 210, 255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 150, 200), thickness=2),
            )

        # ── Détection statique sol ────────────────────────────────────────
        on_floor, floor_score = is_on_floor(landmarks)

        # ── Inférence LSTM ────────────────────────────────────────────────
        prob_fall     = 0.0
        buffer_filled = len(frame_buffer) == WINDOW

        if buffer_filled:
            prob_fall = predict(model, frame_buffer, norm_mean, norm_std)

        # ── Logique confirmation anti-false-positive ───────────────────────
        # Signal = LSTM détecte chute  OU  heuristique sol
        fall_signal = (prob_fall > FALL_THRESHOLD) or on_floor

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
            if prob_fall > FALL_THRESHOLD:
                causes.append(f"LSTM={prob_fall:.2f}")
            if on_floor:
                causes.append(f"sol(score={floor_score:.2f})")
            print(f"[{time.strftime('%H:%M:%S')}] 🚨 ALARME — {' + '.join(causes)}")
        elif alarm_active and (now - last_alarm_t > 2.5):
            alarm_active = False

        # ── Interface ─────────────────────────────────────────────────────
        frame = draw_ui(
            frame, prob_fall, on_floor, floor_score,
            confirm_count if buffer_filled else len(frame_buffer),
            alarm_active, fps, buffer_filled
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


if __name__ == "__main__":
    main()
