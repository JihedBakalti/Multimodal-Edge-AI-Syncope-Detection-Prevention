import time
import cv2
import numpy as np
import torch
from collections import deque

from config import Config
from pose_extractor import PoseExtractor, normalize_pose
from model_tcn import PoseTCN

# MediaPipe indices
L_HIP, R_HIP = 23, 24
L_SH, R_SH = 11, 12


# ---------- Real-time plotting (OpenCV panel) ----------

def draw_timeseries_panel(frame, series_dict, panel_w=340, panel_h=190, max_points=120):
    """
    series_dict: {"name": (deque_values, (B,G,R), y_min, y_max)}
    Draw a right-side panel with curves.
    """
    h, w = frame.shape[:2]
    x0 = w - panel_w - 10
    y0 = 10

    # background
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (200, 200, 200), 1)

    # axes area
    gx0 = x0 + 40
    gy0 = y0 + 10
    gw = panel_w - 55
    gh = panel_h - 35

    # axes
    cv2.line(frame, (gx0, gy0), (gx0, gy0 + gh), (120, 120, 120), 1)
    cv2.line(frame, (gx0, gy0 + gh), (gx0 + gw, gy0 + gh), (120, 120, 120), 1)

    # ticks
    for t in [0.0, 0.5, 1.0]:
        yy = int(gy0 + (1.0 - t) * gh)
        cv2.line(frame, (gx0 - 4, yy), (gx0 + 4, yy), (120, 120, 120), 1)

    def to_xy(i, val, y_min, y_max):
        x = int(gx0 + (i / max(1, max_points - 1)) * gw)
        val = float(np.clip(val, y_min, y_max))
        t = (val - y_min) / (y_max - y_min + 1e-6)
        y = int(gy0 + (1.0 - t) * gh)
        return x, y

    # plot series
    legend_y = y0 + panel_h + 18
    lx = x0 + 5
    for name, (vals, color, y_min, y_max) in series_dict.items():
        arr = list(vals)[-max_points:]
        if len(arr) >= 2:
            pts = [to_xy(i, v, y_min, y_max) for i, v in enumerate(arr)]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)

        # legend
        cv2.putText(frame, name, (lx, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        lx += 115


# ---------- Improved lying + immobile logic ----------

def lying_score_from_pose(kps_frame: np.ndarray) -> float:
    """
    Robust lying score using:
    - torso angle (shoulders-hips)
    - bbox aspect ratio (width/height)
    Returns score in [0..1]
    """
    pts = kps_frame[:, :2]
    vis = kps_frame[:, 2] > 0.3
    ptsv = pts[vis]
    if len(ptsv) < 8:
        return 0.0

    # bbox aspect ratio
    min_xy = ptsv.min(axis=0)
    max_xy = ptsv.max(axis=0)
    w = max_xy[0] - min_xy[0]
    h = max_xy[1] - min_xy[1]
    ar = w / (h + 1e-6)  # > 1 => more horizontal

    # torso angle
    sh = 0.5 * (kps_frame[L_SH, :2] + kps_frame[R_SH, :2])
    hip = 0.5 * (kps_frame[L_HIP, :2] + kps_frame[R_HIP, :2])
    v = sh - hip
    v_norm = np.linalg.norm(v) + 1e-6
    cos_vert = abs(v[1]) / v_norm
    angle_score = 1.0 - cos_vert  # horizontal => high

    # map aspect ratio to [0..1] with soft thresholding
    # ar ~ 0.7..2.0 typical; tweak if needed
    ar_score = np.clip((ar - 0.8) / 1.2, 0.0, 1.0)

    score = 0.55 * angle_score + 0.45 * ar_score
    return float(np.clip(score, 0.0, 1.0))


def mean_motion(prev_kps: np.ndarray, cur_kps: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(cur_kps[:, :2] - prev_kps[:, :2], axis=1)))


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "checkpoints/best_pose_tcn.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    model = PoseTCN(
        in_dim=cfg.num_keypoints * cfg.in_features_per_kp,
        channels=cfg.tcn_channels,
        layers=cfg.tcn_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    extractor = PoseExtractor(cfg.pose_model_complexity, cfg.pose_min_det_conf, cfg.pose_min_track_conf)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    # buffers
    T = cfg.clip_len
    buf = []

    # time series history
    max_points = 120
    p_hist = deque(maxlen=max_points)
    lying_hist = deque(maxlen=max_points)
    motion_hist = deque(maxlen=max_points)

    # fall confirmation (anti-false positives)
    fall_hits = 0
    fall_hits_needed = 3  # require N consecutive hits

    # post-fall state
    fall_mode = False
    fall_time = None
    immobile_start = None
    prev_kps = None

    # stable immobility over a small window
    motion_win = deque(maxlen=10)  # roughly 10 inferences ~ 2-3s depending infer_every

    last_infer = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        kps = extractor.extract(frame)  # (33,3) in 0..1
        buf.append(kps)
        if len(buf) > T:
            buf.pop(0)

        now = time.time()
        fainted = False

        p_fall = None
        lying_score = None
        motion = None

        if len(buf) == T and (now - last_infer) > cfg.infer_every_seconds:
            last_infer = now

            seq = np.stack(buf, axis=0).astype(np.float32)      # (T,33,3)
            seq_n = normalize_pose(seq)                         # (T,33,3)
            x = torch.from_numpy(seq_n.reshape(T, -1)[None, ...]).to(device)  # (1,T,99)

            with torch.no_grad():
                logit = model(x).squeeze(0).squeeze(0)
                p_fall = float(torch.sigmoid(logit).item())

            # --- fall confirmation ---
            if p_fall > cfg.fall_threshold:
                fall_hits += 1
            else:
                fall_hits = max(0, fall_hits - 1)

            if fall_hits >= fall_hits_needed and not fall_mode:
                fall_mode = True
                fall_time = now
                immobile_start = None
                prev_kps = None
                motion_win.clear()

            # --- post-fall checks ---
            if fall_mode:
                last_pose = seq_n[-1]  # normalized pose
                lying_score = lying_score_from_pose(last_pose)
                lying = lying_score > cfg.lying_threshold

                # motion
                if prev_kps is not None:
                    motion = mean_motion(prev_kps, buf[-1])
                else:
                    motion = 999.0
                prev_kps = buf[-1].copy()

                motion_win.append(motion)
                motion_avg = float(np.mean(motion_win)) if len(motion_win) > 0 else motion
                immobile = motion_avg < cfg.immobile_epsilon

                if lying and immobile:
                    if immobile_start is None:
                        immobile_start = now
                    if (now - immobile_start) >= cfg.hold_seconds:
                        fainted = True
                else:
                    immobile_start = None

                # reset if no confirmation after timeout
                if fall_time and (now - fall_time) > (cfg.hold_seconds + 8.0) and not fainted:
                    fall_mode = False
                    fall_time = None
                    immobile_start = None
                    prev_kps = None
                    fall_hits = 0
                    motion_win.clear()

            # push histories
            p_hist.append(p_fall if p_fall is not None else 0.0)
            lying_hist.append(lying_score if lying_score is not None else 0.0)
            motion_hist.append(motion if motion is not None and motion < 999 else 0.0)

        # overlay text
        y0 = 25
        if p_fall is not None:
            cv2.putText(frame, f"p_fall={p_fall:.2f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            y0 += 28
        cv2.putText(frame, f"fall_mode={fall_mode}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        y0 += 28

        if lying_score is not None:
            cv2.putText(frame, f"lying_score={lying_score:.2f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            y0 += 28
        if motion is not None and motion < 999:
            cv2.putText(frame, f"motion={motion:.4f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            y0 += 28

        if fainted:
            cv2.putText(frame, "FAINTED DETECTED!", (10, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,0,255), 3)

        # draw plot panel
        series = {
            "p_fall": (p_hist, (0, 255, 0), 0.0, 1.0),
            "lying": (lying_hist, (255, 255, 0), 0.0, 1.0),
            "motion": (motion_hist, (0, 200, 255), 0.0, 0.25),  # tweak y_max if needed
        }
        draw_timeseries_panel(frame, series, panel_w=340, panel_h=190, max_points=max_points)

        cv2.imshow("Fainting (Fall+Lying+Immobile) Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
