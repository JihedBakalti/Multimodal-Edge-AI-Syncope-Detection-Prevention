import cv2
import numpy as np
import mediapipe as mp

class PoseExtractor:
    def __init__(self, model_complexity=1, min_det=0.5, min_track=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )

    def extract(self, bgr_img):
        """
        Returns (33,3) => [x, y, visibility] normalized in [0,1]
        If no pose detected => zeros.
        """
        img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)
        if not res.pose_landmarks:
            return np.zeros((33, 3), dtype=np.float32)

        out = np.zeros((33, 3), dtype=np.float32)
        for i, lm in enumerate(res.pose_landmarks.landmark):
            out[i, 0] = float(np.clip(lm.x, 0.0, 1.0))
            out[i, 1] = float(np.clip(lm.y, 0.0, 1.0))
            out[i, 2] = float(np.clip(lm.visibility, 0.0, 1.0))
        return out

def normalize_pose(seq_kps: np.ndarray):
    """
    seq_kps: (T,33,3) normalized coords
    - translation: subtract mid-hip
    - scale: divide by shoulder-hip distance
    keeps visibility as-is
    """
    kps = seq_kps.copy()
    # mid-hip
    hip = 0.5 * (kps[:, 23, :2] + kps[:, 24, :2])  # (T,2)
    kps[:, :, 0] -= hip[:, None, 0]
    kps[:, :, 1] -= hip[:, None, 1]

    # scale (mid-shoulder to mid-hip)
    sh = 0.5 * (kps[:, 11, :2] + kps[:, 12, :2])
    dist = np.linalg.norm(sh - hip, axis=1) + 1e-6
    kps[:, :, 0] /= dist[:, None]
    kps[:, :, 1] /= dist[:, None]
    return kps
