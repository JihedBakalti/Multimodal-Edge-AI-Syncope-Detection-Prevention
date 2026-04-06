import os
import json
import cv2
import numpy as np
from tqdm import tqdm

from config import Config
from pose_extractor import PoseExtractor, normalize_pose

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")

def list_videos(folder: str):
    if not os.path.isdir(folder):
        return []
    vids = []
    for f in os.listdir(folder):
        if f.lower().endswith(VIDEO_EXTS):
            vids.append(os.path.join(folder, f))
    return sorted(vids)

def extract_pose_from_video(video_path: str, extractor: PoseExtractor, frame_step: int, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    kps_all = []
    frame_idx = 0
    kept = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            kps_all.append(extractor.extract(frame))
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(kps_all) == 0:
        return None

    return np.stack(kps_all, axis=0).astype(np.float32)  # (T,33,3)

def main():
    cfg = Config()
    extractor = PoseExtractor(
        model_complexity=cfg.pose_model_complexity,
        min_det=cfg.pose_min_det_conf,
        min_track=cfg.pose_min_track_conf,
    )

    falls_dir = os.path.join(cfg.urfd_root, "falls")
    adl_dir   = os.path.join(cfg.urfd_root, "adl")

    out_root = os.path.join(cfg.urfd_root, "processed_pose_npz")
    os.makedirs(out_root, exist_ok=True)

    index = {"falls": [], "adl": []}

    for label_name, folder, y in [("falls", falls_dir, 1), ("adl", adl_dir, 0)]:
        vids = list_videos(folder)
        if not vids:
            print(f"[WARN] No videos found in: {folder}")
            continue

        for vp in tqdm(vids, desc=f"Processing {label_name}"):
            kps_all = extract_pose_from_video(
                vp,
                extractor,
                frame_step=cfg.frame_step,
                max_frames=cfg.max_frames_per_seq,
            )
            if kps_all is None or kps_all.shape[0] < cfg.min_frames_per_seq:
                continue

            kps_all = normalize_pose(kps_all)

            # seq_id from filename (without extension)
            seq_id = os.path.splitext(os.path.basename(vp))[0]
            out_path = os.path.join(out_root, f"{label_name}-{seq_id}.npz")

            np.savez_compressed(
                out_path,
                kps=kps_all,
                y=np.int64(y),
                seq=seq_id,
                video_path=vp,
                frame_step=np.int32(cfg.frame_step),
            )
            index[label_name].append(out_path)

    with open(os.path.join(out_root, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("Done.")
    print(f"Saved pose NPZs to: {out_root}")
    print(f"Index: {os.path.join(out_root, 'index.json')}")
    print(f"Falls processed: {len(index['falls'])} | ADL processed: {len(index['adl'])}")

if __name__ == "__main__":
    main()
