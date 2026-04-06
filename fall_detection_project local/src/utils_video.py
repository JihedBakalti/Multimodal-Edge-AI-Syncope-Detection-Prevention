import os
import glob

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.mpeg", "*.mpg")

def find_video_file(seq_root: str, camera_hint: str = "cam0"):
    """
    Finds a video file under seq_root. Prefers filenames containing 'cam0'.
    Returns absolute/relative path or None.
    """
    vids = []
    for ext in VIDEO_EXTS:
        vids.extend(glob.glob(os.path.join(seq_root, ext)))
        vids.extend(glob.glob(os.path.join(seq_root, "**", ext), recursive=True))

    if not vids:
        return None

    # Prefer cam0 in filename
    cam0 = [v for v in vids if camera_hint.lower() in os.path.basename(v).lower()]
    if cam0:
        return sorted(cam0)[0]

    # Otherwise pick first (stable order)
    return sorted(vids)[0]
