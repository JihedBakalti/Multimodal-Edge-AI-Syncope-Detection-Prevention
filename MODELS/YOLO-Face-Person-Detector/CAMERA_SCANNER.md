# Multi-camera person detection (InterSense)

This folder contains `camera_person_scanner.py`, which:

1. Tries every camera index from **0** to **`max_camera_index - 1`** (default 10) using OpenCV.
2. Loads the local YOLO weights `yolov8x_person_face.pt` (person + face classes).
3. Captures a short **warmup** on each open device, runs one inference per camera.
4. Records which indices have at least one **person** or **face** above confidence.
5. Writes **`active_cameras_state.json`** in this directory (see fields below) so the syncope model / orchestrator can read **which camera IDs to use** and **`count_cameras_with_people`**.

## Run

From the project root (with `ultralytics`, `torch`, `opencv-python` installed):

```bash
python MODELS/YOLO-Face-Person-Detector/camera_person_scanner.py
```

Options:

- `--max-cameras 10` — try indices 0..9.
- `--conf 0.4` / `--iou 0.7` — YOLO thresholds.
- `--output path/to/state.json` — custom JSON path.
- `--no-save` — print JSON only (no file).

## Output JSON (for orchestrator integration)

| Field | Meaning |
|--------|--------|
| `human_detected` | `true` if at least one camera shows a person or face. |
| `count_cameras_with_people` | How many devices currently show a person/face. |
| `camera_indices_with_people` | List of OpenCV camera indices to feed into the next stage. |
| `camera_indices_opened` | Indices that successfully opened. |
| `per_camera` | Per-index details (`opened`, `has_person`, `num_detections`). |
| `person_class_ids` | YOLO class ids for `person` / `face` in this model. |
| `timestamp_utc` | When the scan finished. |

## Python API

```python
from pathlib import Path
import sys
sys.path.insert(0, r"path/to/MODELS/YOLO-Face-Person-Detector")
from camera_person_scanner import scan_all_cameras, load_active_cameras_state

state = scan_all_cameras(max_camera_index=8)
# or
data = load_active_cameras_state()
n = data["count_cameras_with_people"]
indices = data["camera_indices_with_people"]
```

**Note:** On Windows, webcams are opened with **DirectShow** first, then the default backend if that fails.
