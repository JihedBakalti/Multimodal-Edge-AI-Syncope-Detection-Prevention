#!/usr/bin/env python3
"""
Scan all OpenCV-accessible cameras on the PC with the local YOLO Face & Person model.

- Opens each camera index sequentially (avoids locking/bandwidth issues on many systems).
- Marks cameras where at least one **person** or **face** detection passes confidence.
- Writes `active_cameras_state.json` for the InterSense orchestrator / downstream models.

Usage (from repo root or this folder):
    python MODELS/YOLO-Face-Person-Detector/camera_person_scanner.py
    python camera_person_scanner.py --max-cameras 6 --conf 0.4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = _THIS_DIR / "yolov8x_person_face.pt"
_DEFAULT_STATE_PATH = _THIS_DIR / "active_cameras_state.json"
_CACHED_MODEL = None
_CACHED_DEVICE = None


@dataclass
class CameraScanResult:
    """One frame batch result for orchestrator / syncope pipeline."""

    timestamp_utc: str
    model_weights: str
    conf: float
    iou: float
    max_camera_index_tried: int
    human_detected: bool
    camera_indices_opened: List[int]
    camera_indices_with_people: List[int]
    #: Cameras where YOLO saw at least one ``face`` detection during the scan window.
    camera_indices_with_faces: List[int]
    #: Cameras with ``person`` (body) visible but never ``face`` in that window — use body/fall pipeline.
    camera_indices_body_only: List[int]
    count_cameras_with_people: int
    per_camera: Dict[str, Any]
    person_class_ids: List[int]

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _resolve_weights(path: Optional[Path] = None) -> Path:
    p = path or _DEFAULT_WEIGHTS
    if not p.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {p}\n"
            "Place yolov8x_person_face.pt next to this script or pass --weights."
        )
    return p


def _load_yolo(weights: Path):
    from ultralytics import YOLO
    import torch

    model = YOLO(str(weights))
    model.fuse()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        model.to(device)
    return model, device


def preload_yolo_model(weights: Optional[Path] = None):
    """
    Warm-load YOLO model once; keeps model in memory (warm model, cold camera).
    """
    global _CACHED_MODEL, _CACHED_DEVICE
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_DEVICE
    w = _resolve_weights(weights)
    _CACHED_MODEL, _CACHED_DEVICE = _load_yolo(w)
    return _CACHED_MODEL, _CACHED_DEVICE


def _open_camera(cv2, index: int):
    """Open camera by index; on Windows use DirectShow for webcams that fail with default API."""
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def _init_cv2():
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "OpenCV is required. Install: pip install opencv-python"
        ) from e
    return cv2


def _detection_face_and_person_flags(result) -> Tuple[bool, bool]:
    """Return ``(seen_face, seen_person_body)`` for one YOLO result."""
    seen_face = False
    seen_person = False
    if result.boxes is None or len(result.boxes) == 0:
        return seen_face, seen_person
    names = result.names
    for cls_tensor in result.boxes.cls:
        c = int(cls_tensor.item()) if hasattr(cls_tensor, "item") else int(cls_tensor)
        label = str(names.get(c, names[c]) if isinstance(names, dict) else names[c]).lower()
        if label == "face":
            seen_face = True
        elif label == "person":
            seen_person = True
    return seen_face, seen_person


def _detection_has_person_or_face(result, class_whitelist: Tuple[str, ...] = ("person", "face")) -> bool:
    f, p = _detection_face_and_person_flags(result)
    return f or p


def _read_frame_warmup(cv2, cap, warm_frames: int = 3) -> Optional[Any]:
    frame = None
    for _ in range(max(1, warm_frames)):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    return frame


def _scan_camera_for_people(model, cap, conf: float, iou: float, scan_seconds: float, max_frames: int):
    """
    Sample several frames per camera. Accumulates separate face vs body (person) flags
    so the orchestrator can choose syncope (face) vs body-fall pipelines.
    """
    start = time.time()
    tested = 0
    best_detections = 0
    seen_face = False
    seen_person = False
    last_error = None
    while tested < max_frames and (time.time() - start) < scan_seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            last_error = "no frame"
            time.sleep(0.02)
            continue
        tested += 1
        results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
        r0 = results[0] if results else None
        dets = int(len(r0.boxes)) if r0 and r0.boxes is not None else 0
        best_detections = max(best_detections, dets)
        if r0 is not None:
            f, p = _detection_face_and_person_flags(r0)
            seen_face = seen_face or f
            seen_person = seen_person or p
            if seen_face and seen_person:
                # Both classes seen — no need to keep sampling for classification.
                break
    has_human = seen_face or seen_person
    return has_human, seen_face, seen_person, tested, best_detections, last_error


def scan_all_cameras(
    weights: Optional[Path] = None,
    state_path: Optional[Path] = None,
    max_camera_index: int = 10,
    conf: float = 0.4,
    iou: float = 0.7,
    warm_frames: int = 3,
    scan_seconds_per_camera: float = 1.5,
    max_frames_per_camera: int = 20,
    save: bool = True,
) -> CameraScanResult:
    """
    Probe camera indices ``0 .. max_camera_index-1`` and run YOLO on one frame each.

    Returns a :class:`CameraScanResult` and, if ``save`` is True, writes JSON to
    ``state_path`` (default: ``active_cameras_state.json`` in this directory).
    """
    weights = _resolve_weights(weights)
    model, _device = preload_yolo_model(weights)
    cv2 = _init_cv2()

    per_camera: Dict[str, Any] = {}
    opened: List[int] = []
    with_people: List[int] = []
    with_faces: List[int] = []
    body_only: List[int] = []
    # Class ids for person/face (for downstream syncope / orchestrator)
    person_class_ids: List[int] = []
    yolo_names = getattr(model, "names", None) or {}
    if isinstance(yolo_names, dict):
        for k, v in yolo_names.items():
            if str(v).lower() in ("person", "face"):
                person_class_ids.append(int(k))
    if not person_class_ids:
        person_class_ids = [0, 1]

    for index in range(max_camera_index):
        cap = _open_camera(cv2, index)
        if not cap.isOpened():
            per_camera[str(index)] = {
                "opened": False,
                "has_person": False,
                "error": "could not open",
            }
            cap.release()
            continue

        opened.append(index)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        frame = _read_frame_warmup(cv2, cap, warm_frames=warm_frames)
        if frame is None:
            cap.release()
            per_camera[str(index)] = {
                "opened": True,
                "has_person": False,
                "error": "no frame",
            }
            continue

        has_h, seen_face, seen_person, frames_tested, best_dets, last_error = _scan_camera_for_people(
            model=model,
            cap=cap,
            conf=conf,
            iou=iou,
            scan_seconds=scan_seconds_per_camera,
            max_frames=max_frames_per_camera,
        )
        cap.release()

        if has_h:
            with_people.append(index)
        if seen_face:
            with_faces.append(index)
        elif seen_person:
            # Person/body without a face box in this window → body fall pipeline.
            body_only.append(index)

        per_camera[str(index)] = {
            "opened": True,
            "has_person": has_h,
            "has_face": seen_face,
            "has_person_class": seen_person,
            "num_detections": best_dets,
            "frames_tested": frames_tested,
            "error": None if has_h else last_error,
        }

    result = CameraScanResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model_weights=str(weights),
        conf=conf,
        iou=iou,
        max_camera_index_tried=max_camera_index,
        human_detected=len(with_people) > 0,
        camera_indices_opened=opened,
        camera_indices_with_people=with_people,
        camera_indices_with_faces=sorted(set(with_faces)),
        camera_indices_body_only=sorted(set(body_only)),
        count_cameras_with_people=len(with_people),
        per_camera=per_camera,
        person_class_ids=sorted(set(person_class_ids)) or [0, 1],
    )

    if save:
        out = state_path or _DEFAULT_STATE_PATH
        out.write_text(
            json.dumps(result.to_json_dict(), indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {out}", file=sys.stderr)

    return result


def load_active_cameras_state(
    state_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load the last saved JSON, or None if missing."""
    path = state_path or _DEFAULT_STATE_PATH
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=f"Path to yolov8x_person_face.pt (default: {_DEFAULT_WEIGHTS.name} next to this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_STATE_PATH,
        help=f"JSON state file (default: {_DEFAULT_STATE_PATH.name})",
    )
    parser.add_argument("--max-cameras", type=int, default=10, help="Try indices 0..N-1")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--scan-seconds", type=float, default=1.5)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--no-save", action="store_true", help="Print only, do not write JSON")
    args = parser.parse_args()

    r = scan_all_cameras(
        weights=args.weights,
        state_path=args.output,
        max_camera_index=args.max_cameras,
        conf=args.conf,
        iou=args.iou,
        scan_seconds_per_camera=args.scan_seconds,
        max_frames_per_camera=args.max_frames,
        save=not args.no_save,
    )
    print(json.dumps(r.to_json_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
