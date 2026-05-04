# -*- coding: utf-8 -*-
"""
TORGO dysarthria voice risk scoring for WAV bytes (e.g. from SpeechRecognition).
Adds MODELS/Voice Anomaly detection to sys.path and reuses test_voice.predict_from_array.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VOICE_DIR = ROOT / "MODELS" / "Voice Anomaly detection"


def _ensure_path() -> None:
    p = str(VOICE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


_model = None


def _get_model(tv):
    global _model
    if _model is None:
        _model = tv.load_model()
    return _model


def score_wav_bytes_torgo(wav_bytes: bytes) -> dict | None:
    """
    Return dict with pred, prob_normal, prob_dysarthric, label — or None if deps/audio missing.
    On recoverable errors, returns {"error": str, "prob_normal": 0.0, "prob_dysarthric": 0.0, ...}.
    """
    if not wav_bytes:
        return None
    _ensure_path()
    try:
        import librosa
        import numpy as np
    except ImportError as e:
        return {"error": f"librosa: {e}", "prob_normal": 0.0, "prob_dysarthric": 0.0, "pred": 0, "label": "n/a"}

    try:
        import test_voice as tv
    except ImportError as e:
        return {"error": f"test_voice: {e}", "prob_normal": 0.0, "prob_dysarthric": 0.0, "pred": 0, "label": "n/a"}

    try:
        y, _sr = librosa.load(io.BytesIO(wav_bytes), sr=int(tv.SR), mono=True)
    except Exception as e:
        return {"error": f"decode: {e}", "prob_normal": 0.0, "prob_dysarthric": 0.0, "pred": 0, "label": "n/a"}

    clip_sec = float(os.getenv("VOICE_SAFETY_ANALYSIS_SECONDS", "10"))
    cal = tv.load_calibration()
    mic_gain = float(cal.get("mic_gain", 1.0))
    model = _get_model(tv)
    y32 = np.asarray(y, dtype=np.float32)
    pred, p0, p1 = tv.predict_from_array(model, y32, mic_gain=mic_gain, clip_seconds=clip_sec)
    return {
        "pred": int(pred),
        "prob_normal": float(p0),
        "prob_dysarthric": float(p1),
        "label": str(tv.LABELS.get(pred, str(pred))),
        "clip_seconds": clip_sec,
    }
