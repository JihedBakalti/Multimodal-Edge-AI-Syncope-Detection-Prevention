
# ═══════════════════════════════════════════════════════════════
# test_voice.py — Real-time voice test with trained model
# Run: python test_voice.py
# Calibrate mic (low gain / wrong device):  python test_voice.py --calibrate
# File scoring: python test_voice.py -a clip.wav   (or: python test_voice.py clip.wav)
# Default recording length via env VOICE_RECORD_SECS (longer clips OK — inference uses TORGO-matched windows below).
# ═══════════════════════════════════════════════════════════════

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import sounddevice as sd
import sys
from pathlib import Path
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_FILE = SCRIPT_DIR / "voice_input_calibration.json"

_CALIBRATE_HINT = (
    "Calibration is a standalone flag - use:\n"
    f"  python {Path(__file__).name} --calibrate\n"
    "Do not put `--calibrate` after `-a/--audio` or `--audio=`; "
    'give an actual audio path (e.g. `--audio recording.wav`).'
)


def _strip_audio_path(raw: object) -> str:
    """Normalize user/glue-code strings (BOM/spaces); does not validate existence."""
    return str(raw).strip().lstrip("\ufeff").strip()


def _reject_cli_flag_used_as_filepath(path_like: object) -> str:
    """
    Paths like `--calibrate` commonly come from quoting mistakes or `-a --calibrate`.
    Raises ValueError before librosa/audioread tries to open()- them.
    """
    s = _strip_audio_path(path_like)
    if not s:
        raise ValueError("Empty audio path.")
    if s.startswith("--"):
        raise ValueError(_CALIBRATE_HINT + f"\n(Refusing to open flag-like path {s!r}.)")
    return s


def _argparse_audio_file_value(s: str) -> str:
    try:
        return _reject_cli_flag_used_as_filepath(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e

# ── Model definition (must match training) ──
class VoiceClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ── Config (must match training) ──
SR = 16000
# Training (torgo-notebook-2) used fixed 3 s clips. Longer recordings are split into overlapping 3 s windows
# and the max dysarthria score is taken so brief cues are not smeared across the whole mel time axis.
DURATION = float(os.getenv("VOICE_RECORD_SECS", "10.0"))
CLIP_SECONDS_MAX = float(os.getenv("VOICE_CLIP_MAX_SECS", "30.0"))
TORGO_CLIP_SECS = float(os.getenv("VOICE_TORGO_WINDOW_SECS", "3.0"))
WIN_HOP_SECS = float(os.getenv("VOICE_WIN_HOP_SECS", "1.5"))
# Fraction of short-time frames that must exceed a relative RMS floor (post peak-norm) for a "reliable" clip.
MIN_SPEECH_FRAME_FRAC = float(os.getenv("VOICE_MIN_SPEECH_FRAC", "0.12"))


def clip_sample_count(seconds: float) -> int:
    sec = float(np.clip(seconds, 0.5, CLIP_SECONDS_MAX))
    return int(SR * sec)


def clamp_record_seconds(seconds: float) -> float:
    return float(np.clip(seconds, 0.5, CLIP_SECONDS_MAX))
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMAX = 8000
IMG_H, IMG_W = 128, 128
LABELS = {0: '✅ Normal', 1: '⚠️  Dysarthric / Risk Detected'}
MODEL_PATH = SCRIPT_DIR / "voice_model_export.pt"

# Aim for RMS similar to trimmed speech clips often used in training (~0.05–0.12 in float [-1,1]).
TARGET_RMS = float(os.getenv("VOICE_TARGET_RMS", "0.075"))
MIC_GAIN_CLIP = (0.35, 30.0)
PEAK_NORM_TARGET = float(os.getenv("VOICE_PEAK_NORM", "0.95"))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_calibration():
    if not CALIBRATION_FILE.is_file():
        return {"mic_gain": 1.0, "rms_ref": None, "device": None}
    try:
        return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"mic_gain": 1.0, "rms_ref": None, "device": None}


def save_calibration(mic_gain, rms_measured, device_index):
    data = {
        "mic_gain": float(mic_gain),
        "rms_measured_raw": float(rms_measured),
        "device": device_index,
    }
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved calibration → {CALIBRATION_FILE}")
    print(f"  mic_gain={mic_gain:.3f} (use this multiplier before peak-normalize)")


def preprocess_waveform(y, mic_gain=1.0):
    """
    Loudness alignment: TORGO-like clips are usually normalized; laptop mics often sit ~10× quieter,
    which makes mel + min-max look like silence/noise to the classifier.
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    if mic_gain != 1.0:
        y = y * float(mic_gain)
    peak = np.max(np.abs(y)) + 1e-9
    if peak < 1e-7:
        return y.astype(np.float32), {
            "rms": 0.0,
            "peak": 0.0,
            "warn": True,
            "hint": "No usable audio — enable mic / pick device with --list-devices.",
        }
    y = y / peak * PEAK_NORM_TARGET
    rms = float(np.sqrt(np.mean(np.square(y))))
    hint = ""
    if rms < 0.012:
        hint = (
            "(Input still very quiet — run `--calibrate` or check Windows mic level / `--list-devices`.)"
        )
    return np.clip(y, -1.0, 1.0).astype(np.float32), {"rms": rms, "peak": float(peak), "warn": False, "hint": hint}


def load_model():
    mp = MODEL_PATH
    if not mp.is_file():
        raise FileNotFoundError(f"Missing model: {mp}")
    checkpoint = torch.load(mp, map_location=DEVICE, weights_only=False)
    model = VoiceClassifier().to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model loaded — Val AUC: {checkpoint.get('val_auc', 'N/A')}")
    return model

def audio_to_melspec(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_db = np.array([
        np.interp(np.linspace(0, mel_db.shape[1]-1, IMG_W),
                  np.arange(mel_db.shape[1]), row)
        for row in mel_db
    ])
    return mel_db.astype(np.float32)


def speech_frame_fraction(y_f32):
    """Fraction of short-time frames with RMS well above noise floor (waveform already scaled)."""
    y_f32 = np.asarray(y_f32, dtype=np.float32).flatten()
    if len(y_f32) < 512:
        return 0.0
    rms = librosa.feature.rms(y=y_f32, frame_length=512, hop_length=256, center=True)[0]
    peak = float(np.max(np.abs(y_f32)) + 1e-9)
    thr = max(1e-5, 0.01 * peak)
    return float(np.mean(rms > thr))


def _forward_mel(model, mel_db):
    x = torch.tensor(mel_db).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        pred = int(out.argmax(1).item())
    return pred, float(probs[0].item()), float(probs[1].item())


def predict_from_array(model, y, mic_gain=1.0, clip_seconds=None):
    """Predict from raw numpy audio array (float32 ±1 mic samples).

    clip_seconds: how many seconds of y to analyze (pads/truncates). Defaults to ``DURATION``.
    Must match recording length when you pass a longer ``--duration`` recording.
    """
    clip_sec = float(DURATION if clip_seconds is None else clip_seconds)
    n_samples = clip_sample_count(clip_sec)
    y = np.asarray(y, dtype=np.float32).flatten()
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]

    win_samples = int(SR * clamp_record_seconds(TORGO_CLIP_SECS))
    hop_samples = max(1, int(SR * clamp_record_seconds(WIN_HOP_SECS)))

    # Clips longer than TORGO training window: score overlapping subwindows and max-pool dysarthria prob
    # (full-length mel smears brief events across the fixed IMG_W time axis).
    if n_samples > win_samples:
        preds_norm_dys = []
        skipped_silent = 0
        max_start = n_samples - win_samples
        start = 0
        while start <= max_start:
            chunk = np.array(y[start : start + win_samples], dtype=np.float32, copy=True)
            y_w, lvl = preprocess_waveform(chunk, mic_gain=mic_gain)
            if lvl.get("rms", 1) < 1e-5:
                skipped_silent += 1
                start += hop_samples
                continue
            frac = speech_frame_fraction(y_w)
            if frac < MIN_SPEECH_FRAME_FRAC:
                skipped_silent += 1
                start += hop_samples
                continue
            mel = audio_to_melspec(y_w)
            preds_norm_dys.append(_forward_mel(model, mel))
            start += hop_samples

        if preds_norm_dys:
            # Risk = strongest dysarthria signal in any voiced window (softmax is per-window).
            best = max(preds_norm_dys, key=lambda t: t[2])
            pred, prob_normal, prob_dys = best[0], best[1], best[2]
            if skipped_silent:
                print(
                    f"ℹ️  Used {len(preds_norm_dys)} voiced {TORGO_CLIP_SECS:g}s window(s); "
                    f"skipped {skipped_silent} low-speech window(s) (silence skews mel + ref=np.max)."
                )
            return pred, prob_normal, prob_dys

        print(
            "⚠️  No window had enough voiced audio — falling back to full-clip score "
            "(long silence can falsely elevate dysarthria; try speaking steadily or shorten `--duration`)."
        )

    y, lvl = preprocess_waveform(y, mic_gain=mic_gain)
    if lvl.get("rms", 1) < 1e-5:
        print("⚠️  Near-zero signal after normalization — mic may be muted or wrong device.")
    else:
        frac = speech_frame_fraction(y)
        if frac < MIN_SPEECH_FRAME_FRAC:
            print(
                f"⚠️  Low voiced-frame fraction (~{frac:.0%}) — "
                "scores may be unreliable (mostly silence / mic too quiet)."
            )

    mel = audio_to_melspec(y)
    return _forward_mel(model, mel)


def record_clip(duration, device=None):
    frames = int(duration * SR)
    audio = sd.rec(frames, samplerate=SR, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio.flatten()


def run_calibration(duration=5.0, device=None):
    print("\n📏 Microphone calibration")
    print("  1. Stay quiet… (2 s room noise)")
    print("  2. Then speak at your NORMAL volume until the beep finishes.\n")
    noise = record_clip(2.0, device=device)
    speech = record_clip(duration, device=device)
    n_rms = float(np.sqrt(np.mean(np.square(noise))))
    s_rms = float(np.sqrt(np.mean(np.square(speech))))
    if s_rms < n_rms * 1.5:
        print(
            "⚠️  Speech RMS is barely above silence — boost OS mic gain, move closer,"
            " or pick another device (`--list-devices`). Using conservative gain."
        )
    effective_rms = max(s_rms - n_rms * 0.5, 1e-6)
    gain = TARGET_RMS / effective_rms
    gain = float(np.clip(gain, MIC_GAIN_CLIP[0], MIC_GAIN_CLIP[1]))
    print(f"  Noise RMS≈{n_rms:.5f}, speech RMS≈{s_rms:.5f}")
    print(f"  Applying mic_gain={gain:.3f} (TARGET_RMS={TARGET_RMS})")
    idx = device if device is not None else sd.default.device[0]
    save_calibration(gain, s_rms, idx)
    return gain


def record_and_predict(model, duration=DURATION, device=None, mic_gain=1.0):
    duration = clamp_record_seconds(duration)
    print(f"\n🎙️  Recording {duration}s — speak now (any language, any words)...")
    y = record_clip(duration, device=device)

    dbg, _dbg2 = preprocess_waveform(y.copy(), mic_gain=mic_gain)
    r_dbg = float(np.sqrt(np.mean(np.square(dbg))))
    print(f"✅ Recording done | post-gain+RMS RMS≈{r_dbg:.4f} — analyzing {duration:g}s…")
    if _dbg2.get("hint"):
        print("   ", _dbg2["hint"])

    pred, prob_normal, prob_dys = predict_from_array(
        model, y, mic_gain=mic_gain, clip_seconds=duration
    )
    
    print("\n" + "="*45)
    print(f"  RESULT: {LABELS[pred]}")
    print(f"  Confidence — Normal:     {prob_normal*100:.1f}%")
    print(f"  Confidence — Dysarthric: {prob_dys*100:.1f}%")
    print("="*45)
    
    # Risk level for your fainting detection pipeline
    if prob_dys > 0.7:
        print("  🚨 HIGH RISK — Escalate to emergency system")
    elif prob_dys > 0.4:
        print("  ⚠️  MODERATE RISK — Trigger confirmation dialogue")
    else:
        print("  ✅ LOW RISK — Continue monitoring")
    
    return pred, prob_dys

def predict_from_file(model, path, mic_gain=1.0, clip_seconds=None):
    """Predict from an existing audio file."""
    path_str = _reject_cli_flag_used_as_filepath(path)
    p = Path(path_str).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}\n({_CALIBRATE_HINT})")

    y, _ = librosa.load(os.fspath(p), sr=SR, mono=True)
    pred, prob_normal, prob_dys = predict_from_array(
        model, y, mic_gain=mic_gain, clip_seconds=clip_seconds
    )
    print(f"File: {p}")
    print(f"Result: {LABELS[pred]} | Normal: {prob_normal:.3f} | Dysarthric: {prob_dys:.3f}")
    return pred, prob_dys

def _pick_device_explicit(args):
    if args.device is not None:
        return int(args.device)
    cal = load_calibration()
    d = cal.get("device")
    return int(d) if d is not None else None


def _cli():
    ap = argparse.ArgumentParser(
        description=(
            "TORGO dysarthria vs control classifier (motor/slurred speech cues — "
            "not trained for stuttering; use max over 3 s windows on longer clips)."
        )
    )
    ap.add_argument("--calibrate", action="store_true", help="Microphone calibration (saves beside script)")
    ap.add_argument("--list-devices", action="store_true", help="Print sounddevice inputs and exit")
    ap.add_argument("--device", type=int, default=None, help="Input device index")
    ap.add_argument(
        "--duration",
        type=float,
        default=DURATION,
        help=f"Recording / file analysis window (seconds, capped at {CLIP_SECONDS_MAX:g}; env VOICE_RECORD_SECS)",
    )
    ap.add_argument(
        "--mic-gain",
        type=float,
        default=None,
        help="Force multiplier (skip calibration file if set)",
    )
    ap.add_argument(
        "--audio",
        "-a",
        dest="audio_path",
        default=None,
        metavar="PATH",
        type=_argparse_audio_file_value,
        help="WAV/MP3 file to score (alternative to positional path below)",
    )
    # Optional positional last; type= rejects mistaken tokens like `--calibrate` early.
    ap.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        metavar="AUDIO_FILE",
        type=_argparse_audio_file_value,
        help="Optional path to WAV/MP3; omit for realtime microphone",
    )
    args = ap.parse_args()
    args.duration = clamp_record_seconds(args.duration)

    audio_in = args.audio_path or args.audio_file
    try:
        if audio_in is not None:
            audio_in = _reject_cli_flag_used_as_filepath(audio_in)
    except ValueError as e:
        ap.error(str(e))
    args.audio_in = audio_in

    if args.list_devices:
        print(sd.query_devices())
        return

    cal = load_calibration()
    mic_gain = float(cal.get("mic_gain", 1.0)) if args.mic_gain is None else float(args.mic_gain)

    device = _pick_device_explicit(args)

    if args.calibrate:
        run_calibration(duration=max(8.0, args.duration), device=device)
        cal = load_calibration()
        mic_gain = float(cal.get("mic_gain", 1.0))

    print(f"Using mic_gain={mic_gain:.4f}" + (" (saved calibration)" if args.mic_gain is None else ""))
    if device is not None:
        print(f"Input device index: {device}")

    model = load_model()

    if args.audio_in:
        predict_from_file(model, args.audio_in, mic_gain=mic_gain, clip_seconds=args.duration)
        return

    print("\n🎤 Real-time Voice Risk Detector")
    print("Speak any language — model is language agnostic")
    print("Tips: OS mic level up, `--calibrate` once, `--list-devices` if wrong mic")
    print("Press Enter to record | Ctrl+C to exit\n")

    while True:
        try:
            input("Press Enter to start recording...")
            record_and_predict(model, duration=args.duration, device=device, mic_gain=mic_gain)
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break


# ── Main ──
if __name__ == "__main__":
    _cli()
