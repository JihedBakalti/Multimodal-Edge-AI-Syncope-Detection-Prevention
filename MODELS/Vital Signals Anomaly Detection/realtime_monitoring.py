import json
import time
from collections import deque
from typing import Deque, Optional, Tuple

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

INPUT_FILE = "spo2.json"
OUTPUT_FILE = "scores.json"


# =========================
# Baseline Tracker
# =========================
class SpO2BaselineTracker:
    def __init__(self, window_sec=60.0, dt=1.0, spike_threshold=4.0):
        self.alpha = dt / max(window_sec, dt)
        self.spike_threshold = spike_threshold
        self.baseline: Optional[float] = None

    def update(self, spo2: float) -> float:
        if self.baseline is None:
            self.baseline = spo2
            return spo2

        lo = self.baseline - self.spike_threshold
        hi = self.baseline + self.spike_threshold
        clipped = max(lo, min(hi, spo2))

        self.baseline = (1 - self.alpha) * self.baseline + self.alpha * clipped
        return self.baseline


class BpmBaselineTracker:
    def __init__(self, window_sec=60.0, dt=1.0, spike_threshold=25.0):
        self.alpha = dt / max(window_sec, dt)
        self.spike_threshold = spike_threshold
        self.baseline: Optional[float] = None

    def update(self, bpm: float) -> float:
        if self.baseline is None:
            self.baseline = bpm
            return bpm

        lo = self.baseline - self.spike_threshold
        hi = self.baseline + self.spike_threshold
        clipped = max(lo, min(hi, bpm))

        self.baseline = (1 - self.alpha) * self.baseline + self.alpha * clipped
        return self.baseline


class BpmFaintingDetector:
    def __init__(self, peak_window_sec: float = 120.0) -> None:
        self.peak_window_sec = peak_window_sec
        self.history: Deque[Tuple[float, float, float]] = deque()
        self.last_label: Optional[str] = None
        self.last_alert_time: Optional[float] = None
        self.cooldown_sec = 5.0

    def process(self, t: float, bpm: float, baseline: float) -> str:
        self.history.append((t, bpm, baseline))
        while self.history and (t - self.history[0][0]) > self.peak_window_sec:
            self.history.popleft()

        if self.history:
            peak_time, peak_bpm, peak_base = max(self.history, key=lambda x: x[1])
        else:
            peak_time, peak_bpm, peak_base = t, bpm, baseline

        peak_base_safe = peak_base if peak_base > 0 else 1.0
        increase_from_baseline_at_peak = (peak_bpm - peak_base_safe) / peak_base_safe

        peak_bpm_safe = peak_bpm if peak_bpm > 0 else 1.0
        drop_from_peak_rel = (peak_bpm_safe - bpm) / peak_bpm_safe
        abs_drop_bpm = peak_bpm - bpm
        time_since_peak = t - peak_time

        label: Optional[str] = None

        if baseline > 0 and bpm <= 0.65 * baseline:
            label = "CRITICAL FAINTING"
        elif baseline > 0 and (bpm <= 0.80 * baseline) and (abs_drop_bpm >= 20.0):
            label = "HIGH RISK"
        elif (
            increase_from_baseline_at_peak >= 0.20
            and drop_from_peak_rel >= 0.30
            and 0.0 <= time_since_peak <= 60.0
        ):
            label = "EARLY WARNING"

        now = t
        if label is not None:
            if (
                self.last_label == label
                and self.last_alert_time is not None
                and (now - self.last_alert_time) < self.cooldown_sec
            ):
                return label
            self.last_label = label
            self.last_alert_time = now
            return label

        baseline_safe = baseline if baseline > 0 else 1.0
        percent_increase_now = (bpm - baseline_safe) / baseline_safe
        if percent_increase_now >= 0.20:
            return "Rising"
        return "Normal"


# =========================
# Detector (time-aware)
# =========================
class PreSyncopeDetector:
    def __init__(self, window_sec=120):
        self.window_sec = window_sec
        self.history: Deque[Tuple[float, float, float]] = deque()

    def process(self, t, spo2, baseline):
        self.history.append((t, spo2, baseline))

        # remove old
        while self.history and (t - self.history[0][0]) > self.window_sec:
            self.history.popleft()

        peak_time, peak_spo2, _ = max(self.history, key=lambda x: x[1])

        drop = peak_spo2 - spo2
        time_since_peak = t - peak_time
        below_baseline = baseline - spo2

        if spo2 < 85:
            return "CRITICAL"
        elif spo2 < 88 and below_baseline >= 5:
            return "HIGH RISK"
        elif drop >= 5 and time_since_peak <= 45:
            return "HIGH RISK"
        elif drop >= 3 and time_since_peak <= 60:
            return "EARLY WARNING"
        elif below_baseline >= 2:
            return "DECLINING"
        else:
            return "NORMAL"


# =========================
# Score computation
# =========================

def label_to_score(label):
    if label == "CRITICAL":
        return 1.0
    elif label == "HIGH RISK":
        return 0.8
    elif label == "EARLY WARNING":
        return 0.6
    elif label == "DECLINING":
        return 0.3
    else:
        return 0.0


def bpm_label_to_score(label):
    if label == "CRITICAL FAINTING":
        return 1.0
    elif label == "HIGH RISK":
        return 0.8
    elif label == "EARLY WARNING":
        return 0.6
    elif label == "Rising":
        return 0.3
    else:
        return 0.0


# =========================
# Globals (same session state as watchdog mode)
# =========================
baseline_tracker = SpO2BaselineTracker()
bpm_baseline_tracker = BpmBaselineTracker()
detector = PreSyncopeDetector()
bpm_detector = BpmFaintingDetector()

last_timestamp = -1


def _compute_clinical_dict(spo2: float, bpm: float, timestamp: float) -> dict:
    """Single place for clinical math — used by process_data and compute_clinical_for_merge."""
    spo2_baseline = baseline_tracker.update(spo2)
    bpm_baseline = bpm_baseline_tracker.update(bpm)

    spo2_label = detector.process(timestamp, spo2, spo2_baseline)
    spo2_score = label_to_score(spo2_label)
    bpm_label = bpm_detector.process(timestamp, bpm, bpm_baseline)
    bpm_score = bpm_label_to_score(bpm_label)
    score = max(spo2_score, bpm_score)

    return {
        "score": round(score, 3),
        "spo2_state": spo2_label,
        "bpm_state": bpm_label,
        "spo2_score": round(spo2_score, 3),
        "bpm_score": round(bpm_score, 3),
    }


def compute_clinical_for_merge(spo2: float, bpm: float, timestamp: float) -> dict:
    """
    Same outcome as process_data's clinical block (no file write).
    Import from main2 to merge with PPGPT into results.json.
    """
    return _compute_clinical_dict(spo2, bpm, timestamp)


# =========================
# JSON IO
# =========================
def read_data():
    try:
        with open(INPUT_FILE, "r") as f:
            data = json.load(f)
            return float(data["spo2"]), float(data["bpm"]), float(data["timestamp"])
    except Exception:
        return None, None, None


def write_scores(latest_result):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(latest_result, f, indent=2)


# =========================
# Processing
# =========================
def process_data(spo2, bpm, timestamp):
    clinical = _compute_clinical_dict(spo2, bpm, timestamp)
    latest_result = {
        "timestamp": round(timestamp, 3),
        **clinical,
    }
    write_scores(latest_result)

    print(
        f"[{timestamp:.1f}] "
        f"SpO2={spo2:.1f} (base={baseline_tracker.baseline:.1f}) [{clinical['spo2_state']}] | "
        f"BPM={bpm:.1f} (base={bpm_baseline_tracker.baseline:.1f}) [{clinical['bpm_state']}] | "
        f"score={clinical['score']}"
    )


# =========================
# File Watcher
# =========================
class Spo2Handler(FileSystemEventHandler):
    def on_modified(self, event):
        global last_timestamp

        if not event.src_path.endswith(INPUT_FILE):
            return

        spo2, bpm, timestamp = read_data()

        if timestamp is None:
            return

        if timestamp <= last_timestamp:
            return

        last_timestamp = timestamp
        process_data(spo2, bpm, timestamp)


# =========================
# MAIN
# =========================
def main():
    observer = Observer()
    handler = Spo2Handler()

    observer.schedule(handler, ".", recursive=False)
    observer.start()

    print("Monitoring started...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()
