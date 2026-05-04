"""
Streaming HR + SpO2 samples with monotonic timestamps for clinical anomaly scoring.

Reuses baseline + pattern detectors from MODELS/Vital Signals Anomaly Detection/realtime_monitoring.py
without reading spo2.json. Per (user_id, session_id) state for baselines and deduplication.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

_VS_DIR = Path(__file__).resolve().parent / "MODELS" / "Vital Signals Anomaly Detection"
if str(_VS_DIR) not in sys.path:
    sys.path.insert(0, str(_VS_DIR))

from realtime_monitoring import (  # noqa: E402
    BpmBaselineTracker,
    BpmFaintingDetector,
    PreSyncopeDetector,
    SpO2BaselineTracker,
    bpm_label_to_score,
    label_to_score,
)


def _clinical_dict(
    baseline_tracker: SpO2BaselineTracker,
    bpm_baseline_tracker: BpmBaselineTracker,
    detector: PreSyncopeDetector,
    bpm_detector: BpmFaintingDetector,
    spo2: float,
    bpm: float,
    timestamp: float,
) -> dict[str, Any]:
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


class VitalStreamSession:
    """Isolated trackers for one patient/session (no module-level globals)."""

    def __init__(self) -> None:
        self.baseline_tracker = SpO2BaselineTracker()
        self.bpm_baseline_tracker = BpmBaselineTracker()
        self.detector = PreSyncopeDetector()
        self.bpm_detector = BpmFaintingDetector()

    def process(self, spo2: float, bpm: float, timestamp: float) -> dict[str, Any]:
        return _clinical_dict(
            self.baseline_tracker,
            self.bpm_baseline_tracker,
            self.detector,
            self.bpm_detector,
            spo2,
            bpm,
            timestamp,
        )


class VitalSessionState:
    def __init__(self) -> None:
        self.stream = VitalStreamSession()
        self.last_timestamp: float = -1.0
        self.last_orchestrate_mono: float = 0.0
        self.last_sample: dict[str, Any] = {}
        # After a successful auto-orchestration at full escalation, block repeats until
        # max(spo2_score, bpm_score) drops below ORCH_VITAL_ORCHESTRATE_MIN_CHANNEL_SCORE (default 1.0).
        self._orchestration_latched: bool = False

    def reset_clinical_trackers(self) -> None:
        """New baselines & empty pattern history; keeps last bpm/spo2 for live-vitals display."""
        self.stream = VitalStreamSession()
        self.last_timestamp = -1.0
        # Intentionally do NOT clear _orchestration_latched — prevents alert spam after session_reset.
        if self.last_sample:
            self.last_sample = {
                "bpm": self.last_sample.get("bpm"),
                "spo2": self.last_sample.get("spo2"),
                "timestamp": self.last_sample.get("timestamp"),
                "clinical": None,
                "trackers_reset": True,
            }

    def apply_sample(
        self,
        *,
        spo2: float,
        bpm: float,
        timestamp: float,
        clinical_threshold: float,
        orchestrate_cooldown_sec: float,
        should_orchestrate: bool,
        orchestrate_fn: Callable[[dict[str, Any]], dict[str, Any]],
        build_orchestrator_payload: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """
        orchestrate_fn: e.g. submit_orchestrator_task or run_simulation_orchestrator.
        build_orchestrator_payload: receives clinical dict, returns orchestrator payload.
        """
        if timestamp <= self.last_timestamp:
            return {
                "ok": True,
                "deduplicated": True,
                "reason": "timestamp_not_increasing",
                "last_timestamp": self.last_timestamp,
                "clinical": self.last_sample.get("clinical"),
                "orchestrator_triggered": False,
            }

        self.last_timestamp = timestamp
        clinical = self.stream.process(spo2, bpm, timestamp)
        self.last_sample = {
            "bpm": bpm,
            "spo2": spo2,
            "timestamp": round(timestamp, 3),
            "clinical": clinical,
        }

        score = float(clinical.get("score", 0.0))
        score_above_threshold = score >= clinical_threshold
        lo = _orchestrate_min_channel_score()
        try:
            ss = float(clinical.get("spo2_score", 0.0))
            bs = float(clinical.get("bpm_score", 0.0))
        except (TypeError, ValueError):
            ss, bs = 0.0, 0.0
        max_ch = max(ss, bs)

        if self._orchestration_latched and max_ch < lo - 1e-9:
            self._orchestration_latched = False

        channel_gate = max_ch >= lo - 1e-9
        orchestrate_eligible = channel_gate and not self._orchestration_latched

        orch_result: dict[str, Any] | None = None
        triggered = False
        session_reset = False

        if orchestrate_eligible and should_orchestrate:
            now_m = time.monotonic()
            if now_m - self.last_orchestrate_mono >= orchestrate_cooldown_sec:
                orch_result = orchestrate_fn(build_orchestrator_payload(clinical))
                self.last_orchestrate_mono = time.monotonic()
                triggered = True
                self._orchestration_latched = True
                if _should_reset_trackers_after_run(clinical, orch_result):
                    self.reset_clinical_trackers()
                    session_reset = True

        suppressed_repeat = bool(channel_gate and self._orchestration_latched and not triggered)

        return {
            "ok": True,
            "deduplicated": False,
            "bpm": bpm,
            "spo2": spo2,
            "timestamp": round(timestamp, 3),
            "clinical": clinical,
            "clinical_anomaly": channel_gate,
            "clinical_orchestrate_eligible": orchestrate_eligible,
            "clinical_score_above_threshold": score_above_threshold,
            "clinical_channel_gate": channel_gate,
            "clinical_escalation_min_channel_score": lo,
            "clinical_threshold": clinical_threshold,
            "orchestration_latch_active": self._orchestration_latched,
            "orchestration_suppressed_repeat": suppressed_repeat,
            "orchestrator_triggered": triggered,
            "orchestrator_result": orch_result,
            "clinical_session_reset": session_reset,
        }


_lock = threading.Lock()
_states: dict[tuple[str, str], VitalSessionState] = {}


def get_vital_session_state(user_id: str, session_id: str) -> VitalSessionState:
    key = (str(user_id or "1"), str(session_id or "default-session"))
    with _lock:
        if key not in _states:
            _states[key] = VitalSessionState()
        return _states[key]


def get_last_simulated_vitals(user_id: str, session_id: str) -> dict[str, Any]:
    key = (str(user_id or "1"), str(session_id or "default-session"))
    with _lock:
        st = _states.get(key)
        if not st or not st.last_sample:
            return {"ok": False}
        s = st.last_sample
        return {
            "ok": True,
            "heart_rate": int(round(s["bpm"])),
            "blood_oxygen": int(round(s["spo2"])),
            "clinical": s.get("clinical"),
            "timestamp": s.get("timestamp"),
        }


def default_clinical_threshold() -> float:
    return float(os.getenv("ORCH_VITAL_CLINICAL_THRESHOLD", "0.5"))


def default_orchestrate_cooldown_sec() -> float:
    return float(os.getenv("ORCH_VITAL_ORCH_COOLDOWN_SEC", "45"))


def _vital_auto_reset_enabled() -> bool:
    return os.getenv("ORCH_VITAL_AUTO_RESET", "1").strip().lower() not in ("0", "false", "no")


def _orchestrate_min_channel_score() -> float:
    """Orchestrate when max(spo2_score, bpm_score) >= this value (default 1.0 = CRITICAL channel only)."""
    return float(os.getenv("ORCH_VITAL_ORCHESTRATE_MIN_CHANNEL_SCORE", "1.0"))


def _should_reset_trackers_after_run(
    clinical: dict[str, Any],
    orch_result: dict[str, Any] | None,
) -> bool:
    """Break feedback loops: fresh baselines after critical vitals or critical pipeline."""
    if not _vital_auto_reset_enabled():
        return False
    if orch_result and str(orch_result.get("state", "")).lower() == "critical_emergency":
        return True
    spo = str(clinical.get("spo2_state", "")).upper()
    bpm = str(clinical.get("bpm_state", "")).upper()
    if "CRITICAL" in spo or "CRITICAL" in bpm:
        return True
    try:
        if float(clinical.get("score", 0.0)) >= 0.99:
            return True
    except (TypeError, ValueError):
        pass
    return False
