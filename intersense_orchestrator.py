from enum import Enum
from typing import Dict, Any


class InterSenseState(str, Enum):
    NORMAL = "normal"
    CRITICAL_EMERGENCY = "critical_emergency"
    WARNING = "warning"
    NO_ACTION = "no_action"


def build_dynamic_warning_prompt(heart_rate: int) -> str:
    return (
        "SYSTEM ALERT: The user's wearable detected a physiological anomaly. "
        f"Their current simulated heart rate is {heart_rate} BPM. "
        "Initiate a verbal check immediately. Ask the user if they are feeling dizzy, "
        "lightheaded, or experiencing symptoms. If they confirm symptoms or sound "
        "impaired, trigger your WhatsApp alert tool."
    )


def evaluate_simulation_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    heart_rate = int(payload.get("heart_rate", 70))
    anomaly_value = float(payload.get("anomaly_value", 0.0))
    dl_risk_score = float(payload.get("dl_risk_score", 0.0))
    wearable_anomaly = bool(payload.get("wearable_anomaly", False))
    human_detected = bool(payload.get("human_detected", False))
    fainting_detected = bool(payload.get("fainting_detected", False))
    dl_detected_fainting = fainting_detected or dl_risk_score >= 0.7

    if not wearable_anomaly:
        return {
            "state": InterSenseState.NORMAL.value,
            "message": "System Normal. Monitoring...",
            "actions": [],
        }

    if wearable_anomaly and human_detected and dl_detected_fainting:
        return {
            "state": InterSenseState.CRITICAL_EMERGENCY.value,
            "message": "Critical emergency detected by deep learning signal. Dispatching WhatsApp alert now.",
            "actions": ["send_whatsapp_alert"],
            "dl_risk_score": dl_risk_score,
        }

    if wearable_anomaly and human_detected and not dl_detected_fainting:
        return {
            "state": InterSenseState.WARNING.value,
            "message": "Human detected and no critical DL event. Starting proactive voice safety check.",
            "actions": ["wake_voice_agent"],
            "dynamic_prompt": build_dynamic_warning_prompt(heart_rate),
            "dl_risk_score": dl_risk_score,
        }

    return {
        "state": InterSenseState.NO_ACTION.value,
        "message": "Anomaly detected but no human target confirmed. Starting voice safety check.",
        "actions": ["voice_safety_check"],
        "anomaly_value": anomaly_value,
        "heart_rate": heart_rate,
    }
