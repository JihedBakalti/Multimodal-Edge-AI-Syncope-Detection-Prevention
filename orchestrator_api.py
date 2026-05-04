from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import os
import urllib.request
import time
import hashlib
from datetime import datetime, timezone

from medical_assistant import run_simulation_orchestrator
import vital_sample_session as vsess


class SimulationRequest(BaseModel):
    heart_rate: int = Field(..., ge=40, le=180)
    blood_oxygen: int | None = Field(None, ge=70, le=100)
    prefer_simulator_vitals: bool = False
    anomaly_value: float = 0.0
    dl_risk_score: float = 0.0
    wearable_anomaly: bool
    human_detected: bool
    fainting_detected: bool
    language: str = "en"
    user_id: str = "1"
    session_id: str = "default-session"


class VitalSampleRequest(BaseModel):
    heart_rate: int = Field(..., ge=40, le=180)
    blood_oxygen: int = Field(..., ge=70, le=100)
    timestamp: float | None = None
    user_id: str = "1"
    session_id: str = "default-session"
    auto_orchestrate: bool = True
    clinical_threshold: float | None = Field(None, ge=0.0, le=1.0)
    human_detected: bool = False
    fainting_detected: bool = False


app = FastAPI(title="InterSense Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _push_to_platform(payload: dict) -> None:
    url = os.getenv("PLATFORM_INGEST_URL", "http://127.0.0.1:8100/api/ingestion/orchestrator-event/")
    token = os.getenv("PLATFORM_INGEST_TOKEN", "platform-dev-token")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-Platform-Ingest-Token": token},
    )
    retries = int(os.getenv("PLATFORM_INGEST_RETRIES", "3"))
    timeout_sec = float(os.getenv("PLATFORM_INGEST_TIMEOUT_SEC", "4"))
    backoff = float(os.getenv("PLATFORM_INGEST_BACKOFF_SEC", "1.0"))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec):
                return
        except Exception as exc:
            if attempt == retries - 1:
                print(f"[Platform] ingest failed: {exc}")
            else:
                time.sleep(backoff * (attempt + 1))


@app.post("/api/simulate")
def simulate(payload: SimulationRequest):
    input_payload = payload.model_dump()
    result = run_simulation_orchestrator(input_payload)
    timestamp = datetime.now(timezone.utc).isoformat()
    idempotency_key = hashlib.sha256(
        f"{input_payload.get('user_id','1')}|{input_payload.get('session_id','default-session')}|{timestamp}|{result.get('state','unknown')}".encode("utf-8")
    ).hexdigest()
    ingest_payload = {
        "user_id": input_payload.get("user_id", "1"),
        "session_id": input_payload.get("session_id", "default-session"),
        "heart_rate": result.get("effective_heart_rate", input_payload.get("heart_rate")),
        "blood_oxygen": result.get("effective_blood_oxygen"),
        "anomaly_value": input_payload.get("anomaly_value", 0.0),
        "dl_risk_score": max(
            float((result.get("syncope_result") or {}).get("max_risk_score", 0.0)),
            float((result.get("body_fall_result") or {}).get("max_risk_score", 0.0)),
        ),
        "state": result.get("state", "unknown"),
        "message": result.get("message", ""),
        "actions": (result.get("warning_result") or {}).get("actions", []),
        "alert": {
            "type": "critical_escalation",
            "channel": "whatsapp",
            "status": (result.get("alert_result") or {}).get("status", "not_sent"),
            "details": result.get("alert_result", {}),
        }
        if result.get("alert_result")
        else {},
        "model_name": "intersense_orchestrator",
        "threshold": 0.7,
        "timestamp": timestamp,
        "idempotency_key": idempotency_key,
        "metadata": {"orchestrator": "fastapi", "language": input_payload.get("language", "en")},
        "source": "orchestrator_api",
    }
    _push_to_platform(ingest_payload)
    return result


@app.post("/api/vital-sample")
def vital_sample(body: VitalSampleRequest):
    ts = float(body.timestamp) if body.timestamp is not None else time.time()
    thr = (
        float(body.clinical_threshold)
        if body.clinical_threshold is not None
        else vsess.default_clinical_threshold()
    )
    cool = vsess.default_orchestrate_cooldown_sec()
    state = vsess.get_vital_session_state(body.user_id, body.session_id)

    def _build_orch_payload(clinical: dict) -> dict:
        sc = float(clinical.get("score", 0.0))
        return {
            "heart_rate": body.heart_rate,
            "blood_oxygen": body.blood_oxygen,
            "wearable_anomaly": True,
            "prefer_simulator_vitals": True,
            "anomaly_value": sc,
            "dl_risk_score": min(1.0, sc),
            "human_detected": body.human_detected,
            "fainting_detected": body.fainting_detected,
            "language": "en",
            "user_id": body.user_id,
            "session_id": body.session_id,
        }

    out = state.apply_sample(
        spo2=float(body.blood_oxygen),
        bpm=float(body.heart_rate),
        timestamp=ts,
        clinical_threshold=thr,
        orchestrate_cooldown_sec=cool,
        should_orchestrate=body.auto_orchestrate,
        orchestrate_fn=run_simulation_orchestrator,
        build_orchestrator_payload=_build_orch_payload,
    )
    orch = out.get("orchestrator_result")
    if isinstance(orch, dict) and orch:
        timestamp = datetime.now(timezone.utc).isoformat()
        idempotency_key = hashlib.sha256(
            f"{body.user_id}|{body.session_id}|vital-sample|{timestamp}|{orch.get('state','unknown')}".encode(
                "utf-8"
            )
        ).hexdigest()
        ingest_payload = {
            "user_id": body.user_id,
            "session_id": body.session_id,
            "heart_rate": orch.get("effective_heart_rate", body.heart_rate),
            "blood_oxygen": orch.get("effective_blood_oxygen", body.blood_oxygen),
            "anomaly_value": float((out.get("clinical") or {}).get("score", 0.0)),
            "dl_risk_score": max(
                float((orch.get("syncope_result") or {}).get("max_risk_score", 0.0)),
                float((orch.get("body_fall_result") or {}).get("max_risk_score", 0.0)),
            ),
            "state": orch.get("state", "unknown"),
            "message": orch.get("message", ""),
            "actions": (orch.get("warning_result") or {}).get("actions", []),
            "alert": {
                "type": "critical_escalation",
                "channel": "whatsapp",
                "status": (orch.get("alert_result") or {}).get("status", "not_sent"),
                "details": orch.get("alert_result", {}),
            }
            if orch.get("alert_result")
            else {},
            "model_name": "intersense_orchestrator",
            "threshold": thr,
            "timestamp": timestamp,
            "idempotency_key": idempotency_key,
            "metadata": {"orchestrator": "fastapi", "trigger": "vital_sample", "clinical": out.get("clinical")},
            "source": "orchestrator_api",
        }
        _push_to_platform(ingest_payload)
    return out


@app.get("/api/live-vitals")
def live_vitals(user_id: str = "1", session_id: str = "default-session"):
    sim = vsess.get_last_simulated_vitals(user_id, session_id)
    ok_sim = bool(sim.get("ok"))
    return {
        "ok": ok_sim,
        "firebase_vitals": {"ok": False, "message": "Firebase not used in standalone orchestrator_api."},
        "simulator_vitals": sim,
        "effective_heart_rate": sim.get("heart_rate"),
        "effective_blood_oxygen": sim.get("blood_oxygen"),
    }
