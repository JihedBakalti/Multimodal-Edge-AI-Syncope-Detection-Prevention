from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from medical_assistant import run_simulation_orchestrator


class SimulationRequest(BaseModel):
    heart_rate: int = Field(..., ge=40, le=180)
    wearable_anomaly: bool
    human_detected: bool
    fainting_detected: bool
    language: str = "en"


app = FastAPI(title="InterSense Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/simulate")
def simulate(payload: SimulationRequest):
    result = run_simulation_orchestrator(payload.model_dump())
    return result
