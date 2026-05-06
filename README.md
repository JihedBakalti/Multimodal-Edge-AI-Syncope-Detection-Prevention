# InterSense 🫀🧠 — Adaptive multimodal orchestration for syncope-risk monitoring

**InterSense** is a safety-oriented medical AI orchestration framework developed as coursework for the **AI Integrated Project** at **Esprit School of Engineering**. It explores compute-efficient continuous monitoring through **hierarchical inference orchestration**: wearable vital-sign analysis, computer vision, and an agentic voice pipeline are coordinated so that expensive models run only when clinical signals justify escalation.

> **Disclaimer.** This repository is for **research, education, and integration prototyping**. It is **not** a medical device and does not replace emergency services or professional clinical judgment. Treat voice data, transcripts, and logs as sensitive.

---

## Overview

InterSense targets **continuous syncope-risk supervision** without running heavy deep learning pipelines at full duty cycle. An **adaptive compute strategy** activates downstream stages (camera vision, syncope / fall models, full voice safety flows) only after upstream gates indicate justified risk. That keeps GPU and CPU load comparatively low while preserving **sub-minute** reaction paths for escalations.

The system combines:

- **Wearable vitals** (heart rate, SpO₂) with session-scoped anomaly scoring and orchestration gating  
- **Camera workflows** (presence scan, syncope face route, body fall route) invoked on demand  
- **Agentic voice**: speech recognition, optional **RAG** over medical knowledge (Qdrant), LLM reasoning, TTS, and structured escalation  

A single orchestration path ties these modalities together; see `intersense_orchestrator.py`, `medical_assistant.py`, and `orchestrator_api.py`.

---

## Features

- **Staged inference cascade** — Wearable-driven escalation gates YOLO-style presence scanning, then deeper syncope / fall models only when policy allows.  
- **Agentic voice pipeline** — ASR → (optional) RAG retrieval → LLM reasoning → TTS, with tool-style orchestration hooks where configured.  
- **Wake-word subsystem (“Elysa”)** — OpenWakeWord-based activation with custom `Elysa/Elysa.onnx`; integrates with the same voice entry points as the interactive assistant.  
- **Tiered escalation** — Non-normal orchestrator outcomes can drive voice safety dialogue; **critical** paths can trigger **Twilio WhatsApp** alerts and structured logging for clinician review.  
- **Clinical event ingestion** — Orchestrator events can be posted to the **Platform** Django API with retries and idempotency-oriented payloads (`PLATFORM_INGEST_*`).  
- **Configurable control surface** — Thresholds, cooldowns, latch behaviour, and post-critical silence windows are tunable via **environment variables** (see `vital_sample_session.py` and orchestrator settings).  
- **Operator interfaces** — Optional **Streamlit** UI (`app.py`), root **Vite** simulator frontend (`frontend/`), and the **InterSense Platform** (Django + React under `Platform/`).

---

## Tech stack

### AI / machine learning

| Component | Role in this repo |
|-----------|-------------------|
| **YOLO / camera scanners** | Human presence and routing under `MODELS/YOLO-Face-Person-Detector/` |
| **Syncope face model (V2)** | Deep syncope-risk path under `MODELS/V2 SYNCOPE FACE DETECTION MODEL/` |
| **Body fall detector** | Body-route fall logic under `MODELS/BODY FALL DETECTION/` |
| **Whisper (Groq)** | ASR via Groq `whisper-large-v3` in `medical_assistant.py` when `GROQ_API_KEY` is set |
| **HeartGPT (experimental)** | Cardiac sequence tooling bundled under vital-signs research subtree (`MODELS/Vital Signals Anomaly Detection/.../HeartGPT/`) |
| **TORGO / voice anomaly** | Speech-risk scoring via `voice_scoring.py` and related assets under `MODELS/Voice Anomaly detection/` |
| **OpenWakeWord** | Wake-word runtime for **Elysa** (`Elysa/wakeword_service.py`) |

### Backend and orchestration

| Component | Notes |
|-----------|--------|
| **Python** | Core runtime: `medical_assistant.py`, `intersense_orchestrator.py` |
| **FastAPI** | `orchestrator_api.py` — simulation and `/api/vital-sample` style triggers |
| **Firebase** | Optional live vitals source when enabled (`ORCH_USE_FIREBASE_VITALS`, `FIREBASE_RTDB_URL` in `medical_assistant.py`) |
| **Qdrant** | Vector store for RAG; indexing via `medical_index.py` |
| **Google Gemini / Groq** | LLM backends as configured by API keys in `.env` |

### Infrastructure and integrations

| Component | Notes |
|-----------|--------|
| **Twilio (WhatsApp / SMS)** | Alert dispatch when Twilio env vars are set |
| **ElevenLabs** | TTS for assistant and cached prompts |
| **Docker** | Useful for isolating model workloads; see `Platform/DEPLOYMENT.md` for deployment patterns |
| **Platform ingest API** | Django app under `Platform/backend/apps/ingestion/` |

### Other tools

- **python-dotenv** — Configuration from environment variables  
- **asyncio** — Async orchestration and I/O in the assistant and API layers  

---

## Architecture

InterSense is organised into five functional layers. Deeper, costlier layers run only when upstream evidence supports escalation (see also `SOLUTION_ARCHITECTURE.md`).

```
┌─────────────────────────────────────────────────────┐
│              INTERACTION LAYER                      │
│   Voice / text UI  ·  Wake-word (Elysa)  ·  RAG    │
├─────────────────────────────────────────────────────┤
│             ORCHESTRATION LAYER                     │
│   State machine  ·  Policy  ·  FastAPI (`orchestrator_api`) │
├─────────────────────────────────────────────────────┤
│            CLINICAL SIGNAL LAYER                  │
│    BPM / SpO₂ scoring  ·  Gate / latch / cooldown   │
├─────────────────────────────────────────────────────┤
│         VISION & SAFETY MODELS (on demand)          │
│   YOLO scan  ·  Syncope face  ·  Fall detector      │
│                  Voice safety scoring               │
├─────────────────────────────────────────────────────┤
│          ESCALATION & PERSISTENCE LAYER             │
│   Twilio alerts  ·  Platform ingest  ·  audit logs  │
└─────────────────────────────────────────────────────┘
```

### Decision map (`ELIZA.html`)

[`ELIZA.html`](ELIZA.html) is a **static SVG** overview of triggers, branching, voice check-ups, and dashboard flows. **Naming:** the diagram label **“ELIZA”** refers to the orchestration concept; the hands-free wake phrase in code is **“Elysa”** (OpenWakeWord + `Elysa/Elysa.onnx`). Open the file in a browser (wide layout or print to PDF). When behaviour changes in code, update the SVG to match `medical_assistant.py` / `orchestrator_api.py`.

---

## Directory structure

```
MEDICAL_ASSISSTANT_AGENT/
├── medical_assistant.py       # Voice pipeline, orchestration entrypoints, Twilio hooks
├── intersense_orchestrator.py # State machine and baseline policy
├── orchestrator_api.py        # FastAPI: simulate, vital-sample, ingest hooks
├── vital_sample_session.py    # Per-session vital gating, cooldowns, post-critical silence
├── voice_scoring.py           # TORGO-oriented speech risk scoring for safety dialogue
├── medical_index.py           # Qdrant indexing from `dataset/` for RAG
├── Elysa/
│   ├── wakeword_service.py    # Wake-word listener and audio concurrency
│   └── Elysa.onnx             # Custom wake-word model
├── MODELS/
│   ├── YOLO-Face-Person-Detector/
│   ├── V2 SYNCOPE FACE DETECTION MODEL/
│   ├── BODY FALL DETECTION/
│   ├── Vital Signals Anomaly Detection/
│   └── Voice Anomaly detection/
├── Platform/
│   ├── backend/               # Django REST, ingestion, clinical APIs
│   └── frontend/              # Vite React “InterSense Platform” (doctor / patient UI)
├── frontend/                  # Agent simulator UI (e.g. vital stream controls)
├── dataset/                   # Knowledge-base sources for RAG indexing (optional)
├── ELIZA.html                 # Orchestration decision map (SVG)
├── requirements.txt           # Python dependencies (root)
├── SOLUTION_ARCHITECTURE.md   # Extended architecture narrative
└── README.md
```

---

## Getting started

### Prerequisites

- **Python 3.10+**  
- **Node.js** (for `frontend/` or `Platform/frontend`)  
- **Qdrant** (local or remote) if using RAG  
- **Twilio** account and numbers if using WhatsApp alerts  
- **API keys** as required: Gemini and/or Groq, ElevenLabs, optional Qdrant API key  

### Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd MEDICAL_ASSISSTANT_AGENT
pip install -r requirements.txt
```

Create a `.env` file in the project root (see table below). There is no committed `.env`; copy variables from your team’s secure template or build from the defaults documented in source.

### Key environment variables

| Variable | Description |
|----------|-------------|
| `ORCH_VITAL_CLINICAL_THRESHOLD` | Minimum combined clinical score before escalation logic considers orchestration (see `vital_sample_session.py`). |
| `ORCH_VITAL_ORCH_COOLDOWN_SEC` | Minimum seconds between orchestration attempts per session. |
| `ORCH_VITAL_ORCHESTRATE_MIN_CHANNEL_SCORE` | Channel score threshold (SpO₂ / HR) for auto-orchestration eligibility (default **1.0** = CRITICAL channel). |
| `ORCH_VITAL_POST_CRITICAL_SILENCE_SEC` | After a **critical_emergency** auto-orchestration, wall-clock pause before another auto-orchestration (default **90**). |
| `ORCH_VITAL_AUTO_RESET` | Whether clinical trackers reset after critical paths (`1` / `true` vs `0` / `false`). |
| `PLATFORM_INGEST_URL` | Base URL for orchestrator event ingestion (default in code targets local Django). |
| `PLATFORM_INGEST_TOKEN` | Bearer-style token expected by the Platform ingest endpoint. |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `TWILIO_TO_NUMBER` | WhatsApp / messaging alerts. |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | LLM providers as used by `medical_assistant.py`. |
| `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | Text-to-speech. |
| `QDRANT_HOST`, `QDRANT_API_KEY` | Vector database for RAG. |

---

## Running locally

```bash
# Orchestration API (choose port; example 9000)
uvicorn orchestrator_api:app --host 127.0.0.1 --port 9000 --reload

# Medical assistant (voice / orchestration loop)
python medical_assistant.py

# Wake-word service
python Elysa/wakeword_service.py
```

**Typical ports**

| Service | Default | Notes |
|---------|---------|--------|
| Platform Django API | **8100** | REST + ingestion; override in frontend env if needed. |
| Platform Vite dev | **5174** | See `Platform/frontend/vite.config.js`. |
| Qdrant | **6333** | HTTP API default. |
| Orchestrator FastAPI | *configurable* | Match the URL used by simulators and integrations. |

Further deployment notes: [`Platform/DEPLOYMENT.md`](Platform/DEPLOYMENT.md).

Optional: `streamlit run app.py` for the Streamlit shell; `npm run dev` inside `frontend/` or `Platform/frontend/` for web UIs.

---

## Operational modes

| Mode | Behaviour |
|------|-----------|
| **Manual clinical assistant** | User-driven voice or text; RAG-enriched answers; heavy vision paths only when invoked by workflow. |
| **Automatic monitoring** | Vital samples via `/api/vital-sample` (simulator or Firebase); scoring, latch, cooldowns, and post-critical silence gate orchestration. |
| **Automatic escalation** | Full multimodal pipeline when policy triggers; critical outcomes can drive WhatsApp dispatch and Platform ingestion. |

---

## Escalation logic (high level)

```
Wearable anomaly / clinical gate satisfied?
        │ YES
        ▼
Human presence? (YOLO scan)
   │ YES                │ NO
   ▼                    ▼
Deep models critical?   Voice safety check → failure / timeout → WhatsApp alert
   │ YES    │ NO
   ▼        ▼
ALERT      Voice safety check
```

Exact branching is implemented in `intersense_orchestrator.py` and `medical_assistant.py` (`run_simulation_orchestrator` and related helpers). Voice safety uses `voice_scoring.py` where configured.

---

## Acknowledgments

This project was completed under the guidance of **Prof. Dorsaf Hrizi** and **Prof. Oumayma Guasmi** at Esprit School of Engineering. Medical-domain guidance was provided by **Dr. Abdeljelil Maatougui**.

### Team

| Name |
|------|
| Jihed El Hak Bakalti |
| Hamza Zighni |
| Mohamed Aziz Ayadi |
| Mohamed Malek Manai |
| Sara Bessaad |
| Alaeddine Ben Hassine |
| Cyrine Belfguira |

---

## License

Developed for **academic purposes** as part of the **AI Integrated Project** program at **Esprit School of Engineering**, Tunisia.

Use secure transport, access control, and data-retention practices appropriate to your jurisdiction before any real patient deployment.
