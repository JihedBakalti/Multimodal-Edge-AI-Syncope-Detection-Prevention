# Medical Assistant Agent (InterSense / ELIZA architecture)

Voice-first medical **assistant** and **orchestration** stack for syncope / fainting risk workflows: wearable vitals, optional camera CV, voice check-ups, LLM reasoning with optional RAG, TTS, alerts, and a clinician **Platform** (Django + React).

> **Not a medical device.** This repository is for research, education, and integration prototyping. It does not replace emergency services or professional care.

---

## Solution overview (see `ELIZA.html`)

[`ELIZA.html`](ELIZA.html) is a **static SVG decision map** of the intended system: triggers, branching, convergence on voice check-up, logging, and the doctor dashboard. Open it in a browser (wide viewport or print to PDF).

**Naming**

- The diagram uses **“ELIZA”** as the orchestrator label. In code, the hands-free wake word is **“Elysa”** (OpenWakeWord + `Elysa/Elysa.onnx`). The diagram trigger text matches that in the HTML (`Say "Elysa"`).

**Where the diagram is aspirational vs literal**

- **RAG**: Retrieval uses **Qdrant** + embeddings (see `medical_index.py`, `medical_assistant.py`), not a generic “any vector DB” requirement. If Qdrant or the collection is missing, the assistant can still run with RAG disabled.
- **Multilingual**: Supported languages depend on STT/TTS and app configuration (e.g. Streamlit `app.py` language map); not “all languages” in the strict sense.
- **Wearable ML path**: Vital anomaly scoring in the simulator path uses rules under `MODELS/Vital Signals Anomaly Detection/` (e.g. `realtime_monitoring.py`), not necessarily every separate ML stack named on the diagram at once.
- **Platform ingest**: Orchestrator pushes events to the Platform API when `PLATFORM_INGEST_URL` is set (default base in code points at the Django service).

---

## Main components

| Area | Role |
|------|------|
| [`medical_assistant.py`](medical_assistant.py) | Core loop: voice command `\voice`, transcription, optional **RAG** (Qdrant), Gemini, ElevenLabs TTS, orchestration, camera/voice/vital hooks. |
| [`Elysa/wakeword_service.py`](Elysa/wakeword_service.py) | Background listener for **“Elysa”**; queues the same `\voice` path as the UI mic. |
| [`medical_index.py`](medical_index.py) | Build/populate Qdrant from `dataset/` for RAG (`hannibal_kb` collection). |
| [`orchestrator_api.py`](orchestrator_api.py) | **FastAPI**: simulation and vital-sample endpoints; optional push to Platform ingest. |
| [`vital_sample_session.py`](vital_sample_session.py) | Session state for vital streaming / gating / orchestration latch (env-tunable). |
| [`MODELS/`](MODELS/) | Subprojects: vital anomaly helpers, voice anomaly, YOLO person/camera scan, syncope face, body fall, etc. |
| [`Platform/backend/`](Platform/backend/) | **Django REST** + Channels: accounts, clinical logs, messaging, reports, ingestion from orchestrator. |
| [`Platform/frontend/`](Platform/frontend/) | **Vite + React** “InterSense Platform” portal (doctor/patient dashboards). |
| [`app.py`](app.py) | Optional **Streamlit** UI wiring `medical_assistant` helpers. |

---

## Typical local ports

| Service | Default | Notes |
|---------|---------|--------|
| Platform Django API | **8100** | Frontend expects `http://127.0.0.1:8100/api` unless `VITE_PLATFORM_API_BASE` is set (`Platform/frontend/src/services/api.js`). |
| Platform Vite dev | **5174** | `Platform/frontend/vite.config.js`; CORS allows 5173/5174 in `Platform/backend/config/settings.py`. |
| Qdrant | **6333** | `medical_index.py` / env `QDRANT_HOST`. |
| Orchestrator FastAPI | *(you choose)* | Run e.g. `uvicorn orchestrator_api:app --host 127.0.0.1 --port 9000` and point clients at that port. |

More deployment notes: [`Platform/DEPLOYMENT.md`](Platform/DEPLOYMENT.md).

---

## Quick start (high level)

1. **Python env** — Install project dependencies (see any `requirements.txt` you use for this repo).
2. **`.env`** — API keys and URLs (Gemini, ElevenLabs, Qdrant, Platform ingest token, etc.). Repo `.env` is gitignored.
3. **RAG (optional)** — Run Qdrant; execute `medical_index.py` to load `dataset/` into the collection.
4. **Assistant** — Run `medical_assistant.py` (or Streamlit `streamlit run app.py`).
5. **Platform** — From `Platform/backend`: migrate/runserver on **8100**; from `Platform/frontend`: `npm install` / `npm run dev` (port **5174**).
6. **Orchestrator** — Run `uvicorn orchestrator_api:app --reload` on your chosen port; set `PLATFORM_INGEST_URL` if the Django server should receive events.

---

## Diagram maintenance (`ELIZA.html`)

The SVG is hand-authored. If arrows or copy drift from the code, prefer updating the HTML to match this README and `medical_assistant.py` / `orchestrator_api.py`. Known fragile spots: long dashed “logging” lines and diamond decision coordinates when box sizes change.

---

## License / ethics

Treat voice data, transcripts, and clinical logs as **sensitive**. Use secure transport, access control, and retention policies appropriate to your jurisdiction before any real patient use.
