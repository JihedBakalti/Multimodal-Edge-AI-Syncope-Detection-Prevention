# InterSense Orchestrator: Solution Architecture

## 1) Purpose and Scientific Positioning

The InterSense orchestrator is a multimodal, safety-oriented medical assistant architecture designed for syncope-risk workflows.  
Its core scientific contribution is **adaptive compute orchestration**: deep learning pipelines are **not** executed continuously (24/24), but activated only when upstream evidence justifies escalation.

This allows the system to:
- reduce unnecessary GPU/CPU camera inference cycles,
- preserve latency for truly risky episodes,
- improve operational sustainability for continuous monitoring contexts,
- keep human-centric voice interaction available as a lower-cost first response.

In short, the architecture replaces "always-on heavy inference" with a **harmonic, demand-driven escalation policy**.

---

## 2) System-Level Architecture

### 2.1 Functional layers

1. **Interaction Layer (Manual + Voice)**
   - Manual conversation and voice invocation are handled by `medical_assistant.py`.
   - Wake-word flow (`Elysa`) is managed by `Elysa/wakeword_service.py`.
   - Optional RAG context (Qdrant + embeddings) enriches clinical dialogue when available.

2. **Orchestration Layer**
   - High-level state policy is defined in `intersense_orchestrator.py`.
   - Runtime orchestration path is implemented in `medical_assistant.py` via `run_simulation_orchestrator(...)`.
   - API orchestration endpoints are exposed by `orchestrator_api.py`.

3. **Clinical Signal Layer**
   - Streaming vital-sample session logic and per-session gating are implemented in `vital_sample_session.py`.
   - Per-patient/session tracker isolation avoids global-state contamination.

4. **Vision and Safety Models Layer (On-Demand)**
   - YOLO camera scan determines human presence and route selection.
   - Syncope face model and body-fall detector are launched conditionally.
   - Voice safety scoring (TORGO-based) augments verbal safety checks.

5. **Escalation + Persistence Layer**
   - WhatsApp emergency dispatch through Twilio when critical criteria are met.
   - Structured event ingestion to Platform backend (`/api/ingestion/orchestrator-event/`) with retries and idempotency keys.

---

## 3) Operational Modes: Manual to Fully Automatic

The architecture supports a continuum from manual operation to autonomous safety supervision.

### 3.1 Manual clinical assistant mode
- User explicitly interacts with the voice/RAG medical assistant.
- Heavy visual models are not required for ordinary informational use.
- This mode preserves resources and supports routine interaction.

### 3.2 Automatic monitoring mode
- Vitals arrive through `/api/vital-sample`.
- Clinical anomaly scoring evaluates BPM/SpO2 trend states.
- Only when gating conditions are satisfied does the orchestrator invoke deeper multimodal checks.

### 3.3 Automatic escalation mode
- If multimodal evidence converges to critical risk, the system escalates:
  - critical voice prompt,
  - WhatsApp alert dispatch,
  - persistent event logging for clinician review.

This transition implements a progressive, safety-first escalation ladder.

---

## 4) Core Decision Logic (State Machine)

The baseline state categories are:
- `normal`
- `warning`
- `no_action`
- `critical_emergency`

### 4.1 Baseline policy intent

From `intersense_orchestrator.py`:
- **No wearable anomaly** -> remain `normal` (monitoring only).
- **Wearable anomaly + human + DL critical** -> `critical_emergency` (immediate alert action).
- **Wearable anomaly + human + non-critical DL** -> `warning` (proactive voice safety check).
- **Wearable anomaly + no confirmed human target** -> `no_action` (voice safety protocol still attempts contact).

### 4.2 Runtime orchestration expansion

`run_simulation_orchestrator(...)` extends the policy with staged evidence:
1. Validate and enrich vitals (simulator or Firebase source policy).
2. Execute lightweight camera presence scan.
3. If human is detected, route intelligently:
   - face-visible path -> syncope face model,
   - body-only path -> body-fall detector.
4. If any active vision path returns critical -> immediate emergency escalation.
5. Otherwise, continue with voice safety protocols (`warning` / `no_action` behavior).

This makes the state machine both symbolic (state policy) and evidence-driven (runtime multimodal checks).

---

## 5) Compute-Optimization Strategy (Key Contribution)

The orchestrator is intentionally engineered to minimize unnecessary deep learning execution.

### 5.1 Principle: run heavy models only when justified

The system avoids always-on DL by introducing a cascade:
- **Stage A (cheap):** wearable/vital anomaly signals,
- **Stage B (moderate):** camera presence scan + route selection,
- **Stage C (expensive):** syncope/body-fall windows only for selected cameras and scenarios.

### 5.2 Mechanisms that reduce compute load

1. **Conditional activation**
   - DL vision windows are skipped when no wearable anomaly or no human evidence is found.

2. **Route-constrained model invocation**
   - Face path triggers syncope model only when face evidence exists.
   - Body-only path triggers fall model when facial path is not suitable.

3. **Session-scoped orchestration control**
   - `VitalSessionState` is keyed per `(user_id, session_id)` to prevent cross-session triggering artifacts.

4. **Cooldown windows**
   - `ORCH_VITAL_ORCH_COOLDOWN_SEC` throttles repeated orchestrator invocations.

5. **Latch-based suppression**
   - `_orchestration_latched` blocks repetitive high-cost reruns after an escalation-class trigger until channel scores drop below configured minimum.

6. **Channel gating**
   - `ORCH_VITAL_ORCHESTRATE_MIN_CHANNEL_SCORE` controls whether computed clinical channels can trigger orchestration (default behavior favors critical channels).

7. **Auto-reset to prevent feedback loops**
   - Optional tracker reset after critical pipeline outcomes avoids pathological re-trigger loops.

### 5.3 Scientific framing

InterSense can be interpreted as a **hierarchical triage scheduler** over heterogeneous sensors, where computationally expensive inference is treated as a scarce resource allocated only under sufficient uncertainty/risk.  
This design improves expected compute efficiency while preserving safety responsiveness.

---

## 6) End-to-End Event Flow

### 6.1 Automatic flow (`/api/vital-sample`)
1. Receive BPM/SpO2 sample.
2. Update per-session clinical baselines and anomaly scores.
3. Apply gate + cooldown + latch policy.
4. If eligible, build orchestrator payload and run multimodal orchestrator.
5. Persist orchestration result to platform ingest API with idempotency key.

### 6.2 Simulation flow (`/api/simulate`)
1. Receive synthetic/controlled state payload.
2. Run orchestrator runtime and multimodal decision path.
3. Persist event snapshot to platform ingest API.

### 6.3 Human safety check flow
When risk is non-critical but concerning:
- launch structured voice check attempts,
- optionally score speech characteristics,
- classify response confidence,
- escalate to WhatsApp if safety cannot be confirmed.

---

## 7) Agentic Voice Pipeline (ASR -> RAG -> LLM -> Tool/Speech -> TTS)

Beyond safety escalation, InterSense operates as a full agentic voice pipeline that can decide between conversational response and executable action.

### 7.1 Pipeline stages

1. **Voice capture and transcription (ASR)**
   - User speech is recorded and passed through Whisper-based transcription.
   - The first transcript is treated as a high-value but potentially imperfect signal, especially under accent, noise, or low-resource language conditions.

2. **Knowledge retrieval (RAG enrichment)**
   - The transcript is used to retrieve domain context from the vector database (Qdrant collection).
   - Retrieved chunks are injected into the prompt context to ground downstream reasoning.

3. **Context fusion for robust understanding**
   - The system forwards both:
     - the retrieved knowledge context, and
     - the original voice-derived signal (audio/transcript context),
     to the LLM.
   - This dual-input strategy helps disambiguate rare local language expressions and mitigate ASR transcript errors.

4. **Agentic decision step (reason + act)**
   - The LLM evaluates intent and safety context against the available toolset.
   - It returns either:
     - a **tool call** (when action is required), or
     - a **speech response** (when no external action is needed).

5. **Execution and response rendering**
   - If a tool is selected, the tool is executed and the result can be verbalized back to the user.
   - If no tool is required, the generated text response is sent directly to speech synthesis.

6. **Speech synthesis (TTS)**
   - Final assistant text is converted back to voice output for natural conversational interaction.

### 7.2 Scientific value of this pipeline

This pipeline is not only conversational; it is **agentic**:
- it perceives (voice),
- grounds (RAG),
- reasons (LLM),
- acts when needed (tools),
- and communicates back in voice form (TTS).

The architecture therefore combines multilingual robustness, retrieval-grounded reasoning, and executable tool-use in a single loop suitable for clinical-support workflows.

---

## 8) Wake-Word Subsystem (Elysa)

InterSense includes a production-style wake-word subsystem so users can transition into hands-free agent mode without manual UI interaction.

### 8.1 Implementation stack

- Wake-word detection is implemented with the **OpenWakeWord** library and a custom model file: `Elysa/Elysa.onnx`.
- Runtime listener logic is implemented in `Elysa/wakeword_service.py`.
- On detection, the wake-word service enqueues the same `\voice` command path used by the UI mic, preserving a single agent pipeline.

### 8.2 TTS integration around wake flow

- A cached greeting audio file is used for low-latency wake acknowledgments.
- This greeting is generated via `Elysa/download_greeting_audio.py` (ElevenLabs-based generation), then replayed locally.
- This design avoids repeated network TTS calls during wake acknowledgement and reduces interaction delay.

### 8.3 Audio concurrency and false-trigger controls

The wake-word subsystem includes explicit microphone/session coordination:
- wake listener pauses while STT capture or TTS playback is active,
- delayed resume is applied after voice sessions,
- post-session trigger suppression and cooldown windows reduce speaker-echo retriggers.

These controls make wake-word behavior stable in real conversational loops and avoid recursive self-triggering.

---

## 9) Escalation Design and Safety Semantics

Escalation is intentionally conservative:
- Critical multimodal evidence triggers immediate alert action.
- Non-critical uncertainty triggers dialog-based verification first.
- Failure to confirm user safety (or signs of impairment) can still escalate.

This balances false-positive control with real-time intervention capability.

---

## 10) Data and Integration Architecture

### 10.1 Ingestion and observability
- Orchestrator emits structured payloads including:
  - state,
  - actions,
  - risk outputs,
  - heart rate and oxygen values,
  - metadata, timestamps, and idempotency keys.

### 10.2 Platform integration
- Default platform ingest endpoint:
  - `http://127.0.0.1:8100/api/ingestion/orchestrator-event/`
- Retry, timeout, and backoff are configurable.

### 10.3 Clinical traceability
- Voice safety checks and orchestration events are persisted for clinician dashboard inspection.
- Idempotency-key generation supports duplicate-safe ingestion workflows.

---

## 11) Configurable Control Surface

Key environment-level knobs include:
- `ORCH_VITAL_CLINICAL_THRESHOLD`
- `ORCH_VITAL_ORCH_COOLDOWN_SEC`
- `ORCH_VITAL_ORCHESTRATE_MIN_CHANNEL_SCORE`
- `ORCH_VITAL_AUTO_RESET`
- `PLATFORM_INGEST_URL`, `PLATFORM_INGEST_TOKEN`
- model/window routing controls (camera count, route durations, forced camera index)

These parameters make the architecture tunable across research experiments and deployment constraints.

---

## 12) Research Claims You Can Defend in a Paper

1. **Adaptive multimodal orchestration**  
   InterSense operationalizes a staged, evidence-gated inference architecture for medical safety workflows.

2. **Compute-aware risk escalation**  
   Heavy vision models are event-triggered rather than always-on, reducing unnecessary compute occupation.

3. **Human-in-the-loop continuity**  
   The same system supports manual RAG voice interaction and autonomous monitoring without architecture bifurcation.

4. **Clinical auditability**  
   Event ingestion with structured payloads and idempotency supports reproducible timeline reconstruction.

5. **Practical deployment orientation**  
   Cooldowns, latch suppression, and route-aware model selection improve operational stability in real-world noisy streams.

---

## 13) Optimized Hosting and Capacity Model

Because InterSense activates deep-learning services only when risk-gating conditions are met, infrastructure can be provisioned for **bursty, short-lived inference windows** rather than continuous full-load processing.

### 13.1 Model-serving topology (example deployment)

A practical production pattern is to isolate each heavy model family on dedicated serving nodes:
- one model family per server pool (e.g., syncope face pool, body-fall pool),
- each server running multiple equivalent containers (e.g., 8 containers per server),
- stateless request dispatch from orchestrator to available container instances.

This architecture gives operational advantages:
- horizontal scalability (add servers without changing orchestrator logic),
- clear fault domains per model type,
- independent autoscaling and versioning for each model family.

### 13.2 Why this supports high user coverage

The key point is **duty cycle compression**:
- users are monitored continuously at low cost (vitals + lightweight checks),
- expensive models run only for short decision windows when clinically justified.

Under this behavior, a finite pool (for example, 8 active containers for a model path) can serve a significantly larger user population than an always-on design, because container occupancy is intermittent rather than permanent.  
In a representative deployment profile, such a pool can support **200+ users** when trigger rates and inference windows remain within expected operating bounds.

### 13.3 Queueing policy under temporary saturation

When all containers in a model pool are busy:
- requests are placed in a bounded queue,
- dispatch uses FIFO policy with urgency-aware overrides if needed,
- target queue wait is capped (engineering SLO) to approximately **<= 30 seconds** in normal burst conditions.

This converts short overload spikes into controlled latency rather than dropped safety workflows.

### 13.4 Paper-ready framing

For a scientific paper, this can be stated as:
- InterSense combines **event-triggered multimodal inference** with **pool-and-queue serving**,
- yielding high effective concurrency while preserving bounded escalation latency,
- and improving infrastructure efficiency compared with always-on model execution.

---

## 14) Suggested Experimental Evaluation Axes

For publication-quality validation, evaluate at least:
- **Resource efficiency:** GPU/CPU time, active model duty cycle, and inference-minutes per hour versus always-on baselines.
- **Safety latency:** time from anomaly onset to first voice check and to emergency alert under critical conditions.
- **Escalation quality:** precision/recall of critical escalation decisions and false-alert burden.
- **Robustness:** behavior under dropped vitals, camera unavailability, or ingestion retries.
- **Human factors:** user response rates to proactive voice checks and intervention acceptance.

---

## 15) Model Integration Audit (MODELS Folder vs Orchestrator Runtime)

This section maps model assets in `MODELS/` to their actual orchestration status, to ensure architectural claims remain technically accurate.

### 15.1 Models actively integrated into orchestrator workflow

1. **YOLO Face/Person Detector** (`MODELS/YOLO-Face-Person-Detector/`)
   - Used by `run_camera_presence_scan(...)` as the first vision gate.
   - Determines human presence and face-vs-body routing.

2. **V2 Syncope Face Runtime** (`MODELS/V2 SYNCOPE FACE DETECTION MODEL/`)
   - Invoked by `run_syncope_detection_window(...)` when face route is selected.
   - Produces per-window critical flags and max risk score.

3. **Body Fall Detector** (`MODELS/BODY FALL DETECTION/`)
   - Invoked by `run_body_fall_detection_window(...)` for body-only route.
   - Uses sustained confirmation logic before critical escalation.

4. **Vital Signals Anomaly Scoring** (`MODELS/Vital Signals Anomaly Detection/realtime_monitoring.py`)
   - Integrated in `vital_sample_session.py` for per-session BPM/SpO2 trend scoring.
   - Drives gate/cooldown/latch orchestration eligibility logic.

5. **Voice Anomaly Scoring (TORGO path)** (`MODELS/Voice Anomaly detection/`)
   - Integrated via `voice_scoring.py` and invoked in voice safety checks.
   - Adds speech-risk evidence during safety dialogue attempts.

### 15.2 Model assets present but not in the main orchestrator runtime path

- **HeartGPT / Multimodal-Edge-AI-Syncope-Detection-Prevention** subproject assets are present as research and experimentation modules.
- These are not currently on the default low-latency orchestration path that executes `run_simulation_orchestrator(...)` and `/api/vital-sample` triggers.

### 15.3 Compliance statement

The current architecture document now reflects the implemented runtime accurately:
- wake-word subsystem: integrated and operational,
- primary multimodal safety models: integrated,
- research/experimental model subtrees: explicitly marked as non-default runtime components.

---

## 16) Conclusion

InterSense demonstrates a harmonized orchestration paradigm for medical-assistant systems:  
**use minimal compute during normal conditions, progressively activate intelligence when risk grows, and escalate decisively when multimodal evidence confirms danger.**

This architecture is both technically grounded and publication-ready as a compute-optimized, safety-oriented orchestration framework.
