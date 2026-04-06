"""
Background wake-word listener for "Elysa" using OpenWakeWord.
Queues the same \\voice command as the Web UI mic button so the main assistant loop
handles STT → RAG → Gemini → ElevenLabs.

Coordinates the microphone with SpeechRecognition: pause wake listening while the
assistant records or plays TTS (call pause_wakeword_mic / resume_wakeword_mic).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

# Audio format (OpenWakeWord expects 16 kHz mono int16, 1280 samples default chunk)
CHUNK = 1280
CHANNELS = 1
RATE = 16000

_ENABLED = False
_stop = threading.Event()
_thread: Optional[threading.Thread] = None
_stream = None
_audio = None
_model = None
_input_queue = None
_get_language: Optional[Callable[[], str]] = None

_mic_lock = threading.Lock()
_session_lock = threading.Lock()
_WAKE_ALLOWED = threading.Event()
_WAKE_ALLOWED.set()

# After TTS, speakers → mic retriggers "Elysa" unless we wait and suppress scoring.
_SUPPRESS_TRIGGERS_UNTIL = 0.0
_resume_timer: Optional[threading.Timer] = None
_resume_timer_lock = threading.Lock()

_THRESHOLD = float(os.getenv("ELYSA_THRESHOLD", "0.02"))
_COOLDOWN_S = float(os.getenv("ELYSA_COOLDOWN", "5.0"))
_RESUME_DELAY_SEC = float(os.getenv("ELYSA_RESUME_DELAY_SEC", "2.5"))
_POST_VOICE_COOLDOWN_SEC = float(os.getenv("ELYSA_POST_VOICE_COOLDOWN_SEC", "6.0"))


def _model_path() -> Path:
    return Path(__file__).resolve().parent / "Elysa.onnx"


def _cancel_resume_timer() -> None:
    global _resume_timer
    with _resume_timer_lock:
        if _resume_timer is not None:
            _resume_timer.cancel()
            _resume_timer = None


def pause_wakeword_mic() -> None:
    """Release the wake-word stream so SpeechRecognition / TTS can use the mic/speaker path."""
    _cancel_resume_timer()
    _WAKE_ALLOWED.clear()
    _release_stream()


def resume_wakeword_mic() -> None:
    """Re-enable wake listening immediately (orchestrator paths, Streamlit, shutdown)."""
    _cancel_resume_timer()
    _WAKE_ALLOWED.set()


def schedule_wakeword_resume_after_voice_session() -> None:
    """
    Call when a \\voice session fully finished (incl. ElevenLabs TTS).
    Waits ELYSA_RESUME_DELAY_SEC before reopening the mic, and ignores wake scores
    until ELYSA_POST_VOICE_COOLDOWN_SEC after *now* so speaker echo cannot retrigger Elysa.
    """
    global _resume_timer, _SUPPRESS_TRIGGERS_UNTIL
    _cancel_resume_timer()
    _WAKE_ALLOWED.clear()
    _release_stream()
    now = time.time()
    with _session_lock:
        _SUPPRESS_TRIGGERS_UNTIL = now + _POST_VOICE_COOLDOWN_SEC

    def _open_mic() -> None:
        _WAKE_ALLOWED.set()

    with _resume_timer_lock:
        _resume_timer = threading.Timer(_RESUME_DELAY_SEC, _open_mic)
        _resume_timer.daemon = True
        _resume_timer.start()


def _release_stream() -> None:
    global _stream
    with _mic_lock:
        if _stream is not None:
            try:
                _stream.stop_stream()
            except Exception:
                pass
            try:
                _stream.close()
            except Exception:
                pass
            _stream = None


def _score_from_prediction(prediction: dict) -> float:
    if not prediction:
        return 0.0
    if "Elysa" in prediction:
        return float(prediction["Elysa"])
    for key, val in prediction.items():
        if "elysa" in str(key).lower():
            return float(val)
    try:
        return float(max(prediction.values()))
    except (TypeError, ValueError):
        return 0.0


def _listener_loop() -> None:
    global _stream, _audio, _model
    import numpy as np
    import pyaudio
    from openwakeword.model import Model

    pa_format = pyaudio.paInt16
    last_trigger = 0.0

    model_path = _model_path()
    if not model_path.is_file():
        print(f"[Elysa] Wake word disabled: missing model file {model_path}")
        return

    try:
        _model = Model(wakeword_models=[str(model_path)])
    except Exception as e:
        print(f"[Elysa] Wake word failed to load model: {e}")
        return

    print(f"[Elysa] Listening for wake word (threshold={_THRESHOLD})…")

    while not _stop.is_set():
        if not _WAKE_ALLOWED.is_set():
            _release_stream()
            time.sleep(0.05)
            continue

        try:
            with _mic_lock:
                if _stream is None:
                    if _audio is None:
                        _audio = pyaudio.PyAudio()
                    _stream = _audio.open(
                        format=pa_format,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                    )
            data = _stream.read(CHUNK, exception_on_overflow=False)
        except Exception:
            _release_stream()
            time.sleep(0.3)
            continue

        try:
            audio_data = np.frombuffer(data, dtype=np.int16)
            prediction = _model.predict(audio_data)
            score = _score_from_prediction(prediction)
        except Exception:
            continue

        now = time.time()
        with _session_lock:
            if now < _SUPPRESS_TRIGGERS_UNTIL:
                continue
        if score <= _THRESHOLD or (now - last_trigger) < _COOLDOWN_S:
            continue

        last_trigger = now
        _WAKE_ALLOWED.clear()
        _release_stream()

        lang = "en"
        if _get_language:
            try:
                lang = _get_language() or "en"
            except Exception:
                lang = "en"

        if _input_queue is not None:
            try:
                _input_queue.put(
                    {"text": "\\voice", "language": lang, "elysa_greeting": True}
                )
                print(f"\n[Elysa] Wake word detected (score={score:.3f}) → voice mode")
            except Exception as e:
                print(f"[Elysa] Failed to queue voice command: {e}")
                _WAKE_ALLOWED.set()

    _release_stream()
    if _audio is not None:
        try:
            _audio.terminate()
        except Exception:
            pass
        _audio = None
    print("[Elysa] Wake listener stopped.")


def start_elysa_wakeword_listener(
    input_queue,
    get_language: Optional[Callable[[], str]] = None,
) -> bool:
    """
    Start the background thread if ELYSA_WAKE_WORD is not '0'/'false' and Elysa.onnx exists.
    Returns True if the thread was started.
    """
    global _ENABLED, _input_queue, _get_language, _thread

    raw = os.getenv("ELYSA_WAKE_WORD", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        print("[Elysa] Wake word disabled (ELYSA_WAKE_WORD).")
        return False

    if not _model_path().is_file():
        print(f"[Elysa] Wake word not started: add custom model at {_model_path()}")
        return False

    _input_queue = input_queue
    _get_language = get_language
    _stop.clear()
    _WAKE_ALLOWED.set()

    _thread = threading.Thread(target=_listener_loop, name="ElysaWakeWord", daemon=True)
    _thread.start()
    _ENABLED = True
    return True


def stop_elysa_wakeword_listener() -> None:
    _stop.set()
    _cancel_resume_timer()
    resume_wakeword_mic()
    _release_stream()
    if _thread and _thread.is_alive():
        _thread.join(timeout=2.0)


def is_wakeword_enabled() -> bool:
    return _ENABLED
