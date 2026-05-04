#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("Starting medical_assistant.py...")

# Suppress warnings - these NumPy warnings on Windows are harmless
import os
import sys
import warnings
from datetime import datetime, timezone
import hashlib

print("Imported os, sys, warnings")

# Set environment variable to suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress all warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

print("Loading dependencies...")

try:
    import time
    import asyncio
    import threading
    import json
    import base64
    import queue
    import websockets
    import re
    import ast
    import webbrowser
    import subprocess
    import importlib.util
    import pathlib
    import urllib.request
    print("  [OK] time, asyncio, threading, json, queue, websockets, re, ast, webbrowser")
except ImportError as e:
    print(f"  [FAIL] websockets: {e}")
    print("   Install with: pip install websockets")
    sys.exit(1)
except Exception as e:
    print(f"  [FAIL] Standard libs: {e}")
    sys.exit(1)

try:
    from langdetect import detect, LangDetectException
    print("  [OK] langdetect")
except Exception as e:
    print(f"  [FAIL] langdetect: {e}")
    print("   Install with: pip install langdetect")
    sys.exit(1)

print("  Attempting to import qdrant_client...")
sys.stdout.flush()  # Force output
try:
    from qdrant_client import QdrantClient
    print("  [OK] qdrant_client")
    sys.stdout.flush()
except ImportError as e:
    print(f"  [FAIL] qdrant_client ImportError: {e}")
    print("   Install with: pip install qdrant-client")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"  [FAIL] qdrant_client Exception: {type(e).__name__}: {e}")
    print("   This might be a dependency issue.")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
except BaseException as e:
    print(f"  [FAIL] qdrant_client BaseException: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print("  [OK] sentence_transformers")
except Exception as e:
    print(f"  [WARN] sentence_transformers unavailable: {e}")
    print("   RAG retrieval will be disabled; core voice/orchestrator still works.")
    print("   To re-enable retrieval later: pip install tf-keras sentence-transformers")
    SentenceTransformer = None

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    print("  [OK] google-generativeai")
except Exception as e:
    print(f"  [FAIL] google-generativeai: {e}")
    print("   Install with: pip install google-generativeai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("  [OK] dotenv")
except Exception as e:
    print(f"  [FAIL] dotenv: {e}")
    print("   Install with: pip install python-dotenv")
    sys.exit(1)

try:
    from groq import Groq
    print("  [OK] groq")
except ImportError as e:
    print(f"  [FAIL] groq: {e}")
    print("   Install with: pip install groq")
    sys.exit(1)

try:
    import speech_recognition as sr
    import io
    print("  [OK] speech_recognition")
except ImportError as e:
    print(f"  [FAIL] speech_recognition: {e}")
    print("   Install with: pip install SpeechRecognition pyaudio")
    # Don't exit, just disable voice
    sr = None

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.play import play
    print("  [OK] elevenlabs")
except ImportError as e:
    print(f"  [FAIL] elevenlabs: {e}")
    print("   Install with: pip install elevenlabs")
    print(f"  [FAIL] elevenlabs: {e}")
    print("   Install with: pip install elevenlabs")
    ElevenLabs = None

try:
    import pyaudio
    print("  [OK] pyaudio")
except ImportError as e:
    print(f"  [FAIL] pyaudio: {e}")
    print("   Install with: pip install pyaudio")
    pyaudio = None

try:
    import pygame
    print("  [OK] pygame")
except ImportError as e:
    print(f"  [FAIL] pygame: {e}")
    print("   Install with: pip install pygame")
    pygame = None

try:
    from twilio.rest import Client
    print("  [OK] twilio")
except ImportError as e:
    print(f"  [FAIL] twilio: {e}")
    # print("   Install with: pip install twilio")
    Client = None

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    print("  [OK] fastapi/uvicorn")
except ImportError as e:
    print(f"  [WARN] fastapi/uvicorn not installed: {e}")
    FastAPI = None
    CORSMiddleware = None
    BaseModel = None
    Field = None
    uvicorn = None

from intersense_orchestrator import evaluate_simulation_state

try:
    from Elysa.wakeword_service import (
        pause_wakeword_mic,
        resume_wakeword_mic,
        schedule_wakeword_resume_after_voice_session,
        start_elysa_wakeword_listener,
    )
except ImportError:
    def pause_wakeword_mic():
        pass

    def resume_wakeword_mic():
        pass

    def schedule_wakeword_resume_after_voice_session():
        pass

    def start_elysa_wakeword_listener(*args, **kwargs):
        return False


# ----------------------------
# Agent Actions & Tools (Inlined)
# ----------------------------

def play_alert_sound(**kwargs):
    """Play a sound effect using Pygame."""
    # Flexible argument handling
    level = kwargs.get("level") or kwargs.get("alert_type") or kwargs.get("type") or "high"
    print(f"🔊 [ACTION] Playing {level} alert sound")
    
    # Check pygame mixer
    if pygame:
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
                print("  [Init] Pygame Mixer Initialized for Action")
            except Exception as e:
                print(f"⚠️ Mixer Init Failed: {e}")
                return {"status": "failed", "error": "Mixer init failed"}

        # Try to find a sound file
        sound_path = os.path.join("frontend", "assets", "alert.mp3") 
        if os.path.exists(sound_path):
            try:
                # Use Sound object for SFX so it doesn't conflict with music (TTS)
                effect = pygame.mixer.Sound(sound_path)
                effect.set_volume(0.5) # Sentient volume
                effect.play()
            except Exception as e:
                print(f"⚠️ Sound Error: {e}")
        else:
            print(f"⚠️ Sound file missing: {sound_path}")
            # Fallback beep
            print("\a") # System bell
    else:
        print("⚠️ Pygame mixer not initialized")

    return {"status": "played", "level": level}

def send_whatsapp_alert(message_text="EMERGENCY: The user needs help!", **kwargs):
    """Send a WhatsApp alert to the family."""
    print(f"📱 [ACTION] Sending WhatsApp alert: {message_text}")
    
    if not Client:
        print("⚠️ Twilio library not installed.")
        return {"status": "failed", "error": "Twilio not installed"}

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number = os.getenv("TWILIO_TO_NUMBER")
    
    if not all([account_sid, auth_token, from_number, to_number]):
        print("⚠️ Twilio credentials missing in .env")
        return {"status": "failed", "error": "Twilio credentials missing"}
    
    try:
        client = Client(account_sid, auth_token)
        # Use simple free-form text instead of content templates for flexibility
        message = client.messages.create(
            from_=from_number,
            body=message_text,
            to=to_number
        )
        print(f"✅ WhatsApp Message sent! SID: {message.sid}")
        return {"status": "sent", "sid": message.sid}
    except Exception as e:
        print(f"⚠️ Twilio Error: {e}")
        return {"status": "failed", "error": str(e)}



# Register Actions
ACTIONS = {
    "send_whatsapp_alert": send_whatsapp_alert
}

# Tool Definitions for LLM
TOOLS = [
    {
        "name": "send_whatsapp_alert",
        "description": "Send a WhatsApp alert to family members. You MUST provide a 'message_text' describing the situation (e.g., 'User is feeling faint').",
        "parameters": {
            "type": "object",
            "properties": {
                "message_text": {
                    "type": "string",
                    "description": "The specific message to send to the family."
                }
            },
            "required": ["message_text"]
        }
    },

]

def execute_action(action_name, args):
    """Execute a registered action."""
    if action_name in ACTIONS:
        try:
            return ACTIONS[action_name](**args)
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    return {"status": "failed", "error": "Unknown action"}

def handle_agent_response(response):
    """Normalize agent response."""
    if isinstance(response, dict) and response.get("type") == "action":
        return {"type": "action", "name": response.get("name"), "args": response.get("args", {})}
    return {"type": "speech", "text": response.get("text", str(response))}


def parse_agent_json(raw_reply):
    """Parse raw LLM output as action/speech JSON."""
    parsed_response = None

    def try_parse(text):
        try:
            return json.loads(text)
        except Exception:
            pass

        text_py = text.replace("false", "False").replace("true", "True").replace("null", "None")
        try:
            val = ast.literal_eval(text_py)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        return None

    parsed_response = try_parse(raw_reply)
    if not parsed_response:
        match = re.search(r"\{.*\}", raw_reply, re.DOTALL)
        if match:
            parsed_response = try_parse(match.group(0))

    if not parsed_response:
        return {"type": "speech", "text": raw_reply}
    return parsed_response

print("Loading environment variables...")
load_dotenv()
print("[OK] Environment loaded")

# ----------------------------
# Setup
# ----------------------------
# Qdrant configuration - can use local or cloud
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")  # Local default
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # Optional for local
COLLECTION_NAME = "hannibal_kb"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Xb7hH8MSUJpSbSDYk0k2")

# ----------------------------
# WebSocket Server & Input Handling
# ----------------------------
CONNECTED_CLIENTS = set()
WS_PORT = 8765
INPUT_QUEUE = queue.Queue()
ORCHESTRATOR_QUEUE = queue.Queue()
WS_LOOP = None

# Last UI-selected voice language (WebSocket); used for "Elysa" wake → \voice
_VOICE_CLIENT_LANG = ["en"]
_VOICE_CLIENT_LANG_LOCK = threading.Lock()


def get_client_voice_language():
    with _VOICE_CLIENT_LANG_LOCK:
        return _VOICE_CLIENT_LANG[0]


def broadcast_state(state, audio_level=0.0):
    """Send state update to all connected clients in a thread-safe way."""
    if not CONNECTED_CLIENTS or WS_LOOP is None:
        return
    
    message = json.dumps({
        "state": state,
        "audioData": audio_level
    })
    
    async def _send():
        websockets.broadcast(CONNECTED_CLIENTS, message)

    try:
        # Schedule the broadcast on the WebSocket thread's event loop
        asyncio.run_coroutine_threadsafe(_send(), WS_LOOP)
    except Exception as e:
        print(f"⚠️ Broadcast Error: {e}")

async def ws_handler(websocket):
    print(f" [WS] Client connected")
    CONNECTED_CLIENTS.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                lang = data.get("language")
                if isinstance(lang, str) and lang.strip():
                    with _VOICE_CLIENT_LANG_LOCK:
                        _VOICE_CLIENT_LANG[0] = lang.strip().lower()[:16]
                if msg_type == "language":
                    continue
                if msg_type == "command":
                    cmd = data.get("content")
                    language = data.get("language", "en")  # Default to english
                    print(f" [WS] Remote command received: {cmd} (lang: {language})")
                    INPUT_QUEUE.put({"text": cmd, "language": language})
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f" [WS] Error parsing: {e}")
    except Exception as e:
        print(f" [WS] Connection closed: {e}")
    finally:
        CONNECTED_CLIENTS.discard(websocket)
        print(f" [WS] Client disconnected")

async def run_ws_server():
    print(f" [WS] Starting WebSocket server on port {WS_PORT}...")
    async with websockets.serve(ws_handler, "127.0.0.1", WS_PORT):
        await asyncio.Future()  # run forever

def start_websocket_server():
    """Start WS server in a daemon thread."""
    def thread_run():
        global WS_LOOP
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        WS_LOOP = loop
        
        loop.run_until_complete(run_ws_server())
    
    t = threading.Thread(target=thread_run, daemon=True)
    t.start()
    time.sleep(1) # Give it a sec to start

if not GEMINI_API_KEY:
    print("[ERROR] GEMINI_API_KEY not found!")
    print("   Please set it in .env file: GEMINI_API_KEY=your_key_here")
    # sys.exit(1) # Don't exit yet to allow import, but will fail on query

# Initialize Qdrant client with error handling
try:
    print("[INFO] Connecting to Qdrant...")
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(url=QDRANT_HOST)
    
    # Test connection
    collections = client.get_collections()
    print(f"[OK] Connected to Qdrant at {QDRANT_HOST}\n")
except Exception as e:
    print(f"[ERROR] Cannot connect to Qdrant at {QDRANT_HOST}")
    print(f"   Error: {e}")
    print("\n[INFO] Solutions:")
    print("   1. Check internet connection")
    print("   2. Verify QDRANT_HOST in .env includes port (e.g. :6333)")
    print("   3. Verify API key")
    sys.exit(1)

# ----------------------------
# Embedding Model (Local, Free)
# ----------------------------
model = None
if SentenceTransformer is not None:
    try:
        print("[INFO] Loading embedding model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[OK] Embedding model loaded.\n")
    except Exception as e:
        print(f"[WARN] Embedding model unavailable: {e}")
        print("       Retrieval disabled; assistant will continue without RAG context.\n")
        model = None
else:
    print("[WARN] Embedding model skipped (sentence_transformers unavailable).")
    print("       Retrieval disabled; assistant will continue without RAG context.\n")

def get_embedding(text):
    """Generate embedding using local SentenceTransformer model."""
    if model is None:
        return None
    return model.encode(text).tolist()

# ----------------------------
# LLM Setup (Gemini)
# ----------------------------
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def query_gemini(messages, voice_wav_bytes=None, voice_language=None):
    """Query Gemini API."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not configured."

    # Extract system message
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
    
    # Filter for chat messages
    chat_messages = [m for m in messages if m["role"] != "system"]
    
    if not chat_messages:
        return "Error: No user message found."
    
    # Configure model with system instruction
    # Using gemini-2.5-flash for speed/cost efficiency
    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_msg)
    
    # Split into history and last message
    last_msg = chat_messages[-1]
    history_msgs = chat_messages[:-1]
    
    # Build Gemini history format
    gemini_history = []
    for m in history_msgs:
        role = "user" if m["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [m["content"]]})
        
    try:
        chat = model.start_chat(history=gemini_history)
        if voice_wav_bytes:
            # Give Gemini the transcript + original WAV so it can correct STT mistakes.
            language_hint = voice_language or "auto"
            text_part = (
                f"Voice language hint: {language_hint}\n"
                "User transcript (may contain STT mistakes):\n"
                f"{last_msg['content']}\n\n"
                "Please use the attached WAV audio as primary evidence when unclear."
            )
            audio_part = {
                "inline_data": {
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(voice_wav_bytes).decode("utf-8"),
                }
            }
            response = chat.send_message([text_part, audio_part])
        else:
            response = chat.send_message(last_msg["content"])
        text = response.text.strip()
        # Clean up markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```python"):
            text = text[9:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    except google_exceptions.InvalidArgument as e:
        return f"Gemini Error (Invalid Argument): {e}"
    except Exception as e:
        return f"Gemini Error: {e}"

# ----------------------------
# Retriever
# ----------------------------
def retrieve_docs(user_query, k=3):
    """Retrieve relevant documents from Qdrant vector database."""
    query_vec = get_embedding(user_query)
    if query_vec is None:
        return []
    try:
        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            print(f"⚠️ Collection '{COLLECTION_NAME}' not found!")
            print(f"   Available collections: {collections}")
            print(f"   Please run: python farm_index_data.py")
            return []
        
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=k
        ).points
        return [r.payload["content"] for r in results]
    except Exception as e:
        print(f"⚠️ Error retrieving documents: {e}")
        return []

        return []

# ----------------------------
# Voice Transcription (Groq)
# ----------------------------
def listen_and_transcribe(return_audio=False):
    """Listen to microphone and transcribe using Groq Whisper."""
    if not sr:
        print("❌ SpeechRecognition not installed.")
        return None
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env.")
        return None

    r = sr.Recognizer()
    # Balance speed + sentence completeness (avoid mid-sentence cutoffs)
    r.energy_threshold = 300  # Default 300, can adjust dynamic
    r.pause_threshold = 1.0   # Allow longer short pauses while speaking
    r.non_speaking_duration = 0.7  # Keep stream open a bit longer before endpointing
    r.dynamic_energy_threshold = True

    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        with sr.Microphone() as source:
            broadcast_state("listening")
            print("\n🎤 Listening... (Speak now!)")
            # Slightly longer calibration improves stability in noisy rooms
            r.adjust_for_ambient_noise(source, duration=0.3)
            
            # Listen (stops when silence is detected)
            try:
                # Reduced phrase_time_limit to avoid broken open mics
                audio_data = r.listen(source, timeout=10, phrase_time_limit=15)
                # Immediately switch to thinking state to show user we heard them
                broadcast_state("thinking")
            except sr.WaitTimeoutError:
                broadcast_state("neutral")  # Reset if they didn't speak
                return None

            print("⏳ Transcribing...")

            # Convert to WAV in memory
            wav_data = audio_data.get_wav_data()
            audio_stream = io.BytesIO(wav_data)
            audio_stream.name = "audio.wav" # Groq needs a filename
            
            transcription = groq_client.audio.transcriptions.create(
                file=("audio.wav", audio_stream),
                model="whisper-large-v3",
                temperature=0,
                response_format="verbose_json",
            )
            transcript_text = transcription.text.strip()
            if return_audio:
                return transcript_text, wav_data
            return transcript_text
            
    except sr.RequestError as e:
        print(f"❌ Microphone error: {e}")
        broadcast_state("neutral")
        return None
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        broadcast_state("neutral")
        return None

# ----------------------------
# Text-to-Speech (ElevenLabs)
# ----------------------------
def speak_response(text):
    """Convert text to speech and play it using Pygame (MP3) in chunks."""
    if not ElevenLabs or not ELEVENLABS_API_KEY:
        return
    
    if not pygame:
        print("⚠️ Pygame not available for playback.")
        return

    # Initialize Pygame Mixer if needed
    if not pygame.mixer.get_init():
        # Lower buffer size for lower latency (default is 4096)
        pygame.mixer.init(buffer=512)

    import re
    # Split text into sentences to play immediately (Time-to-first-byte reduction)
    # This regex splits by . ! ? but keeps the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    try:
        for sentence in sentences:
            # Skip very short fragments that might just be noise
            if len(sentence) < 2:
                continue

            # Generate audio for this sentence
            # We fetch while the previous one might still be playing (overlapping IO/Playback)
            try:
                audio_generator = client.text_to_speech.convert(
                    text=sentence,
                    voice_id=ELEVENLABS_VOICE_ID,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128", 
                )
                
                audio_data = b"".join([chunk for chunk in audio_generator if chunk])
                audio_io = io.BytesIO(audio_data)

                # Wait for previous playback to finish before starting new one
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10) # Check every 10ms for faster response

                # Play current content
                broadcast_state("speaking", audio_level=0.8) # Simulate level
                pygame.mixer.music.load(audio_io)
                pygame.mixer.music.play()

            except Exception as e:
                print(f"⚠️ TTS Error on sentence '{sentence[:10]}...': {e}")
                continue

        # Wait for the final sentence to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        print(f"⚠️ TTS General Error: {e}")
    finally:
        broadcast_state("neutral") # Always reset to neutral

# Cached wake greeting (generate once: python Elysa/download_greeting_audio.py)
ELYSA_WAKE_GREETING_MP3 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Elysa", "elysa_greeting.mp3"
)
WHATSAPP_ALERT_SENT_MP3 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Elysa", "whatsapp_alert_sent.mp3"
)
CRITICAL_FAINTING_ESCALATION_MP3 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Elysa", "critical_fainting_escalation.mp3"
)
_ELYSA_DIR = os.path.dirname(ELYSA_WAKE_GREETING_MP3)
# Single English cached thanks reply (any-language thanks from user → English audio).
THANKS_REPLY_MP3 = os.path.join(_ELYSA_DIR, "thanks_reply_en.mp3")
THANKS_HISTORY_TEXT = "You're welcome. I'm here if you need anything else."
_GRATITUDE_BAD_SUBSTRINGS = (
    "dizzy", "faint", "pain", "hurt", "nausea", "chest", "breath", "bleed",
    "emergency", "help me", "not ok", "not okay", "feel bad", "symptom",
    "دوار", "وجع", "غمى", "إغماء",
)
_GRATITUDE_TOKEN_REQUIRED = frozenset(
    {
        "thank", "thanks", "thx", "ty", "merci", "gracias", "gracia",
        "shukran", "choukran", "شكرا", "شكرًا", "thankyou",
    }
)
_GRATITUDE_ALLOWED_TOKENS = frozenset(
    {
        "thank", "you", "thanks", "thx", "ty", "thankyou", "so", "very", "much",
        "a", "lot", "the", "for", "your", "my", "all", "and", "to", "too",
        "merci", "beaucoup", "bien", "gracias", "gracia", "muchas",
        "shukran", "choukran", "ya", "yes", "ok", "okay", "buddy", "mate",
        "شكرا", "شكرًا", "شكراً", "جزيلا", "يعيشك", "بارك", "الله", "فيك",
    }
)


def _normalize_thanks_text(text):
    t = text.strip().lower()
    t = re.sub(r"[\s,.!?;:،؟]+", " ", t)
    return t.strip()


def is_gratitude_only_message(text):
    """
    True if the user message is a short thanks-only utterance (no medical distress cues).
    Skips LLM; plays cached ElevenLabs-generated MP3 instead.
    """
    if not text or not str(text).strip():
        return False
    raw = str(text).strip()
    if len(raw) > 140:
        return False
    low = raw.lower()
    if any(b in low for b in _GRATITUDE_BAD_SUBSTRINGS):
        return False
    # Arabic / mixed thanks without strict Latin tokenization
    if any(x in raw for x in ("شكرا", "شكرًا", "شكراً", "يعيشك", "بارك الله")):
        if "لكن" in low or " but " in low or "however" in low:
            return False
        if len(raw) <= 90:
            return True
    norm = _normalize_thanks_text(raw)
    if not norm:
        return False
    tokens = re.findall(r"[\w']+|[^\x00-\x7F]+", norm)
    if not tokens:
        return False
    flat = " ".join(tokens)
    if not any(req in flat for req in _GRATITUDE_TOKEN_REQUIRED):
        return False
    for tok in tokens:
        t0 = tok.lower().strip("'")
        if t0 in _GRATITUDE_ALLOWED_TOKENS:
            continue
        if re.fullmatch(r"[\W_]+", tok):
            continue
        return False
    return True


def play_thanks_reply_cached():
    """Play pre-generated English thank-you reply MP3 — no LLM, no live ElevenLabs API."""
    path = THANKS_REPLY_MP3
    if not os.path.isfile(path):
        print(
            f"⚠️ Cached thanks audio missing: {path}\n"
            "   Run: python Elysa/download_greeting_audio.py"
        )
        return
    if not pygame:
        return
    if not pygame.mixer.get_init():
        try:
            pygame.mixer.init(buffer=512)
        except Exception as e:
            print(f"⚠️ Thanks reply: mixer init failed: {e}")
            return
    try:
        broadcast_state("speaking", audio_level=0.8)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"⚠️ Thanks reply playback failed: {e}")
    finally:
        broadcast_state("neutral")


def play_elysa_wake_greeting():
    """Play local MP3 greeting after wake word — no ElevenLabs round-trip."""
    if not os.path.isfile(ELYSA_WAKE_GREETING_MP3):
        print(
            f"⚠️ Elysa greeting audio missing: {ELYSA_WAKE_GREETING_MP3}\n"
            "   Run: python Elysa/download_greeting_audio.py"
        )
        return
    if not pygame:
        return
    if not pygame.mixer.get_init():
        try:
            pygame.mixer.init(buffer=512)
        except Exception as e:
            print(f"⚠️ Elysa greeting: mixer init failed: {e}")
            return
    try:
        broadcast_state("speaking", audio_level=0.8)
        pygame.mixer.music.load(ELYSA_WAKE_GREETING_MP3)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"⚠️ Elysa greeting playback failed: {e}")
    finally:
        broadcast_state("neutral")

def play_critical_fainting_escalation_prompt():
    """Play cached critical-fainting escalation line (no live ElevenLabs call)."""
    if not os.path.isfile(CRITICAL_FAINTING_ESCALATION_MP3):
        print(
            f"⚠️ Cached critical escalation audio missing: {CRITICAL_FAINTING_ESCALATION_MP3}\n"
            "   Run: python Elysa/download_greeting_audio.py"
        )
        return
    if not pygame:
        return
    if not pygame.mixer.get_init():
        try:
            pygame.mixer.init(buffer=512)
        except Exception as e:
            print(f"⚠️ Critical escalation prompt: mixer init failed: {e}")
            return
    try:
        broadcast_state("speaking", audio_level=0.8)
        pygame.mixer.music.load(CRITICAL_FAINTING_ESCALATION_MP3)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"⚠️ Critical escalation prompt playback failed: {e}")
    finally:
        broadcast_state("neutral")


def play_whatsapp_alert_sent_prompt():
    """Play cached WhatsApp alert confirmation (no ElevenLabs call)."""
    if not os.path.isfile(WHATSAPP_ALERT_SENT_MP3):
        print(
            f"⚠️ Cached WhatsApp confirmation missing: {WHATSAPP_ALERT_SENT_MP3}\n"
            "   Run: python Elysa/download_greeting_audio.py"
        )
        return
    if not pygame:
        return
    if not pygame.mixer.get_init():
        try:
            pygame.mixer.init(buffer=512)
        except Exception as e:
            print(f"⚠️ WhatsApp confirmation: mixer init failed: {e}")
            return
    try:
        broadcast_state("speaking", audio_level=0.8)
        pygame.mixer.music.load(WHATSAPP_ALERT_SENT_MP3)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"⚠️ WhatsApp confirmation playback failed: {e}")
    finally:
        broadcast_state("neutral")


# Maximum number of conversation exchanges to keep in history
MAX_HISTORY_EXCHANGES = 5

def get_response(
    user_message,
    history,
    language=None,
    system_prompt_append=None,
    voice_wav_bytes=None,
):
    """Generate response using RAG pipeline."""
    broadcast_state("thinking")
    if not language:
        try:
            language = detect(user_message)
        except LangDetectException:
            language = "en"

    # Retrieve relevant context
    context_docs = retrieve_docs(user_message)
    context_text = "\n".join(context_docs) if context_docs else "No relevant context found."

    # Single English-output persona: understand any input language, always reply in English.
    persona = (
        "You are a Medical Assistant specialized in Sudden Fainting (Syncope). "
        "Your role is to calm the user down and provide clear, step-by-step instructions. "
        "You are reassuring, direct, and professional. "
        "If the user says they feel faint, guide them immediately: sit down, lie down, elevate legs. "
        "Use your tools if necessary (escalation/notification). "
        "Focus ONLY on the medical emergency."
    )

    # Build system message with context
    tools_json = json.dumps(TOOLS, indent=2)
    system_message = f"""{persona}

DETECTED_USER_LANGUAGE_HINT (for understanding only, not for output): {language}

INSTRUCTIONS:
- You are a Medical Assistant.
- The user may speak or write in any language (Arabic, French, Tunisian Darija, English, etc.). Understand their intent fully.
- **OUTPUT LANGUAGE**: Always write the JSON `"text"` field for speech in **English only**. Never reply in Arabic, French, or other languages in `"text"`. Tool string fields (e.g. message_text) should also be in English unless a system rule says otherwise.
- **TOOL-FIRST POLICY (STRICT)**: If any available tool can directly solve or materially improve the user's stated problem, choose a tool action first instead of speech-only advice.
- **PRIORITY**: If the user needs immediate attention, emergency escalation, or external notification, USE THE TOOL immediately.
- Prefer `send_whatsapp_alert` when contacting others can improve safety.
- Do not ask unnecessary follow-up questions before using a suitable tool in urgent scenarios.
- **CALM INSTRUCTIONS**: Guide the user step-by-step.
    - `send_whatsapp_alert` = "Notify family/friends."
- Do NOT mention being a General or Stratageist.
- Only use context provided.
- Keep wording simple and clear.
- Admit missing info.
- Reject off-topic questions politely.
- Keep responses <60 words.
- Provide practical, actionable advice.

AVAILABLE TOOLS:
{tools_json}

RESPONSE FORMAT:
You must respond in JSON format.
If a suitable tool exists for the user's stated need, return an action JSON.
If you want to speak, return:
{{ "type": "speech", "text": "Your response here (English only)" }}

If you want to perform an action, return:
{{ "type": "action", "name": "tool_name", "args": {{ "arg1": "value" }} }}

CONVERSATION CONTEXT:
{context_text}"""
    if system_prompt_append:
        system_message = f"{system_message}\n\n{system_prompt_append}"

    # Build messages array with conversation history
    messages = [{"role": "system", "content": system_message}]
    
    # Limit history to most recent exchanges
    recent_history = history[-MAX_HISTORY_EXCHANGES:] if len(history) > MAX_HISTORY_EXCHANGES else history
    
    # Add conversation history
    for user_msg, bot_msg in recent_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    # Simulate typing
    if __name__ == "__main__":
        print("Medical Assistant is typing...", end="\r")
        
    
    return query_gemini(messages, voice_wav_bytes=voice_wav_bytes, voice_language=language)


def run_warning_voice_check(heart_rate, language="en"):
    """Wake the voice agent with dynamic prompt injection."""
    pause_wakeword_mic()
    try:
        return _run_warning_voice_check_impl(heart_rate, language=language)
    finally:
        resume_wakeword_mic()


def _run_warning_voice_check_impl(heart_rate, language="en"):
    dynamic_prompt = (
        "SYSTEM ALERT: The user's wearable detected a physiological anomaly. "
        f"Their current simulated heart rate is {heart_rate} BPM. "
        "Initiate a verbal check immediately. Ask the user if they are feeling dizzy, "
        "lightheaded, or experiencing symptoms. If they confirm symptoms or sound "
        "impaired, trigger your WhatsApp alert tool."
    )
    trigger_message = "Start an immediate verbal check-in with the user now."
    raw_reply = get_response(
        trigger_message,
        history=[],
        language=language,
        system_prompt_append=dynamic_prompt,
    )
    parsed = parse_agent_json(raw_reply)
    normalized = handle_agent_response(parsed)

    if normalized["type"] == "action":
        action_result = execute_action(normalized["name"], normalized.get("args", {}))
        return {
            "status": "action_executed",
            "agent_action": normalized["name"],
            "action_result": action_result,
            "dynamic_prompt": dynamic_prompt,
        }

    reply_text = normalized.get("text", "Checking your condition now.")
    broadcast_state("speaking")
    speak_response(reply_text)

    # Continue through the existing Whisper -> Gemini -> ElevenLabs flow for one follow-up turn.
    follow_up_user_text = listen_and_transcribe()
    if follow_up_user_text:
        follow_up_raw = get_response(
            follow_up_user_text,
            history=[(trigger_message, reply_text)],
            language=language,
            system_prompt_append=dynamic_prompt,
        )
        follow_up_parsed = parse_agent_json(follow_up_raw)
        follow_up_normalized = handle_agent_response(follow_up_parsed)

        if follow_up_normalized["type"] == "action":
            follow_up_action_result = execute_action(
                follow_up_normalized["name"], follow_up_normalized.get("args", {})
            )
            return {
                "status": "follow_up_action_executed",
                "agent_action": follow_up_normalized["name"],
                "action_result": follow_up_action_result,
                "dynamic_prompt": dynamic_prompt,
            }

        follow_up_reply_text = follow_up_normalized.get("text", "")
        if follow_up_reply_text:
            broadcast_state("speaking")
            speak_response(follow_up_reply_text)

    return {
        "status": "voice_check_started",
        "agent_reply": reply_text,
        "dynamic_prompt": dynamic_prompt,
    }


def classify_symptom_response_with_llm(text, language="en"):
    """Use Gemini to decide safety-check response class: ok/symptoms/unclear."""
    if not text:
        return "unclear"

    system_prompt = (
        "You are a medical triage classifier. "
        "Return exactly one token: ok OR symptoms OR unclear. "
        "Rules: "
        "ok = user clearly denies concerning symptoms and confirms feeling fine; "
        "symptoms = user reports dizziness, lightheadedness, weakness, chest pain, breathing issues, faintness, "
        "or any condition suggesting risk; "
        "unclear = ambiguous, irrelevant, too short, contradictory, or uncertain."
    )
    user_prompt = (
        f"Language hint: {language}\n"
        f"User response: {text}\n"
        "Output one token only: ok / symptoms / unclear"
    )
    raw = get_response(
        user_prompt,
        history=[],
        language=language,
        system_prompt_append=system_prompt,
    )
    token = (raw or "").strip().lower()
    if "symptoms" in token:
        return "symptoms"
    if token == "ok" or " ok" in token or token.startswith("ok"):
        return "ok"
    if "unclear" in token:
        return "unclear"
    return "unclear"


def run_no_human_safety_check(
    heart_rate,
    anomaly_value,
    language="en",
    user_id=None,
    session_id=None,
    orchestrator_state="no_action",
):
    """
    Ask up to 3 verbal safety checks.
    Escalate to WhatsApp alert if user is symptomatic or does not answer clearly.
    """
    pause_wakeword_mic()
    try:
        return _run_no_human_safety_check_impl(
            heart_rate,
            anomaly_value,
            language=language,
            user_id=user_id,
            session_id=session_id,
            orchestrator_state=orchestrator_state,
        )
    finally:
        resume_wakeword_mic()


def _run_no_human_safety_check_impl(
    heart_rate,
    anomaly_value,
    language="en",
    user_id=None,
    session_id=None,
    orchestrator_state="no_action",
):
    max_attempts = 3
    uid = str(user_id or os.getenv("ORCH_DEFAULT_USER_ID", "1"))
    sid = str(session_id or "")

    for attempt in range(1, max_attempts + 1):
        check_prompt = (
            f"Safety check attempt {attempt} of {max_attempts}. "
            f"A vital signal anomaly was detected with value {anomaly_value:.2f} and heart rate {heart_rate} BPM. "
            "Are you feeling okay? Are you dizzy, lightheaded, weak, or experiencing any symptoms? "
            "Please answer clearly."
        )
        broadcast_state("speaking")
        speak_response(check_prompt)

        block = listen_safety_check_user_turn()
        user_reply = block.get("text")
        voice = block.get("voice")
        if user_reply:
            print(f"🎤 [Safety Check] User said: {user_reply}")
        else:
            print("🎤 [Safety Check] No speech captured after listen + retry.")
        response_class = classify_symptom_response_with_llm(user_reply, language=language)
        push_voice_safety_check_event(
            user_id=uid,
            session_id=sid,
            check_kind="no_human",
            orchestrator_state=orchestrator_state,
            attempt=attempt,
            heart_rate=heart_rate,
            anomaly_value=anomaly_value,
            dl_risk_score=None,
            transcript=user_reply,
            response_class=response_class,
            voice=voice,
        )

        if response_class == "ok":
            confirm_text = "Understood. I will continue monitoring you. Please stay seated and alert us if anything changes."
            broadcast_state("speaking")
            speak_response(confirm_text)
            return {
                "status": "resolved_user_ok",
                "attempts_used": attempt,
                "last_user_reply": user_reply,
            }

        if response_class == "symptoms":
            alert_message = (
                "INTERSENSE ESCALATION: User reported symptoms during anomaly safety check. "
                f"Anomaly value: {anomaly_value:.2f}, heart rate: {heart_rate} BPM."
            )
            alert_result = send_whatsapp_alert(message_text=alert_message)
            return {
                "status": "escalated_symptoms_reported",
                "attempts_used": attempt,
                "last_user_reply": user_reply,
                "alert_result": alert_result,
            }

    alert_message = (
        "INTERSENSE ESCALATION: No clear response after 3 anomaly safety checks. "
        f"Anomaly value: {anomaly_value:.2f}, heart rate: {heart_rate} BPM."
    )
    alert_result = send_whatsapp_alert(message_text=alert_message)
    return {
        "status": "escalated_no_clear_response",
        "attempts_used": max_attempts,
        "alert_result": alert_result,
    }


def run_human_detected_safety_check(
    heart_rate,
    anomaly_value,
    dl_risk_score,
    language="en",
    user_id=None,
    session_id=None,
    orchestrator_state="warning",
):
    """
    Human detected with no critical DL event:
    contact user up to 3 times and escalate on symptoms/unclear responses.
    """
    pause_wakeword_mic()
    try:
        return _run_human_detected_safety_check_impl(
            heart_rate,
            anomaly_value,
            dl_risk_score,
            language=language,
            user_id=user_id,
            session_id=session_id,
            orchestrator_state=orchestrator_state,
        )
    finally:
        resume_wakeword_mic()


def _run_human_detected_safety_check_impl(
    heart_rate,
    anomaly_value,
    dl_risk_score,
    language="en",
    user_id=None,
    session_id=None,
    orchestrator_state="warning",
):
    max_attempts = 3
    history = []
    uid = str(user_id or os.getenv("ORCH_DEFAULT_USER_ID", "1"))
    sid = str(session_id or "")

    for attempt in range(1, max_attempts + 1):
        user_prompt = (
            f"Human was detected by camera and deep learning risk score is {dl_risk_score:.2f}. "
            f"Wearable anomaly value is {anomaly_value:.2f} with heart rate {heart_rate} BPM. "
            f"This is proactive check attempt {attempt} of {max_attempts}. "
            "Ask if the user feels okay and screen for dizziness, lightheadedness, weakness, chest pain, "
            "or breathing difficulty."
        )
        dynamic_prompt = (
            "SYSTEM ALERT: Human detected. Deep learning did not confirm critical fainting yet. "
            f"DL risk score is {dl_risk_score:.2f}. Wearable anomaly value is {anomaly_value:.2f}. "
            f"Heart rate is {heart_rate} BPM. You must perform a proactive verbal safety check. "
            "If the user reports symptoms or gives unclear responses repeatedly, trigger WhatsApp alert."
        )

        raw_reply = get_response(
            user_prompt,
            history=history,
            language=language,
            system_prompt_append=dynamic_prompt,
        )
        parsed = parse_agent_json(raw_reply)
        normalized = handle_agent_response(parsed)

        if normalized["type"] == "action":
            action_result = execute_action(normalized["name"], normalized.get("args", {}))
            return {
                "status": "action_executed",
                "attempts_used": attempt,
                "agent_action": normalized["name"],
                "action_result": action_result,
            }

        reply_text = normalized.get("text", "Please confirm if you are feeling okay.")
        broadcast_state("speaking")
        speak_response(reply_text)

        block = listen_safety_check_user_turn()
        user_reply = block.get("text")
        voice = block.get("voice")
        if user_reply:
            print(f"🎤 [Proactive Check] User said: {user_reply}")
        else:
            print("🎤 [Proactive Check] No speech captured after listen + retry.")
        response_class = classify_symptom_response_with_llm(user_reply, language=language)
        push_voice_safety_check_event(
            user_id=uid,
            session_id=sid,
            check_kind="human_proactive",
            orchestrator_state=orchestrator_state,
            attempt=attempt,
            heart_rate=heart_rate,
            anomaly_value=anomaly_value,
            dl_risk_score=dl_risk_score,
            transcript=user_reply,
            response_class=response_class,
            voice=voice,
        )
        history.append((user_prompt, reply_text))
        if user_reply:
            history.append((user_reply, "Acknowledged."))

        if response_class == "ok":
            broadcast_state("speaking")
            speak_response("Thank you. I will keep monitoring. Please call out immediately if symptoms begin.")
            return {
                "status": "resolved_user_ok",
                "attempts_used": attempt,
                "last_user_reply": user_reply,
            }

        if response_class == "symptoms":
            alert_message = (
                "INTERSENSE ESCALATION: Human detected, user reported symptoms during proactive safety check. "
                f"DL risk score: {dl_risk_score:.2f}, anomaly value: {anomaly_value:.2f}, heart rate: {heart_rate} BPM."
            )
            alert_result = send_whatsapp_alert(message_text=alert_message)
            return {
                "status": "escalated_symptoms_reported",
                "attempts_used": attempt,
                "last_user_reply": user_reply,
                "alert_result": alert_result,
            }

    alert_message = (
        "INTERSENSE ESCALATION: Human detected but no clear response after 3 proactive checks. "
        f"DL risk score: {dl_risk_score:.2f}, anomaly value: {anomaly_value:.2f}, heart rate: {heart_rate} BPM."
    )
    alert_result = send_whatsapp_alert(message_text=alert_message)
    return {
        "status": "escalated_no_clear_response",
        "attempts_used": max_attempts,
        "alert_result": alert_result,
    }


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ORCH_PYTHON_EXECUTABLE = os.getenv("ORCH_PYTHON_EXECUTABLE", sys.executable)
# Laptop demo defaults: probe only camera 0 (see ORCH_FORCE_CAMERA_INDEX).
ORCH_MAX_CAMERAS = int(os.getenv("ORCH_MAX_CAMERAS", "1"))
ORCH_SCAN_SECONDS = float(os.getenv("ORCH_SCAN_SECONDS", "0.6"))
ORCH_SCAN_MAX_FRAMES = int(os.getenv("ORCH_SCAN_MAX_FRAMES", "8"))
ORCH_SCAN_TIMEOUT = int(os.getenv("ORCH_SCAN_TIMEOUT", "60"))
ORCH_CAMERA_SCAN_CACHE_SEC = int(os.getenv("ORCH_CAMERA_SCAN_CACHE_SEC", "10"))
ORCH_SYNCOPE_MAX_CAMERAS = int(os.getenv("ORCH_SYNCOPE_MAX_CAMERAS", "1"))
ORCH_SYNCOPE_DURATION_SECONDS = int(os.getenv("ORCH_SYNCOPE_DURATION_SECONDS", "30"))
# Subprocess timeout = duration + padding (preload MediaPipe+LSTM+h5 before camera can exceed inference time).
# Allow enough wall time for V2 preload (see SYNCOPE_MODEL_LOAD_TIMEOUT_SEC in syncope_cnn_lstm.py, default 300s).
ORCH_SYNCOPE_SUBPROCESS_PADDING = int(os.getenv("ORCH_SYNCOPE_SUBPROCESS_PADDING", "330"))
ORCH_FORCE_CAMERA_INDEX = int(os.getenv("ORCH_FORCE_CAMERA_INDEX", "0"))
ORCH_USE_FIREBASE_VITALS = os.getenv("ORCH_USE_FIREBASE_VITALS", "1") == "1"
FIREBASE_RTDB_URL = os.getenv(
    "FIREBASE_RTDB_URL",
    "https://syncopedetection-default-rtdb.europe-west1.firebasedatabase.app",
).rstrip("/")
YOLO_SCANNER_SCRIPT = os.path.join(
    PROJECT_ROOT, "MODELS", "YOLO-Face-Person-Detector", "camera_person_scanner.py"
)
YOLO_SCANNER_STATE = os.path.join(
    PROJECT_ROOT, "MODELS", "YOLO-Face-Person-Detector", "active_cameras_state.json"
)
SYNCOPE_MODEL_DIR = os.path.join(PROJECT_ROOT, "MODELS", "V2 SYNCOPE FACE DETECTION MODEL")
SYNCOPE_RUNTIME_SCRIPT = os.path.join(
    SYNCOPE_MODEL_DIR, "syncope_runtime_scan.py"
)
SYNCOPE_UI_RUNTIME_SCRIPT = os.path.join(
    SYNCOPE_MODEL_DIR, "syncope_cnn_lstm.py"
)
SYNCOPE_RUNTIME_STATE = os.path.join(
    SYNCOPE_MODEL_DIR, "syncope_runtime_state.json"
)
BODY_FALL_MODEL_DIR = os.path.join(PROJECT_ROOT, "MODELS", "BODY FALL DETECTION")
BODY_FALL_RUNTIME_SCRIPT = os.path.join(BODY_FALL_MODEL_DIR, "realtime_fall_detector.py")
BODY_FALL_RUNTIME_STATE = os.path.join(BODY_FALL_MODEL_DIR, "body_fall_runtime_state.json")
ORCH_BODY_FALL_DURATION_SECONDS = int(os.getenv("ORCH_BODY_FALL_DURATION_SECONDS", "30"))
ORCH_BODY_FALL_CONFIRM_SECONDS = float(os.getenv("ORCH_BODY_FALL_CONFIRM_SECONDS", "2"))
# Show OpenCV UI when orchestrator runs body fall subprocess (disable on headless CI: ORCH_BODY_FALL_GUI=0).
ORCH_BODY_FALL_GUI = os.getenv("ORCH_BODY_FALL_GUI", "1") == "1"
if not os.path.isfile(SYNCOPE_UI_RUNTIME_SCRIPT):
    raise FileNotFoundError(
        "Syncope UI script is missing. Expected file in "
        f"{SYNCOPE_MODEL_DIR}: syncope_cnn_lstm.py"
    )
print(f" [ORCH] Syncope model directory: {SYNCOPE_MODEL_DIR}")
print(f" [ORCH] Body fall model directory: {BODY_FALL_MODEL_DIR}")
ORCH_WARM_WORKER_ENABLED = os.getenv("ORCH_WARM_WORKER_ENABLED", "1") == "1"
ORCH_WARM_PRELOAD_SYNCOPE = os.getenv("ORCH_WARM_PRELOAD_SYNCOPE", "0") == "1"


class OrchestratorWarmModelWorker:
    """
    Warm model, cold camera worker:
    - Preloads YOLO + syncope model/landmarker once
    - Opens cameras only during explicit scan calls
    - Stays mostly idle (low CPU) between tasks
    """

    def __init__(self):
        self.request_queue = queue.Queue()
        self.ready_event = threading.Event()
        self.thread = None
        self.camera_module = None
        self.syncope_module = None
        self.last_init_error = None

    def _load_module(self, name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # Register before exec so decorators/type resolution can find module metadata.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def _run(self):
        try:
            self.camera_module = self._load_module("camera_person_scanner", YOLO_SCANNER_SCRIPT)
            if ORCH_WARM_PRELOAD_SYNCOPE:
                if os.path.isfile(SYNCOPE_RUNTIME_SCRIPT):
                    print(f" [ORCH] Warm preload syncope script: {SYNCOPE_RUNTIME_SCRIPT}")
                    self.syncope_module = self._load_module(
                        "syncope_runtime_scan", SYNCOPE_RUNTIME_SCRIPT
                    )
                else:
                    print(
                        " [ORCH] Warm preload syncope skipped (syncope_runtime_scan.py not in "
                        f"{SYNCOPE_MODEL_DIR})."
                    )

            # Warm-load models. Cameras stay closed until scan calls.
            if hasattr(self.camera_module, "preload_yolo_model"):
                self.camera_module.preload_yolo_model()
            if self.syncope_module is not None:
                if hasattr(self.syncope_module, "preload_syncope_model"):
                    self.syncope_module.preload_syncope_model()
                if hasattr(self.syncope_module, "get_face_landmarker"):
                    self.syncope_module.get_face_landmarker()

            if ORCH_WARM_PRELOAD_SYNCOPE:
                print(" [ORCH] Warm model worker ready (YOLO+syncope preloaded, cameras cold).")
            else:
                print(" [ORCH] Warm model worker ready (YOLO preloaded, cameras cold).")
        except Exception as e:
            self.last_init_error = str(e)
            print(f" [ORCH] Warm model worker init failed: {e}")
        finally:
            self.ready_event.set()

        while True:
            item = self.request_queue.get()
            if item is None:
                break
            task_type, payload, response_queue = item
            try:
                if self.last_init_error:
                    raise RuntimeError(self.last_init_error)

                if task_type == "camera_scan":
                    result = self.camera_module.scan_all_cameras(
                        max_camera_index=int(payload.get("max_cameras", ORCH_MAX_CAMERAS)),
                        conf=float(payload.get("conf", 0.4)),
                        iou=float(payload.get("iou", 0.7)),
                        scan_seconds_per_camera=float(payload.get("scan_seconds", ORCH_SCAN_SECONDS)),
                        max_frames_per_camera=int(payload.get("max_frames", ORCH_SCAN_MAX_FRAMES)),
                        state_path=pathlib.Path(YOLO_SCANNER_STATE),
                        save=True,
                    )
                    response_queue.put({"ok": True, "data": result.to_json_dict()})
                elif task_type == "syncope_scan":
                    result = self.syncope_module.run_syncope_scan(
                        camera_indices=list(payload.get("camera_indices", [])),
                        duration_seconds=int(payload.get("duration_seconds", 30)),
                    )
                    pathlib.Path(SYNCOPE_RUNTIME_STATE).write_text(
                        json.dumps(result, indent=2), encoding="utf-8"
                    )
                    response_queue.put({"ok": True, "data": result})
                else:
                    response_queue.put({"ok": False, "error": f"Unknown task type: {task_type}"})
            except Exception as e:
                response_queue.put({"ok": False, "error": str(e)})

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, task_type, payload, timeout_seconds):
        self.ready_event.wait(timeout=60)
        response_queue = queue.Queue(maxsize=1)
        self.request_queue.put((task_type, payload, response_queue))
        return response_queue.get(timeout=timeout_seconds)


ORCH_MODEL_WORKER = None


def start_orch_warm_worker():
    global ORCH_MODEL_WORKER
    if not ORCH_WARM_WORKER_ENABLED:
        print(" [ORCH] Warm model worker disabled (ORCH_WARM_WORKER_ENABLED=0).")
        return
    ORCH_MODEL_WORKER = OrchestratorWarmModelWorker()
    ORCH_MODEL_WORKER.start()


def fetch_latest_firebase_vitals(user_id="1"):
    """
    Pull latest vitals from Firebase RTDB path used by ESP32:
      /users/{user_id}/vitals.json
    """
    if not ORCH_USE_FIREBASE_VITALS:
        return {"ok": False, "message": "Firebase vitals disabled by ORCH_USE_FIREBASE_VITALS=0"}
    if not FIREBASE_RTDB_URL:
        return {"ok": False, "message": "FIREBASE_RTDB_URL is not configured."}

    user_id = str(user_id or "1")
    url_direct = f"{FIREBASE_RTDB_URL}/users/{user_id}/vitals.json"
    url_users = f"{FIREBASE_RTDB_URL}/users.json"
    try:
        def _read_json(url):
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                payload = resp.read().decode("utf-8", errors="ignore")
            return json.loads(payload) if payload else None

        # Primary path used by your ESP sketch.
        direct = _read_json(url_direct)
        if isinstance(direct, dict) and direct:
            hr = direct.get("HeartRate")
            spo2 = direct.get("BloodOxygen")
            return {
                "ok": True,
                "heart_rate": int(hr) if hr is not None else None,
                "blood_oxygen": int(spo2) if spo2 is not None else None,
                "raw": direct,
                "source": f"users/{user_id}/vitals",
            }

        # Fallback: handle array-style users payload (null, {...}) seen in RTDB exports.
        users_payload = _read_json(url_users)
        candidate = None
        if isinstance(users_payload, list):
            idx = int(user_id) if user_id.isdigit() else None
            if idx is not None and len(users_payload) > idx and isinstance(users_payload[idx], dict):
                candidate = (users_payload[idx] or {}).get("vitals")
        elif isinstance(users_payload, dict):
            user_entry = users_payload.get(user_id)
            if isinstance(user_entry, dict):
                candidate = user_entry.get("vitals")

        if isinstance(candidate, dict) and candidate:
            hr = candidate.get("HeartRate")
            spo2 = candidate.get("BloodOxygen")
            return {
                "ok": True,
                "heart_rate": int(hr) if hr is not None else None,
                "blood_oxygen": int(spo2) if spo2 is not None else None,
                "raw": candidate,
                "source": "users fallback",
            }

        return {
            "ok": False,
            "message": f"No vitals found at users/{user_id}/vitals or users fallback.",
            "raw": {"direct": direct, "users": users_payload},
        }
    except Exception as e:
        return {"ok": False, "message": f"Firebase vitals fetch failed: {e}"}


def _normalized_camera_indices(seq):
    """JSON / subprocess may emit indices as strings; routing and YOLO ``per_camera`` keys need consistency."""
    out = []
    for x in seq or []:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def _load_recent_camera_scan_state(max_age_seconds):
    if not os.path.isfile(YOLO_SCANNER_STATE):
        return None
    try:
        age = time.time() - os.path.getmtime(YOLO_SCANNER_STATE)
        if age > max_age_seconds:
            return None
        with open(YOLO_SCANNER_STATE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cameras = _normalized_camera_indices(data.get("camera_indices_with_people", []))
        return {
            "ok": True,
            "human_detected": bool(data.get("human_detected", len(cameras) > 0)),
            "camera_indices_with_people": cameras,
            "count_cameras_with_people": int(data.get("count_cameras_with_people", len(cameras))),
            "raw": data,
            "cached": True,
            "cache_age_seconds": round(age, 2),
        }
    except Exception:
        return None


def run_camera_presence_scan(max_cameras=None):
    """
    Run YOLO face/person scanner and return active camera metadata for orchestrator.
    """
    if max_cameras is None:
        max_cameras = ORCH_MAX_CAMERAS
    if ORCH_FORCE_CAMERA_INDEX >= 0:
        # Single-camera fast path: only probe forced index range.
        max_cameras = max(1, ORCH_FORCE_CAMERA_INDEX + 1)

    cached_state = _load_recent_camera_scan_state(ORCH_CAMERA_SCAN_CACHE_SEC)
    if cached_state is not None:
        print(
            f" [ORCH] Using cached camera scan ({cached_state.get('cache_age_seconds')}s old): "
            f"cameras_with_people={cached_state.get('camera_indices_with_people', [])}"
        )
        return cached_state

    if not os.path.isfile(YOLO_SCANNER_SCRIPT):
        return {
            "ok": False,
            "message": "YOLO camera scanner script missing.",
            "camera_indices_with_people": [],
            "count_cameras_with_people": 0,
            "human_detected": False,
        }

    try:
        if ORCH_MODEL_WORKER is not None:
            print(" [ORCH] Running multi-camera person scan via warm worker...")
            worker_resp = ORCH_MODEL_WORKER.submit(
                "camera_scan",
                {
                    "max_cameras": max_cameras,
                    "conf": 0.4,
                    "iou": 0.7,
                    "scan_seconds": ORCH_SCAN_SECONDS,
                    "max_frames": ORCH_SCAN_MAX_FRAMES,
                },
                timeout_seconds=ORCH_SCAN_TIMEOUT,
            )
            if not worker_resp.get("ok"):
                print(
                    f" [ORCH] Warm worker camera scan failed: {worker_resp.get('error')} "
                    "-> falling back to subprocess scanner."
                )
            else:
                data = worker_resp.get("data", {}) or {}
                cameras = _normalized_camera_indices(data.get("camera_indices_with_people", []))
                if ORCH_FORCE_CAMERA_INDEX >= 0:
                    cameras = [c for c in cameras if c == ORCH_FORCE_CAMERA_INDEX]
                print(
                    f" [ORCH] Camera scan done: human_detected={bool(data.get('human_detected', len(cameras)>0))}, "
                    f"cameras_with_people={cameras}"
                )
                return {
                    "ok": True,
                    "human_detected": bool(data.get("human_detected", len(cameras) > 0)),
                    "camera_indices_with_people": cameras,
                    "count_cameras_with_people": int(data.get("count_cameras_with_people", len(cameras))),
                    "raw": data,
                    "cached": False,
                }

        cmd = [
            ORCH_PYTHON_EXECUTABLE,
            YOLO_SCANNER_SCRIPT,
            "--max-cameras",
            str(max_cameras),
            "--scan-seconds",
            str(ORCH_SCAN_SECONDS),
            "--max-frames",
            str(ORCH_SCAN_MAX_FRAMES),
            "--output",
            YOLO_SCANNER_STATE,
        ]
        print(f" [ORCH] Running multi-camera person scan (python={ORCH_PYTHON_EXECUTABLE})...")
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=ORCH_SCAN_TIMEOUT,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "message": f"YOLO scanner failed: {proc.stderr.strip() or proc.stdout.strip()}",
                "camera_indices_with_people": [],
                "count_cameras_with_people": 0,
                "human_detected": False,
            }

        data = {}
        if os.path.isfile(YOLO_SCANNER_STATE):
            with open(YOLO_SCANNER_STATE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            out = (proc.stdout or "").strip()
            if out:
                data = json.loads(out)

        cameras = _normalized_camera_indices(data.get("camera_indices_with_people", []))
        if ORCH_FORCE_CAMERA_INDEX >= 0:
            cameras = [c for c in cameras if c == ORCH_FORCE_CAMERA_INDEX]
        print(
            f" [ORCH] Camera scan done: human_detected={bool(data.get('human_detected', len(cameras)>0))}, "
            f"cameras_with_people={cameras}"
        )
        return {
            "ok": True,
            "human_detected": bool(data.get("human_detected", len(cameras) > 0)),
            "camera_indices_with_people": cameras,
            "count_cameras_with_people": int(data.get("count_cameras_with_people", len(cameras))),
            "raw": data,
            "cached": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "message": (
                f"YOLO scanner timed out after {ORCH_SCAN_TIMEOUT}s. "
                "Increase ORCH_SCAN_TIMEOUT or reduce ORCH_MAX_CAMERAS."
            ),
            "camera_indices_with_people": [],
            "count_cameras_with_people": 0,
            "human_detected": False,
        }
    except Exception as e:
        return {
            "ok": False,
            "message": f"YOLO scanner exception: {e}",
            "camera_indices_with_people": [],
            "count_cameras_with_people": 0,
            "human_detected": False,
        }


def _orchestrator_pick_cameras(camera_scan, indices, limit):
    """Highest YOLO box count first."""
    raw = camera_scan.get("raw") or {}
    per_cam = raw.get("per_camera") or {}
    if not indices:
        return []
    ranked = sorted(
        _normalized_camera_indices(indices),
        key=lambda idx: int((per_cam.get(str(idx), {}) or {}).get("num_detections", 0)),
        reverse=True,
    )
    return ranked[: max(1, limit)]


def orchestrator_human_vision_route(camera_scan, camera_indices_with_people, max_cameras, force_camera_index):
    """
    ``face``: run syncope on cameras that saw YOLO ``face``.
    ``body``: run pose/fall detector on cameras that saw ``person`` but no ``face`` in the scan.
    Older ``active_cameras_state.json`` without ``camera_indices_with_faces`` ⇒ treat as ``face``.
    """
    raw = camera_scan.get("raw") or {}
    people = _normalized_camera_indices(camera_indices_with_people)
    if force_camera_index >= 0:
        people = [c for c in people if c == force_camera_index]
    legacy = "camera_indices_with_faces" not in raw

    faces_ix = set(_normalized_camera_indices(raw.get("camera_indices_with_faces")))
    body_ix = set(_normalized_camera_indices(raw.get("camera_indices_body_only")))

    if legacy:
        return "face", _orchestrator_pick_cameras(camera_scan, people, max_cameras)

    faces_here = _orchestrator_pick_cameras(camera_scan, [c for c in people if c in faces_ix], max_cameras)
    if faces_here:
        return "face", faces_here

    body_here = _orchestrator_pick_cameras(camera_scan, [c for c in people if c in body_ix], max_cameras)
    if body_here:
        return "body", body_here

    return "face", _orchestrator_pick_cameras(camera_scan, people, max_cameras)


def run_body_fall_detection_window(camera_indices, duration_seconds=None, confirm_seconds=None):
    """
    Run pose + LSTM body fall detector. Requires ``confirm_seconds`` of continuous fall signal
    before marking ``critical_detected`` (default 2s). Ends after ``duration_seconds`` with no alert.
    """
    duration_seconds = ORCH_BODY_FALL_DURATION_SECONDS if duration_seconds is None else duration_seconds
    confirm_seconds = ORCH_BODY_FALL_CONFIRM_SECONDS if confirm_seconds is None else confirm_seconds

    if not camera_indices:
        return {
            "ok": True,
            "critical_detected": False,
            "cameras_critical": [],
            "max_risk_score": 0.0,
            "message": "No active cameras passed to body fall detector.",
            "detection_path": "body_fall",
        }

    if not os.path.isfile(BODY_FALL_RUNTIME_SCRIPT):
        return {
            "ok": False,
            "critical_detected": False,
            "cameras_critical": [],
            "max_risk_score": 0.0,
            "message": "Body fall runtime script missing (realtime_fall_detector.py).",
            "detection_path": "body_fall",
        }

    try:
        primary_camera = int(camera_indices[0])
        print(
            f" [ORCH] Starting body fall detector for {duration_seconds}s "
            f"(confirm≥{confirm_seconds}s sustained) on camera {primary_camera}..."
        )
        cmd = [
            ORCH_PYTHON_EXECUTABLE,
            BODY_FALL_RUNTIME_SCRIPT,
            "--camera-index",
            str(primary_camera),
            "--duration",
            str(int(duration_seconds)),
            "--confirm-seconds",
            str(float(confirm_seconds)),
            "--output",
            BODY_FALL_RUNTIME_STATE,
        ]
        if not ORCH_BODY_FALL_GUI:
            cmd.append("--no-gui")
        print(f" [ORCH] Body fall script path: {BODY_FALL_RUNTIME_SCRIPT}")
        print(f" [ORCH] Body fall GUI: {ORCH_BODY_FALL_GUI}")
        timeout_seconds = int(duration_seconds) + 120
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "critical_detected": False,
                "cameras_critical": [],
                "max_risk_score": 0.0,
                "message": f"Body fall runtime failed: {proc.stderr.strip() or proc.stdout.strip()}",
                "detection_path": "body_fall",
            }

        data = {}
        if os.path.isfile(BODY_FALL_RUNTIME_STATE):
            with open(BODY_FALL_RUNTIME_STATE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            out = (proc.stdout or "").strip()
            if out:
                try:
                    data = json.loads(out.splitlines()[-1])
                except json.JSONDecodeError:
                    data = {}

        mr = float(data.get("max_risk_score", data.get("max_fall_prob", 0.0)))
        result = {
            "ok": True,
            "critical_detected": bool(data.get("critical_detected", False)),
            "cameras_critical": data.get("cameras_critical", []),
            "camera_indices_opened": data.get("camera_indices_opened", []),
            "max_risk_score": mr,
            "raw": data,
            "detection_path": "body_fall",
        }
        print(
            f" [ORCH] Body fall runtime done: opened={result.get('camera_indices_opened', [])}, "
            f"critical={result.get('critical_detected')}, max_risk={result.get('max_risk_score')}"
        )
        return result
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "critical_detected": False,
            "cameras_critical": [],
            "max_risk_score": 0.0,
            "message": "Body fall runtime timeout.",
            "detection_path": "body_fall",
        }
    except Exception as e:
        return {
            "ok": False,
            "critical_detected": False,
            "cameras_critical": [],
            "max_risk_score": 0.0,
            "message": f"Body fall runtime exception: {e}",
            "detection_path": "body_fall",
        }


def run_syncope_detection_window(camera_indices, duration_seconds=30):
    """
    Run syncope model for a fixed window on cameras with detected humans.
    """
    if not camera_indices:
        return {
            "ok": True,
            "critical_detected": False,
            "cameras_critical": [],
            "max_risk_score": 0.0,
            "message": "No active human cameras passed to syncope runtime.",
        }

    if not os.path.isfile(SYNCOPE_UI_RUNTIME_SCRIPT):
        return {
            "ok": False,
            "critical_detected": False,
            "cameras_critical": [],
            "max_risk_score": 0.0,
            "message": "Syncope UI runtime script missing.",
        }

    try:
        primary_camera = int(camera_indices[0])
        print(
            f" [ORCH] Starting syncope UI runtime for {duration_seconds}s on camera {primary_camera}..."
        )
        cmd = [
            ORCH_PYTHON_EXECUTABLE,
            SYNCOPE_UI_RUNTIME_SCRIPT,
            "--camera-index",
            str(primary_camera),
            "--duration",
            str(int(duration_seconds)),
            "--output",
            SYNCOPE_RUNTIME_STATE,
        ]
        print(f" [ORCH] Syncope UI script path: {SYNCOPE_UI_RUNTIME_SCRIPT}")
        timeout_seconds = int(duration_seconds) + ORCH_SYNCOPE_SUBPROCESS_PADDING
        print(f" [ORCH] Syncope subprocess timeout={timeout_seconds}s (padding={ORCH_SYNCOPE_SUBPROCESS_PADDING}s)")
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "critical_detected": False,
                "cameras_critical": [],
                "max_risk_score": 0.0,
                "message": f"Syncope UI runtime failed: {proc.stderr.strip() or proc.stdout.strip()}",
            }

        data = {}
        if os.path.isfile(SYNCOPE_RUNTIME_STATE):
            with open(SYNCOPE_RUNTIME_STATE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            out = (proc.stdout or "").strip()
            if out:
                data = json.loads(out)

        result = {
            "ok": True,
            "critical_detected": bool(data.get("critical_detected", False)),
            "cameras_critical": data.get("cameras_critical", []),
            "camera_indices_opened": data.get("camera_indices_opened", []),
            "max_risk_score": float(data.get("max_risk_score", 0.0)),
            "raw": data,
        }
        print(
            f" [ORCH] Syncope runtime done: opened={result.get('camera_indices_opened', [])}, "
            f"critical={result.get('critical_detected')}, cameras_critical={result.get('cameras_critical', [])}, "
            f"max_risk={result.get('max_risk_score')}"
        )
        return result
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "critical_detected": False,
            "cameras_critical": [],
            "max_risk_score": 0.0,
            "message": "Syncope runtime timeout.",
        }
    except Exception as e:
        return {
            "ok": False,
            "critical_detected": False,
            "cameras_critical": [],
            "max_risk_score": 0.0,
            "message": f"Syncope runtime exception: {e}",
        }


def run_simulation_orchestrator(payload):
    """Process simulator payload and trigger InterSense state actions."""
    verbose_steps = []
    def _step(msg):
        ts = time.strftime("%H:%M:%S")
        verbose_steps.append(f"[{ts}] {msg}")

    wearable_anomaly = bool(payload.get("wearable_anomaly", False))
    prefer_simulator_vitals = bool(payload.get("prefer_simulator_vitals", False))
    heart_rate = int(payload.get("heart_rate", 70))
    anomaly_value = float(payload.get("anomaly_value", 0.0))
    dl_risk_score = float(payload.get("dl_risk_score", 0.0))
    blood_oxygen = None
    user_id = str(payload.get("user_id", os.getenv("ORCH_DEFAULT_USER_ID", "1")))
    firebase_vitals = fetch_latest_firebase_vitals(user_id=user_id)
    if prefer_simulator_vitals:
        heart_rate = int(payload.get("heart_rate", 70))
        bo_raw = payload.get("blood_oxygen")
        if bo_raw is not None and bo_raw != "":
            try:
                blood_oxygen = int(round(float(bo_raw)))
            except (TypeError, ValueError):
                blood_oxygen = None
        _step(
            "prefer_simulator_vitals: using payload heart_rate / blood_oxygen (Firebase not applied)."
        )
    elif firebase_vitals.get("ok"):
        if firebase_vitals.get("heart_rate") is not None and firebase_vitals.get("heart_rate") > 0:
            heart_rate = int(firebase_vitals.get("heart_rate"))
        blood_oxygen = firebase_vitals.get("blood_oxygen")
        _step(
            f"Firebase vitals read OK: heart_rate={firebase_vitals.get('heart_rate')}, "
            f"blood_oxygen={firebase_vitals.get('blood_oxygen')}, source={firebase_vitals.get('source')}"
        )
    else:
        _step(f"Firebase vitals unavailable: {firebase_vitals.get('message')}")
    _step(
        f"Incoming payload: wearable_anomaly={wearable_anomaly}, heart_rate={heart_rate}, "
        f"anomaly_value={anomaly_value:.2f}, dl_risk_score={dl_risk_score:.2f}"
    )

    if not wearable_anomaly:
        _step("No wearable anomaly -> state=normal.")
        return {
            "ok": True,
            "state": "normal",
            "message": "System Normal. Monitoring...",
            "firebase_vitals": firebase_vitals,
            "effective_heart_rate": heart_rate,
            "effective_blood_oxygen": blood_oxygen,
            "verbose_steps": verbose_steps,
        }

    # 1) Wearable anomaly detected -> run YOLO scanner (camera count: ORCH_MAX_CAMERAS, or forced index via ORCH_FORCE_CAMERA_INDEX).
    _step(
        "Wearable anomaly detected -> running YOLO person/face scan "
        f"(ORCH_MAX_CAMERAS={ORCH_MAX_CAMERAS}, ORCH_FORCE_CAMERA_INDEX={ORCH_FORCE_CAMERA_INDEX})."
    )
    camera_scan = run_camera_presence_scan()
    camera_indices_with_people = camera_scan.get("camera_indices_with_people", [])
    human_detected = bool(camera_scan.get("human_detected", False))
    _step(
        f"Camera scan result: ok={camera_scan.get('ok')}, human_detected={human_detected}, "
        f"cameras_with_people={camera_indices_with_people}"
    )
    if not camera_scan.get("ok", False):
        _step("Camera scan failed -> stopping orchestrator safely.")
        return {
            "ok": False,
            "state": "camera_scan_failed",
            "message": camera_scan.get(
                "message",
                "Camera scan failed; cannot continue orchestrator decision safely.",
            ),
            "camera_scan": camera_scan,
            "firebase_vitals": firebase_vitals,
            "effective_heart_rate": heart_rate,
            "effective_blood_oxygen": blood_oxygen,
            "verbose_steps": verbose_steps,
        }

    vision_route = None  # face | body
    selected_escalation_cameras = []

    syncope_result = {
        "ok": True,
        "critical_detected": False,
        "cameras_critical": [],
        "max_risk_score": 0.0,
        "message": "Syncope runtime skipped (no human cameras or body-only routing).",
    }
    body_fall_result = {
        "ok": True,
        "critical_detected": False,
        "cameras_critical": [],
        "max_risk_score": 0.0,
        "message": "Body fall runtime skipped (no human cameras or face routing).",
        "detection_path": "body_fall",
    }

    if human_detected:
        vision_route, selected_escalation_cameras = orchestrator_human_vision_route(
            camera_scan,
            camera_indices_with_people,
            ORCH_SYNCOPE_MAX_CAMERAS,
            ORCH_FORCE_CAMERA_INDEX,
        )
        raw_scan = camera_scan.get("raw") or {}
        legacy_split = "camera_indices_with_faces" not in raw_scan
        faces_hint = raw_scan.get("camera_indices_with_faces") if isinstance(raw_scan.get("camera_indices_with_faces"), list) else None
        bodies_hint = raw_scan.get("camera_indices_body_only") if isinstance(raw_scan.get("camera_indices_body_only"), list) else None
        _step(
            f"Human vision routing: route={vision_route}, cameras={selected_escalation_cameras}, "
            f"yolo_faces={faces_hint if not legacy_split else 'legacy-scan'}, "
            f"yolo_body_only={bodies_hint if not legacy_split else 'n/a'}"
        )
        for idx in sorted(set(_normalized_camera_indices(camera_indices_with_people))):
            pc = raw_scan.get("per_camera", {}).get(str(idx), {}) or {}
            if not isinstance(pc, dict):
                continue
            hf = pc.get("has_face")
            hb = pc.get("has_person_class")
            if hf is not None or hb is not None:
                _step(
                    f"YOLO cam {idx}: has_face={hf}, has_person_class={hb}, "
                    f"num_detections={pc.get('num_detections', '?')}"
                )

        if vision_route == "face":
            _step(
                f"Face visible -> starting face syncope ({ORCH_SYNCOPE_DURATION_SECONDS}s window)."
            )
            syncope_result = run_syncope_detection_window(
                camera_indices=selected_escalation_cameras,
                duration_seconds=ORCH_SYNCOPE_DURATION_SECONDS,
            )
            _step(
                f"Syncope runtime result: ok={syncope_result.get('ok')}, "
                f"critical_detected={syncope_result.get('critical_detected')}, "
                f"max_risk={syncope_result.get('max_risk_score')}, "
                f"opened={syncope_result.get('camera_indices_opened', [])}"
            )
            if not syncope_result.get("ok", False):
                _step("Syncope runtime failed -> stopping orchestrator safely.")
                return {
                    "ok": False,
                    "state": "syncope_runtime_failed",
                    "message": syncope_result.get(
                        "message",
                        "Syncope runtime failed; cannot complete camera check.",
                    ),
                    "camera_scan": camera_scan,
                    "human_vision_route": vision_route,
                    "selected_escalation_cameras": selected_escalation_cameras,
                    "syncope_result": syncope_result,
                    "body_fall_result": body_fall_result,
                    "firebase_vitals": firebase_vitals,
                    "effective_heart_rate": heart_rate,
                    "effective_blood_oxygen": blood_oxygen,
                    "verbose_steps": verbose_steps,
                }
        else:
            _step(
                "Body detected without YOLO face box in scan -> starting body fall detector "
                f"({ORCH_BODY_FALL_DURATION_SECONDS}s max, sustained ≥ {ORCH_BODY_FALL_CONFIRM_SECONDS}s)."
            )
            body_fall_result = run_body_fall_detection_window(
                camera_indices=selected_escalation_cameras,
            )
            _step(
                f"Body fall runtime result: ok={body_fall_result.get('ok')}, "
                f"critical_detected={body_fall_result.get('critical_detected')}, "
                f"max_risk={body_fall_result.get('max_risk_score')}"
            )
            if not body_fall_result.get("ok", False):
                _step("Body fall runtime failed -> stopping orchestrator safely.")
                return {
                    "ok": False,
                    "state": "body_fall_runtime_failed",
                    "message": body_fall_result.get(
                        "message",
                        "Body fall runtime failed; cannot complete camera check.",
                    ),
                    "camera_scan": camera_scan,
                    "human_vision_route": vision_route,
                    "selected_escalation_cameras": selected_escalation_cameras,
                    "syncope_result": syncope_result,
                    "body_fall_result": body_fall_result,
                    "firebase_vitals": firebase_vitals,
                    "effective_heart_rate": heart_rate,
                    "effective_blood_oxygen": blood_oxygen,
                    "verbose_steps": verbose_steps,
                }
    else:
        _step("No human detected in camera scan -> vision DL runtimes skipped.")

    selected_syncope_cameras = selected_escalation_cameras if vision_route == "face" else []
    selected_body_fall_cameras = selected_escalation_cameras if vision_route == "body" else []

    active_vision_critical = syncope_result if vision_route == "face" else body_fall_result
    active_vision_critical = active_vision_critical if vision_route else {"critical_detected": False}

    if active_vision_critical.get("critical_detected"):
        max_dl_risk = float(active_vision_critical.get("max_risk_score", 0.0))
        route_label = "syncope face" if vision_route == "face" else "body pose/fall"
        _step(f"Vision DL CRITICAL ({route_label}) -> immediate escalation (voice + WhatsApp).")
        play_critical_fainting_escalation_prompt()
        alert_message = (
            "INTERSENSE CRITICAL EMERGENCY: Wearable anomaly + vision model detected probable fainting/collapse "
            f"({'face syncope path' if vision_route == 'face' else 'body fall path'}, {max_dl_risk:.2f}). "
            f"Heart rate: {heart_rate} BPM. Cameras: {active_vision_critical.get('cameras_critical', [])}. "
            "Immediate assistance required."
        )
        alert_result = send_whatsapp_alert(message_text=alert_message)
        return {
            "ok": True,
            "state": "critical_emergency",
            "message": f"Critical event ({route_label}): automatic escalation triggered.",
            "alert_result": alert_result,
            "camera_scan": camera_scan,
            "human_vision_route": vision_route,
            "selected_escalation_cameras": selected_escalation_cameras,
            "selected_syncope_cameras": selected_syncope_cameras,
            "selected_body_fall_cameras": selected_body_fall_cameras,
            "syncope_result": syncope_result,
            "body_fall_result": body_fall_result,
            "firebase_vitals": firebase_vitals,
            "effective_heart_rate": heart_rate,
            "effective_blood_oxygen": blood_oxygen,
            "verbose_steps": verbose_steps,
        }

    # 3) No critical in vision window -> continue with current voice-check behavior.
    vision_name = {"face": "Face syncope", "body": "Body fall detector"}.get(vision_route, "Vision DL")
    _step(f"{vision_name} did not reach CRITICAL -> continuing to voice-check logic.")
    enriched_payload = dict(payload)
    enriched_payload["human_detected"] = human_detected
    enriched_payload["fainting_detected"] = False
    enriched_payload["dl_risk_score"] = 0.0
    decision = evaluate_simulation_state(enriched_payload)
    state = decision.get("state")
    _step(f"Post vision-DL orchestrator state decision={state}.")

    if state == "critical_emergency":
        _step("State critical_emergency reached -> immediate escalation.")
        play_critical_fainting_escalation_prompt()
        alert_message = (
            "INTERSENSE CRITICAL EMERGENCY: Wearable anomaly + human detected + critical DL detection. "
            f"Simulated heart rate: {heart_rate} BPM. Immediate assistance required."
        )
        alert_result = send_whatsapp_alert(message_text=alert_message)
        return {
            "ok": True,
            "state": state,
            "message": "Critical emergency from deep learning path: WhatsApp alert sent.",
            "alert_result": alert_result,
            "firebase_vitals": firebase_vitals,
            "effective_heart_rate": heart_rate,
            "effective_blood_oxygen": blood_oxygen,
            "camera_scan": camera_scan,
            "human_vision_route": vision_route,
            "syncope_result": syncope_result,
            "body_fall_result": body_fall_result,
            "selected_syncope_cameras": selected_syncope_cameras,
            "selected_body_fall_cameras": selected_body_fall_cameras,
            "verbose_steps": verbose_steps,
        }

    if state == "warning":
        _step("State warning -> launching proactive voice safety checks.")
        language = payload.get("language", "en")
        warning_result = run_human_detected_safety_check(
            heart_rate=heart_rate,
            anomaly_value=anomaly_value,
            dl_risk_score=dl_risk_score,
            language=language,
            user_id=user_id,
            session_id=str(payload.get("session_id", "") or ""),
            orchestrator_state="warning",
        )
        return {
            "ok": True,
            "state": state,
            "message": "Human detected with non-critical DL result: proactive voice check completed.",
            "warning_result": warning_result,
            "camera_scan": camera_scan,
            "human_vision_route": vision_route,
            "selected_escalation_cameras": selected_escalation_cameras,
            "syncope_result": syncope_result,
            "body_fall_result": body_fall_result,
            "firebase_vitals": firebase_vitals,
            "effective_heart_rate": heart_rate,
            "effective_blood_oxygen": blood_oxygen,
            "selected_syncope_cameras": selected_syncope_cameras,
            "selected_body_fall_cameras": selected_body_fall_cameras,
            "verbose_steps": verbose_steps,
        }

    if state == "no_action":
        _step("State no_action -> launching no-human safety checks.")
        language = payload.get("language", "en")
        safety_result = run_no_human_safety_check(
            heart_rate=heart_rate,
            anomaly_value=anomaly_value,
            language=language,
            user_id=user_id,
            session_id=str(payload.get("session_id", "") or ""),
            orchestrator_state="no_action",
        )
        return {
            "ok": True,
            "state": state,
            "message": "No human target confirmed; completed 3-attempt voice safety protocol.",
            "safety_result": safety_result,
            "camera_scan": camera_scan,
            "human_vision_route": vision_route,
            "selected_escalation_cameras": selected_escalation_cameras,
            "syncope_result": syncope_result,
            "body_fall_result": body_fall_result,
            "firebase_vitals": firebase_vitals,
            "effective_heart_rate": heart_rate,
            "effective_blood_oxygen": blood_oxygen,
            "selected_syncope_cameras": selected_syncope_cameras,
            "selected_body_fall_cameras": selected_body_fall_cameras,
            "verbose_steps": verbose_steps,
        }

    _step("No additional action selected.")
    return {
        "ok": True,
        "state": state,
        "message": decision.get("message", "No action taken."),
        "firebase_vitals": firebase_vitals,
        "effective_heart_rate": heart_rate,
        "effective_blood_oxygen": blood_oxygen,
        "camera_scan": camera_scan,
        "human_vision_route": vision_route,
        "selected_escalation_cameras": selected_escalation_cameras,
        "syncope_result": syncope_result,
        "body_fall_result": body_fall_result,
        "selected_syncope_cameras": selected_syncope_cameras,
        "selected_body_fall_cameras": selected_body_fall_cameras,
        "verbose_steps": verbose_steps,
    }


def submit_orchestrator_task(payload, timeout_seconds=90):
    """
    Ensure orchestrator voice checks run on the main assistant loop thread
    (same runtime path as manual \\voice mode).
    """
    response_queue = queue.Queue(maxsize=1)
    ORCHESTRATOR_QUEUE.put({"payload": payload, "response_queue": response_queue})
    try:
        return response_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        return {
            "ok": False,
            "state": "timeout",
            "message": "Orchestrator task timed out while waiting for main voice loop.",
        }


API_PORT = int(os.getenv("API_PORT", "8000"))
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_THREAD = None


def _post_platform_ingest_json(payload: dict) -> None:
    """POST JSON to Platform ingestion (orchestrator or voice_safety_check)."""
    url = os.getenv("PLATFORM_INGEST_URL", "http://127.0.0.1:8100/api/ingestion/orchestrator-event/")
    token = os.getenv("PLATFORM_INGEST_TOKEN", "platform-dev-token")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
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
        except Exception as e:
            if attempt == retries - 1:
                print(f"[Platform] Failed to ingest event: {e}")
            else:
                time.sleep(backoff * (attempt + 1))


def push_orchestrator_event_to_platform(payload: dict) -> None:
    """Persist orchestrator state snapshots into the Platform Django ingestion API."""
    _post_platform_ingest_json(payload)


def push_voice_safety_check_event(
    *,
    user_id: str,
    session_id: str,
    check_kind: str,
    orchestrator_state: str,
    attempt: int,
    heart_rate,
    anomaly_value: float,
    dl_risk_score,
    transcript: str | None,
    response_class: str,
    voice: dict | None,
) -> None:
    """Log one verbal safety-check attempt with TORGO voice scores to the Platform."""
    ts = datetime.now(timezone.utc).isoformat()
    idempotency_key = hashlib.sha256(
        f"{user_id}|{session_id}|voice_safety|{check_kind}|{attempt}|{ts}".encode("utf-8")
    ).hexdigest()
    vid = voice if isinstance(voice, dict) else {}
    transcript_preview = (transcript or "")[:1800]
    extra = {"voice_model": "torgo_dysarthria_efficientnet_b0"}
    if vid.get("error"):
        extra["voice_error"] = str(vid.get("error"))[:500]
    payload = {
        "event_type": "voice_safety_check",
        "user_id": user_id,
        "session_id": session_id or "",
        "idempotency_key": idempotency_key,
        "timestamp": ts,
        "check_kind": check_kind,
        "orchestrator_state": orchestrator_state,
        "heart_rate": heart_rate,
        "anomaly_value": float(anomaly_value or 0.0),
        "dl_risk_score": dl_risk_score,
        "voice": {
            "attempt": int(attempt),
            "prob_normal": float(vid.get("prob_normal", 0.0) or 0.0),
            "prob_dysarthric": float(vid.get("prob_dysarthric", 0.0) or 0.0),
            "pred_label": str(vid.get("label", "") or ""),
            "transcript_preview": transcript_preview,
            "response_class": str(response_class or ""),
        },
        "voice_extra": extra,
    }
    _post_platform_ingest_json(payload)


def _score_voice_bytes_if_enabled(wav_bytes) -> dict | None:
    if os.getenv("VOICE_SAFETY_SCORING_ENABLED", "1").lower() in ("0", "false", "no"):
        return None
    if not wav_bytes:
        return None
    try:
        from voice_scoring import score_wav_bytes_torgo

        return score_wav_bytes_torgo(wav_bytes)
    except Exception as e:
        print(f"[VoiceSafety] scoring failed: {e}")
        return {"error": str(e), "prob_normal": 0.0, "prob_dysarthric": 0.0, "label": "error"}


def _listen_safety_check_utterance_with_voice():
    """Listen once; optionally run TORGO voice model on the same WAV sent to Whisper."""
    use_voice = os.getenv("VOICE_SAFETY_SCORING_ENABLED", "1").lower() not in ("0", "false", "no")
    if use_voice:
        r = listen_and_transcribe(return_audio=True)
        if r is None:
            return {"text": None, "voice": None}
        text, wav = r
        voice = _score_voice_bytes_if_enabled(wav)
        return {"text": text, "voice": voice}
    text = listen_and_transcribe(return_audio=False)
    return {"text": text, "voice": None}


def listen_safety_check_user_turn():
    """First listen + optional retry (same behavior as previous safety checks)."""
    first = _listen_safety_check_utterance_with_voice()
    if first.get("text"):
        return first
    second = _listen_safety_check_utterance_with_voice()
    return second


def start_orchestrator_api_server():
    """Run /api/simulate on the same process as medical_assistant.py."""
    global API_THREAD

    if FastAPI is None or uvicorn is None or BaseModel is None:
        print("⚠️ Orchestrator API disabled (fastapi/uvicorn missing).")
        return

    class SimulationRequest(BaseModel):
        heart_rate: int = Field(..., ge=40, le=180)
        blood_oxygen: int | None = Field(None, ge=70, le=100)
        prefer_simulator_vitals: bool = False
        anomaly_value: float = Field(0.0, ge=0.0)
        dl_risk_score: float = Field(0.0, ge=0.0, le=1.0)
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

    api_app = FastAPI(title="InterSense Embedded Orchestrator API")
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api_app.post("/api/simulate")
    def simulate(payload: SimulationRequest):
        request_payload = payload.model_dump()
        result = submit_orchestrator_task(request_payload)
        timestamp = datetime.now(timezone.utc).isoformat()
        idempotency_key = hashlib.sha256(
            f"{request_payload.get('user_id','1')}|{request_payload.get('session_id','default-session')}|{timestamp}|{result.get('state','unknown')}".encode("utf-8")
        ).hexdigest()
        push_orchestrator_event_to_platform(
            {
                "user_id": request_payload.get("user_id", "1"),
                "session_id": request_payload.get("session_id", "default-session"),
                "heart_rate": result.get("effective_heart_rate", request_payload.get("heart_rate")),
                "blood_oxygen": result.get("effective_blood_oxygen"),
                "anomaly_value": request_payload.get("anomaly_value", 0.0),
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
                "metadata": {"orchestrator": "embedded_api", "language": request_payload.get("language", "en")},
                "source": "medical_assistant_embedded_api",
            }
        )
        return result

    @api_app.post("/api/vital-sample")
    def vital_sample(body: VitalSampleRequest):
        import vital_sample_session as vsess

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
            orchestrate_fn=submit_orchestrator_task,
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
            push_orchestrator_event_to_platform(
                {
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
                    "metadata": {
                        "orchestrator": "embedded_api",
                        "trigger": "vital_sample",
                        "clinical": out.get("clinical"),
                    },
                    "source": "medical_assistant_embedded_api",
                }
            )
        return out

    @api_app.get("/api/live-vitals")
    def live_vitals(user_id: str = "1", session_id: str = "default-session"):
        import vital_sample_session as vsess

        data = fetch_latest_firebase_vitals(user_id=user_id)
        sim = vsess.get_last_simulated_vitals(user_id, session_id)
        ok_fb = bool(data.get("ok"))
        ok_sim = bool(sim.get("ok"))
        return {
            "ok": ok_fb or ok_sim,
            "firebase_vitals": data,
            "simulator_vitals": sim,
            "effective_heart_rate": data.get("heart_rate") if ok_fb else sim.get("heart_rate"),
            "effective_blood_oxygen": data.get("blood_oxygen") if ok_fb else sim.get("blood_oxygen"),
        }

    def run_api():
        print(f" [API] Starting embedded REST API on http://{API_HOST}:{API_PORT}")
        uvicorn.run(api_app, host=API_HOST, port=API_PORT, log_level="warning")

    API_THREAD = threading.Thread(target=run_api, daemon=True)
    API_THREAD.start()
    time.sleep(0.8)

# ----------------------------
# Main loop
# ----------------------------
def main():
    # Check if collection exists before starting
    try:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            print(f"❌ ERROR: Collection '{COLLECTION_NAME}' not found in Qdrant!")
            print("\n💡 Solution: Index your data first:")
            print("   python index_data.py")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Cannot check collections: {e}")
        sys.exit(1)
    
    # Start WebSocket Server + embedded REST API
    start_websocket_server()
    start_orchestrator_api_server()
    start_orch_warm_worker()
    start_elysa_wakeword_listener(INPUT_QUEUE, get_language=get_client_voice_language)

    # Start Console Input Thread
    def console_input():
        while True:
            try:
                # We use a prompt in the loop below, but for thread we just want raw input
                # However, to keep it clean, we might just print nothing here and let main handle prompts
                # But input() blocks. So we need a way to show prompt.
                # Simple hack: Main loop prints prompt, this waits.
                text = input()
                INPUT_QUEUE.put(text)
            except EOFError:
                break
    
    t_console = threading.Thread(target=console_input, daemon=True)
    t_console.start()

    print("🚑 Medical Assistant: Fainting & Syncope Specialist (type 'exit' to quit)\n")
    
    print("Ask for help with:")
    print("  • Feeling dizzy or faint")
    print("  • Sudden loss of consciousness")
    print("  • Recovery position instructions")
    print("  • Emergency signs")
    print("  • Say \"Elysa\" (wake word) or type '\\voice' to speak\n")
    
    history = []
    
    print("You: ", end="", flush=True) # Initial prompt

    while True:
        try:
            # Process pending orchestrator tasks on main loop thread.
            while True:
                try:
                    task = ORCHESTRATOR_QUEUE.get_nowait()
                except queue.Empty:
                    break

                payload = task.get("payload", {})
                response_queue = task.get("response_queue")
                result = run_simulation_orchestrator(payload)
                if response_queue:
                    try:
                        response_queue.put_nowait(result)
                    except Exception:
                        pass

            # Non-blocking check for input from either Console or WebSocket
            try:
                queue_item = INPUT_QUEUE.get(timeout=0.1)
            except queue.Empty:
                continue

            # Handle both string (console) and dict (websocket) inputs
            current_language = None
            elysa_wake_greeting = False
            if isinstance(queue_item, dict):
                user_input = queue_item.get("text", "")
                current_language = queue_item.get("language", "en")
                elysa_wake_greeting = bool(queue_item.get("elysa_greeting"))
            else:
                user_input = str(queue_item)

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                print("You: ", end="", flush=True) # Reprompt
                continue
            
            # Since we got input, we probably want to reprint prompt for next time after processing
            # But first let's handle the current input
            
            def clear_input_queue():
                """Drain all pending inputs from the queue."""
                while not INPUT_QUEUE.empty():
                    try:
                        INPUT_QUEUE.get_nowait()
                    except queue.Empty:
                        break

            
            # Voice: WebSocket / wake word "Elysa" enqueue \\voice — share mic with OpenWakeWord
            is_voice_cmd = user_input.strip().lower() == "\\voice"
            if is_voice_cmd:
                pause_wakeword_mic()
                # Wake + mic button can stack \\voice while listening; drop duplicate bursts
                to_requeue = []
                while True:
                    try:
                        pending_item = INPUT_QUEUE.get_nowait()
                    except queue.Empty:
                        break
                    pending_text = (
                        pending_item.get("text", "").strip().lower()
                        if isinstance(pending_item, dict)
                        else str(pending_item).strip().lower()
                    )
                    if pending_text == "\\voice":
                        continue
                    to_requeue.append(pending_item)
                for item in to_requeue:
                    INPUT_QUEUE.put(item)
            voice_mode = False
            voice_wav_bytes = None
            try:
                if is_voice_cmd:
                    print(f"🎤 Starting voice mode... (Language: {current_language})")
                    if elysa_wake_greeting:
                        play_elysa_wake_greeting()
                    transcribe_result = listen_and_transcribe(return_audio=True)
                    if transcribe_result:
                        transcribed_text, voice_wav_bytes = transcribe_result
                    else:
                        transcribed_text = None
                    if not transcribed_text:
                        broadcast_state("neutral")
                        continue
                    print(f"🎤 You said: {transcribed_text}")
                    user_input = transcribed_text
                    voice_mode = True

                if is_gratitude_only_message(user_input):
                    history.append((user_input, THANKS_HISTORY_TEXT))
                    print(f"Medical_Assistant: {THANKS_HISTORY_TEXT}\n")
                    play_thanks_reply_cached()
                    if voice_mode:
                        clear_input_queue()
                    continue

                raw_reply = get_response(
                    user_input,
                    history,
                    language=current_language,
                    voice_wav_bytes=voice_wav_bytes,
                )

                # Debug actual output
                print(f"DEBUG: Raw Reply: {raw_reply!r}")

                parsed_response = parse_agent_json(raw_reply)
                print(f"DEBUG: Parsed Successfully: {parsed_response}")

                agent_response = handle_agent_response(parsed_response)

                if agent_response["type"] == "action":
                    broadcast_state("thinking")

                    action_name = agent_response["name"]
                    action_args = agent_response["args"]
                    print(f"⚙️ EXECUTING ACTION: {action_name} {action_args}")

                    result = execute_action(action_name, action_args)
                    print(f"✅ ACTION RESULT: {result}")

                    if result.get("status") == "played":
                        reply_text = "I have sounded the alert."
                    elif action_name == "send_whatsapp_alert" and result.get("status") == "sent":
                        reply_text = "Action completed, WhatsApp alert sent."
                    else:
                        reply_text = f"Action completed: {result}"

                    broadcast_state("speaking")
                    if voice_mode:
                        if action_name == "send_whatsapp_alert" and result.get("status") == "sent":
                            play_whatsapp_alert_sent_prompt()
                        else:
                            speak_response(reply_text)
                        clear_input_queue()

                    history.append((user_input, reply_text))
                    print(f"Medical_Assisstant: {reply_text}\n")

                else:
                    reply_text = agent_response["text"]
                    history.append((user_input, reply_text))
                    print(f"Medical_Assisstant: {reply_text}\n")

                    if voice_mode:
                        speak_response(reply_text)
                        clear_input_queue()

            finally:
                if is_voice_cmd:
                    schedule_wakeword_resume_after_voice_session()

            print("You: ", end="", flush=True)
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)
