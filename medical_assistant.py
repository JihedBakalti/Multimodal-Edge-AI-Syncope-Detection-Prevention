#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("Starting medical_assistant.py...")

# Suppress warnings - these NumPy warnings on Windows are harmless
import os
import sys
import warnings

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
    import queue
    import websockets
    import re
    import ast
    import webbrowser
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
    print(f"  [FAIL] sentence_transformers: {e}")
    print("   Install with: pip install sentence-transformers")
    # import traceback
    # traceback.print_exc()
    sys.exit(1)

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
    "play_alert_sound": play_alert_sound,
    "play_alert_sound": play_alert_sound,
    "send_whatsapp_alert": send_whatsapp_alert
}

# Tool Definitions for LLM
TOOLS = [
    {
        "name": "play_alert_sound",
        "description": "Play an alert sound (horns, alarm) to signal the user.",
        "parameters": {"level": {"type": "string", "enum": ["low","medium","high"]}}
    },
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

# ----------------------------
# WebSocket Server & Input Handling
# ----------------------------
CONNECTED_CLIENTS = set()
WS_PORT = 8765
INPUT_QUEUE = queue.Queue()
ORCHESTRATOR_QUEUE = queue.Queue()
WS_LOOP = None

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
                if data.get("type") == "command":
                    cmd = data.get("content")
                    language = data.get("language", "en") # Default to english
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
print("[INFO] Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("[OK] Embedding model loaded.\n")

def get_embedding(text):
    """Generate embedding using local SentenceTransformer model."""
    return model.encode(text).tolist()

# ----------------------------
# LLM Setup (Gemini)
# ----------------------------
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def query_gemini(messages):
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
def listen_and_transcribe():
    """Listen to microphone and transcribe using Groq Whisper."""
    if not sr:
        print("❌ SpeechRecognition not installed.")
        return None
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env.")
        return None

    r = sr.Recognizer()
    # Optimize VAD for speed
    r.energy_threshold = 300  # Default 300, can adjust dynamic
    r.pause_threshold = 1.2   # Increased to 1.2 for more natural pauses
    r.non_speaking_duration = 0.5 # Increased stability
    r.dynamic_energy_threshold = True

    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        with sr.Microphone() as source:
            broadcast_state("listening")
            print("\n🎤 Listening... (Speak now!)")
            # Calibrate faster
            r.adjust_for_ambient_noise(source, duration=0.4)
            
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
            return transcription.text.strip()
            
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
                    voice_id="nPczCjzI2devNBz1zQrb",
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

# Maximum number of conversation exchanges to keep in history
MAX_HISTORY_EXCHANGES = 5

def get_response(user_message, history, language=None, system_prompt_append=None):
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

    # Persona based on language
    # Persona based on language
    if language == "fr":
        persona = (
            "Tu es un Assistant Médical spécialisé dans les évanouissements soudains (syncope). "
            "Ton rôle est de calmer l'utilisateur et de donner des instructions claires et étape par étape. "
            "Tu es rassurant, direct et professionnel. "
            "Si l'utilisateur dit qu'il se sent mal, guide-le immédiatement : s'asseoir, s'allonger, lever les jambes. "
            "Utilise tes outils si nécessaire (alarme). "
            "Concentre-toi uniquement sur l'urgence médicale."
        )
    elif language == "ar":
        persona = (
            "أنت مساعد طبي متخصص في حالات الإغماء المفاجئ. "
            "دورك هو تهدئة المستخدم وإعطاء تعليمات واضحة خطوة بخطوة. "
            "أنت مطمئن ومباشر ومحترف. "
            "إذا قال المستخدم إنه يشعر بالدوار، وجهه فورًا: اجلس، استلقِ، ارفع ساقيك. "
            "استخدم أدواتك عند الضرورة (التنبيه). "
            "ركز فقط على الحالة الطبية الطارئة."
        )
    elif language == "tn":
        persona = (
            "انت مساعد طبي (Medical Assistant) مختص في الدوخة والإغماء. "
            "دورك تتكلم برزانة وتوسع بالك، وتعطي تعليمات واضحة بالدارجة التونسية. "
            "طمن العبد اللي معاك، وقوله شنوا يعمل بالضبط (يقعد، يتمد، يهز ساقيه). "
            "ركز كان على صحة السيد اللي معاك. كان لازم، اطلب الاسعاف ولا استعمل التنبيه."
        )
    else:  # Default to English
        persona = (
            "You are a Medical Assistant specialized in Sudden Fainting (Syncope). "
            "Your role is to calm the user down and provide clear, step-by-step instructions. "
            "You are reassuring, direct, and professional. "
            "If the user says they feel faint, guide them immediately: sit down, lie down, elevate legs. "
            "Use your tools if necessary (alert sound). "
            "Focus ONLY on the medical emergency."
        )

    # Build system message with context
    tools_json = json.dumps(TOOLS, indent=2)
    system_message = f"""{persona}

INSTRUCTIONS:
- You are a Medical Assistant.
- **PRIORITY**: If the user needs immediate attention or an alert, USE THE TOOL.
- **CALM INSTRUCTIONS**: Guide the user step-by-step.
    - `play_alert_sound` = "Signal for help/alert others."
    - `send_whatsapp_alert` = "Notify family/friends."
- Do NOT mention being a General or Stratageist.
- Only use context provided.
- Keep language simple and clear for users.
- Admit missing info.
- Reject off-topic questions politely.
- Keep responses <60 words.
- Provide practical, actionable advice.

AVAILABLE TOOLS:
{tools_json}

RESPONSE FORMAT:
You must respond in JSON format.
If you want to speak, return:
{{ "type": "speech", "text": "Your response here" }}

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
        time.sleep(1)
    
    return query_gemini(messages)


def run_warning_voice_check(heart_rate, language="en"):
    """Wake the voice agent with dynamic prompt injection."""
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


def run_no_human_safety_check(heart_rate, anomaly_value, language="en"):
    """
    Ask up to 3 verbal safety checks.
    Escalate to WhatsApp alert if user is symptomatic or does not answer clearly.
    """
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        check_prompt = (
            f"Safety check attempt {attempt} of {max_attempts}. "
            f"A vital signal anomaly was detected with value {anomaly_value:.2f} and heart rate {heart_rate} BPM. "
            "Are you feeling okay? Are you dizzy, lightheaded, weak, or experiencing any symptoms? "
            "Please answer clearly."
        )
        broadcast_state("speaking")
        speak_response(check_prompt)

        user_reply = listen_and_transcribe()
        if user_reply:
            print(f"🎤 [Safety Check] User said: {user_reply}")
        else:
            print("🎤 [Safety Check] No speech captured on first listen, retrying once...")
            user_reply = listen_and_transcribe()
            if user_reply:
                print(f"🎤 [Safety Check] User said (retry): {user_reply}")
            else:
                print("🎤 [Safety Check] No speech captured.")
        response_class = classify_symptom_response_with_llm(user_reply, language=language)

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


def run_human_detected_safety_check(heart_rate, anomaly_value, dl_risk_score, language="en"):
    """
    Human detected with no critical DL event:
    contact user up to 3 times and escalate on symptoms/unclear responses.
    """
    max_attempts = 3
    history = []

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

        user_reply = listen_and_transcribe()
        if user_reply:
            print(f"🎤 [Proactive Check] User said: {user_reply}")
        else:
            print("🎤 [Proactive Check] No speech captured on first listen, retrying once...")
            user_reply = listen_and_transcribe()
            if user_reply:
                print(f"🎤 [Proactive Check] User said (retry): {user_reply}")
            else:
                print("🎤 [Proactive Check] No speech captured.")
        response_class = classify_symptom_response_with_llm(user_reply, language=language)
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


def run_simulation_orchestrator(payload):
    """Process simulator payload and trigger InterSense state actions."""
    decision = evaluate_simulation_state(payload)
    state = decision.get("state")
    heart_rate = int(payload.get("heart_rate", 70))
    anomaly_value = float(payload.get("anomaly_value", 0.0))
    dl_risk_score = float(payload.get("dl_risk_score", 0.0))

    if state == "normal":
        return {
            "ok": True,
            "state": state,
            "message": "System Normal. Monitoring...",
        }

    if state == "critical_emergency":
        alert_message = (
            "INTERSENSE CRITICAL EMERGENCY: Wearable anomaly + human detected + critical DL detection. "
            f"Simulated heart rate: {heart_rate} BPM. DL risk score: {dl_risk_score:.2f}. Immediate assistance required."
        )
        alert_result = send_whatsapp_alert(message_text=alert_message)
        return {
            "ok": True,
            "state": state,
            "message": "Critical emergency from deep learning path: WhatsApp alert sent.",
            "alert_result": alert_result,
        }

    if state == "warning":
        language = payload.get("language", "en")
        warning_result = run_human_detected_safety_check(
            heart_rate=heart_rate,
            anomaly_value=anomaly_value,
            dl_risk_score=dl_risk_score,
            language=language,
        )
        return {
            "ok": True,
            "state": state,
            "message": "Human detected with non-critical DL result: proactive voice check completed.",
            "warning_result": warning_result,
        }

    if state == "no_action":
        language = payload.get("language", "en")
        safety_result = run_no_human_safety_check(
            heart_rate=heart_rate,
            anomaly_value=anomaly_value,
            language=language,
        )
        return {
            "ok": True,
            "state": state,
            "message": "No human target confirmed; completed 3-attempt voice safety protocol.",
            "safety_result": safety_result,
        }

    return {
        "ok": True,
        "state": state,
        "message": decision.get("message", "No action taken."),
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


def start_orchestrator_api_server():
    """Run /api/simulate on the same process as medical_assistant.py."""
    global API_THREAD

    if FastAPI is None or uvicorn is None or BaseModel is None:
        print("⚠️ Orchestrator API disabled (fastapi/uvicorn missing).")
        return

    class SimulationRequest(BaseModel):
        heart_rate: int = Field(..., ge=40, le=180)
        anomaly_value: float = Field(0.0, ge=0.0)
        dl_risk_score: float = Field(0.0, ge=0.0, le=1.0)
        wearable_anomaly: bool
        human_detected: bool
        fainting_detected: bool
        language: str = "en"

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
        return submit_orchestrator_task(payload.model_dump())

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
    print("  • Type '\\voice' to speak\n")
    
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
            if isinstance(queue_item, dict):
                user_input = queue_item.get("text", "")
                current_language = queue_item.get("language", "en")
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

            
            # Voice Input Handler
            voice_mode = False
            if user_input.strip().lower() == "\\voice":
                print(f"🎤 Starting voice mode... (Language: {current_language})") 
                transcribed_text = listen_and_transcribe()
                if transcribed_text:
                    print(f"🎤 You said: {transcribed_text}")
                    user_input = transcribed_text
                    voice_mode = True
                else:
                    broadcast_state("neutral")
                    continue

            raw_reply = get_response(user_input, history, language=current_language)
            
            # Debug actual output
            print(f"DEBUG: Raw Reply: {raw_reply!r}")

            parsed_response = parse_agent_json(raw_reply)
            print(f"DEBUG: Parsed Successfully: {parsed_response}")

            
            # Helper to handle the parsed response
            agent_response = handle_agent_response(parsed_response)
            
            if agent_response["type"] == "action":
                broadcast_state("thinking") # Show we are processing action
                
                # Execute Action
                action_name = agent_response["name"]
                action_args = agent_response["args"]
                print(f"⚙️ EXECUTING ACTION: {action_name} {action_args}")
                
                result = execute_action(action_name, action_args)
                print(f"✅ ACTION RESULT: {result}")
                
                # For this MVP, we just speak a confirmation or the result
                # In a full loops, we'd feed this back to LLM.
                # Construct a simple speech response based on result
                if result.get("status") == "played":
                    reply_text = "I have sounded the alert."
                else:
                    reply_text = f"Action completed: {result}"
                
                # Transition to speaking
                broadcast_state("speaking")
                if voice_mode:
                    speak_response(reply_text)
                    clear_input_queue()
                
                history.append((user_input, reply_text))
                print(f"Medical_Assisstant: {reply_text}\n")
                
            else:
                # Normal Speech
                reply_text = agent_response["text"]
                history.append((user_input, reply_text))
                print(f"Medical_Assisstant: {reply_text}\n")
                
                if voice_mode:
                    speak_response(reply_text)
                    clear_input_queue()
            
            print("You: ", end="", flush=True) # Ready for next input
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
