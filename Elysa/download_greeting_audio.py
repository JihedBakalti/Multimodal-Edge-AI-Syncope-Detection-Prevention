#!/usr/bin/env python3
"""
One-shot: generate cached ElevenLabs prompts used by medical_assistant.
Run from project root:  python Elysa/download_greeting_audio.py
Requires ELEVENLABS_API_KEY in .env or environment.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROMPTS = [
    (
        "Hello, I am Elysa, your medical assistant. How can I help you?",
        "elysa_greeting.mp3",
    ),
    (
        "Action completed. WhatsApp alert sent.",
        "whatsapp_alert_sent.mp3",
    ),
    # English only: user may say thanks in any language; assistant always replies in English.
    (
        "You're welcome. I'm right here if you need anything else.",
        "thanks_reply_en.mp3",
    ),
    (
        "A critical fainting episode is detected. Automatic escalation is starting.",
        "critical_fainting_escalation.mp3",
    ),
]
VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"
MODEL_ID = "eleven_multilingual_v2"


def main():
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        print("Set ELEVENLABS_API_KEY in .env or the environment.", file=sys.stderr)
        sys.exit(1)
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        print("pip install elevenlabs", file=sys.stderr)
        sys.exit(1)

    client = ElevenLabs(api_key=key)
    base_dir = Path(__file__).resolve().parent
    for text, filename in PROMPTS:
        out_path = base_dir / filename
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format="mp3_44100_128",
        )
        data = b"".join(chunk for chunk in audio_generator if chunk)
        out_path.write_bytes(data)
        print(f"Wrote {out_path} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
