#!/usr/bin/env python3
"""
One-shot: generate elysa_greeting.mp3 via ElevenLabs (same voice as medical_assistant).
Run from project root:  python Elysa/download_greeting_audio.py
Requires ELEVENLABS_API_KEY in .env or environment.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TEXT = "hello , I am Elysa your medical Assisstant , How can i help you ?"
OUT_PATH = Path(__file__).resolve().parent / "elysa_greeting.mp3"
VOICE_ID = "nPczCjzI2devNBz1zQrb"
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
    audio_generator = client.text_to_speech.convert(
        text=TEXT,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format="mp3_44100_128",
    )
    data = b"".join(chunk for chunk in audio_generator if chunk)
    OUT_PATH.write_bytes(data)
    print(f"Wrote {OUT_PATH} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
