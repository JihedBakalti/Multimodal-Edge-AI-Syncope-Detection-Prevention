import pyaudio
import numpy as np
from openwakeword.model import Model

# 1. Load your custom model
oww_model = Model(wakeword_models=["Elysa.onnx"])

# 2. Set up the microphone audio stream
CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("\nMicrophone is ON. Listening for 'Elysa'... (Press Ctrl+C to stop)")

# 3. Create an infinite loop to keep listening
try:
    while True:
        # Read audio from the microphone
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Get the prediction score
        prediction = oww_model.predict(audio_data)
        score = prediction['Elysa']
        
        # --- NEW: Print the score constantly on the same line ---
        print(f"Current confidence score: {score:.3f}    ", end='\r')
        
        # If the score crosses the threshold, print the alert!
        if score > 0.3:
            print(f"\n*** WAKE WORD DETECTED! *** (Score: {score:.3f})")

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    audio.terminate()