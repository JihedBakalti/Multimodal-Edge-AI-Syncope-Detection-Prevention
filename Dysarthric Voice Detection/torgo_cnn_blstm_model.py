import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
import time

# --- 1. CONFIGURATION ---
SAMPLING_RATE = 16000
DURATION = 3.0  # Recording window per word

# --- 2. CALIBRATION (CRITICAL) ---
TRAIN_MEAN = -64.862709045410156

TRAIN_STD = 17.011808395385742  

# --- 3. THE CLINICAL WORD LIST ---
# These words test different "articulators" (lips, tongue, breath)
TEST_WORDS = [
    "Mama",          # Labial (Lips)
    "Tip-top",       # Lingual (Tongue tip)
    "Fifty-fifty",   # Coordination
    "Basketball",    # Rhythm/Stutter detection
    "Huckleberry"    # Breath support/Trailing off
]

# --- 4. MODEL LOADING & CUSTOM LAYER ---
@tf.keras.utils.register_keras_serializable()
class SpecAugment(tf.keras.layers.Layer):
    def __init__(self, freq_mask_max=15, time_mask_max=10, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
    def call(self, inputs, training=None):
        return inputs # During test, we don't augment
    def get_config(self):
        config = super().get_config()
        config.update({"freq_mask_max": self.freq_mask_max, "time_mask_max": self.time_mask_max})
        return config

print("--- System Initializing ---")
try:
    model = tf.keras.models.load_model('best_crnn.keras', custom_objects={'SpecAugment': SpecAugment})
    print("✓ Speech Diagnostic Model Loaded")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit()

def analyze_audio(audio_data):
    # 1. Trim leading/trailing silence (TopDB=20 is sensitive)
    trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=20)
    if np.max(np.abs(trimmed_audio)) > 0:
        trimmed_audio = trimmed_audio / np.max(np.abs(trimmed_audio))
    # 2. Extract Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=trimmed_audio, sr=SAMPLING_RATE, n_mels=128, fmax=8000)
    
    # 3. Use np.max to normalize energy within the clip
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # 4. Normalize with your training constants
    log_mel_norm = (log_mel - TRAIN_MEAN) / TRAIN_STD
    
    # 5. Fixed-width Reshape (Your model expects 94 time steps)
    if log_mel_norm.shape[1] < 94:
        pad = 94 - log_mel_norm.shape[1]
        log_mel_norm = np.pad(log_mel_norm, ((0, 0), (0, pad)), mode='constant')
    else:
        log_mel_norm = log_mel_norm[:, :94]

    input_tensor = log_mel_norm.T[np.newaxis, ..., np.newaxis]
    return model.predict(input_tensor, verbose=0)[0][0]

# --- 5. THE ASSESSMENT LOOP ---
print("\n=== STARTING SPEECH LAYER ASSESSMENT ===")
print("Ask the subject to repeat each word clearly.\n")

word_scores = []

for word in TEST_WORDS:
    print(f"READY? Next word is: [{word.upper()}]")
    input(">> Press Enter to start 3s recording...")
    
    print(f"🔴 RECORDING '{word}'...")
    recording = sd.rec(int(DURATION * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1)
    sd.wait()
    
    # Process
    audio_data = recording.flatten()
    
    # Check for empty audio/silence
    if np.max(np.abs(audio_data)) < 0.01:
        print("⚠️ No voice detected. Scoring as HIGH RISK.")
        score = 0.9
    else:
        score = analyze_audio(audio_data)
        status = "⚠️ SLURRED/STUTTER" if score >= 0.6 else "✅ NORMAL"
        print(f"   Result: {status} (Score: {score:.2f})")
    
    word_scores.append(score)
    print("-" * 30)

# --- 6. FINAL DATA FUSION OUTPUT ---
final_risk = np.mean(word_scores)
print("\n" + "="*40)
print(f"FINAL SPEECH RISK SCORE: {final_risk:.4f}")

if final_risk > 0.65:
    print("STATUS: CRITICAL - HIGH SYNCOPE PROBABILITY")
    print("ACTION: Trigger emergency protocol. (Check CV/Vitals logs)")
elif final_risk > 0.45:
    print("STATUS: WARNING - SPEECH IRREGULARITY DETECTED")
else:
    print("STATUS: STABLE - NORMAL SPEECH PATTERNS")
print("="*40)