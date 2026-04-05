
# ═══════════════════════════════════════════════════════════════
# test_voice.py — Real-time voice test with trained model
# Run: python test_voice.py
# ═══════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn as nn
import librosa
import sounddevice as sd
import sys
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Model definition (must match training) ──
class VoiceClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ── Config (must match training) ──
SR = 16000
DURATION = 5.0
N_SAMPLES = int(SR * DURATION)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMAX = 8000
IMG_H, IMG_W = 128, 128
LABELS = {0: '✅ Normal', 1: '⚠️  Dysarthric / Risk Detected'}
MODEL_PATH = 'voice_model_export.pt'  # adjust path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = VoiceClassifier().to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model loaded — Val AUC: {checkpoint.get('val_auc', 'N/A')}")
    return model

def audio_to_melspec(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_db = np.array([
        np.interp(np.linspace(0, mel_db.shape[1]-1, IMG_W),
                  np.arange(mel_db.shape[1]), row)
        for row in mel_db
    ])
    return mel_db.astype(np.float32)

def predict_from_array(model, y):
    """Predict from raw numpy audio array."""
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]
    
    mel = audio_to_melspec(y)
    x = torch.tensor(mel).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        pred = out.argmax(1).item()
    
    return pred, probs[0].item(), probs[1].item()

def record_and_predict(model, duration=DURATION):
    print(f"\n🎙️  Recording {duration}s — speak now (any language, any words)...")
    audio = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    y = audio.flatten()
    print("✅ Recording done. Analyzing...")
    
    pred, prob_normal, prob_dys = predict_from_array(model, y)
    
    print("\n" + "="*45)
    print(f"  RESULT: {LABELS[pred]}")
    print(f"  Confidence — Normal:     {prob_normal*100:.1f}%")
    print(f"  Confidence — Dysarthric: {prob_dys*100:.1f}%")
    print("="*45)
    
    # Risk level for your fainting detection pipeline
    if prob_dys > 0.7:
        print("  🚨 HIGH RISK — Escalate to emergency system")
    elif prob_dys > 0.4:
        print("  ⚠️  MODERATE RISK — Trigger confirmation dialogue")
    else:
        print("  ✅ LOW RISK — Continue monitoring")
    
    return pred, prob_dys

def predict_from_file(model, path):
    """Predict from an existing audio file."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    pred, prob_normal, prob_dys = predict_from_array(model, y)
    print(f"File: {path}")
    print(f"Result: {LABELS[pred]} | Normal: {prob_normal:.3f} | Dysarthric: {prob_dys:.3f}")
    return pred, prob_dys

# ── Main ──
if __name__ == "__main__":
    model = load_model()
    
    # Mode: realtime or file
    if len(sys.argv) > 1:
        # python test_voice.py path/to/audio.wav
        predict_from_file(model, sys.argv[1])
    else:
        # Interactive loop: press Enter to record, Ctrl+C to stop
        print("\n🎤 Real-time Voice Risk Detector")
        print("Speak any language — model is language agnostic")
        print("Press Enter to record | Ctrl+C to exit\n")
        
        while True:
            try:
                input("Press Enter to start recording...")
                record_and_predict(model)
            except KeyboardInterrupt:
                print("\nExiting. Goodbye!")
                break
