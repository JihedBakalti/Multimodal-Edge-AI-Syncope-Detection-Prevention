import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# 1. MODEL ARCHITECTURE
# ============================================================

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa   = MultiHeadAttention(n_head, n_embd // n_head, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class PPGPT(nn.Module):
    def __init__(self, vocab_size=102, n_embd=64, n_head=8, n_layer=8, block_size=500):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + \
            self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        return self.lm_head(x)

# ============================================================
# 2. LOAD PRETRAINED MODEL
# ============================================================

print("Loading pretrained PPG-PT...")
model = PPGPT()
model.load_state_dict(torch.load('Model_files/PPGPT_500k_iters.pth', map_location='cpu'))
model.eval()
print("Ready.\n")

# ============================================================
# 3. CORE FUNCTIONS
# ============================================================

# Threshold calibrated from your test results:
# Window 1 (normal) → 1.89 | Window 2 (anomaly) → 9.48 | Window 3 (normal) → 3.52
THRESHOLD = 5.0

def preprocess(window):
    """Normalize and tokenize a raw PPG window."""
    mn, mx = window.min(), window.max()
    if mx - mn < 1e-6:
        return None  # flat/dead signal
    normalized = (window - mn) / (mx - mn)
    return np.clip((normalized * 101).astype(int), 0, 101)

def compute_mae(tokens):
    """Run tokens through pretrained model, return reconstruction MAE."""
    tensor = torch.tensor(tokens).unsqueeze(0)  # [1, 500]
    with torch.no_grad():
        logits = model(tensor)                  # [1, 500, 102]
    predicted = logits.argmax(dim=-1).squeeze().numpy()  # [500]
    return float(np.mean(np.abs(tokens[1:] - predicted[:-1])))

def predict(window):
    """
    Given a raw PPG window (numpy array, 500 samples),
    return: label, confidence, mae
    """
    tokens = preprocess(window)
    if tokens is None:
        return "ANOMALY", 100.0, 999.0  # flat signal = anomaly

    mae = compute_mae(tokens)

    # Confidence: how far is MAE from threshold (capped at 99%)
    distance   = abs(mae - THRESHOLD)
    confidence = min(99.0, round(50.0 + (distance / THRESHOLD) * 50.0, 1))

    if mae < THRESHOLD:
        return "NORMAL", confidence, round(mae, 2)
    else:
        return "ANOMALY", confidence, round(mae, 2)

# ============================================================
# 4. SCAN YOUR ENTIRE MIMIC SIGNAL
# ============================================================

print("Scanning full PPG signal...")
ppg_raw     = np.load("data/ppg_raw.npy")
WINDOW_SIZE = 500
STEP        = 250   # slide every 2 seconds

results = []
for start in range(0, len(ppg_raw) - WINDOW_SIZE, STEP):
    window          = ppg_raw[start:start + WINDOW_SIZE]
    label, conf, mae = predict(window)
    time_sec        = start / 125.0   # 125 Hz
    results.append({
        'start':    start,
        'time_sec': time_sec,
        'label':    label,
        'conf':     conf,
        'mae':      mae
    })

# ============================================================
# 5. PRINT SUMMARY
# ============================================================

total    = len(results)
anomalies = [r for r in results if r['label'] == 'ANOMALY']
normals   = [r for r in results if r['label'] == 'NORMAL']

print(f"\n{'='*50}")
print(f"  SCAN RESULTS")
print(f"{'='*50}")
print(f"  Total windows scanned : {total}")
print(f"  Normal windows        : {len(normals)}  ({100*len(normals)/total:.1f}%)")
print(f"  Anomaly windows       : {len(anomalies)}  ({100*len(anomalies)/total:.1f}%)")
print(f"  Threshold (MAE)       : {THRESHOLD}")
print(f"{'='*50}")

if anomalies:
    print(f"\n  Anomalies detected at:")
    for r in anomalies[:20]:   # print first 20
        m, s = divmod(int(r['time_sec']), 60)
        print(f"    t={m:02d}:{s:02d}  MAE={r['mae']:.2f}  confidence={r['conf']}%")
    if len(anomalies) > 20:
        print(f"    ... and {len(anomalies)-20} more")

# ============================================================
# 6. PLOT: MAE over time + anomaly markers
# ============================================================

times = [r['time_sec'] for r in results]
maes  = [r['mae']      for r in results]
colors = ['red' if r['label'] == 'ANOMALY' else 'steelblue' for r in results]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("PPG Anomaly Detection — Pretrained PPG-PT", fontsize=13)

# Top: MAE over time
ax1.scatter(times, maes, c=colors, s=10, alpha=0.6)
ax1.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=1.5, label=f'Threshold = {THRESHOLD}')
ax1.set_ylabel("Reconstruction MAE")
ax1.set_xlabel("Time (seconds)")
ax1.set_title("Reconstruction error over time (red = anomaly)")
normal_patch  = mpatches.Patch(color='steelblue', label='Normal')
anomaly_patch = mpatches.Patch(color='red',       label='Anomaly')
ax1.legend(handles=[normal_patch, anomaly_patch, 
           plt.Line2D([0],[0], color='red', linestyle='--', label=f'Threshold={THRESHOLD}')])

# Bottom: raw PPG signal with anomaly regions shaded
ax2.plot(np.arange(len(ppg_raw)) / 125.0, ppg_raw, color='steelblue', linewidth=0.4, alpha=0.8)
for r in anomalies:
    ax2.axvspan(r['time_sec'], r['time_sec'] + WINDOW_SIZE/125.0, color='red', alpha=0.15)
ax2.set_ylabel("PPG Amplitude")
ax2.set_xlabel("Time (seconds)")
ax2.set_title("Raw PPG signal (red shading = detected anomaly regions)")

plt.tight_layout()
plt.savefig("anomaly_scan.png")
plt.show()
print("\nPlot saved to anomaly_scan.png")