import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. MODEL ARCHITECTURE (skeleton to load the weights into)
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
        return self.lm_head(x)  # [B, T, 103] — predicted next token probabilities

# ============================================================
# 2. LOAD PRETRAINED WEIGHTS
# ============================================================

print("Loading pretrained PPG-PT...")
model = PPGPT()
state_dict = torch.load('Model_files/PPGPT_500k_iters.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 3. LOAD YOUR MIMIC PPG SIGNAL
# ============================================================

ppg_raw = np.load("data/ppg_raw.npy")
WINDOW_SIZE = 500

# Take 3 different windows from different parts of the signal
window_starts = [0, 10000, 50000]
windows, tokens_list = [], []

for start in window_starts:
    w = ppg_raw[start:start + WINDOW_SIZE]
    normalized = (w - w.min()) / (w.max() - w.min())
    tokenized  = np.clip((normalized * 101).astype(int), 0, 101)
    windows.append(w)
    tokens_list.append(tokenized)

# Stack into batch [3, 500]
input_tensor = torch.tensor(np.array(tokens_list), dtype=torch.long)
print(f"\nInput tensor shape: {input_tensor.shape}")

# ============================================================
# 4. FORWARD PASS — run through pretrained model
# ============================================================

with torch.no_grad():
    logits = model(input_tensor)   # [3, 500, 103]

print(f"Output (logits) shape: {logits.shape}")

# Convert logits to predicted next tokens
predicted_tokens = logits.argmax(dim=-1)  # [3, 500]
print(f"Predicted tokens shape: {predicted_tokens.shape}")

# ============================================================
# 5. VISUALIZE: input tokens vs predicted next tokens
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("Pretrained PPG-PT — Input vs Predicted next token", fontsize=13)

for i in range(3):
    # Left: raw PPG window
    axes[i, 0].plot(windows[i], color='steelblue')
    axes[i, 0].set_title(f"Window {i+1} — Raw PPG (start={window_starts[i]})")
    axes[i, 0].set_ylabel("Amplitude")

    # Right: input tokens vs model prediction
    axes[i, 1].plot(tokens_list[i],              label='Input tokens',     color='steelblue', alpha=0.7)
    axes[i, 1].plot(predicted_tokens[i].numpy(), label='Predicted tokens', color='orange',    alpha=0.7, linestyle='--')
    axes[i, 1].set_title(f"Window {i+1} — Token prediction (pretrained)")
    axes[i, 1].set_ylabel("Token (0–102)")
    axes[i, 1].legend()

axes[2, 0].set_xlabel("Samples")
axes[2, 1].set_xlabel("Samples")

plt.tight_layout()
plt.savefig("pretrained_test.png")
plt.show()

# ============================================================
# 6. RECONSTRUCTION ERROR — how well does it predict itself?
# ============================================================

print("\n--- Reconstruction error per window ---")
for i in range(3):
    error = np.mean(np.abs(
        tokens_list[i][1:] - predicted_tokens[i].numpy()[:-1]
    ))
    print(f"  Window {i+1} (start={window_starts[i]:6d}): MAE = {error:.2f} tokens")

print("\nDone! Check pretrained_test.png")