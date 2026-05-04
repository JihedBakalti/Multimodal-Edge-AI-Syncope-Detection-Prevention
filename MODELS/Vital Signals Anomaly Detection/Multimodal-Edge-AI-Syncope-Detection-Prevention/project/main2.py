#!/usr/bin/env python3
"""
main2.py — Poll vitals JSON → Synthetic PPG → PPGPT → results.json

Watches --input (default: project/spo2.json) for changes; on each update writes one
results.json: PPGPT (mae_score, anomaly_detected, threshold) plus clinical fields from
realtime_monitoring.compute_clinical_for_merge() (same logic as scores.json).

Usage:
  python project/main2.py
  python project/main2.py -i path/to/vitals.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

_PROJECT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PROJECT_DIR.parent
_ALA_ROOT = _PROJECT_DIR.parent.parent
if str(_ALA_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALA_ROOT))

import realtime_monitoring as rt

# ──────────────────────────────────────────────────────────────────────────────
# 1. PPGPT MODEL (identical to main.py)
# ──────────────────────────────────────────────────────────────────────────────

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
        self.proj = nn.Linear(num_heads * head_size, n_embd)

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
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size, n_embd, block_size)
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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.ln_f(self.blocks(x))
        return self.lm_head(x)  # [B, T, vocab_size]


# ──────────────────────────────────────────────────────────────────────────────
# 2. SYNTHETIC IR PPG GENERATOR (identical)
# ──────────────────────────────────────────────────────────────────────────────

def synthesise_ir_ppg(
    heart_rate_bpm: float,
    spo2_percent: float,
    n_samples: int = 500,
    fs: float = 100.0,
    noise_level: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t = np.arange(n_samples) / fs
    f0 = heart_rate_bpm / 60.0

    fundamental = np.sin(2 * np.pi * f0 * t)
    dicrotic_phase = np.pi * 0.55
    dicrotic = 0.55 * np.sin(2 * np.pi * 2 * f0 * t + dicrotic_phase)

    waveform = fundamental + dicrotic
    waveform = waveform - waveform.min()

    spo2_norm = np.clip(spo2_percent / 100.0, 0.7, 1.0)
    ac_dc_ratio = 0.05 + 0.15 * (spo2_norm - 0.7) / 0.3

    dc_component = 1.0
    ac_amplitude = dc_component * ac_dc_ratio
    peak = waveform.max() if waveform.max() > 0 else 1.0
    waveform = dc_component + ac_amplitude * (waveform / peak)

    waveform += noise_level * rng.standard_normal(n_samples)
    waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-9)
    return waveform.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# 3. TOKENISATION (identical)
# ──────────────────────────────────────────────────────────────────────────────

def tokenise_ppg(signal: np.ndarray) -> np.ndarray:
    tokens = np.clip((signal * 101).astype(int), 0, 101)
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# 4. ANOMALY SCORING (identical)
# ──────────────────────────────────────────────────────────────────────────────

def compute_anomaly_score(model: PPGPT, tokens: np.ndarray) -> tuple[float, np.ndarray]:
    input_tensor = torch.tensor(tokens[np.newaxis, :], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)
    predicted_tokens = logits.argmax(dim=-1).squeeze(0).numpy()

    per_token_error = np.abs(predicted_tokens[:-1] - tokens[1:]).astype(float)
    mae_score = float(per_token_error.mean())
    return mae_score, per_token_error


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLINICAL FLAGS (identical)
# ──────────────────────────────────────────────────────────────────────────────

def clinical_flags(bpm: float, spo2: float) -> list[str]:
    flags = []
    if bpm < 60:
        flags.append(f"Bradycardia — Heart rate {bpm:.0f} bpm (normal: 60–100)")
    elif bpm > 100:
        flags.append(f"Tachycardia — Heart rate {bpm:.0f} bpm (normal: 60–100)")
    else:
        flags.append(f"Normal heart rate: {bpm:.0f} bpm")

    if spo2 < 90:
        flags.append(f"Severe hypoxaemia — SpO2 {spo2:.0f}% (critical: <90%)")
    elif spo2 < 95:
        flags.append(f"Mild hypoxaemia — SpO2 {spo2:.0f}% (normal: ≥95%)")
    else:
        flags.append(f"Normal SpO2: {spo2:.0f}%")

    return flags


# ──────────────────────────────────────────────────────────────────────────────
# 6. PLOTTING (adapted filename)
# ──────────────────────────────────────────────────────────────────────────────

def save_diagnostic_plot(
    signal: np.ndarray,
    tokens: np.ndarray,
    predicted_tokens: np.ndarray,
    per_token_error: np.ndarray,
    mae_score: float,
    anomaly: bool,
    bpm: float,
    spo2: float,
    output_path: str = "ppgpt_anomaly_result_spo2.png",
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9))
    colour = "#d62728" if anomaly else "#2ca02c"
    status = "⚠  ANOMALY DETECTED" if anomaly else "✓  NORMAL"

    fig.suptitle(
        f"PPGPT Anomaly Detection (SpO2 monitor) | BPM={bpm:.0f} SpO2={spo2:.0f}% | "
        f"{status} (MAE={mae_score:.2f})",
        fontsize=13, fontweight="bold", color=colour,
    )

    axes[0].plot(signal, color="steelblue", linewidth=1.0)
    axes[0].set_title("Synthesised IR PPG (from spo2.json)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, len(signal))

    axes[1].plot(tokens, label="Input tokens", color="steelblue", alpha=0.8, linewidth=1.0)
    axes[1].plot(np.arange(1, len(tokens)), predicted_tokens[:-1], label="Predicted", color="orange", alpha=0.8, linestyle="--", linewidth=1.0)
    axes[1].set_title("Tokens vs Predictions")
    axes[1].set_ylabel("Token (0-101)")
    axes[1].legend()
    axes[1].set_xlim(0, len(tokens))

    axes[2].fill_between(np.arange(len(per_token_error)), per_token_error, color=colour, alpha=0.5)
    axes[2].axhline(mae_score, color=colour, linestyle="--", label=f"MAE={mae_score:.2f}")
    axes[2].set_title("Per-token Error")
    axes[2].set_ylabel("|error|")
    axes[2].set_xlabel("Index")
    axes[2].legend()
    axes[2].set_xlim(0, len(per_token_error))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. LOAD SPO2.JSON (adapted for spo2/bpm keys + change tracking)
# ──────────────────────────────────────────────────────────────────────────────

def load_spo2(json_path: str) -> tuple[float, float, float, float]:
    """
    Read vitals JSON. Retries briefly: the writer may truncate the file before
    finishing json.dump (Windows), which yields empty/partial reads and JSONDecodeError.
    """
    path = Path(json_path)
    if not path.exists():
        return 0.0, 0.0, 0.0, 0.0

    mtime = path.stat().st_mtime
    for _ in range(12):
        try:
            raw = path.read_text(encoding="utf-8")
            if not raw.strip():
                time.sleep(0.025)
                continue
            data = json.loads(raw)
            bpm = float(data.get("bpm", 0.0))
            spo2 = float(data.get("spo2", 0.0))
            ts = float(data.get("timestamp", 0.0))
            return bpm, spo2, ts, mtime
        except (json.JSONDecodeError, OSError, UnicodeError, ValueError, TypeError):
            time.sleep(0.025)

    # Skip this poll; next iteration may succeed after writer finishes
    return 0.0, 0.0, 0.0, mtime


# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Poll vitals JSON → PPGPT → results.json")
    p.add_argument(
        "--input", "-i",
        default=str(_PROJECT_DIR / "spo2.json"),
        help="Vitals JSON (bpm, spo2, timestamp)",
    )
    p.add_argument(
        "--model", "-m",
        default=str(_REPO_DIR / "HeartGPT" / "Model_files" / "PPGPT_500k_iters.pth"),
        help="Model path",
    )
    p.add_argument("--threshold", "-t", type=float, default=11.0, help="MAE threshold")
    p.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval (s)")
    p.add_argument("--plot", action="store_true", help="Save plot on each update")
    p.add_argument(
        "--plot-out",
        default=str(_PROJECT_DIR / "ppgpt_anomaly_result_spo2.png"),
        help="Plot path",
    )
    p.add_argument(
        "--results-json",
        default=str(_PROJECT_DIR / "results.json"),
        help="Latest score JSON (overwritten each update)",
    )
    p.add_argument("--fs", type=float, default=100.0, help="Sampling freq")
    return p.parse_args()


def write_merged_results(
    path: str,
    *,
    bpm: float,
    spo2: float,
    timestamp: float,
    mae_score: float,
    anomaly_detected: bool,
    threshold: float,
    clinical: dict,
) -> None:
    """Single results file: PPGPT (mae_score, anomaly_detected) + clinical scores.json fields."""
    payload = {
        "bpm": bpm,
        "spo2": spo2,
        "timestamp": round(timestamp, 3),
        "mae_score": round(mae_score, 4),
        "anomaly_detected": anomaly_detected,
        "threshold": threshold,
        **clinical,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_pipeline(
    model: PPGPT,
    bpm: float,
    spo2: float,
    args,
    counter: int,
    *,
    vitals_timestamp: float,
):
    print(f"\n--- Update #{counter} ---")
    print(f"  BPM: {bpm:.1f}, SpO2: {spo2:.1f}%")

    flags = clinical_flags(bpm, spo2)
    for f in flags:
        print(f"  {f}")

    signal = synthesise_ir_ppg(bpm, spo2, fs=args.fs)
    tokens = tokenise_ppg(signal)
    print(f"  Waveform: {signal.shape} | Tokens [{tokens.min()}-{tokens.max()}]")

    mae_score, per_token_error = compute_anomaly_score(model, tokens)
    print(f"  MAE score: {mae_score:.4f} (threshold: {args.threshold})")

    anomaly = mae_score > args.threshold
    status = "⚠️  ANOMALY" if anomaly else "✅ NORMAL"
    print(f"  Verdict: {status}")

    clinical = rt.compute_clinical_for_merge(spo2, bpm, vitals_timestamp)
    print(
        f"  Clinical: score={clinical['score']} | "
        f"SpO2 {clinical['spo2_state']} | BPM {clinical['bpm_state']}"
    )

    write_merged_results(
        args.results_json,
        bpm=bpm,
        spo2=spo2,
        timestamp=vitals_timestamp,
        mae_score=mae_score,
        anomaly_detected=anomaly,
        threshold=args.threshold,
        clinical=clinical,
    )
    print(
        f"  Saved {args.results_json}: "
        f"mae={round(mae_score, 4)} clinical_score={clinical['score']} ppgpt_anomaly={anomaly}"
    )

    if args.plot:
        input_tensor = torch.tensor(tokens[np.newaxis, :], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_tensor)
        pred_tokens = logits.argmax(-1).squeeze(0).numpy()
        save_diagnostic_plot(signal, tokens, pred_tokens, per_token_error, mae_score, anomaly, bpm, spo2, args.plot_out)

    print("=" * 50)
    return True


def main():
    args = parse_args()

    # Load model once
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"ERROR: Model not found: {args.model}\n"
            "  Download: https://github.com/harryjdavies/HeartGPT/tree/main/Model_files\n"
            "  Or run:  python project/fetch_ppgpt_weights.py",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Loading PPGPT model...")
    model = PPGPT()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("Model ready.")

    counter = 0
    last_mtime = 0.0
    last_ts = 0.0

    print(
        f"\nWatching {args.input} every {args.poll_interval}s → {args.results_json}\n"
        + "(merged: PPGPT + realtime_monitoring clinical)\n"
        + "=" * 60
    )

    try:
        while True:
            bpm, spo2, ts, mtime = load_spo2(args.input)
            if (mtime > last_mtime or ts > last_ts) and bpm > 0:
                vitals_ts = ts if ts > 0 else time.time()
                run_pipeline(
                    model,
                    bpm,
                    spo2,
                    args,
                    counter,
                    vitals_timestamp=vitals_ts,
                )
                counter += 1
                last_mtime = mtime
                last_ts = ts
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
