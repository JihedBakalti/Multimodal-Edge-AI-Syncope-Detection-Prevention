"""
main.py — Firebase vitals → Synthetic IR PPG → PPGPT Anomaly Detection
=======================================================================

Pipeline:
  1. Load vitals (BPM + SpO2) from Firebase JSON
  2. Synthesise a realistic IR-channel PPG waveform from those vitals
  3. Tokenise the waveform (0-101) exactly as PPGPT expects
  4. Run a forward pass through the pre-trained PPGPT model
  5. Compute per-token prediction error (MAE) as the anomaly score
  6. Apply threshold → print verdict + save diagnostic plot

Assumptions / requirements
---------------------------
- Model weights : Model_files/PPGPT_500k_iters.pth  (same as your existing code)
- Python deps   : torch, numpy, matplotlib, scipy
- The JSON file path is passed as a CLI argument (default: vitals.json)

Usage
-----
  python main.py                        # uses vitals.json in current dir
  python main.py --input data/live.json # custom path
  python main.py --threshold 8.0        # override anomaly threshold (tokens)
  python main.py --plot                 # save diagnostic PNG
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PPGPT MODEL  (identical to your existing skeleton)
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
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size) for _ in range(num_heads)]
        )
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
        self.blocks  = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = (self.token_embedding_table(idx)
             + self.position_embedding_table(torch.arange(T, device=idx.device)))
        x = self.ln_f(self.blocks(x))
        return self.lm_head(x)   # [B, T, vocab_size]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SYNTHETIC IR PPG GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def synthesise_ir_ppg(
    heart_rate_bpm: float,
    spo2_percent: float,
    n_samples: int = 500,
    fs: float = 100.0,
    noise_level: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a synthetic IR-channel PPG signal from BPM and SpO2.

    The waveform is built from physiological first principles:
      - Fundamental frequency = BPM / 60 Hz
      - Dicrotic notch via a second harmonic component
      - Amplitude modulated slightly by SpO2 (lower SpO2 → lower AC/DC ratio)
      - Additive white Gaussian noise for realism

    Returns a float64 array of shape (n_samples,) normalised to [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t = np.arange(n_samples) / fs
    f0 = heart_rate_bpm / 60.0          # fundamental cardiac frequency (Hz)

    # ── Systolic peak (main upstroke)
    fundamental = np.sin(2 * np.pi * f0 * t)

    # ── Dicrotic notch: second harmonic at ~60 % of fundamental amplitude,
    #    phase-shifted to place notch on the descending limb
    dicrotic_phase = np.pi * 0.55
    dicrotic = 0.55 * np.sin(2 * np.pi * 2 * f0 * t + dicrotic_phase)

    # ── Combine and shift to positive baseline
    waveform = fundamental + dicrotic
    waveform = waveform - waveform.min()

    # ── SpO2 modulates AC/DC ratio:
    #    normal SpO2 (≥95 %) → high pulsatile fraction (~15 %)
    #    low    SpO2 (<90 %) → reduced pulsatile fraction
    spo2_norm = np.clip(spo2_percent / 100.0, 0.7, 1.0)
    ac_dc_ratio = 0.05 + 0.15 * (spo2_norm - 0.7) / 0.3   # 0.05 – 0.20

    dc_component = 1.0
    ac_amplitude = dc_component * ac_dc_ratio
    peak = waveform.max() if waveform.max() > 0 else 1.0
    waveform = dc_component + ac_amplitude * (waveform / peak)

    # ── Noise
    waveform += noise_level * rng.standard_normal(n_samples)

    # ── Normalise to [0, 1]
    waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-9)
    return waveform.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  TOKENISATION  (mirrors HeartGPT's scheme: 102 tokens, 0–101)
# ──────────────────────────────────────────────────────────────────────────────

def tokenise_ppg(signal: np.ndarray) -> np.ndarray:
    """Map a [0,1]-normalised PPG to integer tokens in {0, …, 101}."""
    tokens = np.clip((signal * 101).astype(int), 0, 101)
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ANOMALY SCORING
# ──────────────────────────────────────────────────────────────────────────────

def compute_anomaly_score(
    model: PPGPT,
    tokens: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Run one forward pass and compute the mean absolute error between
    the input token at position t and the model's predicted token at
    position t-1 (i.e. next-token prediction error).

    Returns
    -------
    mae_score : float
        Mean absolute error over the sequence (in token units 0-101).
    per_token_error : np.ndarray  shape (len(tokens)-1,)
        Absolute error at each position, useful for visualisation.
    """
    input_tensor = torch.tensor(tokens[np.newaxis, :], dtype=torch.long)  # [1, T]

    with torch.no_grad():
        logits = model(input_tensor)   # [1, T, vocab_size]

    predicted_tokens = logits.argmax(dim=-1).squeeze(0).numpy()  # [T]

    # Align: prediction[t] → what comes after token[t]
    # So compare predicted_tokens[:-1] with tokens[1:]
    per_token_error = np.abs(predicted_tokens[:-1] - tokens[1:]).astype(float)
    mae_score = float(per_token_error.mean())
    return mae_score, per_token_error


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CLINICAL CONTEXT  (rule-based flags from raw vitals)
# ──────────────────────────────────────────────────────────────────────────────

def clinical_flags(bpm: float, spo2: float) -> list[str]:
    """
    Return a list of plain-English clinical observations based on known
    normal ranges.  These complement the model-based score.
    """
    flags = []
    if bpm < 60:
        flags.append(f"Bradycardia  — Heart rate {bpm:.0f} bpm  (normal: 60–100)")
    elif bpm > 100:
        flags.append(f"Tachycardia  — Heart rate {bpm:.0f} bpm  (normal: 60–100)")
    else:
        flags.append(f"Normal heart rate: {bpm:.0f} bpm")

    if spo2 < 90:
        flags.append(f"Severe hypoxaemia — SpO2 {spo2:.0f}%  (critical: <90%)")
    elif spo2 < 95:
        flags.append(f"Mild hypoxaemia   — SpO2 {spo2:.0f}%  (normal: ≥95%)")
    else:
        flags.append(f"Normal SpO2: {spo2:.0f}%")

    return flags


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PLOTTING
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
    output_path: str = "ppgpt_anomaly_result.png",
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 9))
    colour = "#d62728" if anomaly else "#2ca02c"
    status = "⚠  ANOMALY DETECTED" if anomaly else "✓  NORMAL"

    fig.suptitle(
        f"PPGPT Anomaly Detection  |  BPM={bpm:.0f}  SpO2={spo2:.0f}%  "
        f"|  {status}  (MAE={mae_score:.2f} tokens)",
        fontsize=13, fontweight="bold", color=colour,
    )

    # ── Synthetic IR PPG waveform
    axes[0].plot(signal, color="steelblue", linewidth=1.0)
    axes[0].set_title("Synthesised IR PPG signal (from Firebase vitals)")
    axes[0].set_ylabel("Amplitude (normalised)")
    axes[0].set_xlim(0, len(signal))

    # ── Token comparison
    axes[1].plot(tokens, label="Input tokens", color="steelblue", alpha=0.8, linewidth=1.0)
    axes[1].plot(
        np.arange(1, len(tokens)),
        predicted_tokens,
        label="Predicted next token",
        color="orange", alpha=0.8, linestyle="--", linewidth=1.0,
    )
    axes[1].set_title("Input tokens vs. PPGPT next-token predictions")
    axes[1].set_ylabel("Token (0–101)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].set_xlim(0, len(tokens))

    # ── Per-token error
    axes[2].fill_between(
        np.arange(len(per_token_error)), per_token_error,
        color=colour, alpha=0.5, linewidth=0,
    )
    axes[2].axhline(mae_score, color=colour, linestyle="--", linewidth=1.2,
                    label=f"MAE = {mae_score:.2f}")
    axes[2].set_title("Per-token absolute prediction error")
    axes[2].set_ylabel("|error| (tokens)")
    axes[2].set_xlabel("Sample index")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].set_xlim(0, len(per_token_error))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [plot] Diagnostic figure saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPGPT anomaly detection from Firebase vitals JSON"
    )
    p.add_argument(
        "--input", "-i",
        default="vitals.json",
        help="Path to Firebase JSON file (default: vitals.json)",
    )
    p.add_argument(
        "--model", "-m",
        default="../HeartGPT/Model_files/PPGPT_500k_iters.pth",
        help="Path to PPGPT weights (default: Model_files/PPGPT_500k_iters.pth)",
    )
    p.add_argument(
        "--threshold", "-t",
        type=float,
        default=11,
        help=(
            "MAE threshold in token units (0–101).  Scores above this are flagged "
            "as anomalies.  Empirical default: 6.5  (tune on your hardware)."
        ),
    )
    p.add_argument(
        "--fs",
        type=float,
        default=100.0,
        help="Sampling frequency used for waveform synthesis (default: 100 Hz)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save a diagnostic PNG alongside the result",
    )
    p.add_argument(
        "--plot-out",
        default="ppgpt_anomaly_result.png",
        help="Output path for the diagnostic plot (default: ppgpt_anomaly_result.png)",
    )
    return p.parse_args()


def load_vitals(json_path: str) -> tuple[float, float]:
    """
    Parse Firebase JSON and return (heart_rate_bpm, blood_oxygen_percent).

    Supports both top-level keys and the nested structure:
        { "users": [ null, { "vitals": { "HeartRate": X, "BloodOxygen": Y } } ] }
    """
    path = Path(json_path)
    if not path.exists():
        print(f"[ERROR] JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r") as fh:
        data = json.load(fh)

    vitals: dict | None = None

    # ── Try flat structure first: { "HeartRate": X, "BloodOxygen": Y }
    if "HeartRate" in data and "BloodOxygen" in data:
        vitals = data

    # ── Try nested Firebase structure: { "users": [ null, { "vitals": {...} } ] }
    elif "users" in data:
        for user in data["users"]:
            if isinstance(user, dict) and "vitals" in user:
                vitals = user["vitals"]
                break

    # ── Try one more level: { "vitals": { ... } }
    elif "vitals" in data:
        vitals = data["vitals"]

    if vitals is None:
        print(
            "[ERROR] Could not find 'HeartRate'/'BloodOxygen' in the JSON.\n"
            "        Expected structure: {\"HeartRate\": X, \"BloodOxygen\": Y}\n"
            "        or Firebase nested: {\"users\": [null, {\"vitals\": {...}}]}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        bpm  = float(vitals["HeartRate"])
        spo2 = float(vitals["BloodOxygen"])
    except KeyError as exc:
        print(f"[ERROR] Missing key in vitals dict: {exc}", file=sys.stderr)
        sys.exit(1)

    return bpm, spo2


def main() -> None:
    args = parse_args()

    # ── 1. Load vitals ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PPGPT — PPG Anomaly Detection from Firebase Vitals")
    print(f"{'='*60}\n")

    bpm, spo2 = load_vitals(args.input)
    print(f"  [vitals]  Heart Rate  : {bpm:.1f} bpm")
    print(f"  [vitals]  Blood Oxygen: {spo2:.1f}%\n")

    # ── 2. Clinical rule-based flags ────────────────────────────────────────
    flags = clinical_flags(bpm, spo2)
    print("  [clinical flags]")
    for f in flags:
        print(f"    • {f}")
    print()

    # ── 3. Synthesise IR PPG waveform ───────────────────────────────────────
    print("  [synthesis]  Generating synthetic IR PPG waveform …")
    signal = synthesise_ir_ppg(
        heart_rate_bpm=bpm,
        spo2_percent=spo2,
        n_samples=500,
        fs=args.fs,
    )
    tokens = tokenise_ppg(signal)
    print(f"              Waveform shape : {signal.shape}  "
          f"| Token range : [{tokens.min()}, {tokens.max()}]\n")

    # ── 4. Load PPGPT model ─────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"[ERROR] Model weights not found at: {args.model}\n"
            "        Please ensure PPGPT_500k_iters.pth is in Model_files/",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  [model]  Loading PPGPT from {args.model} …")
    model = PPGPT()
    state_dict = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"           Parameters : {n_params:,}\n")

    # ── 5. Forward pass + anomaly score ────────────────────────────────────
    print("  [inference]  Running PPGPT forward pass …")
    mae_score, per_token_error = compute_anomaly_score(model, tokens)
    print(f"               MAE score    : {mae_score:.4f} tokens")
    print(f"               Threshold    : {args.threshold:.1f} tokens\n")

    # ── 6. Verdict ──────────────────────────────────────────────────────────
    anomaly = mae_score > args.threshold

    print("  ┌─────────────────────────────────────────────┐")
    if anomaly:
        print(f"  │  ⚠  ANOMALY DETECTED  (MAE = {mae_score:.2f})           │")
    else:
        print(f"  │  ✓  NORMAL SIGNAL     (MAE = {mae_score:.2f})           │")
    print("  └─────────────────────────────────────────────┘\n")

    # ── 7. Optional plot ────────────────────────────────────────────────────
    if args.plot:
        input_tensor = torch.tensor(tokens[np.newaxis, :], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_tensor)
        predicted_tokens = logits.argmax(dim=-1).squeeze(0).numpy()[:-1]

        save_diagnostic_plot(
            signal=signal,
            tokens=tokens,
            predicted_tokens=predicted_tokens,
            per_token_error=per_token_error,
            mae_score=mae_score,
            anomaly=anomaly,
            bpm=bpm,
            spo2=spo2,
            output_path=args.plot_out,
        )

    # ── 8. Machine-readable summary ─────────────────────────────────────────
    result = {
        "heart_rate_bpm"  : bpm,
        "blood_oxygen_pct": spo2,
        "mae_score"       : round(mae_score, 4),
        "threshold"       : args.threshold,
        "anomaly_detected": anomaly,
        "clinical_flags"  : flags,
    }
    print("  [result JSON]")
    print("  " + json.dumps(result, indent=4).replace("\n", "\n  "))
    print()


if __name__ == "__main__":
    main()