# -*- coding: utf-8 -*-
"""Build ../WakeWordTrainingJihed.ipynb (run: python build_wake_word_notebook.py from Elysa/training/)."""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(lines: list[str]) -> dict:
    src = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(True),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    elysa = here.parent
    out = elysa / "WakeWordTrainingJihed.ipynb"

    cells = [
        md(
            "# InterSense — Elysa custom wake-word training\n\n"
            "Train an **openWakeWord** ONNX model for the **Elysa** wake phrase using **Piper** (French TTS) "
            "and the public openWakeWord data recipe.\n\n"
            "The full customised trainer script is **`training/custom_openwakeword_train.py`** (copied over "
            "upstream `openwakeword/openwakeword/train.py` in step 5)."
        ),
        md(
            "## Before you start\n\n"
            "- **Kernel cwd**: launch Jupyter from the **`Elysa`** folder (this notebook expects `./openwakeword`, "
            "`./piper-sample-generator`, `./training/` next to this file).\n"
            "- **Python 3.10+**, **Git**, **`uv`** recommended (`pip install uv`).\n"
            "- **Disk**: multi‑GB downloads (feature `.npy`, AudioSet shard, Piper voice).\n"
            "- **Windows**: shells below use **`wget`** / **`tar`** — use **WSL2**, Git Bash, or replace with curl.\n\n"
            "Dataset licensing is mixed; treat resulting models as **research / personal** unless you complete a full review."
        ),
        code(
            [
                "# --- Confirm we are inside Elysa ---",
                "from pathlib import Path",
                "import os",
                "",
                "WORKDIR = Path.cwd().resolve()",
                'print("Current directory:", WORKDIR)',
                "assert (WORKDIR / 'WakeWordTrainingJihed.ipynb').is_file(), (",
                '    "Open this notebook from the Elysa directory (WakeWordTrainingJihed.ipynb should be in cwd)."',
                ")",
                "os.chdir(WORKDIR)",
            ]
        ),
        md("## 1. Piper sample generator + French ONNX voice"),
        code(
            [
                "# Piper generates synthetic positive utterances via ONNX TTS (Rhasspy community voice).",
                "import pathlib",
                "import subprocess",
                "import urllib.request",
                "",
                "PIP_DIR = pathlib.Path('piper-sample-generator')",
                "if not PIP_DIR.is_dir():",
                "    subprocess.run(",
                "        ['git', 'clone', 'https://github.com/rhasspy/piper-sample-generator', str(PIP_DIR)],",
                "        check=True,",
                "    )",
                "# Optional reproducible checkout:",
                "# subprocess.run(['git', '-C', str(PIP_DIR), 'checkout', 'ded9350eaff558af07f312464ac71baf7de834df'])",
                "",
                "MODEL_DIR = PIP_DIR / 'models'",
                "MODEL_DIR.mkdir(parents=True, exist_ok=True)",
                "urls = [",
                "    'https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx',",
                "    'https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json',",
                "]",
                "for url in urls:",
                "    name = url.rsplit('/', 1)[-1]",
                "    dest = MODEL_DIR / name",
                "    if not dest.is_file():",
                '        print("Downloading", name, "...")',
                "        urllib.request.urlretrieve(url, dest)",
                'print("Piper ONNX + config:", MODEL_DIR.resolve())',
            ]
        ),
        md(
            "## 2. Sanity-check pronunciation\n\n"
            "Edit **`TARGET_WORD`** (underscore phonetics allowed, e.g. `hey_seer_e`). Listen before downloading large datasets."
        ),
        code(
            [
                "import sys",
                "",
                "if 'piper-sample-generator' not in sys.path:",
                "    sys.path.insert(0, 'piper-sample-generator')",
                "",
                "from IPython.display import Audio",
                "import generate_samples",
                "",
                "TARGET_WORD = 'eliisa'  # <-- must match YAML / training phrase",
                "",
                "generate_samples.generate_samples_onnx(",
                "    text=TARGET_WORD,",
                "    max_samples=1,",
                "    length_scales=[1.1],",
                "    noise_scales=[0.7],",
                "    noise_scale_ws=[0.7],",
                "    output_dir='./',",
                "    batch_size=1,",
                "    auto_reduce_batch_size=True,",
                "    file_names=['test_generation.wav'],",
                "    model='piper-sample-generator/models/fr_FR-upmc-medium.onnx',",
                ")",
                "Audio('test_generation.wav', autoplay=False)",
            ]
        ),
        md(
            "## 3. Clone openWakeWord + install deps\n\n"
            "**`--no-deps`** keeps the editable install close to upstream Colab. "
            "If imports fail, `uv pip install openwakeword` extras manually."
        ),
        code(
            [
                "# Some Hugging Face dataset builds query locale encodings incorrectly on macOS/Linux containers.",
                "import locale",
                "",
                "def _utf8(enc=True):",
                '    return "UTF-8"',
                "",
                "locale.getpreferredencoding = _utf8  # type: ignore[method-assign]",
                "",
                "import subprocess",
                "from pathlib import Path",
                "",
                "if not Path('openwakeword').is_dir():",
                "    subprocess.run(['git', 'clone', 'https://github.com/dscripka/openwakeword'], check=True)",
                "",
                "!uv pip install -e ./openwakeword --no-deps",
                "!uv pip install mutagen==1.47.0 torchinfo==1.8.0 torchmetrics==1.2.0 speechbrain==0.5.14",
                "!uv pip install audiomentations==0.33.0 torch-audiomentations==0.11.0 acoustics==0.2.6",
                "!uv pip install onnx_tf==1.10.0 onnx2tf onnx onnx_graphsurgeon sng4onnx",
                "!uv pip install onnxruntime==1.22.1 ai_edge_litert==1.4.0 onnxsim",
                "!uv pip install tensorflow==2.19.0",
                "!uv pip install pronouncing==0.2.0 datasets==2.14.6 deep-phonemizer==0.0.19",
                "",
                "res = Path('openwakeword/openwakeword/resources/models')",
                "res.mkdir(parents=True, exist_ok=True)",
                "BASE = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/'",
                "for asset in ('embedding_model.onnx', 'embedding_model.tflite', 'melspectrogram.onnx', 'melspectrogram.tflite'):",
                "    dest = res / asset",
                "    if dest.is_file():",
                "        continue",
                "    subprocess.run(['wget', BASE + asset, '-O', str(dest)], check=True)",
            ]
        ),
        md(
            "## 4. Download negatives + auxiliary features (~15+ minutes)\n\n"
            "- MIT RIRs (reverb augmentation)\n"
            "- One AudioSet tar shard converted to WAV\n"
            "- One hour of streaming FMA `small`\n"
            "- ACAV validation + feature matrices for mined negatives\n\n"
                "Re-running skips paths that already exist."
        ),
        code(
            [
                "import os",
                "from pathlib import Path",
                "",
                "import numpy as np",
                "import datasets",
                "import scipy",
                "from tqdm import tqdm",
                "",
                "# ---- MIT impulse responses ----",
                "rir_out = Path('./mit_rirs')",
                "if not rir_out.is_dir():",
                "    rir_out.mkdir(parents=True)",
                "    !git lfs install",
                "    !git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses",
                "    wavs = list(Path('./MIT_environmental_impulse_responses/16khz').glob('*.wav'))",
                '    ds = datasets.Dataset.from_dict({"audio": [str(p) for p in wavs]}).cast_column("audio", datasets.Audio())',
                "    for row in tqdm(ds, desc='MIT RIR'):",
                "        name = Path(row['audio']['path']).name",
                "        scipy.io.wavfile.write(str(rir_out / name), 16000, (row['audio']['array'] * 32767).astype(np.int16))",
                "",
                "# ---- AudioSet single balanced train tar ----",
                "if not Path('audioset').is_dir():",
                "    Path('audioset').mkdir()",
                "    fname = 'bal_train09.tar'",
                "    tar_path = Path('audioset') / fname",
                "    link = 'https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/' + fname",
                "    !wget -O {tar_path} {link}",
                "    !cd audioset && tar -xvf {fname}",
                "    wav16_dir = Path('audioset_16k')",
                "    wav16_dir.mkdir(exist_ok=True)",
                '    ads = datasets.Dataset.from_dict({"audio": [str(p) for p in Path("audioset/audio").glob("**/*.flac")]})',
                '    ads = ads.cast_column("audio", datasets.Audio(sampling_rate=16000))',
                "    for row in tqdm(ads, desc='AudioSet WAV'):",
                "        wav_name = Path(row['audio']['path']).stem + '.wav'",
                "        scipy.io.wavfile.write(str(wav16_dir / wav_name), 16000,",
                "                              (row['audio']['array'] * 32767).astype(np.int16))",
                "",
                "# ---- FMA streaming negatives ----",
                "fma_out = Path('./fma')",
                "if not fma_out.is_dir():",
                "    fma_out.mkdir()",
                "    raw = datasets.load_dataset('rudraml/fma', name='small', split='train', streaming=True)",
                "    iterator = iter(raw.cast_column('audio', datasets.Audio(sampling_rate=16000)))",
                "    hours = 1",
                "    n_clips = hours * 3600 // 30",
                "    for _ in tqdm(range(n_clips), desc='FMA'):",
                "        row = next(iterator)",
                "        wav_name = Path(row['audio']['path']).with_suffix('.wav').name",
                "        scipy.io.wavfile.write(str(fma_out / wav_name), 16000,",
                "                              (row['audio']['array'] * 32767).astype(np.int16))",
                "",
                "# ---- Precomputed embeddings for contrastive negatives ----",
                "acav = Path('openwakeword_features_ACAV100M_2000_hrs_16bit.npy')",
                "if not acav.is_file():",
                "    !wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
                "if not Path('validation_set_features.npy').is_file():",
                "    !wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
                'print("Downloads finished.")',
            ]
        ),
        md(
            "## 5. Install the InterSense customised `train.py`\n\n"
            "Copies **`training/custom_openwakeword_train.py`** over **`openwakeword/openwakeword/train.py`** "
            "(same logic that previously lived inline in this notebook)."
        ),
        code(
            [
                "from pathlib import Path",
                "",
                "root = Path.cwd().resolve()",
                "custom = root / 'training' / 'custom_openwakeword_train.py'",
                "dst = root / 'openwakeword' / 'openwakeword' / 'train.py'",
                "assert custom.is_file(), custom",
                "dst.write_text(custom.read_text(encoding='utf-8'), encoding='utf-8')",
                'print("Installed trainer to:", dst)',
            ]
        ),
        md(
            "## 6. YAML config + clip generation → augmentation → training\n\n"
            "Executes **`train.py`** three times (`--generate_clips`, `--augment_clips`, `--train_model`).\n\n"
            "`TARGET_WORD` defaults to **`eliisa`** if you did not rerun section 2 in this kernel."
        ),
        code(
            [
                "import os",
                "import sys",
                "from pathlib import Path",
                "",
                "import yaml",
                "",
                "# --- Tune for dry runs vs production ---",
                "TW = globals().get('TARGET_WORD', 'eliisa')",
                "N_SYNTH = 30000  # drop to e.g. 5000 while debugging",
                "N_STEPS = 10000",
                "MAX_NEG_WEIGHT = 1200",
                "",
                "template = yaml.load(Path('openwakeword/examples/custom_model.yml').read_text(encoding='utf-8'), Loader=yaml.Loader)",
                "cfg = dict(template)",
                "cfg['target_phrase'] = [TW]",
                "cfg['model_name'] = cfg['target_phrase'][0].replace(' ', '_')",
                "cfg['n_samples'] = N_SYNTH",
                "cfg['n_samples_val'] = max(20, N_SYNTH // 10)",
                "cfg['steps'] = N_STEPS",
                "cfg['target_accuracy'] = 0.5",
                "cfg['target_recall'] = 0.25",
                "cfg['output_dir'] = './my_custom_model'",
                "cfg['max_negative_weight'] = MAX_NEG_WEIGHT",
                "cfg['background_paths'] = ['./fma']",
                "cfg['false_positive_validation_data_path'] = 'validation_set_features.npy'",
                r"cfg['feature_data_files'] = {'ACAV100M_sample': 'openwakeword_features_ACAV100M_2000_hrs_16bit.npy'}",
                "Path('my_model.yaml').write_text(yaml.dump(cfg), encoding='utf-8')",
                "",
                'print("=== STEP 1: generate clips ===")',
                "os.system(f'{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --generate_clips')",
                "",
                "!pip install -q librosa soundfile",
                "import librosa",
                "import soundfile as sf",
                "",
                "def force_16k(folder: str):",
                "    n = 0",
                "    for root, _, files in os.walk(folder):",
                "        for fn in files:",
                "            if not fn.endswith('.wav'):",
                "                continue",
                "            path = Path(root) / fn",
                "            audio, sr = librosa.load(str(path), sr=None)",
                "            if sr != 16000:",
                "                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)",
                "                sf.write(str(path), audio, 16000)",
                "                n += 1",
                '    print("Resampled to 16 kHz:", n, "wav files")',
                "",
                "force_16k('generated_clips')",
                "",
                'print("=== STEP 2: augment + cache features ===")',
                "os.system(f'{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips')",
                "",
                'print("=== STEP 3: train & export ONNX/TFLite ===")',
                "os.system(f'{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model')",
            ]
        ),
        md("## 7. (Optional) tightened targets + train-only rerun"),
        code(
            [
                "from pathlib import Path",
                "import yaml",
                "",
                "cfg = yaml.load(Path('my_model.yaml').read_text(encoding='utf-8'), Loader=yaml.Loader)",
                "cfg['target_accuracy'] = 0.8",
                "cfg['target_recall'] = 0.5",
                "cfg['max_negative_weight'] = 200",
                "cfg['steps'] = 15000",
                "cfg['output_dir'] = './my_custom_model'",
                "cfg['model_name'] = 'eliisa'",
                "Path('my_model_v2.yaml').write_text(yaml.dump(cfg), encoding='utf-8')",
                'print("Wrote my_model_v2.yaml")',
            ]
        ),
        code(
            [
                "from pathlib import Path",
                "import yaml",
                "import os",
                "",
                "name = yaml.load(Path('my_model_v2.yaml').read_text(encoding='utf-8'), Loader=yaml.Loader)['model_name']",
                "base = Path('./my_custom_model') / name",
                'print("positive_train:", len(os.listdir(base / "positive_train")))',
                'print("negative_train:", len(os.listdir(base / "negative_train")))',
            ]
        ),
        code(
            [
                "import subprocess",
                "import sys",
                "",
                "subprocess.check_call([",
                "    sys.executable,",
                "    'openwakeword/openwakeword/train.py',",
                "    '--training_config', 'my_model_v2.yaml',",
                "    '--train_model',",
                "])",
            ]
        ),
        md(
            "## 8. (Optional) ONNX → TFLite\n\n"
            "`train.py` already attempts conversion when dependencies succeed.\n\n"
            "Manual template (Linux shell):\n```\n"
            "onnx2tf -i my_custom_model/<model>.onnx -o my_custom_model/ -kat onnx____Flatten_0\n```\n"
            "Rename the `_float32.tflite` artefact if needed, then integrate like **`Elysa.onnx`** in `wakeword_service.py`."
        ),
        code(
            [
                "from pathlib import Path",
                "import yaml",
                "",
                "cfg = yaml.safe_load(Path('my_model.yaml').read_text(encoding='utf-8'))",
                "name = cfg.get('model_name', 'model')",
                "onnx_p = Path('my_custom_model') / f'{name}.onnx'",
                'print("ONNX artefact:", onnx_p.resolve())',
                'print("Copy/rename next to wakeword_service after validation.")',
            ]
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Written:", out)


if __name__ == "__main__":
    main()
