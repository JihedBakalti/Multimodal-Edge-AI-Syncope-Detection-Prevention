# Driver Drowsiness / Pre-syncope Detection (CNN-LSTM)

This repository contains an experimental CNN-LSTM pipeline for detecting driver drowsiness / pre-syncope events from face image sequences. The primary workflow is captured in the notebook [driver-drowsiness-detection-cnn-lstm (2).ipynb](driver-drowsiness-detection-cnn-lstm%20(2).ipynb).

**Project**
- **Goal**: Train a TimeDistributed CNN + LSTM model on short image sequences to detect drowsiness/pre-syncope.
- **Approach**: Extract frames, form sequences (sequence_length=5), process per-frame with CNN (Conv2D + BatchNorm + Pooling + GlobalAveragePooling), then feed sequence features into an LSTM and dense classifier.

**Repository Structure**
- **Notebook**: [driver-drowsiness-detection-cnn-lstm (2).ipynb](driver-drowsiness-detection-cnn-lstm%20(2).ipynb) — full experiment and visualization.
- **Script**: `syncope_cnn_lstm.py` — training / inference script (if provided).
- **Model**: `syncope_lstm_v1.h5` — saved Keras model checkpoint.
- **Dependencies**: `requirements.txt` — Python packages used for the project.

**Dataset**
- The notebook uses a driver drowsiness dataset (example path used in notebook: `/kaggle/input/driver-drowsiness-dataset-ddd/Driver Drowsiness Dataset (DDD)`).
- The dataset is split using `splitfolders` into `train`, `val`, and `test` with a default ratio of 0.8/0.15/0.05.

**Key Hyperparameters**
- Image size: 128x128
- Sequence length: 5 frames
- Batch size: 16 (generator) / 32 (sequence generator)
- Optimizer: Adam (lr=1e-4)
- Loss: Binary crossentropy
- Epochs: 20 (notebook example)

**How to set up (quick)**
1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install split-folders
```

**How to run the notebook**
- Open the notebook in Jupyter / VS Code and run cells in order. The notebook performs:
  - data splitting with `splitfolders`
  - image loading with `ImageDataGenerator`
  - sequence creation using a custom `DataGenerator` (`keras.utils.Sequence`)
  - model definition (TimeDistributed Conv2D -> GlobalAveragePooling2D -> LSTM -> Dense)
  - training and plotting of accuracy/loss curves

**Training from script**
- If `syncope_cnn_lstm.py` contains a runnable training entrypoint, run:

```bash
python syncope_cnn_lstm.py
```

**Model and Evaluation**
- The example notebook saves a trained Keras model (`syncope_lstm_v1.h5`).
- After training, the notebook plots training/validation accuracy and loss and shows sample images from batches. For further evaluation, compute confusion matrix and classification report on test sequences.

**Notes & Tips**
- The provided `DataGenerator` constructs overlapping sequences — tune `sequence_length`, `batch_size`, and `shuffle` to match your dataset and memory constraints.
- Consider using a pretrained backbone (e.g., MobileNetV2) for stronger per-frame feature extraction if accuracy is low.
- For deployment on edge devices, quantize or prune the model and test latency on target hardware.

**Files of interest**
- [driver-drowsiness-detection-cnn-lstm (2).ipynb](driver-drowsiness-detection-cnn-lstm%20(2).ipynb)
- [syncope_cnn_lstm.py](syncope_cnn_lstm.py)
- [syncope_lstm_v1.h5](syncope_lstm_v1.h5)
- [requirements.txt](requirements.txt)

**License**
- No license provided. Add a `LICENSE` file if you plan to publish or share this code.

If you want, I can: run the notebook to produce a trained model, convert the notebook into a runnable script, or add a `requirements.txt` Dockerfile. Which would you like next?
