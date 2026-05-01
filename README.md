# 🛡️ Fall Detection — BiLSTM + Attention

Détection de chute en temps réel via webcam, combinant un modèle LSTM bidirectionnel entraîné sur **URFD + Le2i** et une heuristique géométrique MediaPipe.

Détecte deux cas :
- 👤 **Personne déjà au sol** (effondrée / évanouie)
- 🎬 **Mouvement de chute** en cours

---

## 📁 Structure du projet

```
fall-detection/
├── fall_detection_kaggle.ipynb    # Notebook d'entraînement (Kaggle)
├── realtime_fall_detector.py      # Script de test temps réel (VSCode)
├── fall_lstm_traced.pt            # Modèle TorchScript exporté
├── norm_mean.npy                  # Stats de normalisation
├── norm_std.npy                   # Stats de normalisation
└── README.md
```

---

## 🚀 Étapes

### 1. Entraînement sur Kaggle

1. Ouvrir [Kaggle](https://www.kaggle.com) et créer un nouveau notebook
2. Ajouter les deux datasets :
   - [UR Fall Detection Dataset](https://www.kaggle.com/datasets/shahliza27/ur-fall-detection-dataset)
   - [Le2i Fall Dataset (IMVIA)](https://www.kaggle.com/datasets/tuyennguyenngoc/falldataset-imvia)
3. Importer `fall_detection_kaggle.ipynb` et **Run All**
4. Depuis **Output**, télécharger :
   - `fall_lstm_traced.pt`
   - `norm_mean.npy`
   - `norm_std.npy`

---

### 2. Test temps réel (VSCode)

**Prérequis :**
```bash
pip install torch torchvision mediapipe opencv-python numpy
```

**Placer dans le même dossier :**
```
realtime_fall_detector.py
fall_lstm_traced.pt
norm_mean.npy
norm_std.npy
```

**Lancer :**
```bash
python realtime_fall_detector.py
```

---

## 🎮 Contrôles

| Touche | Action |
|--------|--------|
| `Q` | Quitter |
| `R` | Reset buffer et alarme |
| `S` | Sauvegarder screenshot |

---

## ⚙️ Paramètres (dans `realtime_fall_detector.py`)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `FALL_THRESHOLD` | `0.72` | Seuil probabilité LSTM — augmenter pour moins de fausses alarmes |
| `N_CONFIRM` | `6` | Frames consécutives avant alarme |
| `ALARM_COOLDOWN` | `4.0` | Secondes entre deux alarmes |

---

## 🧠 Architecture

- **Extraction** : MediaPipe Pose → 33 keypoints × xyz = vecteur 99-dim
- **Modèle** : BiLSTM 2 couches + Attention temporelle sur fenêtre de 30 frames
- **Anti-faux-positifs** : confirmation sur N frames + heuristique géométrique sol

---

## 📦 Datasets

- [UR Fall Detection Dataset](https://www.kaggle.com/datasets/shahliza27/ur-fall-detection-dataset)
- [Le2i / IMVIA Fall Dataset](https://www.kaggle.com/datasets/tuyennguyenngoc/falldataset-imvia)
