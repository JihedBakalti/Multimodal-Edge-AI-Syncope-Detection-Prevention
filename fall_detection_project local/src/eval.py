import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config import Config
from datasets import ClipsFromPoseNPZ
from model_tcn import PoseTCN

def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed = os.path.join(cfg.urfd_root, "processed_pose_npz")
    index_json = os.path.join(processed, "index.json")
    ckpt_path = "checkpoints/best_pose_tcn.pt"
    if not os.path.isfile(index_json):
        raise FileNotFoundError("Run dataset_builder_from_videos.py first.")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError("Run train.py first.")

    ds = ClipsFromPoseNPZ(index_json, split="val", clip_len=cfg.clip_len, stride=cfg.stride, seed=cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = PoseTCN(
        in_dim=cfg.num_keypoints * cfg.in_features_per_kp,
        channels=cfg.tcn_channels,
        layers=cfg.tcn_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).squeeze(1)
            p = torch.sigmoid(logits).cpu().numpy()
            pred = (p > 0.5).astype(np.int32)
            y_true.append(y.numpy().astype(np.int32))
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nReport:\n", classification_report(y_true, y_pred, target_names=["ADL", "FALL"], zero_division=0))

if __name__ == "__main__":
    main()
