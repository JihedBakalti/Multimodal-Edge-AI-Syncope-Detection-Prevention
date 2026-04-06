import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from config import Config
from datasets import ClipsFromPoseNPZ
from model_tcn import PoseTCN

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x).squeeze(1)
            p = torch.sigmoid(logits)
            ys.append(y.cpu().numpy())
            ps.append((p.cpu().numpy() > 0.5).astype(np.int32))
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

def main():
    cfg = Config()
    set_seed(cfg.seed)

    processed = os.path.join(cfg.urfd_root, "processed_pose_npz")
    index_json = os.path.join(processed, "index.json")
    if not os.path.isfile(index_json):
        raise FileNotFoundError(f"Missing {index_json}. Run dataset_builder_from_videos.py first.")

    train_ds = ClipsFromPoseNPZ(
    index_json,
    split="train",
    clip_len=cfg.clip_len,
    stride=cfg.stride,
    seed=cfg.seed,
    augment=True,
    noise_std=0.01,
    kp_dropout_prob=0.05,
)
    val_ds   = ClipsFromPoseNPZ(index_json, split="val",   clip_len=cfg.clip_len, stride=cfg.stride, seed=cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PoseTCN(
        in_dim=cfg.num_keypoints * cfg.in_features_per_kp,
        channels=cfg.tcn_channels,
        layers=cfg.tcn_layers,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    os.makedirs("checkpoints", exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | val={metrics}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "checkpoints/best_pose_tcn.pt")
            print("  -> saved checkpoints/best_pose_tcn.pt")

if __name__ == "__main__":
    main()
