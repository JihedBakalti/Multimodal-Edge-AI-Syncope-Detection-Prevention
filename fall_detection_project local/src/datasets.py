import json
import numpy as np
from torch.utils.data import Dataset

class ClipsFromPoseNPZ(Dataset):
    """
    Loads per-sequence pose NPZ and creates many clips (sliding windows).
    Split is sequence-level (not frame-level).
    Optional keypoint augmentation in train.
    """
    def __init__(
        self,
        index_json: str,
        split="train",
        clip_len=48,
        stride=8,
        seed=42,
        train_ratio=0.8,
        augment=False,
        noise_std=0.01,
        kp_dropout_prob=0.05,
    ):
        with open(index_json, "r", encoding="utf-8") as f:
            idx = json.load(f)

        all_files = idx["falls"] + idx["adl"]
        rng = np.random.default_rng(seed)
        rng.shuffle(all_files)

        n = len(all_files)
        n_train = int(train_ratio * n)
        if split == "train":
            self.files = all_files[:n_train]
        else:
            self.files = all_files[n_train:]

        self.clip_len = clip_len
        self.stride = stride

        self.augment = augment
        self.noise_std = noise_std
        self.kp_dropout_prob = kp_dropout_prob
        self.rng = np.random.default_rng(seed + (1 if split == "train" else 999))

        self.clips = []
        for fp in self.files:
            data = np.load(fp, allow_pickle=True)
            T = data["kps"].shape[0]
            y = int(data["y"])
            if T < clip_len:
                continue
            for s in range(0, T - clip_len + 1, stride):
                self.clips.append((fp, s, y))

    def __len__(self):
        return len(self.clips)

    def _augment_clip(self, clip_33x3):
        """
        clip_33x3: (T,33,3) [x,y,vis] already normalized pose
        - add small gaussian noise to x,y
        - randomly drop some keypoints (set x,y,vis=0)
        """
        c = clip_33x3.copy()

        # noise on x,y
        noise = self.rng.normal(0.0, self.noise_std, size=c[:, :, :2].shape).astype(np.float32)
        c[:, :, :2] += noise

        # keypoint dropout
        mask = self.rng.random(size=(c.shape[0], c.shape[1])) < self.kp_dropout_prob
        c[mask, 0] = 0.0
        c[mask, 1] = 0.0
        c[mask, 2] = 0.0

        return c

    def __getitem__(self, i):
        fp, s, y = self.clips[i]
        data = np.load(fp, allow_pickle=True)
        kps = data["kps"]  # (T,33,3)
        clip = kps[s:s+self.clip_len]  # (clip_len,33,3)

        if self.augment:
            clip = self._augment_clip(clip)

        clip = clip.reshape(self.clip_len, -1).astype(np.float32)  # (T,99)
        return clip, np.float32(y)
