from dataclasses import dataclass

@dataclass
class Config:
    # Data
    urfd_root: str = "data"
    use_camera: str = "cam0"

    # Video processing
    frame_step: int = 1            # 1=every frame, 2=every 2nd frame (~15fps), 3 (~10fps)
    max_frames_per_seq: int | None = None  # set e.g. 1500 to cap long videos

    # Clip settings
    clip_len: int = 48             # frames per clip (if 30fps => 1.6s)
    stride: int = 8                # sliding window step
    min_frames_per_seq: int = 60   # discard very short sequences

    # Pose (MediaPipe)
    pose_model_complexity: int = 1
    pose_min_det_conf: float = 0.5
    pose_min_track_conf: float = 0.5

    # Model
    num_keypoints: int = 33        # MediaPipe Pose
    in_features_per_kp: int = 3    # x,y,visibility
    tcn_channels: int = 128
    tcn_layers: int = 4
    dropout: float = 0.2

    # Training
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 0

    # Realtime thresholds
    fall_threshold: float = 0.75
    lying_threshold: float = 0.65
    immobile_epsilon: float = 0.008
    hold_seconds: float = 4.0
    infer_every_seconds: float = 0.25
