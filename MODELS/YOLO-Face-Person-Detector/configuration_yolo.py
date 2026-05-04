"""A HuggingFace-style model configuration."""
from typing import Any, Dict, List
from transformers import PretrainedConfig


class YoloV8Config(PretrainedConfig):
    model_type = 'yolov8'

    def __init__(
        self,
        model_config: str = "yolov8x.yaml",
        task: str = 'detect',
        num_classes: int = 2,
        num_channels: int = 3,
        input_size: int = 640,
        names: Dict = {"0": "person", "1": "face"},
        stride: List[int] = [8, 16, 32],
        verbose: bool = False,
        **kwargs: Any
    ):
        self.input_size = input_size
        self.num_channels = num_channels
        self.task = task
        self.model_config = model_config
        self.num_classes = num_classes
        self.stride = stride
        self.verbose = bool(verbose)
        self.names = {int(key): value for key, value in names.items()}

        super().__init__(**kwargs)
