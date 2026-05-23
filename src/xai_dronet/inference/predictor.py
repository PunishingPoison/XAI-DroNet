"""Checkpoint-backed DroNet inference wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from xai_dronet.inference.preprocessing import ImagePreprocessConfig, preprocess_image
from xai_dronet.models import DroNet, DroNetConfig


@dataclass(frozen=True)
class DroNetPrediction:
    """Single-frame DroNet prediction."""

    steering: float
    collision_probability: float


class DroNetPredictor:
    """Run paper-faithful DroNet inference on image files or tensors."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        model_config: DroNetConfig | None = None,
        preprocess_config: ImagePreprocessConfig | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DroNet(model_config).to(self.device)
        self.preprocess_config = preprocess_config or ImagePreprocessConfig()

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load a PyTorch state dict checkpoint."""

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)

    @torch.inference_mode()
    def predict_tensor(self, image_tensor: Tensor) -> DroNetPrediction:
        """Predict from a preprocessed tensor shaped `[1, 1, 200, 200]`."""

        if image_tensor.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, got shape {tuple(image_tensor.shape)}")

        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        steering, collision = self.model(image_tensor)
        return DroNetPrediction(
            steering=float(steering.squeeze().detach().cpu()),
            collision_probability=float(collision.squeeze().detach().cpu()),
        )

    def predict_image(self, image_path: str | Path) -> DroNetPrediction:
        """Load, preprocess, and predict from an image path."""

        image_tensor = preprocess_image(image_path, config=self.preprocess_config)
        return self.predict_tensor(image_tensor)

