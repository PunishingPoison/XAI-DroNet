"""Image preprocessing for DroNet inference.

The paper uses a single grayscale `200 x 200` frame and rescales pixel values by
`1 / 255`. This module keeps that conversion isolated so AirSim frames and
offline images go through the same path later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ImagePreprocessConfig:
    """Preprocessing parameters matching the DroNet input contract."""

    height: int = 200
    width: int = 200
    rescale: float = 1.0 / 255.0


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk as grayscale `uint8`."""

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a 2D grayscale, BGR, RGB, or BGRA/RGBA image to grayscale."""

    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported channel count: {channels}")


def preprocess_array(
    image: np.ndarray,
    config: ImagePreprocessConfig | None = None,
) -> Tensor:
    """Convert an image array into a `[1, 1, H, W]` float tensor."""

    cfg = config or ImagePreprocessConfig()
    gray = to_grayscale(image)
    resized = cv2.resize(gray, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) * cfg.rescale
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor


def preprocess_image(
    image_path: str | Path,
    config: ImagePreprocessConfig | None = None,
) -> Tensor:
    """Load and preprocess an image from disk for DroNet inference."""

    image = load_grayscale_image(image_path)
    return preprocess_array(image, config=config)

