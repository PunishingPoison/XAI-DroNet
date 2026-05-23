"""Run a no-checkpoint forward-pass smoke test for DroNet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from xai_dronet.models import DroNet


def main() -> None:
    model = DroNet()
    model.eval()
    image = torch.rand(1, 1, 200, 200)

    with torch.inference_mode():
        steering, collision = model(image)

    result = {
        "steering_shape": list(steering.shape),
        "collision_shape": list(collision.shape),
        "steering": float(steering.squeeze()),
        "collision_probability": float(collision.squeeze()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

