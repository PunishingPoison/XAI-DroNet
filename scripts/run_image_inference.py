"""Run DroNet inference on a single image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xai_dronet.inference import DroNetPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DroNet image inference.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--checkpoint", default=None, help="Optional PyTorch checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional device, e.g. cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = DroNetPredictor(checkpoint_path=args.checkpoint, device=args.device)
    prediction = predictor.predict_image(args.image)
    print(json.dumps(prediction.__dict__, indent=2))


if __name__ == "__main__":
    main()

