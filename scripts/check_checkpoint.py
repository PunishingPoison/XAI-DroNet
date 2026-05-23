"""Validate that a checkpoint matches the current DroNet model shape."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xai_dronet.inference import DroNetPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a DroNet checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a PyTorch checkpoint.")
    parser.add_argument("--device", default="cpu", help="Device to load on, e.g. cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = DroNetPredictor(checkpoint_path=args.checkpoint, device=args.device)
    sample = torch.zeros(1, 1, 200, 200)
    prediction = predictor.predict_tensor(sample)

    payload = {
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(predictor.device),
        "num_parameters": sum(p.numel() for p in predictor.model.parameters()),
        "sample_prediction": prediction.__dict__,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

