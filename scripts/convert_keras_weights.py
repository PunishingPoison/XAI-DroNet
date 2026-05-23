"""Convert official DroNet Keras `.h5` weights into a PyTorch checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xai_dronet.models.keras_weights import save_converted_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Keras DroNet weights to PyTorch.")
    parser.add_argument("--keras-weights", required=True, help="Path to official `.h5` weights.")
    parser.add_argument(
        "--output",
        default="checkpoints/dronet_converted.pt",
        help="Output PyTorch checkpoint path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_converted_checkpoint(args.keras_weights, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

