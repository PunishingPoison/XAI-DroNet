"""Generate a DroNet Grad-CAM overlay for one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xai_dronet.explainability import DroNetGradCAM, overlay_heatmap_on_bgr
from xai_dronet.inference.preprocessing import preprocess_image
from xai_dronet.models import DroNet

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DroNet Grad-CAM for a single image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--checkpoint", default="checkpoints/dronet_official.pt")
    parser.add_argument("--target", choices=["collision", "steering"], default="collision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-overlay", default="outputs/gradcam/gradcam_overlay.png")
    parser.add_argument("--output-heatmap", default="outputs/gradcam/gradcam_heatmap.png")
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlay_path = Path(args.output_overlay)
    heatmap_path = Path(args.output_heatmap)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    model = DroNet()
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    tensor = preprocess_image(args.image)
    result = DroNetGradCAM(model, device=args.device).generate(tensor, target=args.target)

    overlay = overlay_heatmap_on_bgr(image_bgr, result.heatmap, alpha=args.alpha)
    heatmap_image = np.uint8(np.clip(result.heatmap, 0.0, 1.0) * 255.0)
    heatmap_image = cv2.resize(heatmap_image, (image_bgr.shape[1], image_bgr.shape[0]))

    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Failed to write overlay: {overlay_path}")
    if not cv2.imwrite(str(heatmap_path), heatmap_image):
        raise RuntimeError(f"Failed to write heatmap: {heatmap_path}")

    print(
        json.dumps(
            {
                "target": result.target,
                "prediction": result.prediction.__dict__,
                "overlay": str(overlay_path),
                "heatmap": str(heatmap_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

