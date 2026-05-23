"""Capture one AirSim frame and run DroNet inference on it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2

from xai_dronet.inference import DroNetPredictor
from xai_dronet.inference.preprocessing import preprocess_array
from xai_dronet.simulator import AirSimCameraClient, AirSimConnectionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture an AirSim frame and run DroNet.")
    parser.add_argument("--checkpoint", default="checkpoints/dronet_official.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera-name", default="0")
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--ip", default="")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--output-frame", default="outputs/airsim/latest_frame.png")
    parser.add_argument("--output-json", default="outputs/airsim/latest_prediction.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_path = Path(args.output_frame)
    json_path = Path(args.output_json)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    config = AirSimConnectionConfig(
        ip=args.ip,
        port=args.port,
        timeout_value=args.timeout,
        camera_name=args.camera_name,
        vehicle_name=args.vehicle_name,
    )
    camera = AirSimCameraClient(config)
    camera.connect()
    frame = camera.capture_scene_frame()

    if not cv2.imwrite(str(frame_path), frame.bgr):
        raise RuntimeError(f"Failed to save frame: {frame_path}")

    predictor = DroNetPredictor(checkpoint_path=args.checkpoint, device=args.device)
    tensor = preprocess_array(frame.bgr)
    prediction = predictor.predict_tensor(tensor)

    result = {
        "frame": {
            "path": str(frame_path),
            "camera_name": frame.camera_name,
            "width": frame.width,
            "height": frame.height,
        },
        "checkpoint": args.checkpoint,
        "prediction": prediction.__dict__,
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

