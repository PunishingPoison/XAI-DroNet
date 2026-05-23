"""Run a read-only live DroNet inference loop on AirSim camera frames."""

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
from xai_dronet.simulator import (
    AirSimCameraClient,
    AirSimConnectionConfig,
    LiveAirSimInferenceRunner,
    LiveInferenceConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run read-only live DroNet inference in AirSim.")
    parser.add_argument("--checkpoint", default="checkpoints/dronet_official.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera-name", default="0")
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--ip", default="")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    parser.add_argument("--output-jsonl", default="outputs/airsim/live_predictions.jsonl")
    parser.add_argument("--frame-output-dir", default="outputs/airsim/frames")
    parser.add_argument("--save-every-n-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_config = AirSimConnectionConfig(
        ip=args.ip,
        port=args.port,
        timeout_value=args.timeout,
        camera_name=args.camera_name,
        vehicle_name=args.vehicle_name,
    )
    loop_config = LiveInferenceConfig(
        num_frames=args.num_frames,
        interval_seconds=args.interval_seconds,
        output_jsonl=Path(args.output_jsonl),
        frame_output_dir=Path(args.frame_output_dir),
        save_every_n_frames=args.save_every_n_frames,
    )

    camera = AirSimCameraClient(camera_config)
    camera.connect()
    predictor = DroNetPredictor(checkpoint_path=args.checkpoint, device=args.device)
    records = LiveAirSimInferenceRunner(camera, predictor, loop_config).run()

    summary = {
        "num_frames": len(records),
        "output_jsonl": str(loop_config.output_jsonl),
        "average_fps": sum(record.fps for record in records) / len(records),
        "last_prediction": {
            "steering": records[-1].steering,
            "collision_probability": records[-1].collision_probability,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

