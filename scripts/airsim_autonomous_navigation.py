"""Run the paper-faithful DroNet autonomous navigation loop in AirSim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xai_dronet.control import DroNetController
from xai_dronet.inference import DroNetPredictor
from xai_dronet.simulator import (
    AirSimCameraClient,
    AirSimConnectionConfig,
    AirSimControlConfig,
    AirSimDroneControlClient,
    AutonomousNavigationConfig,
    AutonomousNavigationRunner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DroNet autonomous navigation in AirSim.")
    parser.add_argument("--checkpoint", default="checkpoints/dronet_official.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera-name", default="0")
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--ip", default="")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    parser.add_argument("--command-duration", type=float, default=0.1)
    parser.add_argument("--max-forward-velocity", type=float, default=1.0)
    parser.add_argument("--velocity-smoothing", type=float, default=0.7)
    parser.add_argument("--yaw-smoothing", type=float, default=0.5)
    parser.add_argument("--output-jsonl", default="outputs/airsim/autonomous_navigation.jsonl")
    parser.add_argument(
        "--execute-control",
        action="store_true",
        help="Actually send AirSim movement commands. Without this flag, the loop is dry-run only.",
    )
    parser.add_argument("--takeoff", action="store_true", help="Take off before executing control.")
    parser.add_argument("--land-on-exit", action="store_true", help="Land after executing control.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = not args.execute_control

    camera_config = AirSimConnectionConfig(
        ip=args.ip,
        port=args.port,
        timeout_value=args.timeout,
        camera_name=args.camera_name,
        vehicle_name=args.vehicle_name,
    )
    control_config = AirSimControlConfig(
        ip=args.ip,
        port=args.port,
        timeout_value=args.timeout,
        vehicle_name=args.vehicle_name,
        command_duration=args.command_duration,
    )
    run_config = AutonomousNavigationConfig(
        num_steps=args.num_steps,
        interval_seconds=args.interval_seconds,
        output_jsonl=Path(args.output_jsonl),
    )

    camera = AirSimCameraClient(camera_config)
    camera.connect()
    predictor = DroNetPredictor(checkpoint_path=args.checkpoint, device=args.device)
    controller = DroNetController(
        max_forward_velocity=args.max_forward_velocity,
        velocity_smoothing=args.velocity_smoothing,
        yaw_smoothing=args.yaw_smoothing,
    )
    control_client = AirSimDroneControlClient(control_config, dry_run=dry_run)

    if not dry_run:
        control_client.connect()
        control_client.prepare_for_flight(takeoff=args.takeoff)

    try:
        records = AutonomousNavigationRunner(
            camera=camera,
            predictor=predictor,
            controller=controller,
            command_sink=control_client,
            config=run_config,
        ).run()
    finally:
        if not dry_run:
            control_client.hover()
            if args.land_on_exit:
                control_client.land()

    summary = {
        "mode": "execute-control" if args.execute_control else "dry-run",
        "num_steps": len(records),
        "output_jsonl": str(run_config.output_jsonl),
        "last_record": records[-1].__dict__,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

