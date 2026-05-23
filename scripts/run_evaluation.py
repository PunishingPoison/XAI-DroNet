"""Run the DroNet autonomous navigation loop with full evaluation metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xai_dronet.control import DroNetController
from xai_dronet.inference import DroNetPredictor
from xai_dronet.inference.preprocessing import preprocess_array
from xai_dronet.explainability.gradcam import DroNetGradCAM, overlay_heatmap_on_bgr
import cv2
from xai_dronet.simulator import (
    AirSimCameraClient,
    AirSimConnectionConfig,
    AirSimControlConfig,
    AirSimDroneControlClient,
)
from xai_dronet.evaluation.evaluator import DroneEvaluator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DroNet evaluation in AirSim.")
    parser.add_argument("--checkpoint", default="checkpoints/dronet_official.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--environment-name", default="Unknown")
    parser.add_argument("--camera-name", default="0")
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--ip", default="")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--max-forward-velocity", type=float, default=1.0)
    parser.add_argument("--velocity-smoothing", type=float, default=0.7)
    parser.add_argument("--yaw-smoothing", type=float, default=0.5)
    parser.add_argument("--output-jsonl", default="outputs/airsim/autonomous_navigation.jsonl")
    parser.add_argument("--report-json", default="outputs/airsim/evaluation_report.json")
    parser.add_argument(
        "--execute-control",
        action="store_true",
        help="Actually send AirSim movement commands. Without this flag, the loop is dry-run only.",
    )
    parser.add_argument("--takeoff", action="store_true", help="Take off before executing control.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = not args.execute_control

    camera_config = AirSimConnectionConfig(
        ip=args.ip, port=args.port, timeout_value=args.timeout,
        camera_name=args.camera_name, vehicle_name=args.vehicle_name,
    )
    control_config = AirSimControlConfig(
        ip=args.ip, port=args.port, timeout_value=args.timeout,
        vehicle_name=args.vehicle_name, command_duration=1.0, wait_for_command=False,
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

    cam_generator = DroNetGradCAM(predictor.model, device=args.device)
    evaluator = DroneEvaluator(environment_name=args.environment_name)
    controller.reset()
    
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting evaluation run for {args.num_steps} steps...")
    
    start_time = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as log_file:
        for step_index in range(args.num_steps):
            loop_start = time.perf_counter()
            
            frame = camera.capture_scene_frame()
            tensor = preprocess_array(frame.bgr)
            prediction = predictor.predict_tensor(tensor)
            command = controller.step(
                steering=prediction.steering,
                collision_probability=prediction.collision_probability,
            )
            applied = control_client.send_command(command)
            
            # Generate and save Grad-CAM
            cam_result = cam_generator.generate(tensor, target="collision")
            heatmap_overlay = overlay_heatmap_on_bgr(frame.bgr, cam_result.heatmap)
            cv2.imwrite("outputs/airsim/latest_frame.jpg", frame.bgr)
            cv2.imwrite("outputs/airsim/latest_gradcam.jpg", heatmap_overlay)
            
            # Get Position (assuming client is available and not dry_run)
            position = (0.0, 0.0, 0.0)
            if not dry_run and control_client._client:
                state = control_client._client.getMultirotorState(vehicle_name=args.vehicle_name)
                pos = state.kinematics_estimated.position
                position = (pos.x_val, pos.y_val, pos.z_val)
                
            frame_time = time.perf_counter() - loop_start
            
            evaluator.log_step(
                position=position,
                steering_angle=prediction.steering,
                has_collided=(prediction.collision_probability > 0.9), # Proxy for collision
                frame_time=frame_time
            )
            
            record = {
                "step_index": step_index,
                "timestamp": time.time(),
                "steering": prediction.steering,
                "collision_probability": prediction.collision_probability,
                "forward_velocity": command.forward_velocity,
                "yaw_radians": command.yaw_radians,
                "executed": applied.executed
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            if step_index % 10 == 0:
                print(f"Step {step_index}/{args.num_steps}: Steering={prediction.steering:.2f}, CollProb={prediction.collision_probability:.2f}")

    if not dry_run:
        control_client.hover()

    evaluator.save_report(args.report_json)
    print("Evaluation run complete.")

if __name__ == "__main__":
    main()
