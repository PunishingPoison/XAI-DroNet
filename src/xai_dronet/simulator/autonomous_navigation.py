"""Autonomous DroNet navigation loop for AirSim."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from xai_dronet.control import DroNetController
from xai_dronet.inference import DroNetPrediction
from xai_dronet.inference.preprocessing import preprocess_array
from xai_dronet.simulator.airsim_camera import AirSimFrame
from xai_dronet.simulator.airsim_control import AirSimAppliedCommand


class FrameSource(Protocol):
    """Protocol for simulator frame sources."""

    def capture_scene_frame(self) -> AirSimFrame:
        """Capture one frame."""


class TensorPredictor(Protocol):
    """Protocol for DroNet predictors."""

    def predict_tensor(self, image_tensor) -> DroNetPrediction:
        """Predict steering and collision probability."""


class CommandSink(Protocol):
    """Protocol for command adapters."""

    def send_command(self, command) -> AirSimAppliedCommand:
        """Send one command and return the values applied."""


@dataclass(frozen=True)
class AutonomousNavigationConfig:
    """Settings for the closed-loop autonomous navigation runner."""

    num_steps: int = 100
    interval_seconds: float = 0.0
    output_jsonl: Path = Path("outputs/airsim/autonomous_navigation.jsonl")


@dataclass(frozen=True)
class AutonomousNavigationRecord:
    """One perception-control record from the navigation loop."""

    step_index: int
    timestamp: float
    elapsed_seconds: float
    width: int
    height: int
    steering: float
    collision_probability: float
    forward_velocity: float
    yaw_radians: float
    yaw_rate_degrees_per_second: float
    command_duration: float
    executed: bool

    def to_json(self) -> str:
        """Serialize record as one JSONL row."""

        return json.dumps(self.__dict__)


class AutonomousNavigationRunner:
    """Closed-loop DroNet navigation runner.

    The policy remains reactive and paper-faithful: a single monocular frame
    drives steering and collision probability, then the DroNet controller maps
    those predictions into smoothed velocity/yaw commands.
    """

    def __init__(
        self,
        camera: FrameSource,
        predictor: TensorPredictor,
        controller: DroNetController,
        command_sink: CommandSink,
        config: AutonomousNavigationConfig | None = None,
    ) -> None:
        self.camera = camera
        self.predictor = predictor
        self.controller = controller
        self.command_sink = command_sink
        self.config = config or AutonomousNavigationConfig()

    def run(self) -> list[AutonomousNavigationRecord]:
        """Run the closed-loop navigation policy."""

        if self.config.num_steps <= 0:
            raise ValueError("num_steps must be greater than 0")

        self.config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.controller.reset()
        records: list[AutonomousNavigationRecord] = []
        start_time = time.perf_counter()

        with self.config.output_jsonl.open("w", encoding="utf-8") as log_file:
            for step_index in range(self.config.num_steps):
                iteration_start = time.perf_counter()
                frame = self.camera.capture_scene_frame()
                prediction = self.predictor.predict_tensor(preprocess_array(frame.bgr))
                command = self.controller.step(
                    steering=prediction.steering,
                    collision_probability=prediction.collision_probability,
                )
                applied = self.command_sink.send_command(command)
                elapsed = time.perf_counter() - start_time

                record = AutonomousNavigationRecord(
                    step_index=step_index,
                    timestamp=time.time(),
                    elapsed_seconds=elapsed,
                    width=frame.width,
                    height=frame.height,
                    steering=prediction.steering,
                    collision_probability=prediction.collision_probability,
                    forward_velocity=command.forward_velocity,
                    yaw_radians=command.yaw_radians,
                    yaw_rate_degrees_per_second=applied.yaw_rate_degrees_per_second,
                    command_duration=applied.duration,
                    executed=applied.executed,
                )
                log_file.write(record.to_json() + "\n")
                log_file.flush()
                records.append(record)

                if self.config.interval_seconds > 0 and step_index < self.config.num_steps - 1:
                    sleep_for = max(0.0, self.config.interval_seconds - (time.perf_counter() - iteration_start))
                    time.sleep(sleep_for)

        return records

