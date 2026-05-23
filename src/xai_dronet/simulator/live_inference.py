"""Read-only live inference loop for AirSim frames."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

from xai_dronet.inference import DroNetPrediction
from xai_dronet.inference.preprocessing import preprocess_array
from xai_dronet.simulator.airsim_camera import AirSimCameraClient, AirSimFrame


class FrameSource(Protocol):
    """Protocol for objects that can provide RGB simulator frames."""

    def capture_scene_frame(self) -> AirSimFrame:
        """Capture one frame."""


class TensorPredictor(Protocol):
    """Protocol for objects that can run DroNet tensor inference."""

    def predict_tensor(self, image_tensor) -> DroNetPrediction:
        """Predict steering and collision probability."""


@dataclass(frozen=True)
class LiveInferenceConfig:
    """Settings for read-only AirSim inference."""

    num_frames: int = 30
    interval_seconds: float = 0.0
    output_jsonl: Path = Path("outputs/airsim/live_predictions.jsonl")
    frame_output_dir: Path = Path("outputs/airsim/frames")
    save_every_n_frames: int = 0


@dataclass(frozen=True)
class LiveInferenceRecord:
    """One logged inference result."""

    frame_index: int
    timestamp: float
    elapsed_seconds: float
    fps: float
    width: int
    height: int
    steering: float
    collision_probability: float
    saved_frame: str | None

    def to_json(self) -> str:
        """Serialize record as one JSONL row."""

        return json.dumps(self.__dict__)


class LiveAirSimInferenceRunner:
    """Run DroNet repeatedly on AirSim camera frames without controlling the drone."""

    def __init__(
        self,
        camera: FrameSource | AirSimCameraClient,
        predictor: TensorPredictor,
        config: LiveInferenceConfig | None = None,
    ) -> None:
        self.camera = camera
        self.predictor = predictor
        self.config = config or LiveInferenceConfig()

    def run(self) -> list[LiveInferenceRecord]:
        """Run the configured inference loop and write JSONL output."""

        if self.config.num_frames <= 0:
            raise ValueError("num_frames must be greater than 0")

        self.config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if self.config.save_every_n_frames > 0:
            self.config.frame_output_dir.mkdir(parents=True, exist_ok=True)

        records: list[LiveInferenceRecord] = []
        start_time = time.perf_counter()

        with self.config.output_jsonl.open("w", encoding="utf-8") as log_file:
            for frame_index in range(self.config.num_frames):
                iteration_start = time.perf_counter()
                frame = self.camera.capture_scene_frame()
                prediction = self.predictor.predict_tensor(preprocess_array(frame.bgr))
                elapsed = time.perf_counter() - start_time
                iteration_elapsed = max(time.perf_counter() - iteration_start, 1e-9)
                saved_frame = self._save_frame(frame, frame_index)

                record = LiveInferenceRecord(
                    frame_index=frame_index,
                    timestamp=time.time(),
                    elapsed_seconds=elapsed,
                    fps=1.0 / iteration_elapsed,
                    width=frame.width,
                    height=frame.height,
                    steering=prediction.steering,
                    collision_probability=prediction.collision_probability,
                    saved_frame=saved_frame,
                )
                log_file.write(record.to_json() + "\n")
                log_file.flush()
                records.append(record)

                if self.config.interval_seconds > 0 and frame_index < self.config.num_frames - 1:
                    sleep_for = max(0.0, self.config.interval_seconds - iteration_elapsed)
                    time.sleep(sleep_for)

        return records

    def _save_frame(self, frame: AirSimFrame, frame_index: int) -> str | None:
        if self.config.save_every_n_frames <= 0:
            return None
        if frame_index % self.config.save_every_n_frames != 0:
            return None

        output_path = self.config.frame_output_dir / f"frame_{frame_index:06d}.png"
        if not cv2.imwrite(str(output_path), frame.bgr):
            raise RuntimeError(f"Failed to save frame: {output_path}")
        return str(output_path)

