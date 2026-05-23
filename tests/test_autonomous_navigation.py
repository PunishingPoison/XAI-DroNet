import json
from pathlib import Path

import numpy as np

from xai_dronet.control import DroNetControlCommand, DroNetController
from xai_dronet.inference import DroNetPrediction
from xai_dronet.simulator import (
    AirSimAppliedCommand,
    AirSimFrame,
    AutonomousNavigationConfig,
    AutonomousNavigationRunner,
)


class FakeFrameSource:
    def __init__(self) -> None:
        self.calls = 0

    def capture_scene_frame(self) -> AirSimFrame:
        self.calls += 1
        rgb = np.full((4, 5, 3), 128, dtype=np.uint8)
        return AirSimFrame(rgb=rgb, camera_name="0", width=5, height=4)


class FakePredictor:
    def __init__(self, predictions: list[DroNetPrediction]) -> None:
        self.predictions = predictions
        self.calls = 0

    def predict_tensor(self, image_tensor) -> DroNetPrediction:
        prediction = self.predictions[self.calls]
        self.calls += 1
        return prediction


class FakeCommandSink:
    def __init__(self) -> None:
        self.commands: list[DroNetControlCommand] = []

    def send_command(self, command: DroNetControlCommand) -> AirSimAppliedCommand:
        self.commands.append(command)
        return AirSimAppliedCommand(
            forward_velocity=command.forward_velocity,
            yaw_rate_degrees_per_second=10.0,
            duration=0.1,
            executed=False,
        )


def test_autonomous_navigation_loop_logs_predictions_and_commands(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "navigation.jsonl"
    camera = FakeFrameSource()
    predictor = FakePredictor(
        [
            DroNetPrediction(steering=0.0, collision_probability=0.0),
            DroNetPrediction(steering=1.0, collision_probability=1.0),
        ]
    )
    controller = DroNetController(max_forward_velocity=1.0)
    command_sink = FakeCommandSink()

    records = AutonomousNavigationRunner(
        camera=camera,
        predictor=predictor,
        controller=controller,
        command_sink=command_sink,
        config=AutonomousNavigationConfig(num_steps=2, output_jsonl=output_jsonl),
    ).run()

    assert len(records) == 2
    assert camera.calls == 2
    assert predictor.calls == 2
    assert len(command_sink.commands) == 2
    assert command_sink.commands[0].forward_velocity > command_sink.commands[1].forward_velocity

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["step_index"] == 0
    assert rows[1]["collision_probability"] == 1.0
    assert rows[1]["executed"] is False

