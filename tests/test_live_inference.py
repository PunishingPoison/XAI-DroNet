import json
from pathlib import Path

import numpy as np

from xai_dronet.inference import DroNetPrediction
from xai_dronet.simulator import AirSimFrame, LiveAirSimInferenceRunner, LiveInferenceConfig


class FakeFrameSource:
    def __init__(self) -> None:
        self.calls = 0

    def capture_scene_frame(self) -> AirSimFrame:
        value = self.calls
        self.calls += 1
        rgb = np.full((4, 5, 3), value, dtype=np.uint8)
        return AirSimFrame(rgb=rgb, camera_name="0", width=5, height=4)


class FakePredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict_tensor(self, image_tensor) -> DroNetPrediction:
        assert tuple(image_tensor.shape) == (1, 1, 200, 200)
        self.calls += 1
        return DroNetPrediction(
            steering=0.1 * self.calls,
            collision_probability=0.2,
        )


def test_live_inference_runner_writes_jsonl(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "predictions.jsonl"
    config = LiveInferenceConfig(num_frames=3, output_jsonl=output_jsonl)
    camera = FakeFrameSource()
    predictor = FakePredictor()

    records = LiveAirSimInferenceRunner(camera, predictor, config).run()

    assert len(records) == 3
    assert camera.calls == 3
    assert predictor.calls == 3
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["frame_index"] for row in rows] == [0, 1, 2]
    assert rows[-1]["steering"] == 0.30000000000000004


def test_live_inference_runner_can_save_selected_frames(tmp_path: Path) -> None:
    config = LiveInferenceConfig(
        num_frames=3,
        output_jsonl=tmp_path / "predictions.jsonl",
        frame_output_dir=tmp_path / "frames",
        save_every_n_frames=2,
    )

    records = LiveAirSimInferenceRunner(FakeFrameSource(), FakePredictor(), config).run()

    saved_frames = [record.saved_frame for record in records if record.saved_frame is not None]
    assert len(saved_frames) == 2
    for saved_frame in saved_frames:
        assert Path(saved_frame).exists()

