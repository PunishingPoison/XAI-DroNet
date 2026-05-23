import torch

from xai_dronet.inference import DroNetPredictor
from xai_dronet.models import DroNet


def test_predictor_returns_python_floats_for_tensor() -> None:
    predictor = DroNetPredictor(device="cpu")
    image = torch.zeros(1, 1, 200, 200)

    prediction = predictor.predict_tensor(image)

    assert isinstance(prediction.steering, float)
    assert isinstance(prediction.collision_probability, float)
    assert 0.0 <= prediction.collision_probability <= 1.0


def test_predictor_loads_state_dict_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "dronet_state_dict.pt"
    model = DroNet()
    torch.save(model.state_dict(), checkpoint_path)

    predictor = DroNetPredictor(checkpoint_path=checkpoint_path, device="cpu")
    prediction = predictor.predict_tensor(torch.zeros(1, 1, 200, 200))

    assert isinstance(prediction.steering, float)
    assert 0.0 <= prediction.collision_probability <= 1.0

