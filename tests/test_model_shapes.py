import torch

from xai_dronet.models import DroNet


def test_dronet_forward_shapes_and_probability_range() -> None:
    model = DroNet()
    model.eval()

    image = torch.rand(2, 1, 200, 200)
    with torch.inference_mode():
        steering, collision = model(image)

    assert steering.shape == (2, 1)
    assert collision.shape == (2, 1)
    assert torch.all(collision >= 0.0)
    assert torch.all(collision <= 1.0)

