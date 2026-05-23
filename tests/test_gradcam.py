import numpy as np
import torch

from xai_dronet.explainability import DroNetGradCAM, overlay_heatmap_on_bgr
from xai_dronet.models import DroNet


def test_gradcam_generates_normalized_heatmap() -> None:
    model = DroNet()
    image = torch.rand(1, 1, 200, 200)

    result = DroNetGradCAM(model).generate(image, target="collision")

    assert result.heatmap.shape == (200, 200)
    assert result.heatmap.dtype == np.float32
    assert float(result.heatmap.min()) >= 0.0
    assert float(result.heatmap.max()) <= 1.0
    assert 0.0 <= result.prediction.collision_probability <= 1.0


def test_gradcam_rejects_batch_size_greater_than_one() -> None:
    model = DroNet()
    image = torch.rand(2, 1, 200, 200)

    try:
        DroNetGradCAM(model).generate(image)
    except ValueError as exc:
        assert "batch size of 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_overlay_heatmap_on_bgr_preserves_image_shape() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    heatmap = np.ones((10, 15), dtype=np.float32)

    overlay = overlay_heatmap_on_bgr(image, heatmap)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8

