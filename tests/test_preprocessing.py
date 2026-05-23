import numpy as np

from xai_dronet.inference.preprocessing import ImagePreprocessConfig, preprocess_array


def test_preprocess_array_returns_dronet_input_tensor() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    tensor = preprocess_array(image, ImagePreprocessConfig(height=200, width=200))

    assert tuple(tensor.shape) == (1, 1, 200, 200)
    assert str(tensor.dtype) == "torch.float32"
    assert float(tensor.min()) == 0.0
    assert float(tensor.max()) == 0.0
