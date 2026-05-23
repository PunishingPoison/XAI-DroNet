from types import SimpleNamespace

import numpy as np
import pytest

from xai_dronet.simulator import AirSimCameraClient, AirSimConnectionConfig, AirSimUnavailableError


class FakeAirSimModule:
    class ImageType:
        Scene = 0

    class ImageRequest:
        def __init__(self, camera_name, image_type, pixels_as_float, compress):
            self.camera_name = camera_name
            self.image_type = image_type
            self.pixels_as_float = pixels_as_float
            self.compress = compress


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.connected = False

    def confirmConnection(self):
        self.connected = True

    def simGetImages(self, requests, vehicle_name=""):
        assert len(requests) == 1
        assert requests[0].camera_name == "0"
        assert vehicle_name == ""
        return [self.response]


def test_capture_scene_frame_from_fake_airsim_response() -> None:
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    response = SimpleNamespace(width=3, height=2, image_data_uint8=rgb.tobytes())
    client = FakeClient(response)
    camera = AirSimCameraClient(
        AirSimConnectionConfig(camera_name="0"),
        client=client,
        airsim_module=FakeAirSimModule,
    )

    camera.connect()
    frame = camera.capture_scene_frame()

    assert client.connected
    assert frame.width == 3
    assert frame.height == 2
    assert np.array_equal(frame.rgb, rgb)


def test_invalid_response_size_raises() -> None:
    response = SimpleNamespace(width=3, height=2, image_data_uint8=b"too-small")
    camera = AirSimCameraClient(
        client=FakeClient(response),
        airsim_module=FakeAirSimModule,
    )

    with pytest.raises(AirSimUnavailableError):
        camera.capture_scene_frame()

