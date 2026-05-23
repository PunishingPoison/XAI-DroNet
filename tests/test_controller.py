from math import pi

import pytest

from xai_dronet.control import DroNetController


def test_controller_clamps_predictions_and_smooths_commands() -> None:
    controller = DroNetController(
        max_forward_velocity=2.0,
        velocity_smoothing=0.7,
        yaw_smoothing=0.5,
        max_yaw_radians=pi / 2,
    )

    command = controller.step(steering=2.0, collision_probability=-1.0)

    assert command.forward_velocity == pytest.approx(0.6)
    assert command.yaw_radians == pytest.approx(pi / 4)
