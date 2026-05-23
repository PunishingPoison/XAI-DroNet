from math import pi

import pytest

from xai_dronet.control import DroNetControlCommand
from xai_dronet.simulator import AirSimControlConfig, AirSimDroneControlClient


class FakeFuture:
    def __init__(self) -> None:
        self.joined = False

    def join(self) -> None:
        self.joined = True


class FakeAirSimModule:
    class DrivetrainType:
        MaxDegreeOfFreedom = 0

    class YawMode:
        def __init__(self, is_rate=True, yaw_or_rate=0.0):
            self.is_rate = is_rate
            self.yaw_or_rate = yaw_or_rate


class FakeClient:
    def __init__(self) -> None:
        self.confirmed = False
        self.api_control = []
        self.armed = []
        self.commands = []
        self.future = FakeFuture()

    def confirmConnection(self):
        self.confirmed = True

    def enableApiControl(self, enabled, vehicle_name=""):
        self.api_control.append((enabled, vehicle_name))

    def armDisarm(self, armed, vehicle_name=""):
        self.armed.append((armed, vehicle_name))

    def takeoffAsync(self, vehicle_name=""):
        return self.future

    def moveByVelocityBodyFrameAsync(
        self,
        vx,
        vy,
        vz,
        duration,
        drivetrain=0,
        yaw_mode=None,
        vehicle_name="",
    ):
        self.commands.append((vx, vy, vz, duration, drivetrain, yaw_mode, vehicle_name))
        return self.future


def test_dry_run_control_client_does_not_call_airsim() -> None:
    fake_client = FakeClient()
    control = AirSimDroneControlClient(client=fake_client, dry_run=True)

    applied = control.send_command(DroNetControlCommand(forward_velocity=1.0, yaw_radians=pi / 2))

    assert applied.executed is False
    assert applied.yaw_rate_degrees_per_second == pytest.approx(90.0)
    assert fake_client.commands == []


def test_control_client_sends_body_frame_velocity_command() -> None:
    fake_client = FakeClient()
    config = AirSimControlConfig(vehicle_name="Drone1", command_duration=0.2)
    control = AirSimDroneControlClient(
        config=config,
        client=fake_client,
        airsim_module=FakeAirSimModule,
        dry_run=False,
    )

    control.connect()
    control.prepare_for_flight(takeoff=True)
    applied = control.send_command(DroNetControlCommand(forward_velocity=0.5, yaw_radians=pi / 4))

    assert fake_client.confirmed
    assert fake_client.api_control == [(True, "Drone1")]
    assert fake_client.armed == [(True, "Drone1")]
    assert applied.executed is True
    assert applied.yaw_rate_degrees_per_second == pytest.approx(45.0)
    assert fake_client.future.joined
    command = fake_client.commands[0]
    assert command[0:4] == (0.5, 0.0, 0.0, 0.2)
    assert command[5].is_rate is True
    assert command[5].yaw_or_rate == pytest.approx(45.0)

