"""AirSim control adapter for DroNet motion commands."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import degrees
from typing import Any

from xai_dronet.control import DroNetControlCommand
from xai_dronet.simulator.airsim_camera import AirSimUnavailableError


@dataclass(frozen=True)
class AirSimControlConfig:
    """Settings for sending body-frame velocity commands to AirSim."""

    ip: str = ""
    port: int = 41451
    timeout_value: int = 10
    vehicle_name: str = ""
    command_duration: float = 0.1
    wait_for_command: bool = True


@dataclass(frozen=True)
class AirSimAppliedCommand:
    """Command values sent to AirSim."""

    forward_velocity: float
    yaw_rate_degrees_per_second: float
    duration: float
    executed: bool


class AirSimDroneControlClient:
    """Send paper-derived DroNet commands to AirSim."""

    def __init__(
        self,
        config: AirSimControlConfig | None = None,
        *,
        client: Any | None = None,
        airsim_module: Any | None = None,
        dry_run: bool = True,
    ) -> None:
        self.config = config or AirSimControlConfig()
        self._client = client
        self._airsim = airsim_module
        self.dry_run = dry_run

    def connect(self) -> None:
        """Create the AirSim client and confirm connection."""

        if self.dry_run:
            return

        if self._client is None:
            self._airsim = self._airsim or self._import_airsim()
            self._client = self._airsim.MultirotorClient(
                ip=self.config.ip,
                port=self.config.port,
                timeout_value=self.config.timeout_value,
            )

        try:
            self._client.confirmConnection()
        except Exception as exc:  # pragma: no cover - exercised only with AirSim running.
            raise AirSimUnavailableError(
                "Could not connect to AirSim. Start an AirSim multirotor environment "
                "before running control."
            ) from exc

    def prepare_for_flight(self, *, takeoff: bool = False) -> None:
        """Enable API control, arm, and optionally take off."""

        if self.dry_run:
            return

        if self._client is None:
            self.connect()

        self._client.enableApiControl(True, vehicle_name=self.config.vehicle_name)
        self._client.armDisarm(True, vehicle_name=self.config.vehicle_name)
        if takeoff:
            self._client.takeoffAsync(vehicle_name=self.config.vehicle_name).join()

    def send_command(self, command: DroNetControlCommand) -> AirSimAppliedCommand:
        """Apply one DroNet control command to AirSim."""

        yaw_rate = degrees(command.yaw_radians)
        applied = AirSimAppliedCommand(
            forward_velocity=command.forward_velocity,
            yaw_rate_degrees_per_second=yaw_rate,
            duration=self.config.command_duration,
            executed=not self.dry_run,
        )

        if self.dry_run:
            return applied

        if self._client is None:
            self.connect()

        airsim = self._airsim or self._import_airsim()
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        future = self._client.moveByVelocityBodyFrameAsync(
            command.forward_velocity,
            0.0,
            0.0,
            self.config.command_duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode,
            vehicle_name=self.config.vehicle_name,
        )
        if self.config.wait_for_command:
            future.join()
        return applied

    def hover(self) -> None:
        """Command AirSim to hover."""

        if self.dry_run:
            return
        if self._client is None:
            self.connect()
        self._client.hoverAsync(vehicle_name=self.config.vehicle_name).join()

    def land(self) -> None:
        """Land and release control."""

        if self.dry_run:
            return
        if self._client is None:
            self.connect()
        self._client.landAsync(vehicle_name=self.config.vehicle_name).join()
        self._client.armDisarm(False, vehicle_name=self.config.vehicle_name)
        self._client.enableApiControl(False, vehicle_name=self.config.vehicle_name)

    @staticmethod
    def _import_airsim() -> Any:
        try:
            return import_module("airsim")
        except ImportError as exc:
            raise AirSimUnavailableError(
                "The AirSim Python package is not installed. Install it with "
                "`pip install -r requirements-airsim.txt`."
            ) from exc

