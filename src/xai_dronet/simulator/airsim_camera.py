"""AirSim monocular camera bridge.

This module handles only the first simulator integration step: acquire one
front-facing scene image from AirSim. It intentionally does not issue movement
commands, keeping control separate until the camera-to-inference path is stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import cv2
import numpy as np


class AirSimUnavailableError(RuntimeError):
    """Raised when the AirSim Python client is missing or unreachable."""


@dataclass(frozen=True)
class AirSimConnectionConfig:
    """Connection and camera settings for AirSim frame capture."""

    ip: str = ""
    port: int = 41451
    timeout_value: int = 10
    camera_name: str = "0"
    vehicle_name: str = ""


@dataclass(frozen=True)
class AirSimFrame:
    """One RGB scene frame captured from AirSim."""

    rgb: np.ndarray
    camera_name: str
    width: int
    height: int

    @property
    def bgr(self) -> np.ndarray:
        """Return the frame converted for OpenCV file writing/preprocessing."""

        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)


class AirSimCameraClient:
    """Small wrapper around AirSim's `MultirotorClient` image API."""

    def __init__(
        self,
        config: AirSimConnectionConfig | None = None,
        *,
        client: Any | None = None,
        airsim_module: Any | None = None,
    ) -> None:
        self.config = config or AirSimConnectionConfig()
        self._airsim = airsim_module
        self._client = client

    def connect(self) -> None:
        """Create the AirSim client and confirm the simulator connection."""

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
                "before running this command."
            ) from exc

    def capture_scene_frame(self) -> AirSimFrame:
        """Capture one uncompressed scene image from the configured camera."""

        airsim = self._airsim or self._import_airsim()
        if self._client is None:
            self.connect()

        request = airsim.ImageRequest(
            self.config.camera_name,
            airsim.ImageType.Scene,
            False,
            False,
        )
        responses = self._client.simGetImages([request], vehicle_name=self.config.vehicle_name)
        if not responses:
            raise AirSimUnavailableError("AirSim returned no image responses.")

        response = responses[0]
        return self._frame_from_response(response)

    def save_scene_frame(self, output_path: str) -> AirSimFrame:
        """Capture one scene frame and save it as an image file."""

        frame = self.capture_scene_frame()
        if not cv2.imwrite(output_path, frame.bgr):
            raise RuntimeError(f"Failed to write AirSim frame: {output_path}")
        return frame

    def _frame_from_response(self, response: Any) -> AirSimFrame:
        width = int(response.width)
        height = int(response.height)
        if width <= 0 or height <= 0:
            raise AirSimUnavailableError(f"Invalid AirSim image size: {width} x {height}")

        image_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        expected_size = width * height * 3
        if image_data.size != expected_size:
            raise AirSimUnavailableError(
                f"Expected {expected_size} RGB bytes from AirSim, got {image_data.size}."
            )

        rgb = image_data.reshape(height, width, 3)
        return AirSimFrame(
            rgb=rgb,
            camera_name=self.config.camera_name,
            width=width,
            height=height,
        )

    @staticmethod
    def _import_airsim() -> Any:
        try:
            return import_module("airsim")
        except ImportError as exc:
            raise AirSimUnavailableError(
                "The AirSim Python package is not installed. Install it with "
                "`pip install -r requirements-airsim.txt`."
            ) from exc

