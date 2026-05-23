"""Paper-style conversion from DroNet outputs to motion commands.

DroNet controls a UAV on a plane using forward velocity and yaw. Collision
probability modulates forward speed, and steering is mapped to a yaw command.
Both are low-pass filtered for smooth motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class DroNetControlCommand:
    """Smoothed motion command derived from one DroNet prediction."""

    forward_velocity: float
    yaw_radians: float


class DroNetController:
    """Low-pass filtered controller matching the DroNet paper equations."""

    def __init__(
        self,
        max_forward_velocity: float = 1.0,
        velocity_smoothing: float = 0.7,
        yaw_smoothing: float = 0.5,
        max_yaw_radians: float = pi / 2,
    ) -> None:
        self.max_forward_velocity = max_forward_velocity
        self.velocity_smoothing = velocity_smoothing
        self.yaw_smoothing = yaw_smoothing
        self.max_yaw_radians = max_yaw_radians
        self._previous_velocity = 0.0
        self._previous_yaw = 0.0

    def reset(self) -> None:
        """Reset filter state between independent navigation episodes."""

        self._previous_velocity = 0.0
        self._previous_yaw = 0.0

    def step(self, steering: float, collision_probability: float) -> DroNetControlCommand:
        """Convert one prediction into a smoothed velocity/yaw command."""

        steering = max(-1.0, min(1.0, steering))
        collision_probability = max(0.0, min(1.0, collision_probability))

        target_velocity = (1.0 - collision_probability) * self.max_forward_velocity
        velocity = (
            self.velocity_smoothing * self._previous_velocity
            + (1.0 - self.velocity_smoothing) * target_velocity
        )

        target_yaw = self.max_yaw_radians * steering
        
        # Panic reflex: If a crash is imminent, force a sharp turn to escape!
        if collision_probability > 0.8:
            target_yaw = self.max_yaw_radians * (1.0 if steering >= 0 else -1.0)
            
        yaw = (
            (1.0 - self.yaw_smoothing) * self._previous_yaw
            + self.yaw_smoothing * target_yaw
        )

        self._previous_velocity = velocity
        self._previous_yaw = yaw
        return DroNetControlCommand(forward_velocity=velocity, yaw_radians=yaw)

