"""AirSim integration package."""

from xai_dronet.simulator.airsim_camera import (
    AirSimCameraClient,
    AirSimConnectionConfig,
    AirSimFrame,
    AirSimUnavailableError,
)
from xai_dronet.simulator.airsim_control import (
    AirSimAppliedCommand,
    AirSimControlConfig,
    AirSimDroneControlClient,
)
from xai_dronet.simulator.autonomous_navigation import (
    AutonomousNavigationConfig,
    AutonomousNavigationRecord,
    AutonomousNavigationRunner,
)
from xai_dronet.simulator.live_inference import (
    LiveAirSimInferenceRunner,
    LiveInferenceConfig,
    LiveInferenceRecord,
)

__all__ = [
    "AirSimCameraClient",
    "AirSimConnectionConfig",
    "AirSimFrame",
    "AirSimUnavailableError",
    "AirSimAppliedCommand",
    "AirSimControlConfig",
    "AirSimDroneControlClient",
    "AutonomousNavigationConfig",
    "AutonomousNavigationRecord",
    "AutonomousNavigationRunner",
    "LiveAirSimInferenceRunner",
    "LiveInferenceConfig",
    "LiveInferenceRecord",
]
