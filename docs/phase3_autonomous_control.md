# Phase 3: Paper-Faithful Autonomous Control

## Purpose

This phase closes the loop:

```text
AirSim frame -> DroNet -> steering + collision probability -> smoothed velocity/yaw command -> AirSim body-frame velocity command
```

The control policy remains reactive, lightweight, and faithful to DroNet. It
does not use SLAM, LiDAR, reinforcement learning, a map, or an external planner.

## Control Mapping

- Collision probability modulates forward speed:
  `speed = (1 - collision_probability) * max_forward_velocity`
- Steering is clamped to `[-1, 1]` and mapped to a bounded yaw command.
- Velocity and yaw are low-pass filtered by `DroNetController`.
- The AirSim adapter sends body-frame forward velocity and a yaw-rate command.

AirSim expects yaw rate in degrees per second for `YawMode(is_rate=True)`, so
the adapter converts the controller's yaw radians to degrees.

## Dry Run

The autonomous script defaults to dry-run mode. It captures frames, runs
inference, computes commands, and writes logs, but does not move the drone:

```powershell
.\.venv\Scripts\python.exe .\scripts\airsim_autonomous_navigation.py --checkpoint .\checkpoints\dronet_official.pt --num-steps 100
```

## Execute Control

Only run this after the read-only live inference loop is stable:

```powershell
.\.venv\Scripts\python.exe .\scripts\airsim_autonomous_navigation.py --checkpoint .\checkpoints\dronet_official.pt --num-steps 100 --execute-control --takeoff --land-on-exit
```

## Outputs

- `outputs/airsim/autonomous_navigation.jsonl`
- each row contains:
  - steering
  - collision probability
  - smoothed forward velocity
  - smoothed yaw command
  - applied AirSim yaw rate
  - whether control was actually executed

## Safety Boundary

Start in Blocks or another simple environment. Use low `--max-forward-velocity`
and short `--num-steps` at first.

