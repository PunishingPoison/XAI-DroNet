# Phase 2: AirSim Camera Bridge

## Purpose

This phase connects the official DroNet checkpoint to AirSim camera frames, but
still does not control the drone.

The target data flow is:

```text
AirSim front camera -> RGB scene frame -> DroNet preprocessing -> official DroNet checkpoint -> steering + collision probability
```

Keeping this bridge read-only makes failures easier to debug before autonomous
movement is introduced.

## Requirements

Install the optional AirSim dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-airsim.txt
```

If AirSim package metadata generation fails because of legacy setup-time
imports, install the compatibility dependencies first and then install AirSim
without build isolation:

```powershell
.\.venv\Scripts\python.exe -m pip install msgpack-rpc-python backports.ssl_match_hostname
.\.venv\Scripts\python.exe -m pip install airsim --no-build-isolation
```

Start an AirSim multirotor environment such as Blocks or a UE 4.27 project with
AirSim enabled. The default RPC port is `41451`.

## Command

After AirSim is running:

```powershell
.\.venv\Scripts\python.exe .\scripts\airsim_capture_inference.py --checkpoint .\checkpoints\dronet_official.pt --device cpu
```

Run a short read-only live inference loop:

```powershell
.\.venv\Scripts\python.exe .\scripts\airsim_live_inference.py --checkpoint .\checkpoints\dronet_official.pt --device cpu --num-frames 60 --save-every-n-frames 10
```

## Expected Outputs

- `outputs/airsim/latest_frame.png`
- `outputs/airsim/latest_prediction.json`
- `outputs/airsim/live_predictions.jsonl`
- optionally saved frames under `outputs/airsim/frames/`
- Console JSON containing:
  - frame path
  - image width and height
  - steering prediction
  - collision probability
  - average loop FPS for live inference

## Current Boundary

This step is camera/inference only. It must pass before adding takeoff,
velocity, yaw, or collision-avoidance control commands.
