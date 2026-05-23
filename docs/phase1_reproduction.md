# Phase 1: DroNet Inference Reproduction

## Purpose

Phase 1 proves the core DroNet inference contract before any simulator code is
introduced:

```text
single monocular image -> grayscale 200 x 200 tensor -> DroNet -> steering + collision probability
```

This keeps the project aligned with the paper and prevents AirSim, dashboard, or
controller complexity from hiding model mistakes.

## Paper-Faithful Choices

- Input is one forward-looking monocular frame.
- Frames are converted to grayscale and resized to `200 x 200`.
- Pixel values are rescaled to `[0, 1]`.
- The model is a lightweight ResNet-8 style CNN with three residual blocks.
- The CNN backbone is shared by two heads:
  - steering regression
  - collision probability classification
- Collision probability is produced through a sigmoid.
- The controller utility maps steering to yaw and collision probability to
  forward velocity, matching the paper's control idea.
- Training-loss utilities preserve the original mixed-task target format:
  steering samples use task flag `1`, collision samples use task flag `0`.

## Current Model State

The model can run with either random weights for plumbing tests or converted
official DroNet weights for paper-aligned inference. Downloaded and converted
weights live under `checkpoints/` and are ignored by git as reproducible
artifacts.

## Commands

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Create a deterministic smoke-test frame:

```powershell
.\.venv\Scripts\python.exe .\scripts\create_sample_frame.py
```

Run offline image inference:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_image_inference.py --image .\data\samples\synthetic_road_frame.png --device cpu
```

Validate a PyTorch checkpoint:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_checkpoint.py --checkpoint .\checkpoints\dronet.pt --device cpu
```

## Expected Outputs

- Tests should pass.
- Smoke inference should print a steering float and a collision probability
  between `0` and `1`.
- Checkpoint validation should print the parameter count and one sample
  prediction.
- Loss tests should confirm task-aware hard mining and the staged BCE weighting
  used for collision prediction.

## Next Gate

Before AirSim integration, model weights must be available through one of these
paths:

1. port official Keras weights to PyTorch,
2. train/fine-tune this PyTorch model on DroNet-compatible data,
3. or temporarily use the official Keras implementation as a reference while
   validating PyTorch parity.

The current repository supports option 1, and this has been verified locally
with:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_official_dronet_model.py
.\.venv\Scripts\python.exe .\scripts\convert_keras_weights.py --keras-weights .\checkpoints\official\dronet\best_weights.h5 --output .\checkpoints\dronet_official.pt
.\.venv\Scripts\python.exe .\scripts\check_checkpoint.py --checkpoint .\checkpoints\dronet_official.pt --device cpu
```
