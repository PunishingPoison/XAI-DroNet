# Phase 4: Grad-CAM Explainability

## Purpose

Grad-CAM is an explainability extension only. It visualizes image regions that
influence DroNet predictions, but it does not affect control, collision
avoidance, or evaluation metrics.

## Current Scope

- Generate Grad-CAM for one offline image.
- Support collision-probability and steering targets.
- Save a grayscale heatmap and a color overlay.
- Use the last convolution in DroNet's third residual block.

## Command

```powershell
.\.venv\Scripts\python.exe .\scripts\run_gradcam.py --image .\data\samples\synthetic_road_frame.png --checkpoint .\checkpoints\dronet_official.pt --target collision
```

## Outputs

- `outputs/gradcam/gradcam_heatmap.png`
- `outputs/gradcam/gradcam_overlay.png`
- console JSON containing the prediction and output paths

## Research Boundary

Grad-CAM answers: "what visual regions influenced this prediction?"

It does not answer whether the navigation policy is correct. That belongs to
the evaluation framework.

