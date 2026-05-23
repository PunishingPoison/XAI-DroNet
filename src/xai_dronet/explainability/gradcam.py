"""Grad-CAM explainability for DroNet.

Grad-CAM is used only for visualization. It must not affect navigation control
or model predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from xai_dronet.inference import DroNetPrediction
from xai_dronet.models import DroNet

GradCAMTarget = Literal["collision", "steering"]


@dataclass(frozen=True)
class GradCAMResult:
    """Output of one Grad-CAM explanation."""

    heatmap: np.ndarray
    prediction: DroNetPrediction
    target: GradCAMTarget


class DroNetGradCAM:
    """Generate Grad-CAM heatmaps from a DroNet model."""

    def __init__(
        self,
        model: DroNet,
        *,
        device: str | torch.device = "cpu",
        target_layer: nn.Module | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.target_layer = target_layer or self.model.residual_blocks[-1].main[5]

    def generate(self, image_tensor: Tensor, target: GradCAMTarget = "collision") -> GradCAMResult:
        """Generate a normalized heatmap for one preprocessed image tensor."""

        if image_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM currently expects a batch size of 1")

        self.model.eval()
        activations: list[Tensor] = []
        gradients: list[Tensor] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

        try:
            self.model.zero_grad(set_to_none=True)
            input_tensor = image_tensor.to(self.device, dtype=torch.float32).detach().requires_grad_(True)
            steering, collision = self.model(input_tensor)
            score = collision[:, 0].sum() if target == "collision" else steering[:, 0].sum()
            score.backward()

            if not activations or not gradients:
                raise RuntimeError("Failed to capture activations or gradients for Grad-CAM")

            heatmap = self._compute_heatmap(
                activations[-1],
                gradients[-1],
                output_size=tuple(input_tensor.shape[-2:]),
            )
            prediction = DroNetPrediction(
                steering=float(steering.squeeze().detach().cpu()),
                collision_probability=float(collision.squeeze().detach().cpu()),
            )
            return GradCAMResult(heatmap=heatmap, prediction=prediction, target=target)
        finally:
            forward_handle.remove()
            backward_handle.remove()
            self.model.zero_grad(set_to_none=True)

    @staticmethod
    def _compute_heatmap(activation: Tensor, gradient: Tensor, output_size: tuple[int, int]) -> np.ndarray:
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu()

        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max - cam_min < 1e-12:
            return np.zeros(tuple(output_size), dtype=np.float32)
        normalized = (cam - cam_min) / (cam_max - cam_min)
        return normalized.numpy().astype(np.float32)


def overlay_heatmap_on_bgr(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a normalized heatmap onto a BGR image."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape HxWx3, got {image_bgr.shape}")

    if heatmap.shape[:2] != image_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, color_heatmap, alpha, 0.0)
    return overlay

