"""PyTorch implementation of the DroNet ResNet-8 architecture.

The architecture follows the original DroNet design:
- one grayscale `200 x 200` monocular input frame,
- a small shared residual CNN backbone,
- one steering regression head,
- one collision-probability head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DroNetConfig:
    """Configuration for the paper-faithful DroNet model."""

    input_channels: int = 1
    output_dim: int = 1
    dropout: float = 0.5


class ResidualBlock(nn.Module):
    """Downsampling residual block used by DroNet's ResNet-8 backbone."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )
        self.shortcut = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + self.shortcut(x)


class ChannelsLastFlatten(nn.Module):
    """Flatten features in Keras/TensorFlow NHWC order.

    This matters when converting official Keras `.h5` weights: dense-layer
    weights are learned against NHWC flattening, while PyTorch tensors are NCHW.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(x.permute(0, 2, 3, 1).contiguous(), start_dim=1)


class DroNet(nn.Module):
    """Forked ResNet-8 model for steering and collision prediction."""

    def __init__(self, config: DroNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or DroNetConfig()

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.config.input_channels,
                32,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=True,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
        )
        self.head_features = nn.Sequential(
            ChannelsLastFlatten(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.dropout),
        )
        self.steering_head = nn.Linear(128 * 7 * 7, self.config.output_dim)
        self.collision_head = nn.Linear(128 * 7 * 7, self.config.output_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return `(steering, collision_probability)` for a batch of frames."""

        features = self.stem(x)
        features = self.residual_blocks(features)
        features = self.head_features(features)

        steering = self.steering_head(features)
        collision_probability = torch.sigmoid(self.collision_head(features))
        return steering, collision_probability
