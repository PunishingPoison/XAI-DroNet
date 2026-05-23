"""DroNet training losses.

The original project mixes steering and collision samples in one stream. Each
target tensor stores a task flag in column 0 and the value in column 1:

- steering target: `task_flag == 1`
- collision target: `task_flag == 0`

The losses below keep that contract explicit for a future dataset loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class DroNetLossOutput:
    """Container for total and per-task DroNet losses."""

    total: Tensor
    steering_mse: Tensor
    collision_bce: Tensor
    collision_weight: float


def collision_loss_weight(epoch: int, decay: float = 0.1, epoch0: int = 10) -> float:
    """Return the curriculum weight applied to collision BCE."""

    return max(0.0, 1.0 - exp(-decay * (epoch - epoch0)))


def _zero_like_loss(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def hard_mining_mse(prediction: Tensor, target: Tensor, k: int) -> Tensor:
    """Compute hard-mined steering MSE for steering samples only."""

    task_flag = target[:, 0]
    steering_mask = task_flag == 1
    num_samples = int(steering_mask.sum().item())
    if num_samples == 0:
        return _zero_like_loss(prediction)

    pred_steering = prediction.squeeze(-1)[steering_mask]
    true_steering = target[:, 1][steering_mask]
    losses = torch.square(pred_steering - true_steering)
    k_min = min(k, num_samples)
    hard_losses = torch.topk(losses, k=k_min).values
    return hard_losses.sum() / float(k)


def hard_mining_bce(prediction: Tensor, target: Tensor, k: int) -> Tensor:
    """Compute hard-mined collision BCE for collision samples only."""

    task_flag = target[:, 0]
    collision_mask = task_flag == 0
    num_samples = int(collision_mask.sum().item())
    if num_samples == 0:
        return _zero_like_loss(prediction)

    pred_collision = prediction.squeeze(-1)[collision_mask].clamp(1e-6, 1.0 - 1e-6)
    true_collision = target[:, 1][collision_mask]
    losses = F.binary_cross_entropy(pred_collision, true_collision, reduction="none")
    k_min = min(k, num_samples)
    hard_losses = torch.topk(losses, k=k_min).values
    return hard_losses.sum() / float(k)


def compute_dronet_loss(
    steering_prediction: Tensor,
    collision_prediction: Tensor,
    steering_target: Tensor,
    collision_target: Tensor,
    *,
    epoch: int,
    hard_mining_k: int,
    decay: float = 0.1,
    epoch0: int = 10,
) -> DroNetLossOutput:
    """Compute the combined DroNet objective for one batch."""

    steering_mse = hard_mining_mse(steering_prediction, steering_target, hard_mining_k)
    collision_bce = hard_mining_bce(collision_prediction, collision_target, hard_mining_k)
    weight = collision_loss_weight(epoch=epoch, decay=decay, epoch0=epoch0)
    total = steering_mse + weight * collision_bce
    return DroNetLossOutput(
        total=total,
        steering_mse=steering_mse,
        collision_bce=collision_bce,
        collision_weight=weight,
    )

