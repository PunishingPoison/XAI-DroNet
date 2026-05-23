from math import exp

import pytest
import torch

from xai_dronet.training import (
    collision_loss_weight,
    compute_dronet_loss,
    hard_mining_bce,
    hard_mining_mse,
)


def test_collision_loss_weight_starts_after_epoch0() -> None:
    assert collision_loss_weight(epoch=0) == 0.0
    assert collision_loss_weight(epoch=10) == 0.0
    assert collision_loss_weight(epoch=20) == pytest.approx(1.0 - exp(-1.0))


def test_hard_mining_mse_uses_only_steering_samples() -> None:
    prediction = torch.tensor([[0.0], [2.0], [10.0]])
    target = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )

    loss = hard_mining_mse(prediction, target, k=2)

    assert float(loss) == pytest.approx(50.0)


def test_hard_mining_bce_uses_only_collision_samples() -> None:
    prediction = torch.tensor([[0.5], [0.9], [0.1]])
    target = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )

    loss = hard_mining_bce(prediction, target, k=2)
    expected = (-torch.log(torch.tensor(0.5)) - torch.log(torch.tensor(0.9))) / 2.0

    assert float(loss) == pytest.approx(float(expected))


def test_compute_dronet_loss_combines_weighted_tasks() -> None:
    steering_prediction = torch.tensor([[1.0], [0.0]])
    collision_prediction = torch.tensor([[0.5], [0.9]])
    steering_target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    collision_target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    loss = compute_dronet_loss(
        steering_prediction,
        collision_prediction,
        steering_target,
        collision_target,
        epoch=20,
        hard_mining_k=2,
    )

    assert loss.collision_weight == pytest.approx(1.0 - exp(-1.0))
    assert loss.total == pytest.approx(loss.steering_mse + loss.collision_weight * loss.collision_bce)

