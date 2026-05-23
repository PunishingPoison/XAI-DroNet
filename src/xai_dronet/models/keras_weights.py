"""Convert official DroNet Keras HDF5 weights to PyTorch.

The original DroNet release is based on Keras 2.x. This module avoids adding a
TensorFlow runtime dependency by reading `.h5` weights directly with `h5py` and
mapping them into the PyTorch model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor

from xai_dronet.models.dronet import DroNet

CONV_LAYER_MAP: tuple[tuple[str, str], ...] = (
    ("conv2d_1", "stem.0"),
    ("conv2d_2", "residual_blocks.0.main.2"),
    ("conv2d_3", "residual_blocks.0.main.5"),
    ("conv2d_4", "residual_blocks.0.shortcut"),
    ("conv2d_5", "residual_blocks.1.main.2"),
    ("conv2d_6", "residual_blocks.1.main.5"),
    ("conv2d_7", "residual_blocks.1.shortcut"),
    ("conv2d_8", "residual_blocks.2.main.2"),
    ("conv2d_9", "residual_blocks.2.main.5"),
    ("conv2d_10", "residual_blocks.2.shortcut"),
)

BN_LAYER_MAP: tuple[tuple[str, str], ...] = (
    ("batch_normalization_1", "residual_blocks.0.main.0"),
    ("batch_normalization_2", "residual_blocks.0.main.3"),
    ("batch_normalization_3", "residual_blocks.1.main.0"),
    ("batch_normalization_4", "residual_blocks.1.main.3"),
    ("batch_normalization_5", "residual_blocks.2.main.0"),
    ("batch_normalization_6", "residual_blocks.2.main.3"),
)

DENSE_LAYER_MAP: tuple[tuple[str, str], ...] = (
    ("dense_1", "steering_head"),
    ("dense_2", "collision_head"),
)


def convert_keras_h5_to_state_dict(weights_path: str | Path) -> dict[str, Tensor]:
    """Return a PyTorch state dict converted from Keras DroNet weights."""

    model = DroNet()
    state_dict = model.state_dict()

    with h5py.File(weights_path, "r") as h5_file:
        root = _get_weights_root(h5_file)

        for keras_name, torch_prefix in CONV_LAYER_MAP:
            weights = _read_layer_weights(root, keras_name)
            _assign_conv(state_dict, torch_prefix, weights)

        for keras_name, torch_prefix in BN_LAYER_MAP:
            weights = _read_layer_weights(root, keras_name)
            _assign_batch_norm(state_dict, torch_prefix, weights)

        for keras_name, torch_prefix in DENSE_LAYER_MAP:
            weights = _read_layer_weights(root, keras_name)
            _assign_dense(state_dict, torch_prefix, weights)

    model.load_state_dict(state_dict, strict=True)
    return state_dict


def save_converted_checkpoint(weights_path: str | Path, output_path: str | Path) -> None:
    """Convert Keras weights and save a PyTorch state dict checkpoint."""

    state_dict = convert_keras_h5_to_state_dict(weights_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output)


def _get_weights_root(h5_file: h5py.File) -> h5py.Group:
    if "model_weights" in h5_file:
        return h5_file["model_weights"]
    return h5_file


def _read_layer_weights(root: h5py.Group, layer_name: str) -> dict[str, np.ndarray]:
    if layer_name not in root:
        available = ", ".join(root.keys())
        raise KeyError(f"Missing Keras layer '{layer_name}'. Available layers: {available}")

    layer_group = root[layer_name]
    datasets = _collect_datasets(layer_group)
    weight_names = _decode_names(layer_group.attrs.get("weight_names", [])) or list(datasets)
    weights: dict[str, np.ndarray] = {}

    for weight_name in weight_names:
        dataset = _find_dataset(datasets, weight_name)
        short_name = Path(weight_name).name.split(":")[0]
        weights[short_name] = np.asarray(dataset)

    return weights


def _collect_datasets(group: h5py.Group) -> dict[str, h5py.Dataset]:
    datasets: dict[str, h5py.Dataset] = {}

    def collect(name: str, item: Any) -> None:
        if isinstance(item, h5py.Dataset):
            datasets[name] = item

    group.visititems(collect)
    return datasets


def _find_dataset(datasets: dict[str, h5py.Dataset], weight_name: str) -> h5py.Dataset:
    short_name = Path(weight_name).name
    candidates = (
        weight_name,
        short_name,
        weight_name.replace(":0", ""),
        short_name.replace(":0", ""),
    )
    for candidate in candidates:
        if candidate in datasets:
            return datasets[candidate]

    for dataset_name, dataset in datasets.items():
        if dataset_name.endswith(weight_name) or dataset_name.endswith(short_name):
            return dataset

    available = ", ".join(datasets)
    raise KeyError(f"Missing Keras weight '{weight_name}'. Available weights: {available}")


def _decode_names(values: Any) -> list[str]:
    names: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def _assign_conv(
    state_dict: dict[str, Tensor],
    torch_prefix: str,
    weights: dict[str, np.ndarray],
) -> None:
    kernel = torch.from_numpy(weights["kernel"].transpose(3, 2, 0, 1)).float()
    bias = torch.from_numpy(weights["bias"]).float()
    _assign_tensor(state_dict, f"{torch_prefix}.weight", kernel)
    _assign_tensor(state_dict, f"{torch_prefix}.bias", bias)


def _assign_dense(
    state_dict: dict[str, Tensor],
    torch_prefix: str,
    weights: dict[str, np.ndarray],
) -> None:
    kernel = torch.from_numpy(weights["kernel"].T).float()
    bias = torch.from_numpy(weights["bias"]).float()
    _assign_tensor(state_dict, f"{torch_prefix}.weight", kernel)
    _assign_tensor(state_dict, f"{torch_prefix}.bias", bias)


def _assign_batch_norm(
    state_dict: dict[str, Tensor],
    torch_prefix: str,
    weights: dict[str, np.ndarray],
) -> None:
    _assign_tensor(state_dict, f"{torch_prefix}.weight", torch.from_numpy(weights["gamma"]).float())
    _assign_tensor(state_dict, f"{torch_prefix}.bias", torch.from_numpy(weights["beta"]).float())
    _assign_tensor(
        state_dict,
        f"{torch_prefix}.running_mean",
        torch.from_numpy(weights["moving_mean"]).float(),
    )
    _assign_tensor(
        state_dict,
        f"{torch_prefix}.running_var",
        torch.from_numpy(weights["moving_variance"]).float(),
    )


def _assign_tensor(state_dict: dict[str, Tensor], key: str, tensor: Tensor) -> None:
    expected_shape = tuple(state_dict[key].shape)
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise ValueError(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
    state_dict[key] = tensor

