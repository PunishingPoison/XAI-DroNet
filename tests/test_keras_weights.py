from pathlib import Path

import h5py
import numpy as np
import torch

from xai_dronet.models import DroNet
from xai_dronet.models.keras_weights import (
    BN_LAYER_MAP,
    CONV_LAYER_MAP,
    DENSE_LAYER_MAP,
    convert_keras_h5_to_state_dict,
)


def test_convert_keras_h5_to_state_dict_round_trip(tmp_path: Path) -> None:
    model = DroNet()
    source_state = model.state_dict()
    h5_path = tmp_path / "dronet_keras_weights.h5"
    _write_synthetic_keras_h5(h5_path, source_state)

    converted_state = convert_keras_h5_to_state_dict(h5_path)

    for key, expected_tensor in source_state.items():
        assert torch.allclose(converted_state[key], expected_tensor)


def _write_synthetic_keras_h5(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    layer_names = [
        *(name for name, _ in CONV_LAYER_MAP),
        *(name for name, _ in BN_LAYER_MAP),
        *(name for name, _ in DENSE_LAYER_MAP),
    ]

    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["layer_names"] = np.array(layer_names, dtype="S")
        for keras_name, torch_prefix in CONV_LAYER_MAP:
            group = h5_file.create_group(keras_name)
            nested = group.create_group(keras_name)
            group.attrs["weight_names"] = np.array(
                [f"{keras_name}/kernel:0", f"{keras_name}/bias:0"],
                dtype="S",
            )
            kernel = state_dict[f"{torch_prefix}.weight"].numpy().transpose(2, 3, 1, 0)
            bias = state_dict[f"{torch_prefix}.bias"].numpy()
            nested.create_dataset("kernel:0", data=kernel)
            nested.create_dataset("bias:0", data=bias)

        for keras_name, torch_prefix in BN_LAYER_MAP:
            group = h5_file.create_group(keras_name)
            nested = group.create_group(keras_name)
            group.attrs["weight_names"] = np.array(
                [
                    f"{keras_name}/gamma:0",
                    f"{keras_name}/beta:0",
                    f"{keras_name}/moving_mean:0",
                    f"{keras_name}/moving_variance:0",
                ],
                dtype="S",
            )
            nested.create_dataset("gamma:0", data=state_dict[f"{torch_prefix}.weight"].numpy())
            nested.create_dataset("beta:0", data=state_dict[f"{torch_prefix}.bias"].numpy())
            nested.create_dataset(
                "moving_mean:0",
                data=state_dict[f"{torch_prefix}.running_mean"].numpy(),
            )
            nested.create_dataset(
                "moving_variance:0",
                data=state_dict[f"{torch_prefix}.running_var"].numpy(),
            )

        for keras_name, torch_prefix in DENSE_LAYER_MAP:
            group = h5_file.create_group(keras_name)
            nested = group.create_group(keras_name)
            group.attrs["weight_names"] = np.array(
                [f"{keras_name}/kernel:0", f"{keras_name}/bias:0"],
                dtype="S",
            )
            kernel = state_dict[f"{torch_prefix}.weight"].numpy().T
            bias = state_dict[f"{torch_prefix}.bias"].numpy()
            nested.create_dataset("kernel:0", data=kernel)
            nested.create_dataset("bias:0", data=bias)

