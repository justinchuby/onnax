# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent Diffusers forward/autograd with source-named adapter hooks."""

import torch
from diffusers import UNet2DConditionModel


def make_reference(config, weights, params, mapping, *, device="cpu"):
    """Attach low-rank branches to corresponding Diffusers Linear modules."""
    model = UNet2DConditionModel.from_config(config).eval().requires_grad_(False)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    parameters = {
        name: torch.nn.Parameter(torch.from_numpy(array.copy()).to(device))
        for name, array in params.items()
    }
    for row in mapping:
        path = row["module"]
        for marker in (".attn1.", ".attn2."):
            path = path.replace(marker, ".transformer_blocks.0" + marker)
        linear = model.get_submodule(path)
        a, b = parameters[row["A"]], parameters[row["B"]]
        scale = row["alpha"] / row["rank"]

        def adapter(_module, inputs, output, a=a, b=b, scale=scale):
            return output + ((inputs[0] @ a) @ b) * scale

        linear.register_forward_hook(adapter)
    return model, parameters
