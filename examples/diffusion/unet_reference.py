# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent Diffusers forward/autograd with source-named adapter hooks."""

from types import MethodType

import torch
from diffusers import UNet2DConditionModel

from examples.torch_precision import projection_product


def make_reference(config, weights, params, mapping, *, device="cpu", precision="fp32"):
    """Attach low-rank branches to corresponding Diffusers Linear modules."""
    product = None if precision == "fp32" else projection_product(precision)
    model = UNet2DConditionModel.from_config(config).eval().requires_grad_(False)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    if product is not None:

        def linear(module, x):
            output = product.apply(x, module.weight.T)
            return output if module.bias is None else output + module.bias

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.forward = MethodType(linear, module)
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
            delta = (
                ((inputs[0] @ a) @ b)
                if product is None
                else product.apply(product.apply(inputs[0], a), b)
            )
            return output + delta * scale

        linear.register_forward_hook(adapter)
    return model, parameters
