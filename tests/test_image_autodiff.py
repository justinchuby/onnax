# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Image workload products against independent PyTorch differentiation."""

import numpy as np
import onnx_ir as ir
import pytest
from test_fused_autodiff import _check, _model_for


@pytest.mark.parametrize(
    "spatial,groups,stride,dilation",
    [(1, 1, 1, 1), (1, 2, 2, 1), (2, 1, 2, 1), (2, 2, 1, 2), (3, 2, 2, 1)],
)
def test_conv_products_dynamic_spatial_shapes(spatial, groups, stride, dilation):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(51)
    feeds = dict(
        x=rng.normal(0, 0.2, (2, 4, *([7] * spatial))).astype(np.float32),
        w=rng.normal(0, 0.2, (6, 4 // groups, *([3] * spatial))).astype(np.float32),
        b=rng.normal(0, 0.2, 6).astype(np.float32),
    )
    pads = [1] * spatial + [2] * spatial
    model = _model_for(
        "Conv",
        feeds,
        dict(
            group=groups, strides=[stride] * spatial, dilations=[dilation] * spatial, pads=pads
        ),
    )
    model.graph.inputs[0].shape = ir.Shape(
        ["batch", 4, *[f"spatial_{i}" for i in range(spatial)]]
    )
    conv = getattr(torch.nn.functional, f"conv{spatial}d")

    def reference(values):
        padded = torch.nn.functional.pad(values["x"], [1, 2] * spatial)
        return conv(
            padded, values["w"], values["b"], stride=stride, dilation=dilation, groups=groups
        )

    _check(model, feeds, reference)


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_groupnorm_products(groups):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(52)
    feeds = dict(
        x=rng.normal(size=(2, 4, 3, 5)).astype(np.float32),
        scale=rng.normal(size=4).astype(np.float32),
        bias=rng.normal(size=4).astype(np.float32),
    )
    model = _model_for(
        "GroupNormalization", feeds, dict(num_groups=groups, epsilon=1e-5), symbolic=True
    )
    _check(
        model,
        feeds,
        lambda v: torch.nn.functional.group_norm(
            v["x"], groups, v["scale"], v["bias"], eps=1e-5
        ),
    )


@pytest.mark.parametrize("factors", [(1, 1, 2, 2), (1, 1, 3, 2)])
def test_nearest_resize_products(factors):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(53)
    array = rng.normal(size=(2, 3, 3, 5)).astype(np.float32)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 3, "height", "width"])
    tape = ir.tape.Tape()
    scales = tape.op("Constant", [], {"value_floats": list(map(float, factors))})
    y = tape.op("Resize", [x, None, scales], {"mode": "nearest"})
    model = ir.Model(
        ir.Graph([x], [y], nodes=tape.nodes, opset_imports={"": 24}, name="resize"),
        ir_version=10,
    )
    _check(
        model,
        {"x": array},
        lambda v: torch.nn.functional.interpolate(
            v["x"], scale_factor=factors[2:], mode="nearest"
        ),
    )


def test_conv_hvp_against_pytorch():
    from _testing import runtime_session

    import onnax

    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(54)
    x = rng.normal(0, 0.2, (1, 2, 4, 5)).astype(np.float32)
    w = rng.normal(0, 0.2, (3, 2, 3, 3)).astype(np.float32)
    arrays = {"x": x, "w": w}
    inputs = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=value.shape)
        for name, value in arrays.items()
    ]
    tape = ir.tape.Tape()
    conv = tape.op("Conv", inputs, {"pads": [1, 1, 1, 1], "strides": [2, 2]})
    loss = tape.op("ReduceSum", [tape.op("Mul", [conv, conv])], {"keepdims": 0})
    model = ir.Model(
        ir.Graph(inputs, [loss], nodes=tape.nodes, name="conv_loss", opset_imports={"": 24}),
        ir_version=10,
    )
    gradient = onnax.grad(model)
    product = onnax.jvp(gradient)
    directions = [rng.normal(size=value.shape).astype(np.float32) for value in arrays.values()]
    feeds = {
        **arrays,
        **{
            value.name: direction
            for value, direction in zip(
                list(product.graph.inputs)[2:], directions, strict=True
            )
        },
    }
    values = runtime_session(product).run(None, feeds)[-2:]
    _, expected = torch.autograd.functional.hvp(
        lambda a, b: torch.nn.functional.conv2d(a, b, stride=2, padding=1).square().sum(),
        (torch.tensor(x), torch.tensor(w)),
        tuple(torch.tensor(d) for d in directions),
    )
    for actual, reference in zip(values, expected, strict=True):
        np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-4, atol=1e-6)
