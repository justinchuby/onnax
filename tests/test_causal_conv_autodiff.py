# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Stateful temporal products through preserved ONNX function boundaries."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from test_fused_autodiff import _check

from onnax import jvp, vjp


def model_and_arrays(length=3, kernel=4, activation="silu", *, explicit_activation=True):
    rng = np.random.default_rng(71)
    shapes = {
        "input": (2, 3, length),
        "weight": (3, 1, kernel),
        "bias": (3,),
        "state": (2, 3, kernel - 1),
    }
    arrays = {
        name: rng.normal(0, 0.2, shape).astype(np.float32) for name, shape in shapes.items()
    }
    formals = [ir.val(name) for name in shapes]
    x, w, b, state = formals
    body = ir.tape.Tape()

    def constant(values):
        return body.op("Constant", [], {"value": ir.tensor(np.asarray(values, np.int64))})

    combined = body.op("Concat", [state, x], {"axis": 2})
    pre = body.op("Conv", [combined, w], {"group": 3, "kernel_shape": [kernel]})
    pre = body.op("Add", [pre, body.op("Reshape", [b, constant([1, -1, 1])])])
    output = pre if activation == "none" else body.op("Mul", [pre, body.op("Sigmoid", [pre])])
    time = body.op("Shape", [x], {"start": 2, "end": 3})
    end = body.op("Add", [time, constant([kernel - 1])])
    present = body.op("Slice", [combined, time, end, constant([2])])
    function = ir.Function(
        "com.microsoft",
        "CausalConvWithState",
        attributes=[ir.Attr("activation", ir.AttributeType.STRING, activation)],
        graph=ir.Graph(
            formals,
            [output, present],
            nodes=body.nodes,
            opset_imports={"": 24},
            name="causal_body",
        ),
    )
    inputs = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=["batch", *shape[1:]])
        for name, shape in shapes.items()
    ]
    # Bias and weight use concrete ranks; only input/state carry a batch axis.
    for value in inputs[1:3]:
        value.shape = ir.Shape(shapes[value.name])
    inputs[0].shape = ir.Shape(["batch", 3, "time"])
    tape = ir.tape.Tape()
    outputs = tape.op_multi_out(
        "CausalConvWithState",
        inputs,
        {"activation": activation} if explicit_activation else {},
        domain="com.microsoft",
        num_outputs=2,
    )
    for index, value in enumerate(outputs):
        value.name = f"output{index}"
        value.dtype = ir.DataType.FLOAT
        value.shape = ir.Shape(
            ["batch", 3, "time"] if index == 0 else ["batch", 3, kernel - 1]
        )
    model = ir.Model(
        ir.Graph(
            inputs,
            outputs,
            nodes=tape.nodes,
            opset_imports={"": 24, "com.microsoft": 1},
            name="causal",
        ),
        ir_version=10,
        functions=[function],
    )
    return model, arrays


def reference(tensors, activation):
    import torch

    joined = torch.cat((tensors["state"], tensors["input"]), dim=2)
    pre = torch.nn.functional.conv1d(joined, tensors["weight"], groups=3)
    pre = pre + tensors["bias"][None, :, None]
    output = pre if activation == "none" else torch.nn.functional.silu(pre)
    return output, joined[:, :, tensors["input"].shape[2] :]


@pytest.mark.parametrize("length,kernel", [(3, 1), (1, 4), (7, 4)])
@pytest.mark.parametrize("activation", ["none", "silu"])
@pytest.mark.parametrize("names", [None, ["input"], ["state"], ["weight", "bias"], ["bias"]])
def test_causal_convolution_products(length, kernel, activation, names):
    model, arrays = model_and_arrays(length, kernel, activation)
    _check(model, arrays, lambda tensors: reference(tensors, activation), names=names)


def test_state_and_output_products_are_dual_without_torch():
    model, arrays = model_and_arrays()
    rng = np.random.default_rng(19)
    directions = [rng.normal(size=x.shape).astype(np.float32) for x in arrays.values()]
    seeds = [rng.normal(size=(2, 3, 3)).astype(np.float32) for _ in range(2)]
    forward, reverse = jvp(model), vjp(model)
    forward_feeds, reverse_feeds = dict(arrays), dict(arrays)
    forward_feeds.update(
        (x.name, d) for x, d in zip(list(forward.graph.inputs)[4:], directions, strict=True)
    )
    reverse_feeds.update(
        (x.name, g) for x, g in zip(list(reverse.graph.inputs)[4:], seeds, strict=True)
    )
    tangents = runtime_session(forward).run(None, forward_feeds)[2:]
    cotangents = runtime_session(reverse).run(None, reverse_feeds)[2:]
    left = sum(np.sum(a * b) for a, b in zip(tangents, seeds, strict=True))
    right = sum(np.sum(a * b) for a, b in zip(cotangents, directions, strict=True))
    np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-6)


def test_omitted_call_attribute_uses_the_function_default():
    source, arrays = model_and_arrays(activation="none", explicit_activation=False)
    before = ir.to_proto(source).SerializeToString()
    product = jvp(source, inputs=["bias"])
    direction = np.array([0.2, -0.3, 0.5], np.float32)
    actual = runtime_session(product).run(
        None, {**arrays, product.graph.inputs[-1].name: direction}
    )
    np.testing.assert_allclose(
        actual[2], np.broadcast_to(direction[None, :, None], (2, 3, 3)), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_array_equal(actual[3], np.zeros((2, 3, 3), np.float32))
    assert ir.to_proto(source).SerializeToString() == before
