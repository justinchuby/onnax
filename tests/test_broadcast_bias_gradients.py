# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Direct axis removal preserves scalar, singleton and empty broadcast gradients."""

import numpy as np
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

import onnax


@pytest.mark.parametrize(
    "output_shape,bias_shape",
    [
        ((2, 3), (3,)),
        ((2, 4, 3), (1, 3)),
        ((0, 3), (3,)),
        ((2, 0, 3), (0, 3)),
        ((0, 4, 3), (1, 3)),
        ((2, 3), ()),
    ],
)
def test_bias_gradient_removes_leading_axes_without_reshape(output_shape, bias_shape):
    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=output_shape)
    bias = ir.val("bias", dtype=ir.DataType.DOUBLE, shape=bias_shape)
    tape = ir.tape.Tape()
    value = tape.op("Add", [x, bias])
    loss = tape.op("ReduceSum", [value], {"keepdims": 0})
    loss.name, loss.dtype, loss.shape = "loss", ir.DataType.DOUBLE, ir.Shape([])
    model = ir.Model(
        ir.Graph([x, bias], [loss], nodes=tape.nodes, opset_imports={"": 18}), ir_version=10
    )
    graph = onnax.grad(model, inputs=["bias"], output="loss")
    actual = ReferenceEvaluator(ir.to_proto(graph)).run(
        None, {"x": np.ones(output_shape), "bias": np.ones(bias_shape)}
    )[-1]
    leading = len(output_shape) - len(bias_shape)
    axes = [
        *range(leading),
        *(
            leading + i
            for i, d in enumerate(bias_shape)
            if d == 1 and output_shape[leading + i] != 1
        ),
    ]
    expected = np.ones(output_shape).sum(axis=tuple(axes), keepdims=True)
    expected = np.squeeze(expected, axis=tuple(range(leading)))
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == bias_shape
    assert "Squeeze" in {n.op_type for n in graph.graph}
    assert "Reshape" not in {n.op_type for n in graph.graph}
