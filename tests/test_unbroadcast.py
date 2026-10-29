# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Concrete broadcast derivatives should not require dynamic axis discovery."""

import numpy as np
import onnx_ir as ir
import pytest
from test_autodiff import _model, _products, _value


@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 3), (2, 3)),
        ((), ()),
        ((), (2, 3)),
        ((3,), (2, 3)),
        ((2, 1), (1, 3)),
        ((0, 3), (1,)),
        ((2, 0), ()),
    ],
)
def test_concrete_broadcast_products_without_dynamic_axes(shapes):
    x, z = _value("x", shapes[0]), _value("z", shapes[1])
    tape = ir.tape.Tape()
    model = _model([x, z], tape, [tape.op("Mul", [x, z])])
    _, reverse = _products(model, {"x": np.full(shapes[0], 1.3), "z": np.full(shapes[1], 0.7)})
    operators = {node.op_type for node in reverse.graph}
    assert not operators & {"NonZero", "ConstantOfShape", "Shape", "Size"}
    if shapes[0] == shapes[1]:
        assert "ReduceSum" not in operators
