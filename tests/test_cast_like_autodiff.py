# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""CastLike depends on its data input; its dtype template carries no gradient."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from onnx_ir.passes.common import CheckerPass

import onnax


@pytest.mark.parametrize("names", [["data"], ["template"], ["data", "template"]])
@pytest.mark.parametrize("mode", [onnax.jvp, onnax.vjp])
def test_cast_like_data_and_dtype_template(mode, names):
    x = ir.val("data", dtype=ir.DataType.FLOAT, shape=[2, 3])
    template = ir.val("template", dtype=ir.DataType.DOUBLE, shape=[5])
    tape = ir.tape.Tape()
    y = tape.op("CastLike", [x, template])
    y.name, y.dtype, y.shape = "result", ir.DataType.DOUBLE, ir.Shape([2, 3])
    source = ir.Model(
        ir.Graph([x, template], [y], nodes=tape.nodes, opset_imports={"": 24}, name="cast"),
        ir_version=10,
    )
    product = mode(source, inputs=names)
    CheckerPass(full_check=True)(product)
    values = {
        "data": np.arange(6, dtype=np.float32).reshape(2, 3),
        "template": np.ones(5, np.float64),
    }
    if mode is onnax.jvp:
        extras = [np.full_like(values[name], 0.5) for name in names]
        expected = [np.full((2, 3), 0.5 if "data" in names else 0, np.float64)]
    else:
        extras = [np.full((2, 3), 0.5, np.float64)]
        expected = [np.full_like(values[name], 0.5 if name == "data" else 0) for name in names]
    feeds = {
        **values,
        **{
            value.name: array
            for value, array in zip(list(product.graph.inputs)[2:], extras, strict=True)
        },
    }
    actual = runtime_session(product).run(None, feeds)
    np.testing.assert_array_equal(actual[0], values["data"].astype(np.float64))
    for result, target in zip(actual[1:], expected, strict=True):
        assert result.dtype == target.dtype
        np.testing.assert_array_equal(result, target)
