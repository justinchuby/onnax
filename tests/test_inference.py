# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""The configured engine refines clones, never the source or its tensor names."""

import numpy as np
import onnx_ir as ir
import pytest
from onnx_shape_inference import ShapeInferenceError

from onnax import GradPass, VmapPass, _inference


def test_transforms_use_symbolic_engine_and_preserve_tensor_metadata(monkeypatch):
    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=[None, 3])
    tape = ir.tape.Tape()
    p = tape.op("Constant", [], {"value": ir.tensor(np.ones((1, 3)))})
    p.producer().attributes["value"].doc_string = "parameter constant"
    loss = tape.op("ReduceSum", [tape.op("Mul", [x, p])], {"keepdims": 0})
    loss.name = "loss"
    source = ir.Model(
        ir.Graph([x], [loss], nodes=tape.nodes, opset_imports={"": 18}, name="inference"),
        ir_version=10,
    )
    source.graph.meta["application"] = {"nested": [1, 2]}
    before = ir.to_proto(source).SerializeToString()
    engine = _inference.infer_symbolic_shapes
    calls = []

    def infer(model, **kwargs):
        assert model is not source
        calls.append(kwargs)
        return engine(model, **kwargs)

    monkeypatch.setattr(_inference, "infer_symbolic_shapes", infer)
    derivative = GradPass(inputs=["x"])(source).model
    batched = VmapPass([0])(derivative).model
    ir.to_proto(batched)
    assert calls == [{"policy": "refine"}, {"policy": "refine"}]
    assert ir.to_proto(source).SerializeToString() == before
    assert source.graph.meta["application"] == {"nested": [1, 2]}
    batched.graph.meta["application"]["nested"].append(3)
    assert source.graph.meta["application"] == {"nested": [1, 2]}
    assert p.producer().attributes["value"].doc_string == "parameter constant"
    assert p.producer().attributes["value"].as_tensor().name is None


def test_engine_errors_are_not_replaced_by_another_inference_engine():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3])
    p = ir.val("p", dtype=ir.DataType.FLOAT, shape=[4, 2])
    node = ir.node("MatMul", [x, p])
    node.outputs[0].name = "y"
    model = ir.Model(
        ir.Graph([x, p], node.outputs, nodes=[node], opset_imports={"": 18}, name="invalid"),
        ir_version=10,
    )
    before = ir.to_proto(model).SerializeToString()
    with pytest.raises(ShapeInferenceError):
        GradPass(inputs=["x"])(model)
    with pytest.raises(ir.passes.PreconditionError, match="Shape inference failed"):
        VmapPass([0, None])(model)
    assert ir.to_proto(model).SerializeToString() == before
