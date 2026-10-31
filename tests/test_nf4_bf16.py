# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Packed NF4 storage with BF16 products and declared FP32 linear derivatives."""

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import InlinePass, ShapeInferencePass

import onnax
from examples.lora.bf16 import round_bf16
from examples.lora.nf4 import dequantize, lower, register_rules


def model(shape):
    rng = np.random.default_rng(23)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=shape)
    tape = ir.tape.Tape()
    w = tape.initializer(ir.tensor(rng.normal(size=(5, 7)).astype(np.float32), name="w"))
    y = tape.op("MatMul", [x, w])
    y.name = "y"
    source = ir.Model(
        ir.Graph(
            [x],
            [y],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="nf4_bf16_product",
        ),
        ir_version=10,
    )
    lowered, manifest = lower(source, names=["w"], precision="bf16")
    row = manifest["w"]
    arrays = lowered.graph.initializers
    decoded = dequantize(
        arrays[row["packed"]].const_value.numpy(),
        arrays[row["scales"]].const_value.numpy(),
        row["shape"],
    )
    return source, lowered, round_bf16(decoded)


@pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 3, 5)])
def test_nf4_bf16_primal_vjp_and_jvp(shape):
    source, lowered, weight = model(shape)
    assert "w" in source.graph.initializers and "w" not in lowered.graph.initializers
    rng = np.random.default_rng(24)
    x = rng.normal(size=shape).astype(np.float32)
    direction = rng.normal(size=shape).astype(np.float32)
    seed = rng.normal(size=(*shape[:-1], 7)).astype(np.float32)
    primal = round_bf16(round_bf16(x) @ weight)
    vjp = onnax.vjp(lowered, inputs=["x"])
    jvp = onnax.jvp(lowered, inputs=["x"])
    for graph in (lowered, vjp, jvp):
        onnx.checker.check_model(ir.to_proto(graph), full_check=True)
        assert all(
            value.dtype == ir.DataType.FLOAT
            for value in (*graph.graph.inputs, *graph.graph.outputs)
        )
        assert not any(
            value.dtype == ir.DataType.FLOAT and value.shape == ir.Shape([5, 7])
            for value in graph.graph.initializers.values()
        )
    actual_vjp = ReferenceEvaluator(ir.to_proto(vjp)).run(None, {"x": x, "y_cotangent": seed})
    actual_jvp = ReferenceEvaluator(ir.to_proto(jvp)).run(
        None, {"x": x, "x_tangent": direction}
    )
    np.testing.assert_array_equal(actual_vjp[0], primal)
    np.testing.assert_array_equal(actual_jvp[0], primal)
    np.testing.assert_allclose(actual_vjp[1], seed @ weight.T, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(actual_jvp[1], direction @ weight, rtol=1e-4, atol=1e-6)
    inlined = ShapeInferencePass()(InlinePass()(lowered.clone()).model).model
    products = [n for n in inlined.graph if n.op_type == "MatMul"]
    assert len(products) == 1
    assert products[0].inputs[0].dtype == products[0].inputs[1].dtype == ir.DataType.BFLOAT16
    assert products[0].outputs[0].dtype == ir.DataType.BFLOAT16


def test_nf4_bf16_loaded_rules_and_linearized_vjp(tmp_path):
    _, lowered, weight = model([2, 5])
    path = tmp_path / "nf4-bf16.onnx"
    ir.save(lowered, path)
    loaded = ir.load(path)
    register_rules(loaded)
    vjp = onnax.vjp(loaded, inputs=["x"])
    composed = onnax.jvp(vjp, inputs=["y_cotangent"], outputs=["x_cotangent"])
    seed = np.arange(14, dtype=np.float32).reshape(2, 7) / 10
    direction = np.full_like(seed, 0.1)
    actual = ReferenceEvaluator(ir.to_proto(composed)).run(
        None,
        {
            "x": np.ones((2, 5), np.float32),
            "y_cotangent": seed,
            "y_cotangent_tangent": direction,
        },
    )
    np.testing.assert_allclose(actual[-1], direction @ weight.T, rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match="frozen"):
        onnax.vjp(loaded, inputs=["w.absmax"])
