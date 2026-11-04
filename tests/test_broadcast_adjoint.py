# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Portable shape-aware reductions retain broadcast and derivative semantics."""

import json

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir.passes.common import InlinePass
from onnxscript import optimizer

import onnax


def _model(dtype=ir.DataType.FLOAT):
    gradient = ir.val("gradient", dtype=dtype)
    reference = ir.val("reference", dtype=dtype)
    graph = ir.Graph(
        [gradient, reference], [], nodes=[], opset_imports={"": 18}, name="sum_to_shape"
    )
    ctx = onnax.AutodiffContext(graph)
    output = ctx.unbroadcast(gradient, reference)
    output.name = "result"
    output.type = ir.TensorType(dtype)
    # Renaming requires the corresponding interface to name the final boundary.
    first = next(iter(graph))
    descriptor = json.loads(first.metadata_props["onnax.unbroadcast.interface"])
    descriptor["output"] = "result"
    first.metadata_props["onnax.unbroadcast.interface"] = json.dumps(descriptor)
    graph.outputs.append(output)
    return ir.Model(graph, ir_version=10)


def _run(model, feeds):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        ir.to_proto(model).SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)


def _expected(gradient, shape):
    padded = (1,) * (gradient.ndim - len(shape)) + shape
    axes = tuple(i for i, extent in enumerate(padded) if extent == 1)
    return gradient.sum(axis=axes, keepdims=True).reshape(shape)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 3), (3,)),
        ((2, 3), (2, 1)),
        ((2, 3), ()),
        ((), ()),
        ((0, 3), (1, 3)),
        ((2, 0), ()),
        ((2, 0), (2, 0)),
        ((2, 3), (2, 3)),
    ],
)
def test_portable_runtime_shapes_and_products(dtype, shapes):
    model = _model(dtype)
    before = ir.to_proto(model).SerializeToString()
    result = onnax.FuseBroadcastAdjointsPass()(model)
    assert result.modified
    fused = result.model
    assert not onnax.FuseBroadcastAdjointsPass()(fused).modified
    assert len(fused.graph) == 2
    assert len(fused.functions) == 1
    assert ir.to_proto(model).SerializeToString() == before
    assert (
        ir.to_proto(onnax.FuseBroadcastAdjointsPass()(fused).model).SerializeToString()
        == ir.to_proto(fused).SerializeToString()
    )
    # Give checker known ranks while preserving runtime extents.
    for value, shape in zip(fused.graph.inputs, shapes, strict=True):
        value.shape = ir.Shape([None] * len(shape))
    fused.graph.outputs[0].shape = ir.Shape([None] * len(shapes[1]))
    onnx.checker.check_model(ir.to_proto(fused), full_check=True)
    gradient = np.arange(np.prod(shapes[0]), dtype=dtype.numpy()).reshape(shapes[0]) / 7
    gradient = np.asarray(gradient, dtype=dtype.numpy())
    reference = np.full(shapes[1], np.nan, dtype=dtype.numpy())
    feeds = {"gradient": gradient, "reference": reference}
    expected = _expected(gradient, shapes[1])
    for candidate in (fused, optimizer.optimize(fused, inline=False)):
        actual = _run(candidate, feeds)[0]
        assert actual.shape == shapes[1]
        assert actual.dtype == gradient.dtype
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    portable = fused.clone()
    InlinePass()(portable)
    product = onnax.jvp(portable, inputs=["gradient"])
    tangent = np.ones_like(gradient)
    feeds[product.graph.inputs[-1].name] = tangent
    np.testing.assert_allclose(_run(product, feeds)[-1], _expected(tangent, shapes[1]))


def test_modified_region_is_not_native_contract():
    model = _model()
    reduction = next(node for node in model.graph if node.op_type == "ReduceSum")
    reduction.attributes["keepdims"] = ir.AttrInt64("keepdims", 0)
    before = ir.to_proto(model).SerializeToString()
    fused = onnax.FuseBroadcastAdjointsPass()(model).model
    assert ir.to_proto(fused).SerializeToString() == before


def test_shared_intermediate_keeps_portable_region():
    model = _model()
    model.graph.outputs.append(next(iter(model.graph)).outputs[0])
    before = ir.to_proto(model).SerializeToString()
    assert (
        ir.to_proto(onnax.FuseBroadcastAdjointsPass()(model).model).SerializeToString()
        == before
    )


def test_fused_nonlinear_vjp_and_second_directional_product():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", "width"])
    z = ir.val("z", dtype=ir.DataType.FLOAT, shape=["batch", "width"])
    tape = ir.tape.Tape()
    y = tape.op("Tanh", [tape.op("Mul", [x, z])])
    y.name = "y"
    model = ir.Model(
        ir.Graph([x, z], [y], nodes=tape.nodes, opset_imports={"": 18}, name="nonlinear"),
        ir_version=10,
    )
    baseline = onnax.vjp(model)
    fused = onnax.FuseBroadcastAdjointsPass()(baseline).model
    name = "SumToShape"
    assert sum(n.op_type == name for n in fused.graph) == 2
    # Equal symbolic names still allow independent singleton broadcast extents.
    feeds = {
        "x": np.array([[0.2], [0.5]], np.float32),
        "z": np.array([[0.1, 0.3, 0.7]], np.float32),
    }
    feeds[baseline.graph.inputs[-1].name] = np.ones((2, 3), np.float32)
    for actual, expected in zip(_run(fused, feeds), _run(baseline, feeds), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    InlinePass()(fused)
    for candidate in (baseline, fused):
        product = onnax.jvp(candidate, inputs=["x", "z"])
        data = dict(feeds)
        data.update(
            {
                v.name: np.ones_like(feeds[name])
                for v, name in zip(product.graph.inputs[-2:], ("x", "z"), strict=True)
            }
        )
        values = _run(product, data)
        if candidate is baseline:
            control = values
        else:
            for actual, expected in zip(values, control, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_connected_regions_share_one_function():
    inputs = [ir.val(name, dtype=ir.DataType.FLOAT) for name in ("gradient", "row", "scalar")]
    graph = ir.Graph(inputs, [], nodes=[], opset_imports={"": 18}, name="connected")
    ctx = onnax.AutodiffContext(graph)
    first = ctx.unbroadcast(inputs[0], inputs[1])
    output = ctx.unbroadcast(first, inputs[2])
    output.type = ir.TensorType(ir.DataType.FLOAT)
    graph.outputs.append(output)
    model = ir.Model(graph, ir_version=10)
    fused = onnax.FuseBroadcastAdjointsPass()(model).model
    assert len(fused.graph) == 2 * (2)
    assert len(fused.functions) == 1
    feeds = {
        "gradient": np.arange(6, dtype=np.float32).reshape(2, 3),
        "row": np.ones((1, 3), np.float32),
        "scalar": np.array(0, np.float32),
    }
    np.testing.assert_array_equal(_run(fused, feeds)[0], np.array(15, np.float32))


def test_public_pass_manager_and_example_compatibility():
    from examples.broadcast_adjoint import fuse_broadcast_adjoints

    model = _model()
    before = ir.to_proto(model).SerializeToString()
    result = ir.passes.PassManager([onnax.FuseBroadcastAdjointsPass()])(model)
    assert result.modified
    assert (
        ir.to_proto(result.model).SerializeToString()
        == ir.to_proto(fuse_broadcast_adjoints(model)).SerializeToString()
    )
    assert ir.to_proto(model).SerializeToString() == before


def test_incompatible_existing_function_preserves_source():
    model = _model()
    packaged = onnax.FuseBroadcastAdjointsPass()(model).model
    function = next(iter(packaged.functions.values()))
    reduction = next(node for node in function.graph if node.op_type == "ReduceSum")
    reduction.attributes["keepdims"] = ir.AttrInt64("keepdims", 0)
    model.functions[function.identifier()] = function
    before = ir.to_proto(model).SerializeToString()
    with pytest.raises(ValueError, match=r"incompatible.*SumToShape"):
        onnax.FuseBroadcastAdjointsPass()(model)
    assert ir.to_proto(model).SerializeToString() == before


def test_unannotated_model_returns_independent_unchanged_result():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2])
    model = ir.Model(ir.Graph([x], [x], nodes=[], opset_imports={"": 18}), ir_version=10)
    before = ir.to_proto(model).SerializeToString()
    result = onnax.FuseBroadcastAdjointsPass()(model)
    assert not result.modified
    assert result.model is not model
    assert result.model.graph.inputs[0] is not x
    assert ir.to_proto(result.model).SerializeToString() == before
