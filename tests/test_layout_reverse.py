# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Runtime-indexed layout pullbacks agree with independent array indexing."""

import json

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from _testing import runtime_session

import onnax


def _graph(data, tape, output, extra):
    output.name, output.dtype, output.shape = (
        "output",
        ir.DataType.FLOAT,
        ir.Shape([None, None]),
    )
    return ir.Model(
        ir.Graph(
            [data, *extra],
            [output],
            nodes=tape.nodes,
            opset_imports={"": 18},
            name="layout_reverse",
        ),
        ir_version=10,
    )


@pytest.mark.parametrize(
    "starts,ends,axes,steps",
    [
        ([1], [4], [0], [1]),
        ([4], [-6], [0], [-2]),
        ([3, 1], [-5, 5], [-1, 0], [-2, 2]),
        ([3], [1], [0], [1]),
        ([1], [4], None, None),
    ],
)
def test_slice_runtime_controls_and_empty_reversed_slices(starts, ends, axes, steps):
    data = ir.val("data", dtype=ir.DataType.FLOAT, shape=["rows", 4])
    controls = [
        ir.val(name, dtype=ir.DataType.INT64, shape=[len(starts)])
        for name in ["starts", "ends"]
    ]
    feeds = {"starts": np.array(starts, np.int64), "ends": np.array(ends, np.int64)}
    if axes is not None:
        controls.append(ir.val("axes", dtype=ir.DataType.INT64, shape=[len(starts)]))
        feeds["axes"] = np.array(axes, np.int64)
    if steps is not None:
        controls.append(ir.val("steps", dtype=ir.DataType.INT64, shape=[len(starts)]))
        feeds["steps"] = np.array(steps, np.int64)
    tape = ir.tape.Tape()
    source = _graph(data, tape, tape.op("Slice", [data, *controls]), controls)
    reverse = onnax.vjp(source, inputs=["data"])
    onnx.checker.check_model(ir.to_proto(reverse), full_check=True)
    rng = np.random.default_rng(7)
    values = rng.normal(size=(5, 4)).astype(np.float32)
    slices = [slice(None)] * 2
    for start, end, axis, step in zip(
        starts,
        ends,
        axes if axes is not None else range(len(starts)),
        steps if steps is not None else [1] * len(starts),
        strict=True,
    ):
        slices[axis] = slice(start, end, step)
    slices = tuple(slices)
    expected_output = values[slices]
    seed = rng.normal(size=expected_output.shape).astype(np.float32)
    info = json.loads(reverse.graph.metadata_props["onnax.autodiff.interface"])
    actual = runtime_session(reverse).run(
        None, {**feeds, "data": values, info["seeds"][0]: seed}
    )
    expected = np.zeros_like(values)
    expected[slices] = seed
    np.testing.assert_array_equal(actual[0], expected_output)
    np.testing.assert_array_equal(actual[1], expected)


@pytest.mark.parametrize(
    "pads,axes",
    [([1, 2, 2, 1], None), ([-1, 2, -2, 0], None), ([1, -1], [1]), ([-2, -1], [0])],
)
@pytest.mark.parametrize("selected", [["data"], ["value"], ["data", "value"]])
def test_constant_pad_cropping_and_value_gradient(pads, axes, selected):
    data = ir.val("data", dtype=ir.DataType.FLOAT, shape=["rows", 4])
    pads_input = ir.val("pads", dtype=ir.DataType.INT64, shape=[len(pads)])
    value = ir.val("value", dtype=ir.DataType.FLOAT, shape=[])
    extra = [pads_input, value]
    if axes is not None:
        extra.append(ir.val("axes", dtype=ir.DataType.INT64, shape=[len(axes)]))
    tape = ir.tape.Tape()
    source = _graph(data, tape, tape.op("Pad", [data, *extra]), extra)
    reverse = onnax.vjp(source, inputs=selected)
    onnx.checker.check_model(ir.to_proto(reverse), full_check=True)
    rng = np.random.default_rng(11)
    values = rng.normal(size=(5, 4)).astype(np.float32)
    widths = [(0, 0)] * 2
    for i, axis in enumerate(axes if axes is not None else range(2)):
        widths[axis] = (pads[i], pads[i + len(pads) // 2])
    crop = tuple(
        slice(max(-left, 0), extent - max(-right, 0))
        for extent, (left, right) in zip(values.shape, widths, strict=True)
    )
    positive = [(max(left, 0), max(right, 0)) for left, right in widths]
    expected_output = np.pad(values[crop], positive, constant_values=0.3)
    seed = rng.normal(size=expected_output.shape).astype(np.float32)
    expected_data = np.zeros_like(values)
    interior = tuple(
        slice(left, left + extent)
        for (left, _), extent in zip(positive, values[crop].shape, strict=True)
    )
    expected_data[crop] = seed[interior]
    expected_value = seed.sum() - seed[interior].sum()
    feeds = {
        "data": values,
        "pads": np.array(pads, np.int64),
        "value": np.array(0.3, np.float32),
    }
    if axes is not None:
        feeds["axes"] = np.array(axes, np.int64)
    info = json.loads(reverse.graph.metadata_props["onnax.autodiff.interface"])
    actual = runtime_session(reverse).run(None, {**feeds, info["seeds"][0]: seed})
    np.testing.assert_allclose(actual[0], expected_output, rtol=1e-6, atol=1e-6)
    expected = {"data": expected_data, "value": expected_value}
    for gradient, name in zip(actual[1:], selected, strict=True):
        np.testing.assert_allclose(gradient, expected[name], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("coordinates", [[[-1]], [[1, 2]]])
def test_replacement_scatter_unique_coordinates(coordinates):
    indices = np.array(coordinates, np.int64)
    shape = (3, 4)
    update_shape = (*indices.shape[:-1], *shape[indices.shape[-1] :])
    data = ir.val("data", dtype=ir.DataType.FLOAT, shape=["rows", 4])
    index = ir.val("indices", dtype=ir.DataType.INT64, shape=indices.shape)
    updates = ir.val("updates", dtype=ir.DataType.FLOAT, shape=update_shape)
    tape = ir.tape.Tape()
    source = _graph(data, tape, tape.op("ScatterND", [data, index, updates]), [index, updates])
    reverse = onnax.vjp(source, inputs=["data", "updates"])
    onnx.checker.check_model(ir.to_proto(reverse), full_check=True)
    rng = np.random.default_rng(13)
    values = rng.normal(size=shape).astype(np.float32)
    changes = rng.normal(size=update_shape).astype(np.float32)
    seed = rng.normal(size=shape).astype(np.float32)
    keys = tuple(np.moveaxis(indices, -1, 0))
    expected = values.copy()
    expected[keys] = changes
    expected_data = seed.copy()
    expected_data[keys] = 0
    info = json.loads(reverse.graph.metadata_props["onnax.autodiff.interface"])
    actual = runtime_session(reverse).run(
        None, {"data": values, "indices": indices, "updates": changes, info["seeds"][0]: seed}
    )
    for gradient, target in zip(actual, [expected, expected_data, seed[keys]], strict=True):
        np.testing.assert_array_equal(gradient, target)
