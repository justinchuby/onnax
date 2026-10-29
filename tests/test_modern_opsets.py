# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Audited schemas and executable semantics at the actual opsets 25 through 28."""

from __future__ import annotations

import re

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from _testing import evaluator
from onnx_ir.passes.common import CheckerPass
from onnxruntime.capi.onnxruntime_pybind11_state import Fail
from test_autodiff import _model as _ad_model
from test_autodiff import _products, _value
from test_batching import _data, _equivalent, _model
from test_higher_order import _hvp_check
from test_per_example_gradients import _regressor

from onnax import grad, jvp, vjp, vmap
from onnax._batching import _elementwise, _indexing, _layout, _linalg, _nn, _reductions

_FAMILIES = (_elementwise, _indexing, _layout, _linalg, _nn, _reductions)


def test_changed_registered_and_generated_schemas_have_an_explicit_audit():
    operators = set().union(*(family.OPS for family in _FAMILIES)) | {"Range"}
    changed = {
        (s.name, s.since_version)
        for s in onnx.defs.get_all_schemas_with_history()
        if s.domain == "" and s.name in operators and 25 <= s.since_version <= 28
    }
    assert changed == {
        ("Identity", 25),
        ("Cast", 25),
        ("Cast", 28),
        ("CastLike", 25),
        ("Constant", 25),
        ("ConstantOfShape", 25),
        ("Flatten", 25),
        ("Pad", 25),
        ("Reshape", 25),
        ("Shape", 25),
        ("Size", 25),
        ("Squeeze", 25),
        ("Transpose", 25),
        ("Unsqueeze", 25),
        ("QuantizeLinear", 25),
        ("QuantizeLinear", 28),
        ("DequantizeLinear", 25),
        ("DequantizeLinear", 28),
        ("Range", 27),
        ("DepthToSpace", 28),
        ("SpaceToDepth", 28),
        ("OneHot", 28),
        ("ReduceLogSum", 28),
        ("ReduceLogSumExp", 28),
        ("Mod", 28),
        ("BitShift", 28),
        ("Einsum", 28),
        ("Celu", 28),
    }
    # No registered/generated operator changed its input/output structure.
    # New attributes must be carried through or explicitly accounted for.
    for op, version in sorted(changed):
        old, new = onnx.defs.get_schema(op, 24), onnx.defs.get_schema(op, version)
        for side in ("inputs", "outputs"):
            assert [(p.name, p.option, p.type_str) for p in getattr(new, side)] == [
                (p.name, p.option, p.type_str) for p in getattr(old, side)
            ]
        added = set(new.attributes) - set(old.attributes)
        assert added == (
            {"mode"} if op == "SpaceToDepth" else {"stash_type"} if op == "Range" else set()
        )
        for name, old_attr in old.attributes.items():
            new_attr = new.attributes[name]
            assert (new_attr.type, new_attr.required, new_attr.default_value) == (
                old_attr.type,
                old_attr.required,
                old_attr.default_value,
            )
        old_types = {t.type_param_str: set(t.allowed_type_strs) for t in old.type_constraints}
        for constraint in new.type_constraints:
            removed = old_types[constraint.type_param_str] - set(constraint.allowed_type_strs)
            assert removed == (
                {"tensor(int32)", "tensor(int64)", "tensor(uint32)", "tensor(uint64)"}
                if op in ("ReduceLogSum", "ReduceLogSumExp")
                else set()
            )
    assert (
        onnx.defs.get_schema("SpaceToDepth", 28).attributes["mode"].default_value.s == b"DCR"
    )
    assert onnx.defs.get_schema("Range", 27).attributes["stash_type"].default_value.i == 1


@pytest.mark.parametrize("opset", [25, 26, 27, 28])
@pytest.mark.parametrize(
    "op,shapes,axes,attributes",
    [
        ("Tanh", [[2, 3]], [0], {}),
        ("Add", [[2, 3], [3]], [0, None], {}),
        ("MatMul", [[2, 3], [3, 4]], [0, None], {}),
        ("Einsum", [[2, 3], [3, 4]], [0, None], {"equation": "ij,jk->ik"}),
        ("ReduceMean", [[2, 3]], [0], {"keepdims": 0}),
        ("ReduceLogSum", [[2, 3]], [0], {"keepdims": 1}),
        ("ReduceLogSumExp", [[2, 3]], [0], {"keepdims": 0}),
        ("Flatten", [[2, 3]], [0], {"axis": 1}),
        ("Transpose", [[2, 3]], [0], {"perm": [1, 0]}),
        ("Cast", [[2, 3]], [0], {"to": int(ir.DataType.FLOAT)}),
        ("Conv", [[1, 1, 4, 4], [2, 1, 2, 2]], [0, None], {}),
        ("LayerNormalization", [[2, 4], [4], [4]], [0, None, None], {"axis": -1}),
        ("DepthToSpace", [[1, 8, 2, 2]], [0], {"blocksize": 2}),
    ],
)
def test_family_batching_matches_independent_reference_examples(
    opset, op, shapes, axes, attributes
):
    source = _model(op, shapes, attributes, opset=opset)
    feeds = {
        f"x{i}": _data([3, *shape] if axis is not None else shape)
        for i, (shape, axis) in enumerate(zip(shapes, axes, strict=True))
    }
    _equivalent(source, axes, feeds, dynamic=True)


@pytest.mark.parametrize("opset", [25, 26, 27, 28])
def test_layout_index_and_quantization_controls_remain_shared(opset):
    cases = [
        (
            "Reshape",
            [[6], [2]],
            [ir.DataType.FLOAT, ir.DataType.INT64],
            ir.DataType.FLOAT,
            {"x0": _data([3, 6]), "x1": np.array([2, 3], np.int64)},
            {},
        ),
        (
            "Gather",
            [[4, 2], [2]],
            [ir.DataType.FLOAT, ir.DataType.INT64],
            ir.DataType.FLOAT,
            {"x0": _data([3, 4, 2]), "x1": np.array([2, 0], np.int64)},
            {"axis": 0},
        ),
        (
            "Pad",
            [[2, 3], [4]],
            [ir.DataType.FLOAT, ir.DataType.INT64],
            ir.DataType.FLOAT,
            {"x0": _data([3, 2, 3]), "x1": np.array([1, 0, 0, 1], np.int64)},
            {},
        ),
        (
            "QuantizeLinear",
            [[2, 3], [], []],
            [ir.DataType.FLOAT, ir.DataType.FLOAT, ir.DataType.UINT8],
            ir.DataType.UINT8,
            {
                "x0": _data([3, 2, 3]),
                "x1": np.array(0.1, np.float32),
                "x2": np.array(2, np.uint8),
            },
            {},
        ),
        (
            "DequantizeLinear",
            [[2, 3], [], []],
            [ir.DataType.UINT8, ir.DataType.FLOAT, ir.DataType.UINT8],
            ir.DataType.FLOAT,
            {
                "x0": np.arange(18, dtype=np.uint8).reshape(3, 2, 3),
                "x1": np.array(0.1, np.float32),
                "x2": np.array(2, np.uint8),
            },
            {},
        ),
    ]
    for op, shapes, dtypes, output_dtype, feeds, attrs in cases:
        source = _model(
            op, shapes, attrs, opset=opset, input_dtypes=dtypes, dtype=output_dtype
        )
        _equivalent(source, [0, *([None] * (len(shapes) - 1))], feeds, dynamic=True)


@pytest.mark.parametrize("mode", [None, "DCR", "CRD"])
def test_space_to_depth_28_preserves_new_mode_and_native_n(mode):
    attributes = {"blocksize": 2, **({} if mode is None else {"mode": mode})}
    source = _model("SpaceToDepth", [[1, 2, 4, 4]], attributes, opset=28)
    data = np.arange(3 * 1 * 2 * 4 * 4, dtype=np.float32).reshape(3, 1, 2, 4, 4)
    mapped = vmap(source, [0])
    node = next(n for n in mapped.graph if n.op_type == "SpaceToDepth")
    assert node.attributes.get_string("mode", "DCR") == (mode or "DCR")
    actual = evaluator(mapped).run(None, {"x0": data})[0]
    split = data.reshape(3, 1, 2, 2, 2, 2, 2)
    permutation = (0, 1, 4, 6, 2, 3, 5) if mode != "CRD" else (0, 1, 2, 4, 6, 3, 5)
    expected = split.transpose(permutation).reshape(3, 1, 8, 2, 2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("direction", ["LEFT", "RIGHT"])
def test_signed_bitshift_28_sign_extension_and_out_of_range_counts(direction):
    source = _model(
        "BitShift",
        [[4], [4]],
        {"direction": direction},
        opset=28,
        input_dtypes=[ir.DataType.INT32] * 2,
        dtype=ir.DataType.INT32,
    )
    data = np.array([[-8, -8, 8, 8], [-1, -8, 8, -8], [-8, 2, 4, 8]], np.int32)
    counts = np.array([-1, 1, 32, 2], np.int32)
    _equivalent(source, [0, None], {"x0": data, "x1": counts}, dynamic=True)
    result = evaluator(vmap(source, [0, None])).run(None, {"x0": data, "x1": counts})[0]
    expected = (
        np.array([[-1, -4, 0, 2], [-1, -4, 0, -2], [-1, 1, 0, 2]], np.int32)
        if direction == "RIGHT"
        else np.array([[0, -16, 0, 32], [0, -16, 0, -32], [0, 4, 0, 32]], np.int32)
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("fmod", [0, 1])
def test_float_mod_28_quotient_rounding_and_signed_zero(fmod):
    source = _model("Mod", [[4], [4]], {"fmod": fmod}, opset=28)
    data = np.array([[-5.5, 5.5, -0.0, 0.0]], np.float32)
    divisor = np.array([2.0, -2.0, 2.0, -2.0], np.float32)
    result = evaluator(vmap(source, [0, None])).run(None, {"x0": data, "x1": divisor})[0]
    expected = (np.fmod if fmod else np.remainder)(data, divisor)
    np.testing.assert_array_equal(result, expected)
    if not fmod:
        np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))


def test_celu_28_extended_double_dtype():
    source = _model(
        "Celu",
        [[3]],
        {"alpha": 0.5},
        opset=28,
        input_dtypes=[ir.DataType.DOUBLE],
        dtype=ir.DataType.DOUBLE,
    )
    data = np.array([[-1.0, 0.0, 2.0], [1.0, -2.0, 0.5]])
    result = evaluator(vmap(source, [0])).run(None, {"x0": data})[0]
    np.testing.assert_allclose(result, np.where(data > 0, data, 0.5 * np.expm1(data / 0.5)))
    assert result.dtype == np.float64


@pytest.mark.parametrize("opset", [25, 26, 27, 28])
@pytest.mark.parametrize("family", ["elementwise", "linalg", "layout", "reductions"])
def test_jvp_vjp_and_hvp_family_semantics_at_new_opsets(opset, family):
    x = _value("x", [2, 2])
    tape = ir.tape.Tape()
    if family == "elementwise":
        y = tape.op("Exp", [tape.op("Mul", [x, x])])
    elif family == "linalg":
        y = tape.op("MatMul", [x, x])
    elif family == "layout":
        y = tape.op("Transpose", [tape.op("Tanh", [x])], {"perm": [1, 0]})
    else:
        y = tape.op("Softmax", [x], {"axis": 1})
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    source = _ad_model([x], tape, [loss], opset=opset)
    feeds = {"x": np.array([[0.2, 0.3], [0.4, 0.5]])}
    _products(source, feeds)
    transformed = _hvp_check(source, feeds)
    assert transformed.graph.opset_imports[""] == opset


@pytest.mark.parametrize("opset", [25, 26, 27, 28])
def test_serialized_per_example_gradients_and_runtime_axes_at_new_opsets(opset, tmp_path):
    source, parameters = _regressor(np.float64, opset, stored=True)
    derivative = grad(source, inputs=list(parameters), output="loss")
    mapped = vmap(derivative, [0, 0])
    path = tmp_path / "per_example.onnx"
    ir.save(mapped, path)
    mapped = ir.load(path)
    CheckerPass(full_check=True)(mapped)
    assert mapped.graph.opset_imports[""] == opset
    independent, batched = evaluator(derivative), evaluator(mapped)
    rng = np.random.default_rng(42)
    for batch, rows in ((1, 1), (3, 2), (7, 4)):
        feeds = {
            "x": rng.normal(size=(batch, rows, 3)),
            "target": rng.normal(size=(batch, rows, 2)),
        }
        result = batched.run(None, feeds)
        expected = [
            independent.run(None, {k: v[i] for k, v in feeds.items()}) for i in range(batch)
        ]
        for index, value in enumerate(result):
            np.testing.assert_allclose(
                value, np.stack([e[index] for e in expected]), atol=1e-12
            )
    # Generated integer Range controls retain their semantics at Range-27.
    hvp = jvp(derivative, inputs=list(parameters))
    for node in hvp.graph:
        if node.op_type == "Range":
            assert all(v is None or v.dtype in (None, ir.DataType.INT64) for v in node.inputs)


@pytest.mark.parametrize("opset", [25, 26, 27, 28])
def test_new_opsets_do_not_enable_unsupported_derivative_types_or_operators(opset):
    source = _model(
        "Identity",
        [[2]],
        opset=opset,
        input_dtypes=[ir.DataType.FLOAT16],
        dtype=ir.DataType.FLOAT16,
    )
    with pytest.raises((TypeError, ValueError), match=r"FLOAT|DOUBLE"):
        jvp(source, inputs=["x0"])
    source = _model("Sin", [[2]], opset=opset)
    source.graph[0].op_type = "Sign"
    with pytest.raises(NotImplementedError, match="Sign"):
        jvp(source)


@pytest.mark.parametrize("opset", [25, 26, 27, 28])
def test_serialized_jvp_over_vjp_keeps_existing_cotangent_fixed(opset, tmp_path):
    x = _value("x", [2])
    tape = ir.tape.Tape()
    source = _ad_model([x], tape, [tape.op("Sin", [x])], opset=opset)
    reverse = vjp(source)
    path = tmp_path / "vjp.onnx"
    ir.save(reverse, path)
    forward = jvp(ir.load(path), inputs=["x"])
    values, cotangent, direction = (
        np.array([0.3, 0.7]),
        np.array([2.0, -1.0]),
        np.array([0.4, 0.2]),
    )
    result = evaluator(forward).run(
        None,
        {
            "x": values,
            reverse.graph.inputs[-1].name: cotangent,
            forward.graph.inputs[-1].name: direction,
        },
    )
    np.testing.assert_allclose(result[-1], -np.sin(values) * cotangent * direction)
    assert forward.graph.opset_imports[""] == opset


@pytest.mark.parametrize("opset", [25, 26, 27, 28])
def test_optimized_runtime_capability_is_explicit_without_opset_downgrade(opset, tmp_path):
    source = _model("Mul", [[2], [2]], opset=opset)
    transformed = vmap(source, [0, None])
    path = tmp_path / "runtime.onnx"
    ir.save(transformed, path)
    CheckerPass(full_check=True)(ir.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Fail as error:
        ceiling = re.search(
            r"Current official support for domain ai.onnx is till opset (\d+)", str(error)
        )
        if ceiling is None or opset <= int(ceiling[1]):
            raise
        pytest.skip(
            f"ONNX Runtime {ort.__version__} supports through opset {ceiling[1]}, not {opset}; "
            "schema/reference tests cover this opset without downgrading it"
        )
    x, w = _data([3, 2]), _data([2])
    np.testing.assert_array_equal(session.run(None, {"x0": x, "x1": w})[0], x * w)
