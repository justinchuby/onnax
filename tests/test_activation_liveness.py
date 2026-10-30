# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Hand-calculated allocation schedules and a recomputation experiment."""

import os

import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from examples.activation_liveness import analyze, tensor_bytes
from examples.mlp_bf16 import graph, problem
from examples.mlp_rematerialize import recompute_preactivation


def branch(*, alias=False, retain=False, bad_order=False):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[4])
    tape = ir.tape.Tape()
    a = tape.op("Neg", [x])
    b = tape.op("Identity" if alias else "Neg", [a])
    c = tape.op("Add", [a, b])
    d = tape.op("Neg", [c])
    for name, value in zip("abcd", [a, b, c, d], strict=True):
        value.name = name
        value.dtype = ir.DataType.FLOAT
        value.shape = ir.Shape([4])
    return ir.Model(
        ir.Graph(
            [x],
            [a, d] if retain else [d],
            nodes=(
                [tape.nodes[1], tape.nodes[0], *tape.nodes[2:]] if bad_order else tape.nodes
            ),
            opset_imports={"": 18},
        ),
        ir_version=9,
    )


def test_branch_last_use_and_output_retention():
    source = branch(retain=True)
    before = ir.to_proto(source).SerializeToString()
    result = analyze(source)
    assert result["persistent_external_bytes"] == 16
    assert [v["activation_bytes"] for v in result["timeline"]] == [16, 32, 48, 48]
    assert result["peak_activation_bytes"] == 48 and result["peak_total_tensor_bytes"] == 64
    assert next(v for v in result["values"] if v["name"] == "a")["last_use"] == 4
    assert ir.to_proto(source).SerializeToString() == before


def test_views_share_one_storage_group_and_extend_lifetime():
    model = branch(alias=True)
    assert analyze(model)["peak_activation_bytes"] == 48
    result = analyze(model, alias_views=True)
    assert result["peak_activation_bytes"] == 32
    values = {v["name"]: v for v in result["values"]}
    assert values["a"]["group"] == values["b"]["group"]
    assert [v["activation_bytes"] for v in result["timeline"]] == [16, 16, 32, 32]


@pytest.mark.parametrize(
    "dtype,shape,size",
    [
        (ir.DataType.FLOAT, [], 4),
        (ir.DataType.FLOAT, [0, 4], 0),
        (ir.DataType.BFLOAT16, [3], 6),
        (ir.DataType.INT4, [3], 2),
        (ir.DataType.BOOL, [3], 3),
    ],
)
def test_static_storage_sizes(dtype, shape, size):
    assert tensor_bytes(ir.val("v", dtype=dtype, shape=shape)) == size


@pytest.mark.parametrize(
    "dtype,shape",
    [
        (ir.DataType.STRING, [1]),
        (None, [1]),
        (ir.DataType.FLOAT, None),
        (ir.DataType.FLOAT, ["batch", 4]),
    ],
)
def test_unknown_storage_is_rejected(dtype, shape):
    with pytest.raises(ValueError, match=r"dtype|shape"):
        tensor_bytes(ir.val("v", dtype=dtype, shape=shape))


def test_reject_bad_order_and_subgraphs():
    bad = branch(bad_order=True)
    with pytest.raises(ValueError, match="topological"):
        analyze(bad)
    condition = ir.val("condition", dtype=ir.DataType.BOOL, shape=[])
    tape = ir.tape.Tape()
    result = tape.op(
        "If", [condition], {"then_branch": branch().graph, "else_branch": branch().graph}
    )
    result.dtype = ir.DataType.FLOAT
    result.shape = ir.Shape([4])
    subgraph = ir.Model(
        ir.Graph([condition], [result], nodes=tape.nodes, opset_imports={"": 18}), ir_version=9
    )
    with pytest.raises(ValueError, match="control flow"):
        analyze(subgraph)


def test_local_functions_need_inlining():
    from examples.onnxscript_autodiff import linear_regression

    with pytest.raises(ValueError, match="Inline local functions"):
        analyze(ir.from_proto(linear_regression.to_model_proto()))


def test_empty_graph_with_external_output():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[4])
    result = analyze(
        ir.Model(ir.Graph([x], [x], nodes=[], opset_imports={"": 18}), ir_version=9)
    )
    assert result["peak_activation_bytes"] == 0 and result["peak_node"] is None
    assert result["peak_total_tensor_bytes"] == 16


def test_recomputation_preserves_source_outputs_and_peak():
    source = graph(precision="fp32", batch=3)
    before = ir.to_proto(source).SerializeToString()
    candidate = recompute_preactivation(source)
    assert ir.to_proto(source).SerializeToString() == before
    assert len(candidate.graph) == len(source.graph) + 2
    onnx.checker.check_model(ir.to_proto(candidate), full_check=True)
    x, y, _, _, params = problem()
    feeds = {"features": x[:3], "targets": y[:3], **params}
    original = ReferenceEvaluator(ir.to_proto(source)).run(None, feeds)
    modified = ReferenceEvaluator(ir.to_proto(candidate)).run(None, feeds)
    for a, b in zip(original, modified, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert [v.name for v in source.graph.inputs] == [v.name for v in candidate.graph.inputs]
    assert [v.name for v in source.graph.outputs] == [v.name for v in candidate.graph.outputs]
    for alias in (False, True):
        assert (
            analyze(source, alias_views=alias)["peak_activation_bytes"]
            == analyze(candidate, alias_views=alias)["peak_activation_bytes"]
        )
    with pytest.raises(ValueError, match="FP32"):
        recompute_preactivation(graph(precision="bf16", batch=3))


@pytest.mark.skipif(os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX opt-in")
def test_native_recompute_training_and_reload(tmp_path):
    from examples.mlp_rematerialize import run

    report = run(tmp_path)
    assert report["steps"] == report["independent_gradients_checked"] == 150
    assert report["all_updates_byte_equal"] and report["checkpoint_replay"]
    assert report["final_training_loss"] < report["initial_loss"] * 0.5
    assert report["native_final_validation_loss"] < report["initial_validation_loss"] * 0.5
    for label in ("baseline", "candidate"):
        assert report["measurements"][label]["samples"] == 100
