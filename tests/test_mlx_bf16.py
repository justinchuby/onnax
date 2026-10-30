# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Portable graph/error reporting and opt-in native MLX BF16 evaluation."""

import os

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from examples.lora.bf16 import reference
from examples.lora.mlx_bf16 import discrepancy, gradient_model
from examples.lora.train import close, problem


def test_static_bf16_graph_and_independent_derivatives():
    w, c, x, y, _, _, params = problem()
    params["lora.B"] = np.full_like(params["lora.B"], 0.05)
    graph = gradient_model(w, c, batch=3)
    onnx.checker.check_model(ir.to_proto(graph), full_check=True)
    matrices = [
        node
        for node in graph.graph
        if node.op_type == "MatMul" and node.outputs[0].dtype == ir.DataType.BFLOAT16
    ]
    assert len(matrices) == 4
    assert all(v.dtype == ir.DataType.BFLOAT16 for node in matrices for v in node.inputs)
    assert not graph.functions
    assert not any(node.op_type in {"Shape", "Size"} for node in graph.graph)
    actual = ReferenceEvaluator(ir.to_proto(graph)).run(
        None, {"features": x[:3], "target": y[:3], **params}
    )
    expected = reference(x[:3], y[:3], params, w, c)
    for a, b in zip(actual, [*expected[:2], *expected[2].values()], strict=True):
        close(a, b)
    for name, value in (("base.W", w), ("base.C", c)):
        np.testing.assert_array_equal(
            graph.graph.initializers[name].const_value.numpy(), value
        )


def test_discrepancy_preserves_tolerance_and_scalar_indices():
    assert discrepancy(np.array(1), np.array(1))["mismatched"] == 0
    result = discrepancy(np.array(-0.1865234375), np.array(-0.1875))
    assert result["mismatched"] == 1
    assert result["max_abs"] == pytest.approx(0.0009765625)
    assert result["first"]["index"] == []
    assert discrepancy(np.array([1.00005]), np.array([1]))["mismatched"] == 0
    with pytest.raises(ValueError, match="shapes"):
        discrepancy(np.zeros(2), np.zeros(1))
    with pytest.raises(ValueError, match="finite"):
        discrepancy(np.array(np.nan), np.array(1))


@pytest.mark.parametrize("batch", [0, -1, True, 2.5])
def test_invalid_batch(batch):
    with pytest.raises(ValueError, match="positive integer"):
        gradient_model(*problem()[:2], batch=batch)


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1",
    reason="Set ONNAX_TEST_MLX_EP=1 with the MLX plugin on Apple Silicon",
)
def test_native_bf16_training_and_checkpoint_report(tmp_path):
    from examples.lora.mlx_bf16 import run

    report = run(tmp_path, inspect_products=True)
    assert report["executed_providers"] == ["MLXExecutionProvider"]
    assert report["steps"] == 200
    assert report["product_bounds"]["all_observed_products_pass"]
    assert all(len(row["product_bounds"]["products"]) == 4 for row in report["history"])
    assert (tmp_path / "checkpoint.npz").is_file()
    assert report["final_validation_loss"] < report["initial_validation_loss"] * 0.5
    passed = all(
        not metric["mismatched"]
        for row in report["history"]
        for group in ("same_parameter", "independent_trajectory", "trajectory_parameters")
        for metric in row[group].values()
    )
    assert report["strict_numerical_acceptance"] == passed
    if report["first_same_parameter_mismatch_step"] is not None:
        assert (tmp_path / "first-mismatch.npz").is_file()
        assert not report["strict_numerical_acceptance"]
